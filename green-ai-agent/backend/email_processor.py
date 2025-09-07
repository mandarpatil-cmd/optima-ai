"""
Advanced email preprocessing and feature extraction
"""
import re
import html
from typing import Dict, List, Tuple, Any
from email.parser import Parser
from email.policy import default
import base64

from .agent_logic import IntelligentEmailAgent


class EmailProcessor:
    def __init__(self):
        self.spam_keywords = [
            "congratulations",
            "winner",
            "click here",
            "limited time",
            "act now",
            "urgent",
            "free",
            "guarantee",
            "$$",
        ]

    def clean_email_text(self, raw_email: str) -> str:
        """Clean and normalize email text"""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", raw_email)

        # Decode HTML entities
        text = html.unescape(text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove email headers artifacts
        text = re.sub(r"^(From|To|Subject|Date|Reply-To):.*", "", text, flags=re.MULTILINE)

        # Remove quoted text (replies)
        text = re.sub(r"^>.*", "", text, flags=re.MULTILINE)

        # Remove signatures (common patterns)
        text = re.sub(r"\n--\s*\n.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\nBest regards.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\nSincerely.*", "", text, flags=re.DOTALL)

        return text.strip()

    def extract_features(self, email_text: str, subject: str = "", sender: str = "") -> Dict:
        """Extract features that help with classification"""

        features: Dict[str, Any] = {}

        # Basic text features
        features["word_count"] = len(email_text.split())
        features["char_count"] = len(email_text)
        features["sentence_count"] = len(re.findall(r"[.!?]+", email_text))

        # Sender analysis
        features["sender_domain"] = sender.split("@")[-1] if "@" in sender else ""
        features["is_gmail"] = "@gmail.com" in sender.lower()
        features["is_corporate"] = not any(
            domain in sender.lower() for domain in ["gmail", "yahoo", "hotmail", "outlook"]
        )

        # Subject analysis
        features["subject_length"] = len(subject)
        features["subject_has_urgent"] = any(
            word in subject.lower() for word in ["urgent", "asap", "immediate", "critical"]
        )
        features["subject_all_caps"] = subject.isupper() and len(subject) > 5

        # Content analysis
        features["has_links"] = bool(re.search(r"http[s]?://", email_text))
        features["link_count"] = len(re.findall(r"http[s]?://", email_text))
        features["has_attachments"] = "attached" in email_text.lower() or "attachment" in email_text.lower()

        # Spam indicators
        features["spam_keyword_count"] = sum(
            1 for keyword in self.spam_keywords if keyword.lower() in email_text.lower()
        )
        features["excessive_punctuation"] = len(re.findall(r"[!]{2,}", email_text)) > 0
        features["has_money_symbols"] = bool(re.search(r"[\$€¥£]", email_text))

        # Sentiment indicators
        features["has_greeting"] = any(
            greeting in email_text.lower()[:100]
            for greeting in ["hello", "hi", "dear", "greetings"]
        )
        features["has_closing"] = any(
            closing in email_text.lower()[-100:] for closing in ["regards", "sincerely", "thanks", "best"]
        )

        # Business indicators
        features["has_meeting_words"] = any(
            word in email_text.lower() for word in ["meeting", "conference", "call", "schedule"]
        )
        features["has_project_words"] = any(
            word in email_text.lower() for word in ["project", "deadline", "milestone", "deliverable"]
        )

        return features

    def parse_email_message(self, raw_email: str) -> Dict:
        """Parse raw email message and extract components"""

        try:
            # Try to parse as email message
            parser = Parser(policy=default)
            msg = parser.parsestr(raw_email)

            return {
                "subject": msg.get("Subject", ""),
                "sender": msg.get("From", ""),
                "recipient": msg.get("To", ""),
                "date": msg.get("Date", ""),
                "body": self._extract_body(msg),
                "has_attachments": len(msg.get_payload()) > 1 if msg.is_multipart() else False,
            }

        except Exception:
            # Fallback: treat as plain text
            return {
                "subject": "",
                "sender": "",
                "recipient": "",
                "date": "",
                "body": self.clean_email_text(raw_email),
                "has_attachments": False,
            }

    def _extract_body(self, msg) -> str:
        """Extract body text from email message"""

        if msg.is_multipart():
            body_parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body_parts.append(part.get_content())
            return " ".join(body_parts)
        else:
            return msg.get_content()

    def enhance_classification_input(self, email_data: Dict) -> Dict:
        """Enhance email data with processed features for better classification"""

        # Parse email if needed
        if "body" not in email_data:
            parsed = self.parse_email_message(email_data.get("text", ""))
            email_data.update(parsed)

        # Clean the text
        clean_text = self.clean_email_text(email_data.get("body", email_data.get("text", "")))

        # Extract features
        features = self.extract_features(
            clean_text,
            email_data.get("subject", ""),
            email_data.get("sender", ""),
        )

        # Create enhanced input
        enhanced_text = self._create_enhanced_text(clean_text, email_data, features)

        return {
            **email_data,
            "original_text": email_data.get("text", ""),
            "clean_text": clean_text,
            "enhanced_text": enhanced_text,
            "features": features,
            "preprocessing_applied": True,
        }

    def _create_enhanced_text(self, clean_text: str, email_data: Dict, features: Dict) -> str:
        """Create enhanced text input for the model with metadata"""

        # Add context markers that help the model
        enhanced_parts: List[str] = []

        # Add subject context
        if email_data.get("subject"):
            enhanced_parts.append(f"[SUBJECT] {email_data['subject']}")

        # Add sender context if corporate
        if features.get("is_corporate"):
            enhanced_parts.append("[CORPORATE_SENDER]")

        # Add urgency markers
        if features.get("subject_has_urgent"):
            enhanced_parts.append("[URGENT]")

        # Add the main content
        enhanced_parts.append(f"[CONTENT] {clean_text}")

        # Add feature markers for obvious cases
        if features.get("spam_keyword_count", 0) > 2:
            enhanced_parts.append("[PROMOTIONAL_CONTENT]")

        if features.get("has_meeting_words"):
            enhanced_parts.append("[MEETING_RELATED]")

        return " ".join(enhanced_parts)


# Integration with main agent
class EnhancedEmailAgent(IntelligentEmailAgent):
    """Enhanced version with preprocessing"""

    def __init__(self):
        super().__init__()
        self.processor = EmailProcessor()

    def classify_email(self, email_text: str, user_preferences: Dict = None) -> Dict[str, Any]:
        """Enhanced classification with preprocessing"""

        # Preprocess email
        email_data = {"text": email_text}
        enhanced_data = self.processor.enhance_classification_input(email_data)

        # Use enhanced text for classification
        enhanced_text = enhanced_data["enhanced_text"]

        # Run original classification logic
        result = super().classify_email(enhanced_text, user_preferences)

        # Add preprocessing info to result
        result["preprocessing"] = {
            "features_extracted": enhanced_data["features"],
            "text_cleaned": True,
            "enhancement_applied": True,
            "original_length": len(email_text),
            "processed_length": len(enhanced_text),
        }

        return result

