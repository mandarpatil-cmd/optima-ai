import logging
from typing import Dict, Any, List

from .config import LIGHT_MODEL, HEAVY_MODEL, EMAIL_CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make transformers optional; fall back if missing or fails
try:
    from transformers import pipeline  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception as e:  # pragma: no cover
    TRANSFORMERS_AVAILABLE = False
    pipeline = None  # type: ignore
    logger.warning(f"Transformers not available; using fallback models only: {e}")


class EmailModel:
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.pipeline = None
        self.is_loaded = False

    def load_model(self):
        """Load the model pipeline or raise to trigger fallback."""
        logger.info(f"Loading {self.model_type} model: {self.model_name}")
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not installed")

        try:
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
                device=-1,  # CPU
            )
            self.is_loaded = True
            logger.info(f"{self.model_type.capitalize()} model loaded successfully")
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"Failed to init pipeline for {self.model_name}: {e}")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError(f"Model {self.model_name} is not loaded")

        # Ask for all scores when available and normalize shape.
        try:
            raw_predictions = self.pipeline(text, return_all_scores=True)
        except TypeError:
            # Some pipelines may not accept return_all_scores; fall back.
            raw_predictions = self.pipeline(text)

        # Flatten list-of-lists to list-of-dicts
        if isinstance(raw_predictions, list) and raw_predictions and isinstance(raw_predictions[0], list):
            raw_predictions = raw_predictions[0]

        # If labels don't match our email categories (e.g., POSITIVE/NEGATIVE),
        # use a heuristic mapping over the raw text to produce email categories.
        if raw_predictions and isinstance(raw_predictions[0], dict):
            first_label = str(raw_predictions[0].get("label", ""))
        else:
            first_label = ""

        if first_label not in EMAIL_CATEGORIES:
            return self._heuristic_email_category_scores(text)

        # If raw labels are already the email categories, pass through.
        scores_by_label = {str(p.get("label")): float(p.get("score", 0.0)) for p in raw_predictions if isinstance(p, dict)}
        preds = [{"label": cat, "score": scores_by_label.get(cat, 0.0)} for cat in EMAIL_CATEGORIES]
        # Normalize and sort
        total = sum(p["score"] for p in preds) or 1.0
        for p in preds:
            p["score"] /= total
        return sorted(preds, key=lambda x: x["score"], reverse=True)

    def _map_to_email_categories(self, raw_predictions: List[Dict[str, Any]]):
        """Legacy index-based mapper (kept for compatibility)."""
        email_predictions = []
        for i, category in enumerate(EMAIL_CATEGORIES):
            base = 1.0 / max(1, len(EMAIL_CATEGORIES))
            if i < len(raw_predictions) and isinstance(raw_predictions[i], dict):
                score = float(raw_predictions[i].get("score", 0.0)) * 0.7 + base * 0.3
            else:
                score = base
            email_predictions.append({"label": category, "score": score})

        total = sum(p["score"] for p in email_predictions) or 1.0
        for p in email_predictions:
            p["score"] /= total
        return sorted(email_predictions, key=lambda x: x["score"], reverse=True)

    def _heuristic_email_category_scores(self, text: str) -> List[Dict[str, Any]]:
        """Keyword-based heuristic classifier to map text to EMAIL_CATEGORIES."""
        text_lower = str(text).lower()
        scores = {c: 0.1 for c in EMAIL_CATEGORIES}

        if any(w in text_lower for w in ["winner", "congratulations", "claim", "$$$", "urgent", "click here", "click now"]):
            scores["spam"] = 0.8
        elif any(w in text_lower for w in ["meeting", "report", "project", "deadline", "team", "schedule", "q3", "q4"]):
            scores["work"] = 0.7
        elif any(w in text_lower for w in ["sale", "discount", "offer", "% off", "limited time"]):
            scores["promotions"] = 0.75
        elif any(w in text_lower for w in ["help", "support", "problem", "issue", "account", "password", "reset"]):
            scores["support"] = 0.7
        elif any(w in text_lower for w in ["newsletter", "weekly", "update"]):
            scores["newsletter"] = 0.6
        else:
            scores["personal"] = 0.6

        total = sum(scores.values()) or 1.0
        preds = [{"label": cat, "score": scores.get(cat, 0.0) / total} for cat in EMAIL_CATEGORIES]
        return sorted(preds, key=lambda x: x["score"], reverse=True)


class ModelManager:
    def __init__(self):
        self.light_model = EmailModel(LIGHT_MODEL, "light")
        self.heavy_model = EmailModel(HEAVY_MODEL, "heavy")
        self.models_initialized = False

    def initialize_models(self):
        logger.info("Initializing email classification models...")

        # Light model
        try:
            self.light_model.load_model()
        except Exception as e:
            logger.warning(f"Light model failed to load: {e}")
            self.light_model = self._create_fallback_model("light")

        # Heavy model
        try:
            self.heavy_model.load_model()
        except Exception as e:
            logger.warning(f"Heavy model failed to load: {e}")
            self.heavy_model = self._create_fallback_model("heavy")

        self.models_initialized = True
        logger.info("Model initialization completed")

    def _create_fallback_model(self, model_type: str):
        class FallbackModel:
            def __init__(self, model_type: str):
                self.model_type = model_type
                self.is_loaded = True
                self.pipeline = self

            def __call__(self, text, return_all_scores=True):
                text_lower = str(text).lower()
                scores = {c: 0.1 for c in EMAIL_CATEGORIES}

                if any(w in text_lower for w in ["winner", "congratulations", "claim", "$$$", "urgent"]):
                    scores["spam"] = 0.8
                elif any(w in text_lower for w in ["meeting", "report", "project", "deadline", "team"]):
                    scores["work"] = 0.7
                elif any(w in text_lower for w in ["sale", "discount", "offer", "% off", "limited time"]):
                    scores["promotions"] = 0.75
                elif any(w in text_lower for w in ["help", "support", "problem", "issue", "account"]):
                    scores["support"] = 0.7
                elif any(w in text_lower for w in ["newsletter", "weekly", "update"]):
                    scores["newsletter"] = 0.6
                else:
                    scores["personal"] = 0.6

                total = sum(scores.values()) or 1.0
                for k in scores:
                    scores[k] /= total

                predictions = [
                    {"label": cat, "score": scores.get(cat, 0.0)} for cat in EMAIL_CATEGORIES
                ]
                return sorted(predictions, key=lambda x: x["score"], reverse=True)

        logger.info(f"Created fallback model for {model_type}")
        return FallbackModel(model_type)

    def get_model_info(self):
        return {
            "light_model": {
                "name": getattr(self.light_model, "model_name", "fallback"),
                "loaded": getattr(self.light_model, "is_loaded", True),
                "type": "light",
            },
            "heavy_model": {
                "name": getattr(self.heavy_model, "model_name", "fallback"),
                "loaded": getattr(self.heavy_model, "is_loaded", True),
                "type": "heavy",
            },
            "initialized": self.models_initialized,
        }
