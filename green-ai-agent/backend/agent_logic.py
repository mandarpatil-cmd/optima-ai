
from typing import Dict, List, Tuple, Any
import numpy as np
import time
import hashlib
from .energy_tracker import EnhancedEnergyTracker
from .email_models import ModelManager
from .config import *
import sqlite3
import json

class IntelligentEmailAgent:
    def __init__(self):
        self.model_manager = ModelManager()
        self.model_manager.initialize_models()
        self.performance_history = []
        
    def classify_email(self, email_text: str, user_preferences: Dict = None) -> Dict[str, Any]:
        """Main classification method with intelligent model selection"""
        
        # Step 1: Quick assessment with light model
        light_result = self._classify_with_model(email_text, model_type="light")
        
        # Step 2: Decide if we need heavy model
        needs_heavy_model = self._should_use_heavy_model(
            light_result, 
            email_text, 
            user_preferences or {}
        )
        
        if needs_heavy_model:
            heavy_result = self._classify_with_model(email_text, model_type="heavy")
            final_result = self._compare_and_select_result(light_result, heavy_result)
            final_result["escalated"] = True
        else:
            final_result = light_result
            final_result["escalated"] = False
            
        # Step 3: Update learning and return
        self._update_performance_history(final_result)
        return final_result
    
    def _classify_with_model(self, email_text: str, model_type: str) -> Dict[str, Any]:
        """Classify email with specified model type using the model wrapper."""
        tracker = EnhancedEnergyTracker()
        tracker.start_tracking()

        if model_type == "light":
            model = self.model_manager.light_model
            model_name = LIGHT_MODEL
        else:
            model = self.model_manager.heavy_model
            model_name = HEAVY_MODEL

        # Indicate fallback in model_used if we don't have a real model_name attribute
        model_used_display = model_name if hasattr(model, "model_name") else f"{model_name} (fallback)"

        # Perform classification via the model wrapper so outputs are
        # normalized to our EMAIL_CATEGORIES regardless of backend.
        predictions = model.predict(email_text)

        # Ensure we have a list of dicts with label/score
        if not predictions or not isinstance(predictions, list):
            predictions = []
        # Defensive sort and normalization
        predictions = [
            {"label": str(p.get("label", "unknown")), "score": float(p.get("score", 0.0))}
            for p in predictions
        ]
        predictions.sort(key=lambda x: x["score"], reverse=True)

        top_pred = predictions[0] if predictions else {"label": "personal", "score": 0.5}

        # Stop tracking
        energy_metrics = tracker.stop_tracking()

        return {
            "email_text": email_text,
            "predicted_category": top_pred['label'],
            "confidence": float(top_pred['score']),
            "all_predictions": [
                {"category": p['label'], "confidence": float(p['score'])}
                for p in predictions
            ],
            "model_used": model_used_display,
            "model_type": model_type,
            "energy_metrics": energy_metrics,
            "timestamp": time.time()
        }
    
    def _should_use_heavy_model(self, light_result: Dict, email_text: str, user_prefs: Dict) -> bool:
        """Intelligent decision on whether to use heavy model"""
        
        # Rule 1: Low confidence threshold
        if light_result["confidence"] < DEFAULT_CONFIDENCE_THRESHOLD:
            return True
            
        # Rule 2: User explicitly wants high accuracy
        if user_prefs.get("priority") == "accuracy":
            return True
            
        # Rule 3: Important email categories (work, support)
        important_categories = ["work", "support"]
        if light_result["predicted_category"] in important_categories:
            return True
            
        # Rule 4: Complex email (long text, multiple topics)
        if len(email_text.split()) > 100:  # Long emails
            return True
            
        # Rule 5: Historical performance suggests heavy model needed
        if self._historical_analysis_suggests_heavy():
            return True
            
        return False
    
    def _compare_and_select_result(self, light_result: Dict, heavy_result: Dict) -> Dict[str, Any]:
        """Compare light and heavy results, select the best"""
        
        confidence_diff = heavy_result["confidence"] - light_result["confidence"]
        energy_diff = (heavy_result["energy_metrics"]["co2_emissions_g"] - 
                      light_result["energy_metrics"]["co2_emissions_g"])
        
        # If heavy model is significantly more confident and energy cost is acceptable
        if confidence_diff > 0.1 and energy_diff < 50:  # Max 50g more CO2
            selected_result = heavy_result
            selected_result["selection_reason"] = "Heavy model more confident"
        # If light model is almost as good, prefer it for energy savings
        elif confidence_diff < 0.05:
            selected_result = light_result
            selected_result["selection_reason"] = "Light model sufficient, energy saved"
        else:
            selected_result = heavy_result
            selected_result["selection_reason"] = "Heavy model for accuracy"
            
        # Add comparison metrics
        selected_result["model_comparison"] = {
            "light_confidence": light_result["confidence"],
            "heavy_confidence": heavy_result["confidence"],
            "confidence_gain": confidence_diff,
            "energy_cost_light": light_result["energy_metrics"]["co2_emissions_g"],
            "energy_cost_heavy": heavy_result["energy_metrics"]["co2_emissions_g"],
            "energy_overhead": energy_diff
        }
        
        return selected_result
    
    def _historical_analysis_suggests_heavy(self) -> bool:
        """Analyze historical performance to suggest model choice"""
        if len(self.performance_history) < 10:
            return False
            
        recent_light_accuracy = np.mean([
            h["confidence"] for h in self.performance_history[-10:] 
            if h["model_type"] == "light"
        ])
        
        return recent_light_accuracy < 0.8  # If recent light model performance is poor
    
    def _update_performance_history(self, result: Dict):
        """Update performance history for learning"""
        self.performance_history.append({
            "timestamp": result["timestamp"],
            "model_type": result["model_type"],
            "confidence": result["confidence"],
            "energy_cost": result["energy_metrics"]["co2_emissions_g"],
            "escalated": result.get("escalated", False)
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

class AgentOrchestrator:
    """Main orchestrator class that manages the entire workflow"""
    
    def __init__(self):
        self.email_agent = IntelligentEmailAgent()
        self.db_path = DB_PATH
        
    def process_email(self, email_data: Dict) -> Dict[str, Any]:
        """Complete email processing pipeline"""
        
        # Extract email content
        email_text = email_data.get("text", "")
        user_preferences = email_data.get("preferences", {})
        
        # Classify using intelligent agent
        classification_result = self.email_agent.classify_email(email_text, user_preferences)
        
        # Log to database
        self._log_classification(classification_result)
        
        # Add suggestions and insights
        enhanced_result = self._add_insights(classification_result)
        
        return enhanced_result
    
    def _log_classification(self, result: Dict):
        """Log classification result to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Use proper hash of email text
                email_hash = hashlib.sha256(result["email_text"].encode()).hexdigest()[:16]
                
                conn.execute("""
                    INSERT INTO email_classifications 
                    (timestamp, email_hash, predicted_category, confidence, model_used, 
                     escalated, co2_emissions_g, processing_time, energy_efficiency_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result["timestamp"],
                    email_hash,
                    result["predicted_category"],
                    result["confidence"],
                    result["model_used"],
                    result.get("escalated", False),
                    result["energy_metrics"]["co2_emissions_g"],
                    result["energy_metrics"]["processing_time_seconds"],
                    result["energy_metrics"]["energy_efficiency_score"]
                ))
        except Exception as e:
            print(f"Database logging error: {e}")
    
    def _add_insights(self, result: Dict) -> Dict[str, Any]:
        """Add AI-driven insights and suggestions"""
        
        insights = {
            "environmental_impact": self._analyze_environmental_impact(result),
            "accuracy_assessment": self._analyze_accuracy(result),
            "suggestions": self._generate_suggestions(result)
        }
        
        result["ai_insights"] = insights
        return result
    
    def _analyze_environmental_impact(self, result: Dict) -> Dict[str, Any]:
        """Analyze environmental impact"""
        co2_g = result["energy_metrics"]["co2_emissions_g"]
        
        # Compare to benchmarks
        daily_emails = 50  # Average person
        yearly_projection = co2_g * daily_emails * 365
        
        return {
            "co2_this_classification": co2_g,
            "yearly_projection_g": yearly_projection,
            "equivalent_km_driven": yearly_projection / 404,  # g CO2 per km car
            "impact_level": "low" if co2_g < 1 else "medium" if co2_g < 5 else "high"
        }
    
    def _analyze_accuracy(self, result: Dict) -> Dict[str, Any]:
        """Analyze accuracy and confidence"""
        confidence = result["confidence"]
        
        return {
            "confidence_level": "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low",
            "accuracy_assessment": "Very reliable" if confidence > 0.95 else "Reliable" if confidence > 0.8 else "Moderate confidence",
            "should_review": confidence < 0.7
        }
    
    def _generate_suggestions(self, result: Dict) -> List[str]:
        """Generate actionable suggestions"""
        suggestions = []
        
        if result.get("escalated"):
            suggestions.append("💡 Heavy model was used for higher accuracy")
            
        if result["energy_metrics"]["co2_emissions_g"] > 5:
            suggestions.append("🌱 Consider using light model for similar emails to save energy")
            
        if result["confidence"] < 0.8:
            suggestions.append("⚠️ Low confidence - you may want to verify this classification")
            
        category = result["predicted_category"]
        if category == "spam":
            suggestions.append("🛡️ Consider adding sender to block list")
        elif category == "work":
            suggestions.append("💼 This looks like a work email - prioritize accordingly")
        elif category == "support":
            suggestions.append("🎧 This appears to be a support request - respond promptly")
            
        return suggestions




