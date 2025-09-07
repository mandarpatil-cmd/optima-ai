"""
Continuous learning system for the email classifier
"""
import sqlite3
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json

class ContinuousLearningSystem:
    def __init__(self, db_path, min_feedback_samples=10):
        self.db_path = db_path
        self.min_samples = min_feedback_samples
        self.learning_metrics = {}
        
    def log_user_feedback(self, classification_id: int, correct_category: str, user_confidence: float):
        """Log user feedback for continuous improvement"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_feedback (
                    classification_id, correct_category, user_confidence, 
                    feedback_timestamp
                ) VALUES (?, ?, ?, ?)
            """, (classification_id, correct_category, user_confidence, datetime.now().timestamp()))
    
    def analyze_model_performance(self) -> Dict[str, float]:
        """Analyze recent model performance and suggest improvements"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get recent classifications with feedback
            cursor.execute("""
                SELECT e.predicted_category, e.confidence, e.model_used,
                       f.correct_category, f.user_confidence
                FROM email_classifications e
                JOIN user_feedback f ON e.id = f.classification_id
                WHERE e.timestamp > ?
            """, (datetime.now().timestamp() - 7*24*3600,))  # Last 7 days
            
            results = cursor.fetchall()
            
        if len(results) < self.min_samples:
            return {"insufficient_data": True, "sample_count": len(results)}
        
        # Calculate performance metrics
        correct_predictions = sum(1 for r in results if r[0] == r[3])
        accuracy = correct_predictions / len(results)
        
        # Analyze by model type
        light_results = [r for r in results if "distil" in r[2].lower()]
        heavy_results = [r for r in results if "distil" not in r[2].lower()]
        
        light_accuracy = sum(1 for r in light_results if r[0] == r[3]) / len(light_results) if light_results else 0
        heavy_accuracy = sum(1 for r in heavy_results if r[0] == r[3]) / len(heavy_results) if heavy_results else 0
        
        # Confidence calibration
        overconfident_cases = sum(1 for r in results if r[1] > 0.9 and r[0] != r[3])
        underconfident_cases = sum(1 for r in results if r[1] < 0.7 and r[0] == r[3])
        
        metrics = {
            "overall_accuracy": accuracy,
            "light_model_accuracy": light_accuracy,
            "heavy_model_accuracy": heavy_accuracy,
            "overconfident_rate": overconfident_cases / len(results),
            "underconfident_rate": underconfident_cases / len(results),
            "sample_count": len(results),
            "improvement_suggestions": self._generate_improvement_suggestions(results)
        }
        
        self.learning_metrics = metrics
        return metrics
    
    def _generate_improvement_suggestions(self, results) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Analyze error patterns
        errors_by_category = {}
        for result in results:
            if result[0] != result[3]:  # Incorrect prediction
                predicted, actual = result[0], result[3]
                key = f"{predicted} -> {actual}"
                errors_by_category[key] = errors_by_category.get(key, 0) + 1
        
        # Common confusion patterns
        if errors_by_category:
            most_common_error = max(errors_by_category.items(), key=lambda x: x[1])
            suggestions.append(f"Most common error: {most_common_error[0]} ({most_common_error[1]} cases)")
            
            if most_common_error[1] >= 3:
                suggestions.append("Consider retraining models with more examples of these categories")
        
        # Model selection suggestions
        light_accuracy = sum(1 for r in results if "distil" in r[2].lower() and r[0] == r[3])
        light_total = sum(1 for r in results if "distil" in r[2].lower())
        
        if light_total > 0 and light_accuracy / light_total < 0.8:
            suggestions.append("Light model accuracy is low - consider adjusting escalation threshold")
        
        return suggestions

    def recommend_threshold_adjustment(self) -> Dict[str, float]:
        """Recommend optimal confidence thresholds based on performance data"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT confidence, (predicted_category = correct_category) as correct
                FROM email_classifications e
                JOIN user_feedback f ON e.id = f.classification_id
                WHERE e.timestamp > ?
                ORDER BY confidence
            """, (datetime.now().timestamp() - 14*24*3600,))  # Last 14 days
            
            results = cursor.fetchall()
        
        if len(results) < 20:
            return {"insufficient_data": True}
        
        # Find optimal threshold using different criteria
        thresholds = np.arange(0.5, 0.99, 0.02)
        best_threshold = 0.85
        best_score = 0
        
        for threshold in thresholds:
            # Calculate metrics at this threshold
            would_escalate = sum(1 for r in results if r[0] < threshold)
            high_conf_correct = sum(1 for r in results if r[0] >= threshold and r[1])
            high_conf_total = sum(1 for r in results if r[0] >= threshold)
            
            if high_conf_total > 0:
                precision = high_conf_correct / high_conf_total
                escalation_rate = would_escalate / len(results)
                
                # Score balances precision and escalation rate
                score = precision * (1 - escalation_rate * 0.3)  # Penalize high escalation
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        return {
            "recommended_threshold": best_threshold,
            "expected_precision": best_score,
            "current_threshold": 0.85  # From config
        }

# NOTE: This module is kept as a pure library.
# The HTTP endpoints for feedback and learning insights are defined in
# backend/main.py to avoid duplicate route registration and undefined
# references to FastAPI app/DB_PATH here.

