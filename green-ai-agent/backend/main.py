from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import time
import sqlite3
import hashlib
from datetime import datetime

# Absolute imports for backend modules
from backend.config import DB_PATH, EMAIL_CATEGORIES

# Initialize the intelligent agent (with error handling)
try:
    # Import inside try so missing module doesn't crash startup
    from backend.agent_logic import AgentOrchestrator  # type: ignore
    orchestrator = AgentOrchestrator()
    ORCHESTRATOR_READY = True
except Exception as e:
    print(f"Warning: Could not initialize orchestrator: {e}")
    print("Starting in basic mode (keyword fallback only)...")
    orchestrator = None
    ORCHESTRATOR_READY = False

app = FastAPI(title="Green AI Email Classification API", version="2.0")


# -------------------------
# Request/Response Models
# -------------------------
class EmailClassificationRequest(BaseModel):
    text: str
    subject: Optional[str] = ""
    sender: Optional[str] = ""
    preferences: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = "default"


class Prediction(BaseModel):
    category: str
    confidence: float


class EmailClassificationResponse(BaseModel):
    predicted_category: str
    confidence: float
    all_predictions: List[Prediction]
    model_used: str
    escalated: bool
    energy_metrics: Dict[str, Any]
    ai_insights: Dict[str, Any]
    processing_time: float
    timestamp: float


# -------------------------
# Simple fallback classifier
# -------------------------
def simple_classification(text: str) -> Dict[str, Any]:
    """Simple keyword-based classification as fallback"""
    text_lower = text.lower()

    if any(word in text_lower for word in ["congratulations", "winner", "claim", "$$$", "urgent", "click now"]):
        category, confidence = "spam", 0.8
    elif any(word in text_lower for word in ["meeting", "report", "project", "deadline", "team"]):
        category, confidence = "work", 0.7
    elif any(word in text_lower for word in ["offer", "sale", "discount", "% off", "limited time"]):
        category, confidence = "promotions", 0.75
    elif any(word in text_lower for word in ["help", "support", "problem", "issue", "account"]):
        category, confidence = "support", 0.7
    elif any(word in text_lower for word in ["newsletter", "weekly", "news", "update"]):
        category, confidence = "newsletter", 0.6
    else:
        category, confidence = "personal", 0.5

    return {
        "predicted_category": category,
        "confidence": confidence,
        "all_predictions": [
            {"category": category, "confidence": confidence},
            {"category": "personal", "confidence": 0.3},
        ],
        "model_used": "keyword_fallback",
        "escalated": False,
        "energy_metrics": {
            "co2_emissions_g": 0.001,
            "co2_emissions_kg": 0.000001,
            "processing_time_seconds": 0.1,
            "memory_used_gb": 0.001,
            "cpu_utilization_start": 5.0,
            "cpu_utilization_end": 5.0,
            "gpu_metrics": {},
            "energy_efficiency_score": 0.1,
        },
        "ai_insights": {
            "environmental_impact": {
                "co2_this_classification": 0.001,
                "yearly_projection_g": 18.25,
                "equivalent_km_driven": 0.045,
                "impact_level": "low",
            },
            "accuracy_assessment": {
                "confidence_level": "low" if confidence < 0.7 else "medium",
                "accuracy_assessment": "Basic keyword matching",
                "should_review": True,
            },
            "suggestions": [
                "System is using keyword-based fallback",
                "AI models are loading - please wait and try again",
            ],
        },
        "timestamp": time.time(),
        "email_text": text,
    }


# -------------------------
# API Routes
# -------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "orchestrator_ready": ORCHESTRATOR_READY,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
    }


@app.post("/classify-email", response_model=EmailClassificationResponse)
def classify_email(request: EmailClassificationRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Email text is required")

    if len(request.text) > 10000:
        raise HTTPException(status_code=413, detail="Email text too long (>10000 chars)")

    start_time = time.time()

    try:
        if ORCHESTRATOR_READY and orchestrator:
            email_data = {
                "text": request.text,
                "subject": request.subject,
                "sender": request.sender,
                "preferences": request.preferences,
                "user_id": request.user_id,
            }
            result = orchestrator.process_email(email_data)
        else:
            result = simple_classification(request.text)

        processing_time = time.time() - start_time
        log_classification(result, processing_time)

        return EmailClassificationResponse(
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            all_predictions=result["all_predictions"],
            model_used=result["model_used"],
            escalated=result["escalated"],
            energy_metrics=result["energy_metrics"],
            ai_insights=result["ai_insights"],
            processing_time=processing_time,
            timestamp=result["timestamp"],
        )

    except Exception as e:
        print(f"Classification error: {e}")
        result = simple_classification(request.text)
        processing_time = time.time() - start_time

        return EmailClassificationResponse(
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            all_predictions=result["all_predictions"],
            model_used="fallback_error",
            escalated=False,
            energy_metrics=result["energy_metrics"],
            ai_insights=result["ai_insights"],
            processing_time=processing_time,
            timestamp=result["timestamp"],
        )


# Backward compatibility endpoint: old clients used /classify
@app.post("/classify", response_model=EmailClassificationResponse)
def classify_compat(request: EmailClassificationRequest):
    return classify_email(request)


# -------------------------
# DB Logging
# -------------------------
def log_classification(result: Dict, processing_time: float):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            email_hash = hashlib.sha256(result.get("email_text", "unknown").encode()).hexdigest()[:16]

            conn.execute(
                """
                INSERT INTO email_classifications 
                (timestamp, email_hash, predicted_category, confidence, model_used, 
                 escalated, co2_emissions_g, processing_time, energy_efficiency_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result["timestamp"],
                    email_hash,
                    result["predicted_category"],
                    result["confidence"],
                    result["model_used"],
                    result["escalated"],
                    result["energy_metrics"]["co2_emissions_g"],
                    processing_time,
                    result["energy_metrics"]["energy_efficiency_score"],
                ),
            )
    except Exception as e:
        print(f"Database logging error: {e}")


# -------------------------
# Other Endpoints (stats, feedback, insights)
# -------------------------
@app.get("/agent-stats")
def get_agent_statistics():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM email_classifications")
            total_classifications = cursor.fetchone()[0] or 0

            cursor.execute(
                "SELECT SUM(co2_emissions_g), AVG(co2_emissions_g) FROM email_classifications"
            )
            result = cursor.fetchone()
            total_co2 = float(result[0] or 0.0)
            avg_co2 = float(result[1] or 0.0)

            cursor.execute(
                """
                SELECT model_used, COUNT(*), AVG(co2_emissions_g), AVG(confidence)
                FROM email_classifications 
                GROUP BY model_used
            """
            )
            model_stats = [
                {
                    "model": row[0],
                    "count": row[1],
                    "avg_co2_g": float(row[2] or 0),
                    "avg_confidence": float(row[3] or 0),
                }
                for row in cursor.fetchall()
            ]

            cursor.execute("SELECT COUNT(*) FROM email_classifications WHERE escalated = 1")
            escalations = cursor.fetchone()[0] or 0

            cursor.execute(
                """
                SELECT predicted_category, COUNT(*) 
                FROM email_classifications 
                GROUP BY predicted_category
            """
            )
            category_stats = [
                {"category": row[0], "count": row[1]} for row in cursor.fetchall()
            ]

        return {
            "total_classifications": total_classifications,
            "escalation_rate": (escalations / total_classifications)
            if total_classifications > 0
            else 0,
            "total_co2_emissions_g": total_co2,
            "avg_co2_per_email_g": avg_co2,
            "model_performance": model_stats,
            "category_distribution": category_stats,
            "energy_savings_estimate": total_co2 * 0.3 if escalations > 0 else 0,
            "orchestrator_ready": ORCHESTRATOR_READY,
        }

    except Exception as e:
        print(f"Stats error: {e}")
        return {
            "total_classifications": 0,
            "escalation_rate": 0,
            "total_co2_emissions_g": 0.0,
            "avg_co2_per_email_g": 0.0,
            "model_performance": [],
            "category_distribution": [],
            "energy_savings_estimate": 0.0,
            "orchestrator_ready": ORCHESTRATOR_READY,
            "error": str(e),
        }


@app.get("/runs/recent")
def recent_runs(limit: Optional[int] = 50):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, email_hash, predicted_category, confidence, 
                       model_used, escalated, co2_emissions_g, processing_time
                FROM email_classifications 
                ORDER BY id DESC 
                LIMIT ?
            """,
                (limit or 50,),
            )

            rows = []
            for row in cursor.fetchall():
                rows.append(
                    {
                        "id": row[0],
                        "ts": row[1],
                        "ts_local": datetime.fromtimestamp(row[1]).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "text_hash": row[2],
                        "predicted_category": row[3],
                        "confidence": float(row[4]),
                        "model_used": row[5],
                        "escalated": bool(row[6]),
                        "co2_g": float(row[7]),
                        "processing_time": float(row[8]),
                    }
                )

            return rows
    except Exception as e:
        print(f"Recent runs error: {e}")
        return []


@app.post("/feedback")
def submit_feedback(
    classification_id: int,
    correct_category: str,
    user_confidence: float,
    user_id: str = "anonymous",
):
    if correct_category not in EMAIL_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {EMAIL_CATEGORIES}",
        )

    if not 0 <= user_confidence <= 1:
        raise HTTPException(
            status_code=400, detail="User confidence must be between 0 and 1"
        )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO user_feedback (
                    classification_id, correct_category, user_confidence, 
                    feedback_timestamp, user_id
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (classification_id, correct_category, user_confidence, time.time(), user_id),
            )

        return {"status": "feedback_recorded", "message": "Thank you for your feedback!"}

    except Exception as e:
        print(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")


@app.get("/learning-insights")
def get_learning_insights():
    return {
        "performance_metrics": {
            "insufficient_data": True,
            "message": "Learning system integration pending",
        },
        "threshold_recommendation": {
            "insufficient_data": True,
            "message": "More data needed for recommendations",
        },
        "last_updated": datetime.now().isoformat(),
        "orchestrator_ready": ORCHESTRATOR_READY,
    }


# -------------------------
# DB Initialization
# -------------------------
def init_email_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS email_classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    email_hash TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    escalated BOOLEAN NOT NULL,
                    co2_emissions_g REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    energy_efficiency_score REAL NOT NULL
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    classification_id INTEGER,
                    correct_category TEXT,
                    user_confidence REAL,
                    feedback_timestamp REAL,
                    user_id TEXT DEFAULT 'anonymous',
                    FOREIGN KEY (classification_id) REFERENCES email_classifications (id)
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_email_timestamp ON email_classifications(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON user_feedback(feedback_timestamp)"
            )
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")


# Initialize DB
init_email_db()


# -------------------------
# Startup/Shutdown Events
# -------------------------
@app.on_event("startup")
async def startup_event():
    print("Green AI Email Classification API Starting...")
    print(f"Orchestrator Ready: {ORCHESTRATOR_READY}")
    if not ORCHESTRATOR_READY:
        print("Running in fallback mode - some features may be limited")
    print("API Ready!")


@app.on_event("shutdown")
async def shutdown_event():
    print("Green AI Email Classification API Shutting Down...")

