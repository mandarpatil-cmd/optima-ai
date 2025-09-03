from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from transformers import pipeline
from codecarbon import EmissionsTracker
import hashlib, sqlite3, time
from pathlib import Path
from typing import Optional

from .config import (
    LIGHT_MODEL, HEAVY_MODEL,
    DEFAULT_THRESHOLD, MAX_CHARS
)

# ---------- helpers ----------
def normalize_sst2(label: str) -> str:
    if label in ("LABEL_1", "POSITIVE", "Positive"):
        return "POSITIVE"
    if label in ("LABEL_0", "NEGATIVE", "Negative"):
        return "NEGATIVE"
    return label

def cache_key(text: str, threshold: float, mode: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{h}|{threshold}|{mode}"

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------- model load ----------
print("[info] loading models... (first time may take a few minutes)")
clf_light = pipeline("text-classification", model=LIGHT_MODEL, device=-1)
clf_heavy = pipeline("text-classification", model=HEAVY_MODEL, device=-1)
_ = clf_light("warm up"); _ = clf_heavy("warm up")
print("[info] models ready ✅")

# ---------- db ----------
DB_PATH = Path(__file__).resolve().parents[1] / "green_metrics.sqlite3"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            threshold REAL NOT NULL,
            mode TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT NOT NULL,
            escalated INTEGER NOT NULL,
            co2_g REAL NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts)")
init_db()

def log_run(req, label: str, conf: float, used: str, escalated: bool, co2_g: float):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO runs (ts, text_hash, threshold, mode, label, confidence, model_used, escalated, co2_g) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                int(time.time()),
                sha256_text(req.text),
                float(req.threshold),
                req.mode,
                label,
                float(conf),
                used,
                int(escalated),
                float(co2_g),
            )
        )

# ---------- api ----------
app = FastAPI(title="Green AI Agent API")
CACHE = {}

class ClassifyReq(BaseModel):
    text: str
    threshold: float = DEFAULT_THRESHOLD
    mode: str = "auto"               # "auto" | "light" | "heavy"
    force_escalate: bool = False     # for testing

class ClassifyRes(BaseModel):
    label: str
    confidence: float
    model_used: str
    escalated: bool
    co2_g: float
    cache_hit: bool
    # debug
    light_label: str
    light_confidence: float
    heavy_label: str | None = None
    heavy_confidence: float | None = None

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/classify", response_model=ClassifyRes)
def classify(req: ClassifyReq):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if len(req.text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"text too long (>{MAX_CHARS} chars)")

    key = cache_key(req.text, req.threshold, req.mode)
    if key in CACHE and not req.force_escalate:
        return {**CACHE[key], "cache_hit": True}

    tracker = EmissionsTracker(save_to_file=False); tracker.start()

    # compute light first for debug + policy
    light_out = clf_light(req.text, truncation=True)[0]
    light_label = normalize_sst2(light_out["label"])
    light_conf  = float(light_out["score"])

    escalated = False
    heavy_label = None
    heavy_conf = None

    if req.mode == "light":
        used = LIGHT_MODEL
        final_label, final_conf = light_label, light_conf

    elif req.mode == "heavy":
        heavy_out = clf_heavy(req.text, truncation=True)[0]
        used = HEAVY_MODEL
        heavy_label = normalize_sst2(heavy_out["label"])
        heavy_conf  = float(heavy_out["score"])
        final_label, final_conf = heavy_label, heavy_conf

    else:
        # AUTO
        if req.force_escalate or light_conf < req.threshold:
            heavy_out = clf_heavy(req.text, truncation=True)[0]
            used = HEAVY_MODEL
            heavy_label = normalize_sst2(heavy_out["label"])
            heavy_conf  = float(heavy_out["score"])
            final_label, final_conf = heavy_label, heavy_conf
            escalated = True
        else:
            used = LIGHT_MODEL
            final_label, final_conf = light_label, light_conf

    co2_g = (tracker.stop() or 0.0) * 1000.0

    resp = ClassifyRes(
        label=final_label,
        confidence=final_conf,
        model_used=used,
        escalated=escalated,
        co2_g=float(co2_g),
        cache_hit=False,
        light_label=light_label,
        light_confidence=light_conf,
        heavy_label=heavy_label,
        heavy_confidence=heavy_conf
    ).model_dump()

    # log then cache
    log_run(req, resp["label"], resp["confidence"], resp["model_used"], resp["escalated"], resp["co2_g"])
    if not req.force_escalate:
        CACHE[key] = resp

    return resp

@app.get("/metrics")
def metrics():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*), SUM(co2_g), AVG(confidence) FROM runs")
        total, total_co2, avg_conf = c.fetchone()
        total = total or 0
        total_co2 = float(total_co2 or 0.0)
        avg_conf = float(avg_conf or 0.0)

        c.execute("SELECT COUNT(*) FROM runs WHERE model_used LIKE 'distil%'")
        light_count = c.fetchone()[0] or 0

        c.execute("SELECT COUNT(*) FROM runs WHERE escalated=1")
        escalations = c.fetchone()[0] or 0

        c.execute("SELECT model_used, COUNT(*), SUM(co2_g) FROM runs GROUP BY model_used")
        by_model = [
            {"model": row[0], "count": int(row[1] or 0), "total_co2_g": float(row[2] or 0.0)}
            for row in c.fetchall()
        ]

    light_pct = (light_count / total) if total else 0.0
    return {
        "requests": total,
        "light_pct": light_pct,
        "escalations": escalations,
        "total_co2_g": total_co2,
        "avg_confidence": avg_conf,
        "by_model": by_model
    }


    

@app.get("/runs/recent")
def recent_runs(limit: Optional[int] = 50):
    import sqlite3, time
    from pathlib import Path
    DB_PATH = Path(__file__).resolve().parents[1] / "green_metrics.sqlite3"
    rows = []
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for r in c.execute("""
            SELECT id, ts, text_hash, mode, label, confidence, model_used, escalated, co2_g
            FROM runs ORDER BY id DESC LIMIT ?
        """, (limit or 50,)):
            rows.append({
                "id": r[0],
                "ts": r[1],
                "ts_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[1])),
                "text_hash": r[2],
                "mode": r[3],
                "label": r[4],
                "confidence": float(r[5]),
                "model_used": r[6],
                "escalated": bool(r[7]),
                "co2_g": float(r[8]),
            })
    return rows





















# from fastapi import FastAPI, HTTPException
# from fastapi.responses import RedirectResponse
# from pydantic import BaseModel
# from transformers import pipeline
# from codecarbon import EmissionsTracker
# import hashlib
# import sqlite3, time
# from pathlib import Path


# from .config import (
#     LIGHT_MODEL, HEAVY_MODEL,
#     DEFAULT_THRESHOLD, MAX_CHARS
# )

# print("[info] loading models... (first time may take a few minutes)")
# clf_light = pipeline("text-classification", model=LIGHT_MODEL, device=-1)
# clf_heavy = pipeline("text-classification", model=HEAVY_MODEL, device=-1)
# _ = clf_light("warm up"); _ = clf_heavy("warm up")
# print("[info] models ready ✅")


# app = FastAPI(title="Green AI Agent API")

# # very simple in-memory cache (keyed by text+threshold+mode)
# CACHE = {}
# def normalize_sst2(label: str) -> str:
#     # unify labels across checkpoints
#     if label in ("LABEL_1", "POSITIVE", "Positive"):
#         return "POSITIVE"
#     if label in ("LABEL_0", "NEGATIVE", "Negative"):
#         return "NEGATIVE"
#     return label

# def cache_key(text: str, threshold: float, mode: str) -> str:
#     h = hashlib.sha256(text.encode("utf-8")).hexdigest()
#     return f"{h}|{threshold}|{mode}"

# class ClassifyReq(BaseModel):
#     text: str
#     threshold: float = DEFAULT_THRESHOLD
#     mode: str = "auto"               # "auto", "light", "heavy"
#     force_escalate: bool = False     # NEW: handy for testing the heavy path

# class ClassifyRes(BaseModel):
#     label: str
#     confidence: float
#     model_used: str
#     escalated: bool
#     co2_g: float
#     cache_hit: bool
#     # debug (so you can see why escalation didn’t trigger)
#     light_label: str
#     light_confidence: float
#     heavy_label: str | None = None
#     heavy_confidence: float | None = None

# @app.get("/")
# def root():
#     return RedirectResponse(url="/docs")

# @app.post("/classify", response_model=ClassifyRes)
# def classify(req: ClassifyReq):
#     if not req.text or not req.text.strip():
#         raise HTTPException(status_code=400, detail="text is required")
#     if len(req.text) > MAX_CHARS:
#         raise HTTPException(status_code=413, detail=f"text too long (>{MAX_CHARS} chars)")

#     key = cache_key(req.text, req.threshold, req.mode)
#     if key in CACHE and not req.force_escalate:
#         return {**CACHE[key], "cache_hit": True}

#     tracker = EmissionsTracker(save_to_file=False); tracker.start()

#     # always compute light first so we can expose its confidence
#     light_out = clf_light(req.text, truncation=True)[0]
#     light_label = normalize_sst2(light_out["label"])

#     light_conf = float(light_out["score"])

#     escalated = False
#     heavy_label = None
#     heavy_conf = None

#     if req.mode == "light":
#         used = LIGHT_MODEL
#         final_label, final_conf = light_label, light_conf

#     elif req.mode == "heavy":
#         heavy_out = clf_heavy(req.text, truncation=True)[0]
#         used = HEAVY_MODEL
#         heavy_label = normalize_sst2(heavy_out["label"])
#         heavy_conf = float(heavy_out["score"])
#         final_label, final_conf = heavy_label, heavy_conf

#     else:
#         # AUTO policy: escalate if forced, or if light confidence below threshold
#         if req.force_escalate or light_conf < req.threshold:
#             heavy_out = clf_heavy(req.text, truncation=True)[0]
#             used = HEAVY_MODEL
#             heavy_label = normalize_sst2(heavy_out["label"])
#             heavy_conf = float(heavy_out["score"])
#             final_label, final_conf = heavy_label, heavy_conf
#             escalated = True

#         else:
#             used = LIGHT_MODEL
#             final_label, final_conf = light_label, light_conf

#     co2_g = (tracker.stop() or 0.0) * 1000.0

#     resp = ClassifyRes(
#         label=final_label,
#         confidence=final_conf,
#         model_used=used,
#         escalated=escalated,
#         co2_g=float(co2_g),
#         cache_hit=False,
#         light_label=light_label,
#         light_confidence=light_conf,
#         heavy_label=heavy_label,
#         heavy_confidence=heavy_conf
#     ).model_dump()

#     # don’t cache when force_escalate is true (so you can test repeatedly)
#     if not req.force_escalate:
#         CACHE[key] = resp

#     return resp
