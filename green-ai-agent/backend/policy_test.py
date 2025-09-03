from pathlib import Path
from transformers import pipeline
from codecarbon import EmissionsTracker
from .config import LIGHT_MODEL, HEAVY_MODEL, SAMPLES_PATH

def load_samples():
    """
    Try to load data/samples.txt from the project root.
    If not found, return a small built-in list so the script still works.
    """
    print(f"[info] Looking for samples at: {SAMPLES_PATH}")
    if SAMPLES_PATH.exists():
        texts = [line.strip() for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        if texts:
            return texts

    print("[warn] samples.txt not found; using built-in examples.")
    return [
        "The product arrived late and support did not help. I am very disappointed.",
        "I love the new update! The app feels faster and the design is clean.",
        "Can someone help me reset my password? I can't log in to my account.",
        "Please cancel my subscription. I was charged twice this month."
    ]

def policy_light_then_heavy(texts, threshold=0.85):
    light = pipeline("text-classification", model=LIGHT_MODEL, device=-1)
    heavy = pipeline("text-classification", model=HEAVY_MODEL, device=-1)
    _ = light("warm up"); _ = heavy("warm up")

    accepted, escalated = [], []
    tracker = EmissionsTracker(save_to_file=False); tracker.start()

    for t in texts:
        o = light(t, truncation=True)[0]
        if o["score"] >= threshold:
            accepted.append((t, o, "light"))
        else:
            o2 = heavy(t, truncation=True)[0]
            escalated.append((t, o2, "heavy"))

    co2_kg = tracker.stop() or 0.0
    return accepted, escalated, co2_kg

if __name__ == "__main__":
    texts = load_samples()
    accepted, escalated, co2 = policy_light_then_heavy(texts, threshold=0.85)
    print(f"Accepted by light: {len(accepted)}")
    print(f"Escalated to heavy: {len(escalated)}")
    print(f"Estimated CO2 (policy run): {co2*1000:.4f} g")
