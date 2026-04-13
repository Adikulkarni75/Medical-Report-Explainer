import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset

REFERENCE_RANGES = {
    "Hemoglobin (Hb)":              {"low": 13.0,   "high": 17.0,   "critical_low": 7.0,   "critical_high": 20.0},
    "Total RBC count":               {"low": 4.5,    "high": 5.5,    "critical_low": 2.5,   "critical_high": 7.0},
    "Packed Cell Volume (PCV)":      {"low": 40.0,   "high": 50.0,   "critical_low": 20.0,  "critical_high": 60.0},
    "Mean Corpuscular Volume (MCV)": {"low": 83.0,   "high": 101.0,  "critical_low": 60.0,  "critical_high": 120.0},
    "MCH":                           {"low": 27.0,   "high": 32.0,   "critical_low": 15.0,  "critical_high": 40.0},
    "MCHC":                          {"low": 32.5,   "high": 34.5,   "critical_low": 28.0,  "critical_high": 38.0},
    "RDW":                           {"low": 11.6,   "high": 14.0,   "critical_low": 9.0,   "critical_high": 20.0},
    "Total WBC count":               {"low": 4000,   "high": 11000,  "critical_low": 2000,  "critical_high": 30000},
    "Neutrophils":                   {"low": 50.0,   "high": 62.0,   "critical_low": 20.0,  "critical_high": 90.0},
    "Lymphocytes":                   {"low": 20.0,   "high": 40.0,   "critical_low": 5.0,   "critical_high": 60.0},
    "Eosinophils":                   {"low": 0.0,    "high": 6.0,    "critical_low": 0.0,   "critical_high": 20.0},
    "Monocytes":                     {"low": 0.0,    "high": 10.0,   "critical_low": 0.0,   "critical_high": 20.0},
    "Basophils":                     {"low": 0.0,    "high": 2.0,    "critical_low": 0.0,   "critical_high": 5.0},
    "Platelet Count":                {"low": 150000, "high": 410000, "critical_low": 50000, "critical_high": 800000},
}

LABELS = {"Normal": 0, "Low": 1, "High": 2, "Critical": 3}
LABEL_NAMES = {v: k for k, v in LABELS.items()}


def make_features(value, ref_low, ref_high):
    ref_mid = (ref_low + ref_high) / 2
    ref_range = ref_high - ref_low if ref_high != ref_low else 1.0
    normalized = (value - ref_mid) / ref_range
    deviation = (value - ref_mid) / ref_mid if ref_mid != 0 else 0
    below_low = max(0, ref_low - value) / ref_range
    above_high = max(0, value - ref_high) / ref_range
    return [value / (ref_high + 1e-6), normalized, deviation, below_low, above_high]


def get_label(value, ref):
    if value <= ref["critical_low"] or value >= ref["critical_high"]:
        return LABELS["Critical"]
    elif value < ref["low"]:
        return LABELS["Low"]
    elif value > ref["high"]:
        return LABELS["High"]
    else:
        return LABELS["Normal"]


def generate_dataset(samples_per_test=300):
    X, y = [], []
    for test, ref in REFERENCE_RANGES.items():
        low, high = ref["low"], ref["high"]
        c_low, c_high = ref["critical_low"], ref["critical_high"]
        ranges = [
            (c_low, low),
            (low, high),
            (high, c_high),
            (c_low * 0.5, c_low),
            (c_high, c_high * 1.5 + 1),
        ]
        for r_low, r_high in ranges:
            if r_low >= r_high:
                continue
            for _ in range(samples_per_test // len(ranges)):
                val = np.random.uniform(r_low, r_high)
                features = make_features(val, low, high)
                label = get_label(val, ref)
                X.append(features)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class LabClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.network(x)


def train():
    print("Generating dataset...")
    X, y = generate_dataset(samples_per_test=300)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LabClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training neural network...")
    for epoch in range(100):
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        if (epoch + 1) % 10 == 0:
            acc = correct / total * 100
            print(f"  Epoch {epoch+1}/100 — Loss: {total_loss:.3f} — Accuracy: {acc:.1f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/classifier.pt")
    print("\nModel saved to models/classifier.pt")
    return model


def load_model():
    model = LabClassifier()
    model.load_state_dict(torch.load("models/classifier.pt", weights_only=True))
    model.eval()
    return model


def classify_value(test_name: str, value: float) -> dict:
    if test_name not in REFERENCE_RANGES:
        return {"status": "Unknown", "confidence": 0.0}

    ref = REFERENCE_RANGES[test_name]
    features = make_features(value, ref["low"], ref["high"])
    x = torch.tensor([features], dtype=torch.float32)

    model = load_model()
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "status": LABEL_NAMES[pred],
        "confidence": round(confidence * 100, 1)
    }


if __name__ == "__main__":
    train()

    print("\nTesting classifier on sample values:")
    tests = [
        ("Hemoglobin (Hb)", 12.5),
        ("Hemoglobin (Hb)", 6.0),
        ("Platelet Count", 150000),
        ("Total WBC count", 9000),
        ("Packed Cell Volume (PCV)", 57.5),
    ]
    for test_name, value in tests:
        result = classify_value(test_name, value)
        print(f"  {test_name}: {value} → {result['status']} ({result['confidence']}% confidence)")