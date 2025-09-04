# train.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# X = [distance_km, planned_stops, traffic_score, weather_score, priority, promised_hours]
X = np.array([
    [30, 4, 0.8, 0.9, 3, 36],
    [45.2, 5, 0.9, 0.8, 3, 36],
    [8, 1, 0.3, 0.2, 12, 24],
    [12.5, 2, 0.7, 0.4, 24, 24],
])
y = np.array([1, 1, 0, 0])  # 1=delay, 0=on‑time (toy labels)

clf = RandomForestClassifier(n_estimators=80, random_state=42)
clf.fit(X, y)
dump(clf, "model.joblib")
print("Saved model.joblib")
