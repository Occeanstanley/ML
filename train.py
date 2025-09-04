# train.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Toy training data (distance_km, planned_stops, traffic_score, weather_score, priority, promised_hours)
X = np.array([
    [12.5, 2, 0.7, 0.4, 0, 24],
    [8.0,  1, 0.3, 0.2, 1, 12],
    [45.2, 5, 0.9, 0.8, 0, 36],
    [30.0, 4, 0.8, 0.9, 1,  3],
    [18.0, 2, 0.6, 0.3, 0, 16],
    [10.0, 1, 0.2, 0.3, 0, 12],
    [25.0, 3, 0.7, 0.6, 1, 10],
    [40.0, 5, 0.9, 0.7, 1, 20],
])
# Label: 1 = delayed, 0 = on-time (toy)
y = np.array([0,0,1,1,0,0,1,1])

clf = RandomForestClassifier(n_estimators=80, random_state=42)
clf.fit(X, y)
dump(clf, "model.joblib")
print("Saved model.joblib")
