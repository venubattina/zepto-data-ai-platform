import joblib
import pandas as pd

# Load saved pipeline
pipeline = joblib.load("analytics/best_classifier_pipeline.joblib")

# Unseen raw test input
raw_passenger = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 150.0,
    "embarked": "S"
}])

prediction = pipeline.predict(raw_passenger)[0]
probability = pipeline.predict_proba(raw_passenger)[0][1]

print("=== Raw Input Verification ===")
print("Predicted Class (0=Died, 1=Survived):", prediction)
print(f"Survival Probability: {probability:.4f}")
assert prediction in [0, 1], "Invalid prediction"
print("Inference test passed successfully!")
