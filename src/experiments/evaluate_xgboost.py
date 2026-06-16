import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# LOAD TEST DATA
test_df = pd.read_csv(
    "data/clean_test_metadata.csv"
)

print("Test Samples:", len(test_df))


# LABEL MAP
label_map = {
    "0–50": 0,
    "51–100": 1,
    "101–150": 2,
    "151–200": 3,
    "201–300": 4,
    "301–600": 5
}

test_df["label"] = test_df["pm25_bin"].map(
    label_map
)


# FEATURES
features = [
    "longitude",
    "latitude",
    "year",
    "month",
    "day",
    "hour"
]

X_test = test_df[features]
y_test = test_df["label"]


# LOAD MODEL
model = joblib.load(
    "models/xgboost_pm25.pkl"
)

print("Model Loaded")


# PREDICTIONS
predictions = model.predict(X_test)


# ACCURACY
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================")
print(
    f"Test Accuracy: {accuracy*100:.2f}%"
)
print("================================")


# CONFUSION MATRIX
cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# PER-CLASS ACCURACY
print("\nPer-Class Accuracy:")

for i in range(6):

    total = cm[i].sum()

    correct = cm[i][i]

    acc = 100 * correct / total

    print(
        f"Class {i}: "
        f"{acc:.2f}% "
        f"({correct}/{total})"
    )