import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# LOAD DATA
train_df = pd.read_csv("data/clean_metadata.csv")

print("Train Samples:", len(train_df))


# ENCODE station_id
station_encoder = LabelEncoder()

train_df["station_id"] = station_encoder.fit_transform(
    train_df["station_id"]
)

joblib.dump(
    station_encoder,
    "models/station_encoder.pkl"
)


# LABEL MAP
label_map = {
    "0–50": 0,
    "51–100": 1,
    "101–150": 2,
    "151–200": 3,
    "201–300": 4,
    "301–600": 5
}

train_df["label"] = train_df["pm25_bin"].map(label_map)


# FEATURES
features = [
    "longitude",
    "latitude",
    "year",
    "month",
    "day",
    "hour"
]

X_train = train_df[features]
y_train = train_df["label"]


# MODEL
model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=6,
    eval_metric="mlogloss",
    random_state=42
)

print("Training Starting...")

model.fit(X_train, y_train)

print("Training Complete")


# SAVE MODEL
joblib.dump(
    model,
    "models/xgboost_pm25.pkl"
)

print("Model Saved")