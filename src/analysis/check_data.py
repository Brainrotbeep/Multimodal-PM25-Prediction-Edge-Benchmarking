import pandas as pd

train_df = pd.read_csv("data/clean_metadata.csv")
test_df  = pd.read_csv("data/clean_test_metadata.csv")

# Class 4 ke samples dekho
print("Train Class 4 (201-300) samples:")
print(train_df[train_df["pm25_bin"] == "201–300"].head(10))

print("\nTest Class 4 (201-300) samples:")
print(test_df[test_df["pm25_bin"] == "201–300"].head(10))

# tabular features ka distribution dekho class 4 ke liye
print("\nTrain Class 4 stats:")
print(train_df[train_df["pm25_bin"] == "201–300"].describe())

print("\nTest Class 4 stats:")
print(test_df[test_df["pm25_bin"] == "201–300"].describe())