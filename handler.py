import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_col = "Target Pressure (bar)"

# Remove rows with missing target
train = train.dropna(subset=[target_col])

# Separate target before preprocessing
y = train[target_col]
X = train.drop(columns=[target_col])

# Numeric columns
numeric_cols = X.select_dtypes(include="number").columns

# Fill missing numeric values
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
test[numeric_cols] = test[numeric_cols].fillna(X[numeric_cols].mean())

# Detect categorical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

# Fill missing categorical values
for col in cat_cols:
    fill_value = X[col].mode().iloc[0] if not X[col].mode().empty else ""

    X[col] = X[col].fillna(fill_value)

    if col in test.columns:
        test[col] = test[col].fillna(fill_value)

# One-hot encode
X = pd.get_dummies(X, columns=cat_cols)
test = pd.get_dummies(test, columns=cat_cols)

# Align columns
X, test = X.align(test, join="left", axis=1, fill_value=0)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Predict
predictions = model.predict(test)

# Save results
results = pd.DataFrame({
    "ID": test["ID"],
    target_col: predictions
})

results.to_csv("sample_prediction.csv", index=False)

print("Predictions written to sample_prediction.csv")