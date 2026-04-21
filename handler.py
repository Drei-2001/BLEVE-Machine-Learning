import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_col = "Target Pressure (bar)"

# Drop rows with missing target values before training
train = train.dropna(subset=[target_col])

#handles missing values
numeric_cols = train.select_dtypes(include="number").columns.drop(target_col)

train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].mean())
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].mean())

# Categorical columns: fill missing values with mode
cat_cols = train.select_dtypes(include=["string", "category"]).columns
for col in cat_cols:
    fill_value = train[col].mode().iloc[0] if not train[col].mode().empty else ""
    train[col] = train[col].fillna(fill_value)
    if col in test.columns:
        test[col] = test[col].fillna(fill_value)

# One-hot encode categorical columns
train = pd.get_dummies(train, columns=cat_cols, dummy_na=False)
test = pd.get_dummies(test, columns=[col for col in cat_cols if col in test.columns], dummy_na=False)

# Align train and test feature columns
train, test = train.align(test, join="left", axis=1, fill_value=0)

# Separate features and target
X = train.drop(target_col, axis=1)
y = train[target_col]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

test = test.drop(columns=[target_col], errors="ignore")
predictions = model.predict(test)
print(predictions)