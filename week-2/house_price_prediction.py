import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ==========================
# Step 1: Load Dataset
# ==========================

df = pd.read_csv("Housing.csv")

print("Dataset Shape:", df.shape)

# ==========================
# Step 2: Display Dataset Info
# ==========================

print("\nFirst 5 Rows:")
print(df.head())

# ==========================
# Step 3: Separate Features & Target
# ==========================

X = df.drop("price", axis=1)
y = df["price"]

# ==========================
# Step 4: Convert Categorical Data
# ==========================

X = pd.get_dummies(X, drop_first=True)

print("\nShape After Encoding:", X.shape)

# ==========================
# Step 5: Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================
# Step 6: Train Linear Regression
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================
# Step 7: Predictions
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Step 8: Evaluation
# ==========================

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n----- Model Evaluation -----")
print("R² Score :", round(r2, 4))
print("MSE      :", round(mse, 2))
print("RMSE     :", round(rmse, 2))

# ==========================
# Step 9: Actual vs Predicted
# ==========================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices")
print(comparison.head(10))

# ==========================
# Step 10: Scatter Plot
# ==========================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()

# ==========================
# Step 11: Predict Sample House
# ==========================

sample_house = X_test.iloc[[0]]

predicted_price = model.predict(sample_house)

print("\nPredicted Price for Sample House:")
print("₹", round(predicted_price[0], 2))