import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv('real_estate_data.csv')

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=[
    'mainroad', 'guestroom', 'basement', 'hotwaterheating',
    'airconditioning', 'prefarea', 'furnishingstatus'
])

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model, scaler, and columns
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model, columns, and scaler saved.")