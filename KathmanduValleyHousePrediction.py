import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import joblib


# Loading the dataset
file_path = "/Users/sabinrajpokhrel/Documents/AI coursework/HousePricePrediction/Kathmandu_Valley_House_Dataset.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("First 5 rows:")
print(df.head())

# Dropping the columns that are irrelevant
df = df.drop(columns=['TITLE', 'BUILDUP AREA', 'PARKING', 'AMENITIES', 'FACING'], errors='ignore')

# Data Cleaning: LAND AREA
df['LAND AREA'] = df['LAND AREA'].astype(str).str.lower().str.strip().str.replace('anna', 'aana')
df['LAND AREA SQFT'] = 0
df['LAND AREA SQFT'] += df['LAND AREA'].str.extract(r'(\d+\.?\d*)\s*ropani', expand=False).astype(float).fillna(0) * 5476
df['LAND AREA SQFT'] += df['LAND AREA'].str.extract(r'(\d+\.?\d*)\s*kattha', expand=False).astype(float).fillna(0) * 1369
df['LAND AREA SQFT'] += df['LAND AREA'].str.extract(r'(\d+\.?\d*)\s*aana', expand=False).astype(float).fillna(0) * 342.25
df = df[df['LAND AREA SQFT'] > 0].copy()
df['LAND AREA'] = df['LAND AREA SQFT']
df.drop(columns=['LAND AREA SQFT'], inplace=True)

# Data Cleaning: PRICE
def convert_price_to_number(price_str):
    if pd.isnull(price_str):
        return np.nan
    price_str = price_str.lower().strip().replace('rs.', '').replace('rs', '').replace(',', '')
    number = re.findall(r'[\d.]+', price_str)
    if not number:
        return np.nan
    number = float(number[0])
    if 'cr' in price_str:
        multiplier = 1e7
    elif 'lac' in price_str or 'lakh' in price_str:
        multiplier = 1e5
    elif 'k' in price_str:
        multiplier = 1e3
    else:
        multiplier = 1
    return number * multiplier

df['PRICE'] = df['PRICE'].apply(convert_price_to_number)
df = df[df['PRICE'].notnull()].copy()

# Data Cleaning: ROAD ACCESS
df['ROAD ACCESS'] = df['ROAD ACCESS'].astype(str).str.lower().str.strip()
df['ROAD ACCESS'] = df['ROAD ACCESS'].apply(
    lambda x: float(re.findall(r'\d+\.\d+|\d+', x)[0]) if re.findall(r'\d+\.\d+|\d+', x) else np.nan
)
df = df[df['ROAD ACCESS'].notnull()].copy()

# Filling the numeric columns with median
df['FLOOR'] = df['FLOOR'].fillna(df['FLOOR'].median())
df['BEDROOM'] = df['BEDROOM'].fillna(df['BEDROOM'].median())
df['BATHROOM'] = df['BATHROOM'].fillna(df['BATHROOM'].median())

# Cleaning BUILT YEAR and Calculation HOUSE AGE
df['BUILT YEAR'] = df['BUILT YEAR'].astype(str).str.lower().str.replace('b.s', '').str.strip()
df['BUILT YEAR'] = df['BUILT YEAR'].apply(lambda x: re.findall(r'\d{4}', x)[0] if re.findall(r'\d{4}', x) else np.nan)
df = df[df['BUILT YEAR'].notnull()].copy()
df['BUILT YEAR'] = df['BUILT YEAR'].astype(int)
df['HOUSE_AGE'] = 2082 - df['BUILT YEAR']
df['HOUSE_AGE'] = df['HOUSE_AGE'].fillna(df['HOUSE_AGE'].median())

# Filling LOCATION missing values with "Unknown"
df['LOCATION'] = df['LOCATION'].fillna('Unknown').astype(str).str.strip()

# Label encoding LOCATION 
le_loc = LabelEncoder()
df['LOCATION_enc'] = le_loc.fit_transform(df['LOCATION'])


# Dropping original string columns after encoding
df.drop(columns=['LOCATION', 'BUILT YEAR'], inplace=True, errors='ignore')

# Finally, Checking Null Values
print("Missing values after cleaning:")
print(df.isnull().sum())

# Preparing Model Inputs
y = df['PRICE']
X = df.drop(columns=['PRICE'])

# Splitting data sets into: 80% training, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=23)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=23)

# Training the model
model = RandomForestRegressor(
    n_estimators=300,
    max_features='sqrt',
    max_depth= 20,
    min_samples_split=2,
    random_state=23
)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt']
}

search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                            param_distributions=param_grid,
                            n_iter=20, cv=3, scoring='neg_mean_absolute_error')
search.fit(X_train, y_train)

print("Best Params:", search.best_params_)

model.fit(X_train, y_train)

# Saving the trained model and label encoder
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(le_loc, 'location_encoder.pkl')


# Predicting and evaluating on validation set
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val, y_val_pred)

# Results
print("\nValidation Set Evaluation Metrics:")
print(f"MAE  (Mean Absolute Error)     : {val_mae:.2f}")
print(f"MSE  (Mean Squared Error)     : {val_mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {val_rmse:.2f}")
print(f"R² Score                      : {val_r2:.2f}")

# Predicting and evaluating on test set
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

# Results
print("\nTest Set Evaluation Metrics:")
print(f"MAE  (Mean Absolute Error)     : {test_mae:.2f}")
print(f"MSE  (Mean Squared Error)     : {test_mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {test_rmse:.2f}")
print(f"R² Score                      : {test_r2:.2f}")

print(df.head())  # Best Params: {'n_estimators': 300, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 20}