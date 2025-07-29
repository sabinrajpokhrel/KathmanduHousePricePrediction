import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# Load model and label encoders
model = joblib.load('house_price_model.pkl')
le_loc = joblib.load('location_encoder.pkl')

st.title("🏠 Kathmandu Valley House Price Predictor")

# Location dropdown
locations = le_loc.classes_.tolist()
location = st.selectbox("Select Location", locations)

# Numeric inputs
land_area = st.number_input("Land Area (sqft)", min_value=100.0, max_value=100000.0, value=1000.0, step=10.0)
road_access = st.number_input("Road Access (feet)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
floor = st.number_input("Floor", min_value=0, max_value=20, value=1, step=1)
bedroom = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
bathroom = st.number_input("Bathrooms", min_value=0, max_value=10, value=1, step=1)
house_age = st.number_input("House Age (years)", min_value=0, max_value=150, value=10, step=1)

# Prediction
if st.button("Predict House Price"):
    try:
        location_enc = le_loc.transform([location])[0]

        # Full feature vector: [land_area, road_access, floor, bedroom, bathroom, house_age, and location_enc]
        features = np.array([[land_area, road_access, floor, bedroom, bathroom, house_age, location_enc]])

        pred_price = model.predict(features)[0]
        st.success(f"Estimated House Price: Rs. {pred_price:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

st.header("📊 Data Visualization & Insights")

@st.cache_data
def load_data():
    df = pd.read_csv("Kathmandu_Valley_House_Dataset.csv")
    
    # Clean similar to model code
    df = df.drop(columns=['TITLE', 'BUILDUP AREA', 'PARKING', 'AMENITIES', 'FACING'], errors='ignore')
    df['LAND AREA'] = df['LAND AREA'].astype(str).str.lower().str.strip().str.replace('anna', 'aana')
    df['LAND AREA SQFT'] = 0
    df['LAND AREA SQFT'] += df['LAND AREA'].str.extract(r'(\d+\.?\d*)\s*ropani', expand=False).astype(float).fillna(0) * 5476
    df['LAND AREA SQFT'] += df['LAND AREA'].str.extract(r'(\d+\.?\d*)\s*kattha', expand=False).astype(float).fillna(0) * 1369
    df['LAND AREA SQFT'] += df['LAND AREA'].str.extract(r'(\d+\.?\d*)\s*aana', expand=False).astype(float).fillna(0) * 342.25
    df = df[df['LAND AREA SQFT'] > 0].copy()
    df['LAND AREA'] = df['LAND AREA SQFT']
    df.drop(columns=['LAND AREA SQFT'], inplace=True)

    def convert_price(price_str):
        if pd.isnull(price_str): return np.nan
        price_str = price_str.lower().replace('rs.', '').replace('rs', '').replace(',', '').strip()
        num = re.findall(r'[\d.]+', price_str)
        if not num: return np.nan
        num = float(num[0])
        if 'cr' in price_str: return num * 1e7
        elif 'lac' in price_str or 'lakh' in price_str: return num * 1e5
        elif 'k' in price_str: return num * 1e3
        else: return num

    df['PRICE'] = df['PRICE'].apply(convert_price)
    df = df[df['PRICE'].notnull()]

    df['ROAD ACCESS'] = df['ROAD ACCESS'].astype(str).str.lower().str.strip()
    df['ROAD ACCESS'] = df['ROAD ACCESS'].apply(lambda x: float(re.findall(r'\d+\.\d+|\d+', x)[0]) if re.findall(r'\d+\.\d+|\d+', x) else np.nan)
    df = df[df['ROAD ACCESS'].notnull()]
    
    df['FLOOR'] = df['FLOOR'].fillna(df['FLOOR'].median())
    df['BEDROOM'] = df['BEDROOM'].fillna(df['BEDROOM'].median())
    df['BATHROOM'] = df['BATHROOM'].fillna(df['BATHROOM'].median())

    df['BUILT YEAR'] = df['BUILT YEAR'].astype(str).str.replace('b.s', '', regex=False).str.extract(r'(\d{4})').astype(float)
    df = df[df['BUILT YEAR'].notnull()]
    df['HOUSE_AGE'] = 2082 - df['BUILT YEAR']

    df['LOCATION'] = df['LOCATION'].fillna('Unknown').astype(str).str.strip()
    
    return df

df = load_data()

# 1. Price Distribution
st.subheader("📈 Price Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['PRICE'], bins=50, kde=True, ax=ax1, color='skyblue')
ax1.set_xlabel("Price (Rs)")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# 2. Land Area vs Price
st.subheader("📐 Land Area vs Price")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='LAND AREA', y='PRICE', hue='FLOOR', palette='viridis', ax=ax2)
ax2.set_xlabel("Land Area (sqft)")
ax2.set_ylabel("Price (Rs)")
st.pyplot(fig2)

# 3. Boxplot: Price by Location (Top 10 Locations)
st.subheader("📍 Price Comparison by Location")
top_locs = df['LOCATION'].value_counts().nlargest(10).index
filtered_df = df[df['LOCATION'].isin(top_locs)]
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=filtered_df, x='LOCATION', y='PRICE', ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
st.pyplot(fig3)

# 4. Feature Importance
st.subheader("🌲 Feature Importance (from Random Forest)")
feature_importances = model.feature_importances_
feature_names = ['LAND AREA', 'ROAD ACCESS', 'FLOOR', 'BEDROOM', 'BATHROOM', 'HOUSE_AGE', 'LOCATION_enc']
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

fig4, ax4 = plt.subplots()
sns.barplot(data=feat_df, x='Importance', y='Feature', ax=ax4, palette='Blues_d')
st.pyplot(fig4)

# 5. Correlation Heatmap
st.subheader("🔗 Correlation Heatmap")
fig5, ax5 = plt.subplots()
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
st.pyplot(fig5)