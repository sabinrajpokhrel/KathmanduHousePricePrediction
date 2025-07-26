import streamlit as st
import numpy as np
import joblib

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