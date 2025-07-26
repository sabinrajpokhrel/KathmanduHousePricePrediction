
# 🏠 Kathmandu House Price Predictor

This project is a **machine learning-powered house price prediction system** built using **Python, Scikit-learn, and Streamlit**. It predicts the price of houses in Kathmandu based on user-defined property attributes.

---

## 🚀 Features

- Predicts house prices using a trained **Random Forest Regressor**
- Real-time web UI built using **Streamlit**
- Encodes location data using **Label Encoding**
- Clean preprocessing: removes unnecessary and unused features
- Includes trained model and label encoder saved via **joblib**
- Easy-to-use inputs: Location, Land Area, Road Access, Floors, Bedrooms, Bathrooms, and House Age

---

## 📊 Dataset

- Source: Custom/Manual Kathmandu housing dataset
- Preprocessing steps:
  - Cleaned `PRICE` column by removing symbols and characters
  - Dropped irrelevant columns like `FACING` and `BUILT YEAR`
  - Encoded `LOCATION` using `LabelEncoder`

---

## 🧠 Model Training

We used a **Random Forest Regressor** for its robustness and high accuracy with tabular data.

### 📁 `train_model.py`

```bash
- Drops columns: ['FACING', 'BUILT YEAR']
- Cleans PRICE column using regex
- Encodes LOCATION using LabelEncoder
- Splits dataset (80% train / 20% test)
- Trains RandomForestRegressor with 100 trees
- Saves model as `house_price_model.pkl`
- Saves encoder as `location_encoder.pkl`
```

**R² Score Achieved:** ~0.67 (decent predictive performance)

---

## 💻 Streamlit Web App

### 📁 `app.py`

The frontend is a simple yet intuitive Streamlit app.

#### Inputs:
- 📍 Location (dropdown)
- 📐 Land Area (sqft)
- 🚧 Road Access (feet)
- 🏢 Floor Count
- 🛏️ Bedrooms
- 🛁 Bathrooms
- ⏳ House Age (in years)

#### Output:
- ✅ Predicted Price (formatted in Rupees)

#### Example:
> Estimated House Price: Rs. 23,50,000.00

---

## 🏗️ File Structure

```
.
├── train_model.py            # Training script
├── app.py                    # Streamlit web interface
├── house_price_model.pkl     # Trained ML model
├── location_encoder.pkl      # Saved label encoder
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

---

## ⚙️ Installation & Running Locally

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/kathmandu-house-price-predictor.git
cd kathmandu-house-price-predictor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 📦 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- streamlit
- re (Python built-in)

Create a `requirements.txt` using:

```bash
pip freeze > requirements.txt
```

---

## 🧠 Future Improvements

- Add confidence intervals or prediction range
- Improve UI/UX with images or map integration
- Use advanced models like XGBoost or LightGBM
- Add batch prediction from CSV upload
- Deploy on Streamlit Cloud or Render

---

## 🙌 Acknowledgments

This project was developed as part of a practical machine learning exercise using real-world housing data. Special thanks to contributors and mentors who helped refine and debug the preprocessing pipeline and frontend integration.

---

## 📬 Contact

Created by **Sabin Raj Pokharel**

📧 sabinrajpokhrel@icloud.com  
🌐 [LinkedIn](https://www.linkedin.com/in/sabin-raj-pokharel-039aa3278/)  
🐙 [GitHub](https://github.com/sabinrajpokhrel)
