import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── 1. Page Configuration & Responsive Setup ──────────────────────────────────
st.set_page_config(
    page_title="Sleep Health Predictor", 
    page_icon="🌙", 
    layout="centered" # Mobile aur desktop dono ke liye perfect layout
)

# Custom Design for Premium Look & Big Touch Buttons for Mobile
st.markdown("""
    <style>
    .stButton>button {
        width: 100% !important;
        height: 50px;
        font-size: 18px !important;
        border-radius: 8px;
    }
    div[data-testid="stForm"] {
        border-radius: 10px;
        padding: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Assets ko instantly load karein
@st.cache_resource
def load_pure_assets():
    with open('light_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns_order.pkl', 'rb') as f:
        X_columns = pickle.load(f)
    return model, scaler, X_columns

try:
    model, scaler, X_columns = load_pure_assets()
except Exception as e:
    st.error("Missing Files! Pehle terminal mein 'python3 train_model.py' chalayein.")
    st.stop()

st.title("🌙 Sleep Health & Lifestyle Predictor")
st.write("Enter your daily routines and stats below to instantly check sleep disorder risks.")
st.divider()

# ── 2. User Input Form ────────────────────────────────────────────────────────
with st.form("sleep_form"):
    st.subheader("📋 Personal & Lifestyle Info")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (Years)", min_value=10, max_value=100, value=35)
        occupation = st.selectbox("Occupation", ["Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Salesperson", "Scientist", "Software Developer", "Teacher", "Sales Representative"])
        sleep_duration = st.slider("Sleep Duration (Hours)", 4.0, 10.0, 7.0, 0.1)
        quality_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
        
    with col2:
        physical_activity = st.slider("Physical Activity (Mins/Day)", 0, 120, 60)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        bmi_category = st.selectbox("BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"])
        blood_pressure = st.selectbox("Blood Pressure", ["120/80", "130/85", "140/90", "115/75", "125/80", "135/88"])
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=72)
        
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=7000)
    submit = st.form_submit_button("🔮 Analyze Sleep Risk")

# ── 3. Prediction & 3-Condition Logic ─────────────────────────────────────────
if submit:
    # 1. Base dictionary banayein saare numerical columns ke sath
    input_data = {
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps
    }
    
    # 2. Training ke saare One-Hot encoded columns ko check karke 0 ya 1 assign karein
    for col in X_columns:
        if col not in input_data:
            if col.startswith("Gender_") and col.split("_")[1] == gender:
                input_data[col] = 1
            elif col.startswith("Occupation_") and col.split("_")[1] == occupation:
                input_data[col] = 1
            elif col.startswith("BMI Category_") and col.split("_")[1] == bmi_category:
                input_data[col] = 1
            elif col.startswith("Blood Pressure_") and col.split("_")[1] == blood_pressure:
                input_data[col] = 1
            else:
                input_data[col] = 0

    # 3. DataFrame banayein aur X_columns ke exact order mein columns ko set karein
    final_features = pd.DataFrame([input_data])[X_columns]
            
    # 4. Standard Scaler se features transform karein
    scaled_features = scaler.transform(final_features)
    
    # 5. Model prediction probabilities calculation
    prob_array = model.predict_proba(scaled_features)
    prob = float(prob_array[0][1]) 
    
    st.write("")
    st.subheader("📊 Assessment Result")
    
    # ==============================================================================
    # 🚨 THREE DYNAMIC CONDITIONS OUTPUT
    # ==============================================================================
    
    # CONDITION 1: HIGH RISK (Probability above 70%)
    if prob >= 0.70:
        st.error(f"🚨 **High Risk:** The model tracks severe signs of a sleep disorder. (Risk: {prob:.1%})")
        st.info("👉 **Advice:** High stress and low sleep window are heavily impacting you. Consider checking with a medical doctor.")
    
    # CONDITION 2: MODERATE RISK (Probability between 35% and 70%)
    elif 0.35 <= prob < 0.70:
        st.warning(f"⚠️ **Moderate Risk:** Elevated risk variables detected. (Risk: {prob:.1%})")
        st.info("👉 **Advice:** Your routine is shifting towards imbalance. Try reducing daily stress, expanding your sleep hours, and increasing daily steps.")
        
    # CONDITION 3: LOW RISK (Probability below 35%)
    else:
        st.success(f"✅ **Low Risk:** Your sleep patterns look completely healthy and stable! (Risk: {prob:.1%})")
        st.info("👉 **Advice:** Keep doing what you are doing! Your activity, weight management, and sleep timing are well balanced.")
        
    st.progress(prob)
