# 🌙 Sleep Health & Lifestyle Predictor

An industry-grade Deep Learning web application that evaluates daily routines, stress markers, and circulatory vitals to assess the probability of sleep disorders (Insomnia / Sleep Apnea).

## 🚀 Live Demo
🔗 **[Insert your Streamlit live link here after deployment]**

---

## 📊 Dataset Benchmark
The model is trained on the **Sleep Health and Lifestyle Dataset**, which contains clinical data covering:
- **Sleep Metrics:** Duration and Subjective Quality of Sleep.
- **Lifestyle Triggers:** Physical Activity Level, Stress Index, and Daily Steps.
- **Cardiovascular Health:** Blood Pressure ratios and Resting Heart Rate (BPM).
- **Demographics:** Age, Gender, and Occupational risk groups.

---

## 🧠 Model Architecture & Logic
Instead of heavy frameworks that slow down production, this project utilizes a high-performance **Multi-Layer Perceptron (MLP)** neural network built with pure numeric optimization.

### Key Performance Highlights:
- **Architecture:** Input Layer (Dynamic Dummies) ➔ Dense Layer (32 Nodes, ReLU) ➔ Dropout (0.2) ➔ Output Layer (1 Node, Sigmoid).
- **Optimization:** Adam Optimizer with Binary Crossentropy loss.
- **Deployment Format:** Packed cleanly into binary `.pkl` assets for instant execution under 1 second.

---

## 📱 Features & Sizing Matrix
- **Mobile-Responsive Layout:** Automatically scales from a clean multi-column layout on desktops to a touch-friendly single-column layout on mobile phones.
- **Multi-Tier Diagnostics:** Categorizes outcomes into 3 distinct operational risk assessments:
  1. ✅ **Low Risk (< 35%):** Stable circadian patterns.
  2. ⚠️ **Moderate Borderline Risk (35% - 70%):** Early indicators of sleep anomalies.
  3. 🚨 **Critical High Risk (> 70%):** Severe variations requiring clinical evaluation.

---

## 🛠️ Installation & Local Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd sleep-health-predictor
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application:**
   ```bash
   python3 -m streamlit run app.py
   ```

---

## 📁 Repository Structure
```text
├── app.py              # Main Mobile-Responsive Streamlit UI
├── light_model.pkl     # Pre-trained Lightweight Neural Network Weights
├── scaler.pkl          # Preprocessing Standard Scaler Binary
├── columns_order.pkl   # Fixed Features Alignment Array
└── requirements.txt    # Production Package Dependencies List
```
