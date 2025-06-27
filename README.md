# ❤️ Heart Disease Predictor

A simple and interactive web application that predicts the risk of heart disease based on user input using a trained Random Forest Classifier.

Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/), this app provides quick predictions using patient health metrics.

---
## 📊 Dataset

- File: `heart_disease.csv`
- Columns:
  - `age`, `sex`, `chest pain type`, `resting bp s`, `cholesterol`,  
    `fasting blood sugar`, `resting ecg`, `max heart rate`,  
    `exercise angina`, `oldpeak`, `ST slope`, `target`


---

## 📘 Notebook

The entire machine learning workflow — including:
- Exploratory Data Analysis (EDA)
- Outlier handling (Winsorization)
- Preprocessing
- Model building (Logistic Regression & Random Forest)
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis

is documented step-by-step in [`heart_disease.ipynb`](./heart_disease.ipynb).

---

## 📦 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/heart-disease-app.git
   cd heart-disease-app
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure
```bash
heart-disease-app/
│
├── app.py                   # Streamlit web app
├── heart_disease.ipynb      # EDA + model building notebook
├── heart_disease.csv        # Input dataset
├── heart_disease_model.pkl  # Trained model (pipeline)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


```

## ⚠️ Disclaimer
>This tool is for educational/demo purposes only.
>It is **not intended for real-world medical use or diagnosis.**
