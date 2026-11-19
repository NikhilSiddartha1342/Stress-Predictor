# train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def generate_synthetic_data(n=5000, random_state=42):
    np.random.seed(random_state)
    # Features:
    # academic: percentage 30 - 100
    # sleep: hours per night 2 - 10
    # app_usage: hours per day 0 - 12
    # attendance: percentage 50 - 100
    # assignments_on_time: 0 or 1
    academic = np.random.normal(loc=75, scale=12, size=n).clip(30,100)
    sleep = np.random.normal(loc=6.5, scale=1.8, size=n).clip(2,10)
    app_usage = np.random.exponential(scale=2.5, size=n).clip(0,12)
    attendance = np.random.normal(loc=85, scale=10, size=n).clip(50,100)
    assignments_on_time = np.random.binomial(1, p=0.8, size=n)

    df = pd.DataFrame({
        "academic": academic,
        "sleep": sleep,
        "app_usage": app_usage,
        "attendance": attendance,
        "assignments_on_time": assignments_on_time
    })

    # Create label: heuristic that combines features
    # score lower -> higher stress
    # normalize features roughly and compute stress_score
    a = (100 - df["academic"]) / 100.0        # worse academic increases stress
    b = (8 - df["sleep"]) / 8.0               # less sleep increases stress
    c = df["app_usage"] / 12.0                # more app usage increases stress
    d = (100 - df["attendance"]) / 100.0      # low attendance increases stress
    e = 1 - df["assignments_on_time"]         # missed assignments increases stress

    stress_score = 0.4*a + 0.25*b + 0.2*c + 0.1*d + 0.05*e + np.random.normal(0, 0.02, n)

    # Map to labels
    # stress_score in [~0, ~1], choose thresholds
    labels = np.where(stress_score < 0.25, 0, np.where(stress_score < 0.55, 1, 2))

    df["stress_level"] = labels
    return df

def train_and_save(df):
    # --------- Prepare Features and Labels ---------
    X = df[['academic', 'sleep', 'app_usage', 'attendance', 'assignments_on_time']].values
    y = df['stress_level'].values  # Keep as NumPy array for modeling
    
    # --------- Check Class Distribution ---------
    y_series = pd.Series(y)
    print("Class distribution:")
    print(y_series.value_counts())
    
    # --------- Decide stratification ---------
    if y_series.value_counts().min() < 2:
        print("Warning: Some classes have very few samples. Using non-stratified split.")
        stratify_option = None
    else:
        stratify_option = y
    
    # --------- Split Dataset ---------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.12, random_state=42, stratify=stratify_option
    )
    
    # --------- Scaling ---------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --------- Train Model ---------
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # --------- Evaluate ---------
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # --------- Save Model and Scaler ---------
    with open("stress_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(n=5000)
    print("Saving dataset to data/synthetic_stress.csv")
    df.to_csv("data/synthetic_stress.csv", index=False)
    print("Training model...")
    train_and_save(df)
    print("Model and scaler saved to models/")
