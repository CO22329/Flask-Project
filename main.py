from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('heart_model.pkl')  
scaler = joblib.load('scaler.pkl')

grade_map = {
    0: "Normal",
    1: "Medium Risk",
    2: "High Risk"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['cholesterol']),
            float(request.form['bp']),
            float(request.form['hr']),
            float(request.form['ecg'])
        ]

        features = scaler.transform([features])
        prediction = model.predict(features)[0]

        result_text = f"Predicted Heart Disease Grade: <b>{grade_map.get(prediction, 'Unknown')}</b>"
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
