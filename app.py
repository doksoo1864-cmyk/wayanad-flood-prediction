from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'flood_model.pkl'))
FEATURES = joblib.load(os.path.join(BASE_DIR, 'feature_names.pkl'))

FEATURE_LABELS = {
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'rainfall_mm': 'Rainfall Today (mm)',
    'cum_rain_since_window_start_mm': 'Cumulative Rain This Window (mm)',
    'rolling_3d_rain_mm': 'Rain Last 3 Days (mm)',
    'rolling_5d_rain_mm': 'Rain Last 5 Days (mm)',
    'rolling_10d_rain_mm': 'Rain Last 10 Days (mm)',
    'month': 'Month (1-12)',
    'day': 'Day of Month',
    'dayofyear': 'Day of Year (1-366)'
}

FEATURE_DEFAULTS = {
    'latitude': 11.6,
    'longitude': 76.1,
    'rainfall_mm': 50,
    'cum_rain_since_window_start_mm': 200,
    'rolling_3d_rain_mm': 100,
    'rolling_5d_rain_mm': 150,
    'rolling_10d_rain_mm': 250,
    'month': 8,
    'day': 15,
    'dayofyear': 227
}

@app.route('/')
def home():
    features_info = [
        {
            'name': f,
            'label': FEATURE_LABELS.get(f, f),
            'default': FEATURE_DEFAULTS.get(f, 0)
        }
        for f in FEATURES
    ]
    return render_template('index.html', features_info=features_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        feature_vector = [float(data[f]) for f in FEATURES]
        prediction = int(model.predict([feature_vector])[0])
        probabilities = model.predict_proba([feature_vector])[0].tolist()
        if hasattr(model, 'classes_') and 1 in list(model.classes_):
            flood_idx = list(model.classes_).index(1)
            flood_prob = probabilities[flood_idx] * 100
        else:
            flood_prob = probabilities[-1] * 100
        if flood_prob >= 70:
            risk = "HIGH"
        elif flood_prob >= 40:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        return jsonify({
            'prediction': 'FLOOD' if prediction == 1 else 'NO FLOOD',
            'flood_probability': round(flood_prob, 2),
            'risk_level': risk
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
