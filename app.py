from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model, encoder, and scaler
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

UPLOAD_FOLDER = 'uploads'
PREDICTED_FOLDER = 'predicted'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read the uploaded file
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        return f"Error reading the file: {str(e)}", 400

    # Encode and preprocess the new data
    try:
        data['socio_economic_status_encoded'] = encoder.transform(data['socio_economic_status'])
    except ValueError:
        return "The file contains unseen labels in 'socio_economic_status'. Please ensure all labels are known.", 400

    # Drop the original 'socio_economic_status' column
    data = data.drop('socio_economic_status', axis=1)

    # Scale the features
    scaled_data = scaler.transform(data[['attendance', 'study_time', 'socio_economic_status_encoded','health_status']])

    # Make predictions
    predictions = model.predict(scaled_data)
    data['predicted_final_grade'] = predictions

    # Save the predicted file
    predicted_file_path = os.path.join(PREDICTED_FOLDER, 'predicted_' + file.filename)
    data.to_csv(predicted_file_path, index=False)

    return send_file(predicted_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
