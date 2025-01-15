import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('student_performance.csv')
# data = data.drop(columns=['health_status'])

# Preprocess data
encoder = LabelEncoder()
data['socio_economic_status_encoded'] = encoder.fit_transform(data['socio_economic_status'])

# Define features and target
X = data[['attendance', 'study_time', 'socio_economic_status_encoded','health_status']]
y = data['final_grades']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model, encoder, and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
