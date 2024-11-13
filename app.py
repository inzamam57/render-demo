from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model, label_encoders = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        input_data = request.form.to_dict()
        features = []

        # Define the correct order of input features based on training
        feature_order = [
            'engine_temperature', 
            'oil_pressure', 
            'tire_pressure', 
            'battery_voltage', 
            'vehicle_speed', 
            'engine_rpm', 
            'coolant_level', 
            'brake_status', 
            'transmission_fluid_level'
        ]

        # Define categorical columns that need label encoding
        categorical_columns = ['coolant_level', 'brake_status', 'transmission_fluid_level']

        # Process input values
        for column in feature_order:
            value = input_data[column].strip().lower()  # Remove whitespace and convert to lowercase
            
            if column in categorical_columns:  # If column is categorical, use label encoder
                le = label_encoders.get(column)
                if le is not None:
                    # Check if the value is within the known labels
                    if value in le.classes_:
                        encoded_value = le.transform([value])[0]
                    else:
                        # Assign to most frequent known class if label is unseen
                        encoded_value = le.transform([le.classes_[0]])[0]
                    features.append(encoded_value)
            else:  # If column is numeric, convert directly to float
                features.append(float(value))

        # Convert features to numpy array
        features = np.array(features).reshape(1, -1)

        # Debug: Print the features to verify correctness
        print("Input features for prediction:", features)

        # Make prediction
        prediction = model.predict(features)
        output = 'Fault Detected' if prediction[0] == 1 else 'No Fault Detected'

        return render_template('index.html', prediction_text=f'Result: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
    
if __name__ == '__main__':
    app.run(debug=True)