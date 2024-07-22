# Import necessary libraries
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)

# Enable CORS for the application to allow cross-origin requests
CORS(app)

# Load the pre-trained model and label encoders from the saved files
try:
    model = joblib.load('rfc_model.pkl')         # Load the trained RandomForestClassifier model
    scaler = joblib.load('scaler.pkl')           # Load any scaler or preprocessing object used during training (if applicable)
    label_encoders = joblib.load('label_encoders.pkl')   # Load label encoders used for categorical features
    le_category = joblib.load('le_category.pkl')   # Load label encoder used for encoding the target variable (Category)
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    raise e

# Route to handle POST requests for classification
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()   # Get the JSON payload from the POST request
    print(f"Received data: {data}")  # Log the received data for debugging purposes

    try:
        df = pd.DataFrame(data)   # Convert JSON data to a pandas DataFrame
        
        # Initialize an empty list for predictions
        predictions = []

        # Iterate through label encoders for categorical features
        for feature, le in label_encoders.items():
            if feature in df.columns:   # Check if the feature exists in the DataFrame columns
                # Check for unseen labels and handle them (replace with a default value)
                unseen_labels = set(df[feature].astype(str)) - set(le.classes_)
                if unseen_labels:
                    df.loc[df[feature].isin(unseen_labels), feature] = 'Unknown'

                # Transform categorical features using label encoder
                df[feature] = le.transform(df[feature].astype(str))
            else:
                return jsonify({'error': f'Missing feature: {feature}'}), 400   # Return error if feature is missing

        # Example of data standardization if needed (using scaler)
        # Example: df_scaled = scaler.transform(df)

        # Make predictions using the loaded model
        predictions = model.predict(df)   # Assuming the model predicts labels

        # Return predictions as JSON response
        return jsonify({'predictions': predictions.tolist()})   # Convert predictions to list and return as JSON

    except KeyError as e:
        return jsonify({'error': f'Key error: {str(e)}'}), 400   # Handle KeyError exceptions
    except Exception as e:
        return jsonify({'error': str(e)}), 400   # Handle other exceptions

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5001)   # Run the application in debug mode on port 5001
