# Importing pandas for data manipulation and analysis
import pandas as pd  
# Importing train_test_split for splitting data into training and testing sets
#Model selection is the process of choosing the best machine learning model from a set of candidate models. 
# It involves evaluating various models and their configurations to identify the one that best meets the performance criteria for a given task. 
# This process ensures that the chosen model generalizes well to unseen data and performs optimally on the problem at hand.
from sklearn.model_selection import train_test_split  
 # Importing RandomForestClassifier to create the random forest model
 #Ensemble learning is a machine learning technique where multiple models 
 # (often of the same or different types) are combined to improve predictive performance. 
 # Instead of relying on a single model's predictions, ensemble methods aggregate predictions from several models to produce a final prediction that is often more accurate and robust than any individual model in the ensemble.
from sklearn.ensemble import RandomForestClassifier 
# Importing metrics for model evaluation
#accuracy_score is used to evaluate the performance of the Random Forest Classifier on the test data. 
#Use accuracy when the dataset has a balanced class distribution.
#accuracy_score is a metric that measures the proportion of correctly predicted instances out of the total instances
#The classification_report function in scikit-learn is used to evaluate the performance of a classification model by providing a detailed summary of various classification metrics.
#  This function generates a text report showing the main classification metrics: precision, recall, F1-score, and support.
#A confusion matrix is a table used to evaluate the performance of a classification model. 
# It allows you to visualize the performance of an algorithm by showing the actual versus predicted classifications, making it easier to see where the model is getting things right and where it is making mistakes.
#Metrics in machine learning are quantitative measures used to evaluate the performance of a model. 
# They provide insights into how well a model is performing and help in comparing different models. Metrics are essential for understanding the effectiveness, accuracy, and robustness of a model in making predictions.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
 # Importing StandardScaler for feature scaling and LabelEncoder for encoding categorical features
 #Preprocessing refers to the series of steps taken to prepare raw data for machine learning models. 
 # It involves transforming data into a clean, suitable format that can be effectively used for training and evaluation. 
 #StandardScaler is a preprocessing technique in machine learning used to standardize features by removing the mean and scaling to unit variance. It transforms the data such that it has a mean of 0 and a standard deviation of 
 # 1. This transformation is applied independently to each feature (column) in the dataset.
 # Preprocessing typically includes tasks such as handling missing values, encoding categorical variables, normalizing or standardizing features, and more.
 #Label encoding is a technique used to convert categorical labels (textual labels) into numerical labels (integers). 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
# Importing joblib to save and load the model and preprocessors
import joblib  

# Function to load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)  # Reading the data from the CSV file and returning a DataFrame

# Function to preprocess data (including label encoding)
def preprocess_data(df):
    # Encode categorical features
    categorical_features = ['Gender', 'Item Purchased', 'Location', 'Size', 'Color', 'Season', 
                            'Subscription Status', 'Shipping Type', 'Discount Applied', 
                            'Promo Code Used', 'Payment Method', 'Frequency of Purchases']
    label_encoders = {}  # Dictionary to store the label encoders for each categorical feature
    
    for feature in categorical_features:
        le = LabelEncoder()  # Initializing a LabelEncoder for each categorical feature
        df[feature] = le.fit_transform(df[feature].astype(str))  # Encoding the categorical feature and updating the DataFrame
        label_encoders[feature] = le  # Storing the encoder for later use
    
    # Encode the target variable (Category)
    le_category = LabelEncoder()  # Initializing a LabelEncoder for the target variable
    df['Category'] = le_category.fit_transform(df['Category'])  # Encoding the target variable and updating the DataFrame
    
    return df, label_encoders, le_category  # Returning the processed DataFrame and the label encoders

# Function to split data into training and testing sets
def split_data(df):
    X = df.drop(columns=['Category'])  # Separating features (X) from the target variable (y)
    y = df['Category']
    return train_test_split(X, y, test_size=0.2, random_state=42)  # Splitting the data into training and testing sets with 80-20 ratio

# Main function
if __name__ == "__main__":
    try:
        # Load and preprocess the data
        df = load_data('shopping_trends_updated.csv')  # Loading the data from the CSV file
        df, label_encoders, le_category = preprocess_data(df)  # Preprocessing the data
        X_train, X_test, y_train, y_test = split_data(df)  # Splitting the data into training and testing sets

        # Standardize the data
        scaler = StandardScaler()  # Initializing a StandardScaler for feature scaling
        X_train_scaled = scaler.fit_transform(X_train)  # Fitting the scaler on training data and transforming it
        X_test_scaled = scaler.transform(X_test)  # Transforming the testing data using the same scaler

        # Train the Random Forest Classifier
        rfc = RandomForestClassifier(n_estimators=100, random_state=42)  # Initializing the RandomForestClassifier with 100 trees
        rfc.fit(X_train_scaled, y_train)  # Training the model on the scaled training data

        # Evaluate the model
        y_pred = rfc.predict(X_test_scaled)  # Predicting the target variable for the testing data
        print("Random Forest Classifier:")
        print("Accuracy:", accuracy_score(y_test, y_pred))  # Printing the accuracy of the model
        print("Classification Report:")
        print(classification_report(y_test, y_pred))  # Printing the classification report
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))  # Printing the confusion matrix

        # Save the model and the scaler
        joblib.dump(rfc, 'rfc_model.pkl')  # Saving the trained model to a file
        joblib.dump(scaler, 'scaler.pkl')  # Saving the scaler to a file
        joblib.dump(label_encoders, 'label_encoders.pkl')  # Saving the label encoders to a file
        joblib.dump(le_category, 'le_category.pkl')  # Saving the target variable label encoder to a file
        print("Model, scaler, and label encoders saved.")

    except Exception as e:
        print(f"An error occurred: {e}")  # Printing any errors that occur during the execution
