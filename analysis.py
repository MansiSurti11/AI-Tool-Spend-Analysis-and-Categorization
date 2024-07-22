# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhanced visualizations
from sklearn.model_selection import train_test_split, cross_val_score  # For splitting data and cross-validation
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction using TF-IDF
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For label encoding and data scaling
from sklearn.ensemble import RandomForestClassifier  # For RandomForest classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For model evaluation
from scipy.stats import ttest_ind, f_oneway  # For statistical tests

# Function to load data from CSV file and preprocess it
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)  # Read CSV file into a DataFrame

    # Combine relevant text columns for better feature extraction
    df['Shipping Type'] = df['Item Purchased'] + ' ' + df['Size'] + ' ' + df['Color'] + ' ' + df['Season']

    # Encode the target labels using LabelEncoder
    label_encoder = LabelEncoder()
    df['Category'] = label_encoder.fit_transform(df['Category'])
    
    return df, label_encoder  # Return preprocessed DataFrame and LabelEncoder object

# Function to prepare features using TF-IDF vectorization
def prepare_features_and_labels(df):
    X = df['Shipping Type']  # Use the combined text column for feature extraction

    # TF-IDF Vectorizer for text feature extraction
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)  # Transform text data into TF-IDF features

    y = df['Category']  # Target variable: Category
    
    return X_tfidf, y, vectorizer  # Return TF-IDF features, target labels, and TF-IDF vectorizer object

# Function to split data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into 80% training and 20% testing

# Function to train a Random Forest Classifier model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize RandomForest Classifier
    model.fit(X_train, y_train)  # Train the model using training data
    return model  # Return trained RandomForest model

# Function to evaluate the trained model
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)  # Predict using the trained model
    y_test_decoded = label_encoder.inverse_transform(y_test)  # Decode numerical labels back to original categories
    y_pred_decoded = label_encoder.inverse_transform(y_pred)  # Decode predicted labels
    
    # Print classification report, confusion matrix, and accuracy score
    print("Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_decoded, y_pred_decoded))
    
    print("Accuracy Score:")
    print(accuracy_score(y_test_decoded, y_pred_decoded))

# Function to plot a histogram
def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))  # Set figure size
    sns.histplot(df[column], kde=True)  # Plot histogram with KDE
    plt.title(f'Histogram of {column}')  # Set plot title
    plt.xlabel(column)  # Set x-axis label
    plt.ylabel('Frequency')  # Set y-axis label
    plt.show()  # Display the plot

# Function to plot a bar chart
def plot_bar_chart(df, column):
    plt.figure(figsize=(10, 6))  # Set figure size
    sns.countplot(data=df, x=column)  # Plot bar chart
    plt.title(f'Bar Chart of {column}')  # Set plot title
    plt.xlabel(column)  # Set x-axis label
    plt.ylabel('Count')  # Set y-axis label
    plt.show()  # Display the plot

# Function to plot a scatter plot
def plot_scatter(df, x_column, y_column):
    plt.figure(figsize=(10, 6))  # Set figure size
    sns.scatterplot(data=df, x=x_column, y=y_column)  # Plot scatter plot
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')  # Set plot title
    plt.xlabel(x_column)  # Set x-axis label
    plt.ylabel(y_column)  # Set y-axis label
    plt.show()  # Display the plot

# Function to plot a correlation matrix
def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))  # Set figure size
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select numeric columns
    correlation_matrix = numeric_df.corr()  # Compute correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)  # Plot heatmap
    plt.title('Correlation Matrix')  # Set plot title
    plt.show()  # Display the plot

# Function to perform a t-test
def perform_t_test(df, column, group_col, group1, group2):
    group1_data = df[df[group_col] == group1][column]  # Filter data for group 1
    group2_data = df[df[group_col] == group2][column]  # Filter data for group 2
    t_stat, p_value = ttest_ind(group1_data, group2_data)  # Perform t-test
    print(f"T-test between {group1} and {group2} for {column}:")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")  # Print t-statistic and p-value

# Function to perform an ANOVA
def perform_anova(df, column, group_col):
    groups = df[group_col].unique()  # Get unique groups from group column
    group_data = [df[df[group_col] == group][column] for group in groups]  # Extract data for each group
    f_stat, p_value = f_oneway(*group_data)  # Perform ANOVA
    print(f"ANOVA for {column} by {group_col}:")
    print(f"F-statistic: {f_stat}, P-value: {p_value}")  # Print F-statistic and p-value

# Function to perform statistical analysis
def perform_statistical_analysis(df):
    # Example t-test: Compare 'Purchase Amount (USD)' between 'Male' and 'Female'
    perform_t_test(df, 'Purchase Amount (USD)', 'Gender', 'Male', 'Female')
    
    # Example ANOVA: Compare 'Purchase Amount (USD)' across different 'Season'
    perform_anova(df, 'Purchase Amount (USD)', 'Season')

# Function to cross-validate the model
def cross_validate_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize RandomForest Classifier
    scores = cross_val_score(model, X, y, cv=5)  # Perform 5-fold cross-validation
    print("Cross-Validation Scores:", scores)  # Print cross-validation scores
    print("Mean Cross-Validation Score:", scores.mean())  # Print mean cross-validation score

if __name__ == "__main__":
    # Load and preprocess data
    df, label_encoder = load_and_preprocess_data('shopping_trends_updated.csv')
    X_tfidf, y, vectorizer = prepare_features_and_labels(df)
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y)
    print("Data loaded and preprocessed.")
    
    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Cross-validate the model
    cross_validate_model(X_tfidf, y)
    
    # Generate multiple graphs for exploratory data analysis
    plot_histogram(df, 'Age')
    plot_bar_chart(df, 'Gender')
    plot_scatter(df, 'Age', 'Purchase Amount (USD)')
    plot_correlation_matrix(df)
    
    # Perform statistical analysis
    perform_statistical_analysis(df)
