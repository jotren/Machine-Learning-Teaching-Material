import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# Read the CSV file
df = pd.read_csv(r'C:\projects\machine-learning-practise\palmer-penguins\data\penguin_data.csv')
df.dropna(inplace=True)

# Define a function to map 'island' values to numeric values
def map_island_to_numeric(island_name):
    island_mapping = {
        'Torgersen': 0,
        'Biscoe': 1,
        'Dream': 2
    }
    
    return island_mapping.get(island_name, -1)  # Return -1 for unknown or missing values

# Apply the function to create a new 'island_numeric' column
df['island_numeric'] = df['island'].apply(map_island_to_numeric)

# Define a function to map 'sex' values to 0 for male and 1 for female
def map_sex_to_binary(sex):
    if sex == 'male':
        return 0
    elif sex == 'female':
        return 1
    else:
        return None  # Handle missing or other cases if necessary

# Apply the function to create a new 'sex_binary' column
df['sex_binary'] = df['sex'].apply(map_sex_to_binary)

# Define the features (X) and target (y)
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_binary', 'island_numeric', 'year']]
y = df['species']

# Split the data into a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the feature values (important for all classifiers)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions
rf_predictions = rf_classifier.predict(X_test_scaled)

# Evaluate the Random Forest classifier
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)

# Display the results
print("Random Forest Classifier Accuracy:", rf_accuracy)
print("Random Forest Classifier Confusion Matrix:\n", rf_confusion_matrix)
print("Random Forest Classifier Classification Report:\n", rf_classification_report)

# Define and train an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_classifier.fit(X_train_scaled, y_train)
xgb_predictions = xgb_classifier.predict(X_test_scaled)

# Evaluate the XGBoost classifier
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_confusion_matrix = confusion_matrix(y_test, xgb_predictions)
xgb_classification_report = classification_report(y_test, xgb_predictions)

print("\nXGBoost Classifier Accuracy:", xgb_accuracy)
print("XGBoost Classifier Confusion Matrix:\n", xgb_confusion_matrix)
print("XGBoost Classifier Classification Report:\n", xgb_classification_report)

# Define and train a Support Vector Machine classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions
svm_predictions = svm_classifier.predict(X_test_scaled)

# Evaluate the SVM classifier
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
svm_classification_report = classification_report(y_test, svm_predictions)

# Display the results
print("Support Vector Machine Classifier Accuracy:", svm_accuracy)
print("Support Vector Machine Classifier Confusion Matrix:\n", svm_confusion_matrix)
print("Support Vector Machine Classifier Classification Report:\n", svm_classification_report)

# Define and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=10)  # You can adjust the number of neighbors (k)
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions
knn_predictions = knn_classifier.predict(X_test_scaled)

# Evaluate the KNN classifier
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions)
knn_classification_report = classification_report(y_test, knn_predictions)

# Display the results
print("K-Nearest Neighbors Classifier Accuracy:", knn_accuracy)
print("K-Nearest Neighbors Classifier Confusion Matrix:\n", knn_confusion_matrix)
print("K-Nearest Neighbors Classifier Classification Report:\n", knn_classification_report)


