import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time

# Step 1: Load and preprocess the data
data = pd.read_csv('C:/Users/anvin/Desktop/Third Year III Internal Project/Machine learning/UNSW_NB15_training-set.csv')
data.dropna(inplace=True)

# Encode the 'attack_cat' column
label_encoder = LabelEncoder()
data['attack_cat_encoded'] = label_encoder.fit_transform(data['attack_cat'])

# Identify and handle columns with non-numeric values
non_numeric_columns = ['proto', 'service', 'state']  # Add column names with non-numeric values

# Handle non-numeric columns (e.g., encoding categorical variables)
for column in non_numeric_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Define the features and target
X = data.drop(['label', 'attack_cat', 'attack_cat_encoded'], axis=1)
y = data['attack_cat_encoded']

# Step 2: Data Preprocessing
scaler = MinMaxScaler()
X[['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']] = scaler.fit_transform(X[['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']])

# Feature Selection
selector = mutual_info_classif  # Change feature selection method
X_selected = selector(X, y)

selected_features = X.columns[X_selected > 0]  # Get the selected feature names

X = X[selected_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Random Forest Classifier
clf = RandomForestClassifier()

start_time = time.time()  # Record start time
clf.fit(X_train, y_train)
end_time = time.time()  # Record end time

# Step 4: Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results and execution time
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Execution Time: {end_time - start_time} seconds")
