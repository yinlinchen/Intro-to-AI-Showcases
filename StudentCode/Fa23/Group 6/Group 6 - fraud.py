import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset with pattern recognition features
data = pd.read_csv('credit_card_fraud_pattern_recognition_data.csv')  # Replace with your dataset path

# One-Hot Encoding for 'transaction_location'
encoder = OneHotEncoder(sparse=False)
encoded_locations = encoder.fit_transform(data[['transaction_location']])

# Convert the encoded location data back to a DataFrame
encoded_locations_df = pd.DataFrame(encoded_locations, columns=encoder.get_feature_names_out(['transaction_location']))

# Add the encoded location data back to the original DataFrame
data = pd.concat([data, encoded_locations_df], axis=1)

# Drop the original 'transaction_location' column
data.drop('transaction_location', axis=1, inplace=True)

# Feature Selection
features = data.drop(['is_fraud', 'user_id'], axis=1)
labels = data['is_fraud']

# Manually split the dataset
split_index = int(len(data) * 0.8)
X_train = data.iloc[:split_index].drop('is_fraud', axis=1)
y_train = data.iloc[:split_index]['is_fraud']
X_test = data.iloc[split_index:].drop('is_fraud', axis=1)
y_test = data.iloc[split_index:]['is_fraud']

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))

# Pattern Recognition - Focus on unusual transaction amounts
average_amount_by_user = data.groupby('user_id')['amount'].mean()

# Check for unusual patterns in the test set
for index, row in X_test.iterrows():
    user_id = row['user_id']
    if row['amount'] > average_amount_by_user[user_id] * 1.5:  # Check if the amount is unusually high
        print(f"Unusual transaction size for user {user_id}")

# Anomaly Detection (based on amount threshold)
anomaly_threshold = 3000
anomalies = X_test[X_test['amount'] > anomaly_threshold]
if not anomalies.empty:
    print("\nAnomaly Found!")
    print(anomalies)
