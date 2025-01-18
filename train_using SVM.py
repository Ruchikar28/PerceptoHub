import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
with open('sign_data.pickle', 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])  # Features (landmarks or images)
labels = np.array(dataset['labels'])  # Labels

# Encode labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Scale the features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Define the SVM model with a radial basis function (RBF) kernel
svm_model = SVC(kernel='rbf', probability=True)

# Train the SVM model
svm_model.fit(X_train_scaled, y_train)

# Print the training completion message
print("SVM model trained successfully!")


# Define the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

print("Logistic Regression model trained successfully!")


# Make predictions on the test set
y_pred1 = svm_model.predict(X_test_scaled)
y_pred2 = logistic_model.predict(X_test_scaled)

# Calculate accuracy
accuracy1 = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"Logistic Regression Test Accuracy: {accuracy2 * 100:.2f}%")
print(f"SVM Test Accuracy: {accuracy1 * 100:.2f}%")

# Save the trained model
with open('svm_sign_detection_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
print("Model saved as 'svm_sign_detection_model.pkl'.")

with open('Logistic_Regression_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
print("Model saved as 'Logistic_Regression_model.pkl'.")


