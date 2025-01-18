import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# import joblib

# Load the preprocessed dataset
with open('sign_data.pickle', 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])  # Features
labels = np.array(dataset['labels'])  # Labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)
score = accuracy_score(y_predict, y_test)

print(f"{score*100:.2f}% of samples were classified accurately ")
print("Classification Report:")
print(classification_report(y_test, y_predict))
print("Model training completed ")

# Save the model to a file
import pickle

# Save the trained model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved as 'random_forest_model.pkl'.")






