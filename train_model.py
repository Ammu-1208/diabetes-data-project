import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

print("Loading dataset...")

# Load dataset
data = pd.read_csv(r"C:\Users\yayav\OneDrive\Desktop\diabetes\diabetes.csv")

# Encode categorical columns
gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()

data['gender'] = gender_encoder.fit_transform(data['gender'])
data['smoking_history'] = smoking_encoder.fit_transform(data['smoking_history'])

# Features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl")