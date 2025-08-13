#!/usr/bin/env python3
# Neural Network for Iris Dataset - Terminal Version

print("🚀 Starting Neural Network in Terminal...")

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"📊 Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📈 Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Build and train neural network
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    verbose=True
)

print("\n🧠 Training neural network...")
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"\n✅ Results:")
print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy:.2%})")
print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")

# Make predictions
y_pred = model.predict(X_test)
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print(f"\n🎉 Neural Network completed successfully!")
print(f"Final Test Accuracy: {test_accuracy:.2%}")
