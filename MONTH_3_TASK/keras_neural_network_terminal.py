#!/usr/bin/env python3
# Keras Neural Network for Iris Dataset - Terminal Version

print("🚀 Starting Keras Neural Network in Terminal...")

try:
    # Import required libraries
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    
    print("✅ TensorFlow imported successfully!")
    
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    
    print(f"📊 Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # One-hot encode target
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Build model
    model = Sequential([
        Dense(16, input_shape=(X.shape[1],), activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n🏗️ Model Architecture:")
    model.summary()
    
    # Train model
    print(f"\n🧠 Training neural network...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✅ Results:")
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy:.2%})")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    
    print(f"\n🎉 Keras Neural Network completed successfully!")
    print(f"Final Test Accuracy: {test_accuracy:.2%}")
    
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("💡 Please run the sklearn version instead: python neural_network_terminal.py")
except Exception as e:
    print(f"⚠️ Error: {e}")
