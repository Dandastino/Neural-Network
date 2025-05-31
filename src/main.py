from layer.dense import Layer_Dense
from activations.relu import Activation_ReLU
from activations.softmax import Activation_Softmax
from losses.crossentropy import CategoricalCrossentropy
from optimizers.sgd import Optimizer_SGD
from utils.data_loader import load_gym_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def preprocess_data(data):
    # Encode the target (body part)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['BodyPart'])
    
    # Use CountVectorizer for better text preprocessing
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(data['Exercise Name']).toarray()
    
    return X, y, label_encoder, vectorizer

def create_model(input_size, hidden_size, output_size):
    dense1 = Layer_Dense(input_size, hidden_size)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(hidden_size, output_size)
    activation2 = Activation_Softmax()
    return dense1, activation1, dense2, activation2

def train_model(X_train, y_train, X_val, y_val, model, loss_fn, optimizer, epochs=100):
    dense1, activation1, dense2, activation2 = model
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        # Forward pass
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        
        # Loss
        loss = loss_fn.calculate(activation2.output, y_train)
        
        # Backward pass
        loss_fn.backward(activation2.output, y_train)
        activation2.backward(loss_fn.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        # Update weights
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        
        # Validation
        if epoch % 10 == 0:
            # Training metrics
            train_predictions = np.argmax(activation2.output, axis=1)
            train_accuracy = np.mean(train_predictions == y_train)
            
            # Validation metrics
            dense1.forward(X_val)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            val_predictions = np.argmax(activation2.output, axis=1)
            val_accuracy = np.mean(val_predictions == y_val)
            
            print(f"Epoch {epoch}, Loss: {loss:.3f}, Train Accuracy: {train_accuracy:.3f}, Val Accuracy: {val_accuracy:.3f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

def evaluate_model(X_test, y_test, model):
    dense1, activation1, dense2, activation2 = model
    
    # Forward pass
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    # Get predictions
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y_test)
    
    return accuracy, predictions

def main():
    # Load and preprocess data
    data = load_gym_data()
    X, y, label_encoder, vectorizer = preprocess_data(data)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Network architecture
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = len(np.unique(y))
    
    # Create model
    model = create_model(input_size, hidden_size, output_size)
    
    # Initialize loss and optimizer
    loss_fn = CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate=0.01)
    
    # Train model
    print("Starting training...")
    train_model(X_train, y_train, X_val, y_val, model, loss_fn, optimizer)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_accuracy, test_predictions = evaluate_model(X_test, y_test, model)
    print(f"Test Accuracy: {test_accuracy:.3f}")

if __name__ == "__main__":
    main()
