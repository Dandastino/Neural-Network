from layer.dense import Layer_Dense
from layer.batch_norm import Layer_BatchNorm
from layer.dropout import Layer_Dropout
from activations.relu import Activation_ReLU
from activations.softmax import Activation_Softmax
from losses.crossentropy import CategoricalCrossentropy
from optimizers.adam import Optimizer_Adam
from utils.data_loader import load_gym_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from tqdm import tqdm
import time
import pandas as pd

def preprocess_data(data):
    print("Preprocessing data...")
    start_time = time.time()
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['BodyPart'])
    
    data['Desc'] = data['Desc'].fillna('')
    data['combined_text'] = data['Title'] + ' ' + data['Desc']
    
    vectorizer = CountVectorizer(
        max_features=2000, 
        stop_words='english',
        ngram_range=(1, 5),  
        min_df=10,
        max_df=0.70  
    )
    
    X = vectorizer.fit_transform(data['combined_text']).toarray()
    
    equipment_encoder = LabelEncoder()
    equipment_encoded = equipment_encoder.fit_transform(data['Equipment'])
    equipment_encoded = equipment_encoded.reshape(-1, 1)
    
    level_encoder = LabelEncoder()
    level_encoded = level_encoder.fit_transform(data['Level'])
    level_encoded = level_encoded.reshape(-1, 1)
    
    X = np.hstack([X, equipment_encoded, level_encoded])
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    return X, y, label_encoder, vectorizer

def create_model(input_size, hidden_size, output_size):
    dense1 = Layer_Dense(input_size, hidden_size * 2)
    batch_norm1 = Layer_BatchNorm()
    activation1 = Activation_ReLU()
    dropout1 = Layer_Dropout(rate=0.3)  
    
    dense2 = Layer_Dense(hidden_size * 2, hidden_size)
    batch_norm2 = Layer_BatchNorm()
    activation2 = Activation_ReLU()
    dropout2 = Layer_Dropout(rate=0.2)
    
    dense3 = Layer_Dense(hidden_size, hidden_size // 2)
    batch_norm3 = Layer_BatchNorm()
    activation3 = Activation_ReLU()
    dropout3 = Layer_Dropout(rate=0.1)
    
    dense4 = Layer_Dense(hidden_size // 2, output_size)
    activation4 = Activation_Softmax()
    
    return (dense1, batch_norm1, activation1, dropout1,
            dense2, batch_norm2, activation2, dropout2,
            dense3, batch_norm3, activation3, dropout3,
            dense4, activation4)

def train_model(X_train, y_train, X_val, y_val, model, loss_fn, optimizer, epochs=200):
    dense1, batch_norm1, activation1, dropout1, dense2, batch_norm2, activation2, dropout2, dense3, batch_norm3, activation3, dropout3, dense4, activation4 = model
    best_val_accuracy = 0
    patience = 20 
    patience_counter = 0
    best_epoch = 0
    
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc="Training"):
        dense1.forward(X_train)
        batch_norm1.forward(dense1.output)
        activation1.forward(batch_norm1.output)
        dropout1.forward(activation1.output)
        
        dense2.forward(dropout1.output)
        batch_norm2.forward(dense2.output)
        activation2.forward(batch_norm2.output)
        dropout2.forward(activation2.output)
        
        dense3.forward(dropout2.output)
        batch_norm3.forward(dense3.output)
        activation3.forward(batch_norm3.output)
        dropout3.forward(activation3.output)
        
        dense4.forward(dropout3.output)
        activation4.forward(dense4.output)
        
        loss = loss_fn.calculate(activation4.output, y_train)
        
        loss_fn.backward(activation4.output, y_train)
        activation4.backward(loss_fn.dinputs)
        dense4.backward(activation4.dinputs)
        
        dropout3.backward(dense4.dinputs)
        activation3.backward(dropout3.dinputs)
        batch_norm3.backward(activation3.dinputs)
        dense3.backward(batch_norm3.dinputs)
        
        dropout2.backward(dense3.dinputs)
        activation2.backward(dropout2.dinputs)
        batch_norm2.backward(activation2.dinputs)
        dense2.backward(batch_norm2.dinputs)
        
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        batch_norm1.backward(activation1.dinputs)
        dense1.backward(batch_norm1.dinputs)
        
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.update_params(dense4)
        
        if epoch % 10 == 0:
            train_predictions = np.argmax(activation4.output, axis=1)
            train_accuracy = np.mean(train_predictions == y_train)
            
            dense1.forward(X_val)
            batch_norm1.forward(dense1.output, training=False)
            activation1.forward(batch_norm1.output)
            dropout1.forward(activation1.output, training=False)
            
            dense2.forward(dropout1.output)
            batch_norm2.forward(dense2.output, training=False)
            activation2.forward(batch_norm2.output)
            dropout2.forward(activation2.output, training=False)
            
            dense3.forward(dropout2.output)
            batch_norm3.forward(dense3.output, training=False)
            activation3.forward(batch_norm3.output)
            dropout3.forward(activation3.output, training=False)
            
            dense4.forward(dropout3.output)
            activation4.forward(dense4.output)
            
            val_predictions = np.argmax(activation4.output, axis=1)
            val_accuracy = np.mean(val_predictions == y_val)
            
            print(f"\nEpoch {epoch}, Loss: {loss:.3f}, Train Accuracy: {train_accuracy:.3f}, Val Accuracy: {val_accuracy:.3f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation accuracy: {best_val_accuracy:.3f} at epoch {best_epoch}")
                break
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")

def evaluate_model(X_test, y_test, model):
    dense1, batch_norm1, activation1, dropout1, dense2, batch_norm2, activation2, dropout2, dense3, batch_norm3, activation3, dropout3, dense4, activation4 = model
    
    dense1.forward(X_test)
    batch_norm1.forward(dense1.output, training=False)
    activation1.forward(batch_norm1.output)
    dropout1.forward(activation1.output, training=False)
    
    dense2.forward(dropout1.output)
    batch_norm2.forward(dense2.output, training=False)
    activation2.forward(batch_norm2.output)
    dropout2.forward(activation2.output, training=False)
    
    dense3.forward(dropout2.output)
    batch_norm3.forward(dense3.output, training=False)
    activation3.forward(batch_norm3.output)
    dropout3.forward(activation3.output, training=False)
    
    dense4.forward(dropout3.output)
    activation4.forward(dense4.output)
    
    predictions = np.argmax(activation4.output, axis=1)
    accuracy = np.mean(predictions == y_test)
    
    return accuracy, predictions

def load_and_preprocess_data():
    df = pd.read_csv('../archive/megaGymDataset.csv')
    
    min_samples = 150 
    class_counts = df['BodyPart'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df = df[df['BodyPart'].isin(valid_classes)]
    
    print("\nClass distribution after filtering:")
    print(df['BodyPart'].value_counts())
    
    df['Title'] = df['Title'].fillna('')
    df['Desc'] = df['Desc'].fillna('')
    df['text'] = df['Title'] + ' ' + df['Desc']
    
    vectorizer = CountVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['text']).toarray()
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['BodyPart'])
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder.classes_

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, classes = load_and_preprocess_data()
    
    model = create_model(
        input_size=X_train.shape[1],
        hidden_size=128,
        output_size=len(np.unique(y_train))
    )
    
    loss_fn = CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.001) 
    
    train_model(X_train, y_train, X_val, y_val, model, loss_fn, optimizer, epochs=200)
    
    print("\nEvaluating model on test set...")
    test_accuracy, test_predictions = evaluate_model(X_test, y_test, model)
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    print("\nClass Distribution in Test Set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"{classes[label]}: {count}")

if __name__ == "__main__":
    main()
