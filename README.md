# Neural Network from Scratch

## 🚀 Features
- Custom implementation of neural network components:
  - Dense layers with Xavier/Glorot initialization
  - ReLU and Softmax activation functions
  - Categorical Cross-Entropy loss
  - Stochastic Gradient Descent optimizer
- Text preprocessing using CountVectorizer
- Data normalization using StandardScaler
- Train/validation/test split for proper model evaluation
- Progress monitoring during training

## 📂 Project Structure
```
src/
├── activations/
│   ├── relu.py         # ReLU activation function
│   └── softmax.py      # Softmax activation function
├── layer/
│   └── dense.py        # Dense layer implementation
├── losses/
│   └── crossentropy.py # Categorical Cross-Entropy loss
├── optimizers/
│   └── sgd.py          # Stochastic Gradient Descent optimizer
├── utils/
│   └── data_loader.py  # Data loading and preprocessing utilities
└── main.py             # Main training script
```

## 🔍 How It Works
1. Data Preprocessing:
   - Load exercise dataset
   - Convert exercise names to numerical features using CountVectorizer
   - Encode body part labels
   - Normalize data using StandardScaler

2. Model Training:
   - Split data into train/validation/test sets
   - Forward pass through the network
   - Calculate loss and accuracy
   - Backpropagate errors
   - Update weights using SGD

3. Evaluation:
   - Monitor training progress every 10 epochs
   - Track validation accuracy
   - Evaluate final performance on test set

## 🧠 Concepts Covered
- Neural Network Architecture
  - Dense Layers
  - Activation Functions (ReLU, Softmax)
  - Loss Functions (Cross-Entropy)
  - Optimizers (SGD)
- Backpropagation
- Gradient Descent
- Text Preprocessing
- Data Normalization
- Model Evaluation

## 💡 Why I Built This
This project was built to:
- Understand neural networks from the ground up
- Implement core deep learning concepts from scratch
- Create a practical application for exercise classification
- Learn about text processing and feature engineering
- Practice proper model evaluation techniques

## 📦 Setup & Usage
1. Create and activate virtual environment:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

2. Install dependencies:
```bash
pip install numpy scikit-learn pandas
```

3. Run the training:
```bash
python src/main.py
```

## 📄 License
This project is open source and available under the MIT License.
