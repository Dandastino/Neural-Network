# 🏋️‍♂️ Neural Network for Exercise Classification

## 📝 Abstract
This project implements a neural network-based classification system for gym exercises. The system utilizes natural language processing techniques to analyze exercise names and descriptions and classify them into appropriate body part categories. The implementation features a custom-built neural network architecture with dense layers, batch normalization, dropout, and modern activation functions, demonstrating the practical application of deep learning in fitness domain classification.

## 🎯 Introduction

### 🚀 Project Overview
This project represents an innovative approach to exercise classification using deep learning techniques. By leveraging neural networks and natural language processing, the system can accurately categorize gym exercises based on their descriptions and characteristics.

### ❓ Problem Statement
The classification of exercises into appropriate body part categories is crucial for:
- Creating balanced workout routines
- Preventing muscle imbalances
- Optimizing training programs
- Ensuring proper exercise selection

### 🎯 Objectives
- Implement a neural network architecture for exercise classification
- Process and analyze exercise descriptions using NLP techniques
- Achieve high accuracy in exercise categorization
- Provide a scalable and maintainable solution

## 🛠️ Technical Architecture

### 🔧 Core Components
- **Neural Network Implementation**
  - Custom dense layers
  - Batch Normalization layers
  - Dropout layers for regularization
  - ReLU activation function
  - Softmax output layer
  - Categorical cross-entropy loss function
  - Adam optimizer

- **Data Processing Pipeline**
  - Text vectorization using CountVectorizer
  - Label encoding for categorical variables
  - Data normalization using StandardScaler
  - Train/validation/test split (70/15/15)

### 💻 Technologies Used
- Python 3.12
- NumPy: Numerical computations
- Pandas: Data manipulation
- Scikit-learn: Machine learning utilities
- Matplotlib & Seaborn: Data visualization
- Custom neural network implementation

### 💾 Hardware Requirements
- CPU: Intel/AMD processor
- RAM: Minimum 8GB recommended
- Storage: 1GB free space
- GPU: Intel Arc Graphics (optional, for acceleration)

## 🏗️ Implementation Details

### 📊 Data Structure
The system processes the following features:
- Exercise Title
- Description
- Type
- Body Part
- Equipment
- Level
- Rating
- Rating Description

### 🧠 Model Architecture
1. **Input Layer**: Vectorized exercise descriptions
2. **Hidden Layer 1**: 256 neurons with BatchNorm, ReLU activation, and Dropout (0.3)
3. **Hidden Layer 2**: 128 neurons with BatchNorm, ReLU activation, and Dropout (0.2)
4. **Hidden Layer 3**: 64 neurons with BatchNorm, ReLU activation, and Dropout (0.1)
5. **Output Layer**: Softmax activation for multi-class classification

### 🚂 Training Process
- Learning rate: 0.001
- Epochs: 200 (with early stopping)
- Batch processing
- Validation monitoring
- Early stopping based on validation accuracy (patience=20)

## 🤔 Design Choices and Rationale

### 🏎️💨 Optimizer Selection
**Adam optimizer over SGD for several reasons:**
- Adaptive learning rates for each parameter
- Faster convergence
- Better handling of sparse gradients
- Built-in momentum and RMSprop features
- More stable training process

### 🏗️ Layer Architecture
1. **Batch Normalization**:
   - Stabilizes training
   - Reduces internal covariate shift
   - Acts as a regularizer
   - Allows higher learning rates

2. **Dropout**:
   - Prevents overfitting
   - Forces the network to learn robust features
   - Different rates for different layers (0.3 → 0.2 → 0.1)
   - Higher rates in early layers to prevent overfitting

3. **ReLU Activation**:
   - Mitigates vanishing gradient problem
   - Computationally efficient
   - Sparse activation
   - Better gradient flow

### 📉 Loss Function
Categorical Cross-Entropy was chosen because:
- Suitable for multi-class classification
- Provides good gradient properties
- Works well with softmax output
- Handles class probabilities effectively

## 📦 Requirements

### 🐍 Python Packages
```
numpy>=1.26.0      # Numerical computations
pandas>=2.1.0      # Data manipulation
scikit-learn>=1.3.0 # Machine learning utilities
matplotlib>=3.8.0   # Data visualization
tqdm>=4.66.0       # Progress bars
seaborn>=0.13.0    # Statistical data visualization
```

### 💻 Installation
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### ▶️ Running the Model
```bash
python src/main.py
```

### 📊 Expected Output
The model will display:
- Preprocessing progress
- Training progress with tqdm bar
- Loss and accuracy metrics every 10 epochs
- Final test accuracy
- Class distribution in test set

## 📈 Performance Metrics

### 🎯 Training Results
- Training Accuracy: 99.6%
- Validation Accuracy: 83.8%
- Test Accuracy: 81.2%
- Training Time: 24.34 seconds
- Preprocessing Time: 0.07 seconds

### 📊 Class Distribution Analysis
The model was tested on a dataset with the following distribution:
- Most Common Classes:
  - Abdominals: 662 samples
  - Quadriceps: 646 samples
  - Shoulders: 340 samples
  - Chest: 262 samples
  - Biceps: 168 samples
  - Triceps: 151 samples

- Less Common Classes:
  - Lats: 124 samples
  - Hamstrings: 121 samples
  - Middle Back: 118 samples
  - Lower Back: 97 samples
  - Glutes: 81 samples
  - Calves: 47 samples
  - Forearms: 31 samples
  - Traps: 24 samples
  - Abductors: 21 samples
  - Adductors: 17 samples

### 📈 Training Progress
The model showed significant improvement during training:
- Initial accuracy (Epoch 0): 6.7%
- Rapid improvement (Epoch 20): 82.8%
- Plateau reached (Epoch 100): 99.0%
- Final accuracy (Epoch 200): 99.6%

### ⚠️ Class Imbalance Challenge
The model faces a significant challenge due to class imbalance in the training data:
1. **Data Quantity Disparity**:
   - Some classes (e.g., Abdominals: 662) have 30x more examples than others (e.g., Adductors: 17)
   - This imbalance affects the model's ability to learn patterns for underrepresented classes

2. **Impact on Performance**:
   - High training accuracy (99.6%) but lower validation accuracy (83.8%)
   - The model tends to perform better on classes with more training examples
   - Classes with fewer examples (e.g., Adductors, Traps) are harder to classify correctly

3. **Current Solutions**:
   - Filtered out classes with less than 100 examples
   - Implemented dropout layers to prevent overfitting
   - Used batch normalization for better training stability

## 📁 Project Structure
```
├── src/
│   ├── main.py
│   ├── layer/
│   │   ├── dense.py
│   │   ├── batch_norm.py
│   │   └── dropout.py
│   ├── activations/
│   │   ├── relu.py
│   │   └── softmax.py
│   ├── losses/
│   │   └── crossentropy.py
│   ├── optimizers/
│   │   └── adam.py
│   └── utils/
│       └── data_loader.py
├── archive/
│   └── megaGymDataset.csv
├── requirements.txt
└── README.md
```

## 📜 License
This project is licensed under the [MIT License](LICENSE) - click to view the full license text.
