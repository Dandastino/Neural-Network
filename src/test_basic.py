import numpy as np
from utils.data_loader import load_gym_data
from layer.dense import Layer_Dense
from activations.relu import Activation_ReLU
from activations.softmax import Activation_Softmax


def test_data_loading():
    """Test if data can be loaded correctly"""
    try:
        data = load_gym_data()
        print("✅ Data loaded successfully")
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def test_network_components():
    """Test if network components can be initialized"""
    try:
        # Create a small test input
        X_test = np.random.randn(10, 5)  # 10 samples, 5 features
        y_test = np.random.randint(0, 3, 10)  # 3 classes
        
        # Initialize network components
        dense1 = Layer_Dense(5, 4)  # input_size=5, hidden_size=4
        activation1 = Activation_ReLU()
        dense2 = Layer_Dense(4, 3)  # hidden_size=4, output_size=3
        activation2 = Activation_Softmax()
        
        # Test forward pass
        dense1.forward(X_test)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        
        print("✅ Network components initialized and forward pass successful")
        return True
    except Exception as e:
        print(f"❌ Error in network components: {str(e)}")
        raise

def main():
    print("Running basic tests...")
    print("\n1. Testing data loading:")
    data = test_data_loading()
    
    print("\n2. Testing network components:")
    test_network_components()
    
    print("\nAll basic tests completed successfully! 🎉")

if __name__ == "__main__":
    main() 