import numpy as np
from utils import relu, relu_grad, sigmoid, sigmoid_grad, softmax

class ThreeLayerMLP:
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10, activation='relu'):
        self.params = {}
        self.activation_name = activation
        
        if activation == 'relu':
            self.activation = relu
            self.activation_grad = relu_grad
            w1_init = np.sqrt(2.0 / input_dim)
            w2_init = np.sqrt(2.0 / hidden_dim)
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_grad = sigmoid_grad
            w1_init = np.sqrt(1.0 / input_dim)
            w2_init = np.sqrt(1.0 / hidden_dim)
        
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * w1_init
        self.params['b1'] = np.zeros((1, hidden_dim))
        self.params['W2'] = np.random.randn(hidden_dim, output_dim) * w2_init
        self.params['b2'] = np.zeros((1, output_dim))
    
    def forward(self, x):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        self.z1 = x @ W1 + b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ W2 + b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def backward(self, x, y_true, y_pred, lambda_l2):
        n = x.shape[0]
        W1, W2 = self.params['W1'], self.params['W2']
        
        dz2 = y_pred.copy()
        dz2[range(n), y_true] -= 1
        dz2 /= n
        
        dW2 = self.a1.T @ dz2 + lambda_l2 * W2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * self.activation_grad(self.z1)
        
        dW1 = x.T @ dz1 + lambda_l2 * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    
    # 导入训练好的权重
    def load_weights(self, weight_path):
        import pickle
        with open(weight_path, 'rb') as f:
            self.params = pickle.load(f)