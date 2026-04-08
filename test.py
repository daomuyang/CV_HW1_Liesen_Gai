import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# 工具函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

# 数据加载
def load_fashion_mnist(val_size=10000):
    print("📥 正在加载原版Fashion-MNIST 数据集...")
    from tensorflow.keras.datasets import fashion_mnist
   
    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train_full = x_train_full.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    
    val_idx = np.random.choice(len(x_train_full), val_size, replace=False)
    train_mask = np.ones(len(x_train_full), dtype=bool)
    train_mask[val_idx] = False
    
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]
    x_train = x_train_full[train_mask]
    y_train = y_train_full[train_mask]
    
    print(f"✅ 原版数据加载成功！")
    return x_train, y_train, x_val, y_val, x_test, y_test

# 模型定义
class ThreeLayerMLP:
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, activation='relu'):
        self.params = {}
        self.activation_name = activation
        
        if activation == 'relu':
            self.activation = relu
            self.activation_grad = relu_grad
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_grad = sigmoid_grad
        
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.params['b1'] = np.zeros((1, hidden_dim))
        self.params['W2'] = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.params['b2'] = np.zeros((1, output_dim))
    
    def forward(self, x):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        self.z1 = x @ W1 + b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ W2 + b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def load_weights(self, weight_path):
        with open(weight_path, 'rb') as f:
            self.params = pickle.load(f)

# 评估函数
def evaluate(model, x, y):
    y_pred = model.forward(x)
    y_pred_cls = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_cls == y)

# 测试相关可视化
def plot_confusion_matrix(model, x_test, y_test, class_names):
    y_pred = np.argmax(model.forward(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(10), class_names, rotation=45)
    plt.yticks(range(10), class_names)
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.show()

def error_analysis(model, x_test, y_test, class_names):
    y_pred = np.argmax(model.forward(x_test), axis=1)
    errs = np.where(y_pred != y_test)[0]
    if len(errs) == 0:
        errs = [0, 1, 2, 3, 4]
    samples = np.random.choice(errs, 5)
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(samples):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis('off')
    plt.show()

# 主程序
if __name__ == '__main__':
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test = load_fashion_mnist()
    
    # 加载最优模型
    print("\n🚀 加载最优模型权重...")
    final_model = ThreeLayerMLP(hidden_dim=768, activation='relu') # 这里hidden_dim需与最优参数一致
    final_model.load_weights('final_best_model.pkl')
    
    # 测试集评估
    test_acc = evaluate(final_model, x_test, y_test)
    print(f"\n🎯 最终测试集准确率：{test_acc:.4f}")
    
    # 生成测试相关可视化
    plot_confusion_matrix(final_model, x_test, y_test, class_names)
    error_analysis(final_model, x_test, y_test, class_names)