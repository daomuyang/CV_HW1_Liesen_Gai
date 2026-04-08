import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')
np.random.seed(42)

# 工具函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true, weights, lambda_l2):
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true] + 1e-10)
    ce_loss = np.sum(log_likelihood) / n
    l2_loss = 0.5 * lambda_l2 * (np.sum(weights[0]**2) + np.sum(weights[1]**2))
    return ce_loss + l2_loss

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
    print(f"训练集: {x_train.shape} | 验证集: {x_val.shape} | 测试集: {x_test.shape}")
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
    
    def load_weights(self, weight_path):
        with open(weight_path, 'rb') as f:
            self.params = pickle.load(f)

# SGD优化器
class SGDOptimizer:
    def __init__(self, init_lr, decay_rate=0.95, decay_step=5):
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step
    
    def step(self, model, grads, epoch):
        current_lr = self.init_lr * (self.decay_rate ** (epoch // self.decay_step))
        for key in model.params:
            model.params[key] -= current_lr * grads[key]
        return current_lr

# 训练与评估
def train(model, x_train, y_train, x_val, y_val, optimizer, epochs, batch_size, lambda_l2, save_path='best_model.pkl'):
    n_train = x_train.shape[0]
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\n🚀 开始训练，总epoch数：{epochs}")
    
    for epoch in range(epochs):
        shuffle_idx = np.random.permutation(n_train)
        x_train_shuffle = x_train[shuffle_idx]
        y_train_shuffle = y_train[shuffle_idx]
        
        epoch_train_loss = 0.0
        for i in tqdm(range(0, n_train, batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            x_batch = x_train_shuffle[i:i+batch_size]
            y_batch = y_train_shuffle[i:i+batch_size]
            
            y_pred = model.forward(x_batch)
            loss = cross_entropy_loss(y_pred, y_batch, [model.params['W1'], model.params['W2']], lambda_l2)
            grads = model.backward(x_batch, y_batch, y_pred, lambda_l2)
            current_lr = optimizer.step(model, grads, epoch)
            
            epoch_train_loss += loss * len(x_batch)
        
        epoch_train_loss /= n_train
        train_acc = evaluate(model, x_train, y_train)
        val_loss = cross_entropy_loss(model.forward(x_val), y_val, [model.params['W1'], model.params['W2']], lambda_l2)
        val_acc = evaluate(model, x_val, y_val)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} | LR: {current_lr:.6f} | Loss: {epoch_train_loss:.4f} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            with open(save_path, 'wb') as f:
                pickle.dump(model.params, f)
            print(f"✅ 最优模型已保存！ValAcc: {best_val_acc:.4f}")
    
    return history

def evaluate(model, x, y):
    y_pred = model.forward(x)
    y_pred_cls = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_cls == y)

# 超参数搜索
def hyperparam_search(x_train, y_train, x_val, y_val, search_epochs=5, batch_size=256):
    print("\n🔍 开始超参数网格搜索...")
    
    lr_list = [0.05, 0.08, 0.1, 0.15]
    hidden_dim_list = [256,512,768,1024]
    lambda_l2_list = [0.001,0.005, 0.01]
    
    best_val_acc = 0.0
    best_params = None
    total = len(lr_list) * len(hidden_dim_list) * len(lambda_l2_list)
    count = 0
    
    for lr in lr_list:
        for hidden_dim in hidden_dim_list:
            for lambda_l2 in lambda_l2_list:
                count += 1
                print(f"\n--- [{count}/{total}] 测试组合：lr={lr}, hidden={hidden_dim}, l2={lambda_l2} ---")
                
                model = ThreeLayerMLP(hidden_dim=hidden_dim, activation='relu')
                optimizer = SGDOptimizer(init_lr=lr)
                
                history = train(
                    model, x_train, y_train, x_val, y_val,
                    optimizer, epochs=search_epochs, batch_size=batch_size,
                    lambda_l2=lambda_l2, save_path='temp_search_model.pkl'
                )
                
                current_best_acc = max(history['val_acc'])
                print(f"该组合最高验证准确率：{current_best_acc:.4f}")
                
                if current_best_acc > best_val_acc:
                    best_val_acc = current_best_acc
                    best_params = (lr, hidden_dim, lambda_l2)
                    print(f"🏆 发现新的最优参数！")
    
    print(f"\n🏆 超参数搜索完成！")
    print(f"最优组合：lr={best_params[0]}, hidden_dim={best_params[1]}, lambda_l2={best_params[2]}")
    print(f"最高验证准确率：{best_val_acc:.4f}")
    
    return best_params

# 训练相关可视化
def plot_history(history):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='#1f77b4')
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
    plt.title('Cross-Entropy Loss Curve', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='#1f77b4')
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2, color='#ff7f0e')
    plt.title('Classification Accuracy Curve', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Accuracy (0-1)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_weights(model):
    W1 = model.params['W1']
    hidden_dim = W1.shape[1]
    
    n_cols = int(np.ceil(np.sqrt(hidden_dim)))
    n_rows = int(np.ceil(hidden_dim / n_cols))
    
    plt.figure(figsize=(n_cols * 0.8, n_rows * 0.8), dpi=150)
    plt.suptitle('First Layer Weight Visualization (All Neurons)', fontsize=8, y=0.99)
    
    for i in range(hidden_dim):
        plt.subplot(n_rows, n_cols, i+1)
        weight_img = W1[:, i].reshape(28, 28)
        plt.imshow(weight_img, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == '__main__':
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test = load_fashion_mnist()
    
    # 超参数搜索
    best_lr, best_hidden_dim, best_l2 = hyperparam_search(
        x_train, y_train, x_val, y_val,
        search_epochs=5, batch_size=256
    )
    
    # 训练最终最优模型
    print(f"\n🚀 训练最终最优模型...")
    final_model = ThreeLayerMLP(hidden_dim=best_hidden_dim, activation='relu')
    final_optimizer = SGDOptimizer(init_lr=best_lr)
    history = train(
        final_model, x_train, y_train, x_val, y_val,
        final_optimizer, epochs=40, batch_size=128,
        lambda_l2=best_l2, save_path='final_best_model.pkl'
    )
    
    # 可视化训练曲线与权重
    plot_history(history)
    final_model.load_weights('final_best_model.pkl')
    visualize_weights(final_model)