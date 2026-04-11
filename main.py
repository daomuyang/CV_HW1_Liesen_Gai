import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import itertools
import warnings
import os
# 使用Tensorflow只是为了加载数据，后续的数据处理和训练是手动完成的
from tensorflow.keras.datasets import fashion_mnist
warnings.filterwarnings('ignore')
np.random.seed(42)

# 1. 工具函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 交叉熵损失函数
def cross_entropy_loss(y_pred, y_true, weights, lambda_l2):
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true] + 1e-10)
    ce_loss = np.sum(log_likelihood) / n
    l2_loss = 0.5 * lambda_l2 * (np.sum(weights[0]**2) + np.sum(weights[1]**2))
    return ce_loss + l2_loss

# 支持两种激活函数：ReLU和Sigmoid
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

# 余弦退火学习率调度函数
def cosine_lr(epoch, total_epoch, lr_max, lr_min=1e-6):
    if total_epoch == 0:
        return lr_max
    denom = max(total_epoch - 1, 1)
    cos_term = np.cos(np.pi * epoch / denom)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos_term)

# 2. 数据加载
def load_fashion_mnist(val_size=10000):
    print("📥 正在加载原版Fashion-MNIST 数据集...")
   
    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    
    # 预处理
    x_train_full = x_train_full.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    
    # 划分验证集
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

# 3. 三层神经网络模型
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
    
    # 支持导入训练好的最优模型权重
    def load_weights(self, weight_path):
        with open(weight_path, 'rb') as f:
            self.params = pickle.load(f)

# 优化器简化成只带动量，学习率外部传入
class SGDOptimizer:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = {}
    def _init_velocity(self, model):
        if not self.velocity:
            for key in model.params:
                self.velocity[key] = np.zeros_like(model.params[key])
    def step(self, model, grads, lr): # 直接接收当前lr
        self._init_velocity(model)
        for key in model.params:
            self.velocity[key] = self.momentum * self.velocity[key] - lr * grads[key]
            model.params[key] += self.velocity[key]
    
# 5. 训练函数，支持早停，支持选择是否使用验证集加入早停判别和权重保存（全量训练时设为False），支持传入余弦退火的最大学习率
def train(model, x_train, y_train, x_val, y_val, optimizer, epochs, batch_size, lambda_l2, lr_max, use_validation=True, save_path='best_model.pkl', patience=15, clip_norm=1.0, min_delta=1e-4):
    n_train = x_train.shape[0]
    best_val_acc = 0.0
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    if use_validation:
        print(f"\n🚀 开始训练（带验证集），总epoch数：{epochs}，早停耐心值：{patience}")
    else:
        print(f"\n🚀 开始全量数据训练，总epoch数：{epochs}")
    
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
            
            # 梯度裁剪
            clip_norm = 1.0
            total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            if total_norm > clip_norm:
                for key in grads:
                    grads[key] *= clip_norm / (total_norm + 1e-6)
            
            # 使用传入的lr_max 
            current_lr = cosine_lr(epoch, epochs, lr_max=lr_max, lr_min=1e-6)
            optimizer.step(model, grads, current_lr)
            epoch_train_loss += loss * len(x_batch)
        
        epoch_train_loss /= n_train
        train_acc = evaluate(model, x_train, y_train)
        
        if use_validation:
            val_loss = cross_entropy_loss(model.forward(x_val), y_val, [model.params['W1'], model.params['W2']], lambda_l2)
            val_acc = evaluate(model, x_val, y_val)
            
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1} | LR: {current_lr:.6f} | Loss: {epoch_train_loss:.4f} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")
            
            # 早停逻辑
            min_delta = 1e-4
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                counter = 0
                with open(save_path, 'wb') as f:
                    pickle.dump(model.params, f)
                print(f"✅ 最优模型已保存！ValAcc: {best_val_acc:.4f}")
            else:
                counter += 1
                print(f"⏳ 验证集准确率未提升，耐心值：{counter}/{patience}")
                if counter >= patience:
                    print(f"\n🛑 早停触发！训练提前结束。")
                    break
        else:
            # 全量训练时，不验证，直接保存最后一轮
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(train_acc)
            print(f"Epoch {epoch+1} | LR: {current_lr:.6f} | Loss: {epoch_train_loss:.4f} | TrainAcc: {train_acc:.4f}")
    
    # 全量训练时，保存最后一轮权重
    if not use_validation:
        with open(save_path, 'wb') as f:
            pickle.dump(model.params, f)
        print(f"✅ 全量训练完成，模型已保存！")
    
    return history

# 6. 评估
def evaluate(model, x, y):
    y_pred = model.forward(x)
    y_pred_cls = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_cls == y)

# 7. 网格搜索超参数
def hyperparam_search(x_train, y_train, x_val, y_val, search_epochs=5):
    print("\n🔍 开始超参数网格搜索...")
    
    # 记录不同超参数组合下的验证集准确率，最终输出最优组合
    lr_list = [0.1, 0.15, 0.2]
    hidden_dim_list = [512,768,1024]
    batch_size_list = [64, 128, 256] 
    lambda_l2_list = [5e-5, 1e-4, 0.001]
    
    best_val_acc = 0.0
    best_params = None
    total = len(lr_list) * len(hidden_dim_list) * len(batch_size_list) * len(lambda_l2_list)
    count = 0
    
    for lr in lr_list:
        for hidden_dim in hidden_dim_list:
            for batch_size in batch_size_list: 
                for lambda_l2 in lambda_l2_list:
                    count += 1
                    print(f"\n--- [{count}/{total}] 测试组合：lr={lr}, hidden={hidden_dim}, batch={batch_size}, l2={lambda_l2} ---")
                    
                    model = ThreeLayerMLP(hidden_dim=hidden_dim, activation='relu')
                    optimizer = SGDOptimizer(momentum=0.9)
                    
                    history = train(
                        model, x_train, y_train, x_val, y_val,
                        optimizer, epochs=search_epochs, batch_size=batch_size, 
                        lambda_l2=lambda_l2, lr_max=lr, use_validation=True,
                        save_path='temp_search_model.pkl', patience=5
                    )
                    
                    current_best_acc = max(history['val_acc'])
                    print(f"该组合最高验证准确率：{current_best_acc:.4f}")
                    
                    if current_best_acc > best_val_acc:
                        best_val_acc = current_best_acc
                        best_params = (lr, hidden_dim, batch_size, lambda_l2) 
                        print(f"🏆 发现新的最优参数！")
                    
                    if os.path.exists('temp_search_model.pkl'):
                        os.remove('temp_search_model.pkl')
    
    print(f"\n🏆 超参数搜索完成！")
    print(f"最优组合：lr={best_params[0]}, hidden_dim={best_params[1]}, batch_size={best_params[2]}, lambda_l2={best_params[3]}")
    print(f"最高验证准确率：{best_val_acc:.4f}")
    
    return best_params

# 8. 可视化
# 可视化训练过程中在训练集和验证集上的Loss曲线，以及验证集上的Accuracy曲线
def plot_history(history):
    plt.figure(figsize=(14, 6))
    
    # 左侧Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='#1f77b4')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
    plt.title('Cross-Entropy Loss Curve', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # 右侧Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='#1f77b4')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2, color='#ff7f0e')
    plt.title('Classification Accuracy Curve', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Accuracy (0-1)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# 打印各分类的混淆矩阵
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
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

# 将训练好的第一层隐藏层权重矩阵恢复成图像尺寸，并将其作为图像进行可视化
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
    plt.savefig('weight_visualization_all.png', dpi=300, bbox_inches='tight')
    plt.show()

# 挑选几个分错的进行误差分析
def error_analysis(model, x_test, y_test, class_names):
    y_pred = np.argmax(model.forward(x_test), axis=1)
    errs = np.where(y_pred != y_test)[0]

    if len(errs) == 0:
        print("无错误样本，展示前5个测试样本")
        samples = np.arange(5)
    elif len(errs) < 5:
        print(f"错误样本仅{len(errs)}个，全部展示")
        samples = errs  # 不足5个时取所有错误样本
    else:
        samples = np.random.choice(errs, 5, replace=False)  # 避免重复采样
    
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(samples):
        plt.subplot(1, len(samples), i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis('off')
    plt.savefig('error_analysis.png', dpi=300)
    plt.show()

# 主程序
if __name__ == '__main__':
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 数据加载
    x_train, y_train, x_val, y_val, x_test, y_test = load_fashion_mnist()
    
    # 1. 超参数搜索
    best_lr, best_hidden_dim, best_batch_size, best_l2 = hyperparam_search(
        x_train, y_train, x_val, y_val,
        search_epochs=5
    )

    # 2. 用最优超参进行训练
    print(f"\n🚀 训练验证集最优模型...")
    # 如需要自定义隐藏层大小，可在此处修改best_hidden_dim的值，保持和训练一致即可
    final_model = ThreeLayerMLP(hidden_dim=best_hidden_dim, activation='relu')
    final_optimizer = SGDOptimizer(momentum=0.9)
    
    history = train(
        final_model, x_train, y_train, x_val, y_val,
        final_optimizer, epochs=50, batch_size=best_batch_size, 
        lambda_l2=best_l2, lr_max=best_lr, use_validation=True,
        save_path='val_best_model.pkl', patience=15
    )
    
    # 3. 全量数据训练
    print(f"\n🚀 开始全量数据训练（训练集+验证集）...")
    x_full = np.concatenate([x_train, x_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    
    full_model = ThreeLayerMLP(hidden_dim=best_hidden_dim, activation='relu')
    full_optimizer = SGDOptimizer(momentum=0.9)
    
    full_history = train(
        full_model, x_full, y_full, None, None,
        full_optimizer, epochs=40, batch_size=best_batch_size, 
        lambda_l2=best_l2, lr_max=best_lr, use_validation=False,
        save_path='final_best_model.pkl'
    )
    
    # 可视化
    plot_history(history)
    
    # 测试
    full_model.load_weights('final_best_model.pkl')
    test_acc = evaluate(full_model, x_test, y_test)
    print(f"\n🎯 最终测试集准确率：{test_acc:.4f}")
    
    # 绘图
    plot_confusion_matrix(full_model, x_test, y_test, class_names)
    visualize_weights(full_model)
    error_analysis(full_model, x_test, y_test, class_names)