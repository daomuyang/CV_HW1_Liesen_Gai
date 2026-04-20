import numpy as np
from tensorflow.keras.datasets import fashion_mnist

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