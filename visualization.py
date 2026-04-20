import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# 训练曲线可视化
def plot_history(history):
    plt.figure(figsize=(14, 6))
    
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='#1f77b4')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
    plt.title('Cross-Entropy Loss Curve', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Accuracy曲线
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

# 混淆矩阵可视化
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

# 第一层权重可视化
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

# 误差分析可视化
def error_analysis(model, x_test, y_test, class_names):
    y_pred = np.argmax(model.forward(x_test), axis=1)
    errs = np.where(y_pred != y_test)[0]

    if len(errs) == 0:
        print("无错误样本，展示前5个测试样本")
        samples = np.arange(5)
    elif len(errs) < 5:
        print(f"错误样本仅{len(errs)}个，全部展示")
        samples = errs
    else:
        samples = np.random.choice(errs, 5, replace=False)
    
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(samples):
        plt.subplot(1, len(samples), i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis('off')
    plt.savefig('error_analysis.png', dpi=300)
    plt.show()