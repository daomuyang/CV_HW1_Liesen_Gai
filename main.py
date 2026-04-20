import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

from data_loader import load_fashion_mnist
from models import ThreeLayerMLP
from optimizers import SGDOptimizer
from trainer import train, evaluate, hyperparam_search
from visualization import plot_history, plot_confusion_matrix, visualize_weights, error_analysis

if __name__ == '__main__':
    # 类别名称
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 1. 数据加载
    x_train, y_train, x_val, y_val, x_test, y_test = load_fashion_mnist()
    
    # 2. 超参数搜索
    best_lr, best_hidden_dim, best_batch_size, best_l2 = hyperparam_search(
        x_train, y_train, x_val, y_val,
        search_epochs=5
    )

    # 3. 用最优超参训练（带验证集）
    print(f"\n🚀 训练验证集最优模型...")
    final_model = ThreeLayerMLP(hidden_dim=best_hidden_dim, activation='relu')
    final_optimizer = SGDOptimizer(momentum=0.9)
    
    history = train(
        final_model, x_train, y_train, x_val, y_val,
        final_optimizer, epochs=50, batch_size=best_batch_size, 
        lambda_l2=best_l2, lr_max=best_lr, use_validation=True,
        save_path='val_best_model.pkl', patience=15
    )
    
    # 4. 全量数据训练（训练集+验证集）
    print(f"\n🚀 开始全量数据训练...")
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
    
    # 5. 可视化训练曲线
    plot_history(history)
    
    # 6. 测试集评估
    full_model.load_weights('final_best_model.pkl')
    test_acc = evaluate(full_model, x_test, y_test)
    print(f"\n🎯 最终测试集准确率：{test_acc:.4f}")
    
    # 7. 可视化结果
    plot_confusion_matrix(full_model, x_test, y_test, class_names)
    visualize_weights(full_model)
    error_analysis(full_model, x_test, y_test, class_names)