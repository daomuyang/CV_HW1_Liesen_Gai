import numpy as np
import pickle
import os
from tqdm import tqdm
from utils import cross_entropy_loss, cosine_lr
from models import ThreeLayerMLP
from optimizers import SGDOptimizer

# 评估函数
def evaluate(model, x, y):
    y_pred = model.forward(x)
    y_pred_cls = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_cls == y)

# 核心训练函数
def train(model, x_train, y_train, x_val, y_val, optimizer, epochs, batch_size, lambda_l2, lr_max, 
          use_validation=True, save_path='best_model.pkl', patience=15, clip_norm=1.0, min_delta=1e-4):
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
            total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            if total_norm > clip_norm:
                for key in grads:
                    grads[key] *= clip_norm / (total_norm + 1e-6)
            
            # 余弦退火学习率
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
            # 全量训练时记录
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(train_acc)
            print(f"Epoch {epoch+1} | LR: {current_lr:.6f} | Loss: {epoch_train_loss:.4f} | TrainAcc: {train_acc:.4f}")
    
    # 全量训练保存最后权重
    if not use_validation:
        with open(save_path, 'wb') as f:
            pickle.dump(model.params, f)
        print(f"✅ 全量训练完成，模型已保存！")
    
    return history

# 超参数网格搜索
def hyperparam_search(x_train, y_train, x_val, y_val, search_epochs=5):
    print("\n🔍 开始超参数网格搜索...")
    
    # 超参候选
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
                    
                    # 清理临时文件
                    if os.path.exists('temp_search_model.pkl'):
                        os.remove('temp_search_model.pkl')
    
    print(f"\n🏆 超参数搜索完成！")
    print(f"最优组合：lr={best_params[0]}, hidden_dim={best_params[1]}, batch_size={best_params[2]}, lambda_l2={best_params[3]}")
    print(f"最高验证准确率：{best_val_acc:.4f}")
    
    return best_params