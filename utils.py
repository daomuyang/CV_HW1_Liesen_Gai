import numpy as np

# 激活函数及梯度
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

# Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 损失函数（含L2正则）
def cross_entropy_loss(y_pred, y_true, weights, lambda_l2):
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true] + 1e-10)
    ce_loss = np.sum(log_likelihood) / n
    l2_loss = 0.5 * lambda_l2 * (np.sum(weights[0]**2) + np.sum(weights[1]**2))
    return ce_loss + l2_loss

# 余弦退火学习率
def cosine_lr(epoch, total_epoch, lr_max, lr_min=1e-6):
    if total_epoch == 0:
        return lr_max
    denom = max(total_epoch - 1, 1)
    cos_term = np.cos(np.pi * epoch / denom)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos_term)