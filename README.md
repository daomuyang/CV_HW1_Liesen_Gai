# 计算机视觉HW1：三层全连接神经网络Fashion-MNIST图像分类
盖烈森 | 学号：23307130013

---

## 一、项目概述
本项目基于纯NumPy手动实现三层全连接神经网络，完成Fashion-MNIST服装10分类任务，全程未使用PyTorch/TensorFlow等框架的自动微分功能。自主实现前向传播、反向传播、SGD优化器、学习率衰减、L2正则化交叉熵损失;完成48组超参数网格搜索，基于验证集筛选最优模型;最终最优模型在独立测试集上准确率达88.21%，完成所有作业要求的可视化与分析

---

## 二、环境依赖
```bash
pip install numpy matplotlib scikit-learn tensorflow
```
> 注：TensorFlow仅用于Fashion-MNIST数据集加载，不参与模型训练与推理。

---

## 三、仓库文件说明
| 文件名 | 功能说明 |
|--------|----------|
| `main.py` | 完整实验代码，含数据加载、模型实现、训练、超参搜索、可视化全流程 |
| `train.py`| 展示训练过程，含数据加载、模型实现、训练、超参搜索，同时对Loss/Accuracy曲线以及第一层隐藏层权重矩阵进行了可视化|
| `test.py`| 展示测试过程，含数据加载、模型定义、测试，同时对混淆矩阵和几个分类错误案例进行了可视化|
| `hw1盖烈森23307130013.pdf` | 报告文件 |
| `hw1盖烈森23307130013.ipynb` | 实验notebook版本，可不看 |
| `README.md` | 项目说明文档 |
| `final_best_model.pkl` | 最优模型权重文件 |
| `temp_search_model.pkl` | 超参搜索过程临时模型文件 |
| `training_curve.png` | 训练Loss/准确率曲线图片 |
| `confusion_matrix.png` | 测试集混淆矩阵图片 |
| `weight_visualization.png` | 第一层隐藏层权重可视化结果图片 |
| `error_analysis.png` | 错分样本可视化分析 |
| `dataset/` | 数据集存放目录 |

---

## 四、代码运行方式
1. 一键运行完整实验：
```bash
python main.py
```
2. 查看训练过程：
```bash
python train.py
```
3. 查看测试过程：
```bash
python test.py
```

---

## 五、核心实验结果
最优超参数为初始学习率0.15、隐藏层维度768、L2正则化强度0.001；测试集最终准确率为88.21%
