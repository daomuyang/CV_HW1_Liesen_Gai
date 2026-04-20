# 计算机视觉HW1：三层全连接神经网络Fashion-MNIST图像分类
盖烈森 | 学号：23307130013

---

## 一、项目概述
本项目基于纯NumPy手动实现三层全连接神经网络，完成Fashion-MNIST服装10分类任务，全程未使用PyTorch/TensorFlow等框架的自动微分功能。自主实现前向传播、反向传播、SGD优化器、学习率衰减、L2正则化交叉熵损失;完成81组超参数网格搜索，基于验证集筛选最优模型;最终最优模型在独立测试集上准确率达90.08%，完成所有作业要求的可视化与分析，并且进行了模块化设计。

---

## 二、环境依赖

Python 3.x

numpy >= 1.21.0

matplotlib >= 3.4.0

tqdm >= 4.62.0

scikit-learn >= 1.0.0

tensorflow >= 2.8.0 (仅用于加载Fashion-MNIST数据集)

pickle 

warnings 

os 

```bash
pip install numpy matplotlib tqdm scikit-learn tensorflow
```
> 注：TensorFlow仅用于Fashion-MNIST数据集加载，不参与模型训练与推理。

---

## 三、仓库文件说明
| 文件名 | 功能说明 |
|--------|----------|
| `main.py` | 主程序入口，整合所有模块，一键完成数据加载、超参数搜索、模型训练、测试评估与全流程可视化 |
| `utils.py` | 通用工具函数库，包含激活函数、损失函数、数值稳定Softmax、余弦退火学习率调度等核心工具 |
| `data_loader.py` | Fashion-MNIST数据集加载与预处理模块，自动完成数据归一化与训练集/验证集划分 |
| `models.py` | 三层全连接神经网络（ThreeLayerMLP）定义，包含手动实现的前向传播、反向传播与权重加载逻辑 |
| `optimizers.py` | 带动量的SGD随机梯度下降优化器实现 |
| `trainer.py` | 训练核心模块，包含批量训练循环、梯度裁剪、早停机制、模型保存与超参数网格搜索 |
| `visualization.py` | 实验可视化模块，包含训练曲线、混淆矩阵、权重可视化、错例分析等所有绘图功能 |
| `final_best_model.pkl` | 使用训练集+验证集全量数据训练后的最终模型权重（用于最终测试集评估） |
| `val_best_model.pkl` | 基于验证集准确率筛选保存的最优模型权重（带早停机制） |
| `training_curve.png` | 训练过程中训练集与验证集的Loss曲线和Accuracy曲线 |
| `confusion_matrix.png` | 最终模型在测试集上的分类混淆矩阵 |
| `weight_visualization_all.png` | 第一层隐藏层全部神经元权重重塑为28×28灰度图的可视化结果 |
| `error_analysis.png` | 测试集随机错分样本的真实标签与预测标签对比分析 |
| `hw1盖烈森23307130013.pdf` | 正式实验报告，包含实验原理、实现细节与结果分析 |
| `hw1盖烈森23307130013.ipynb` | 实验Jupyter Notebook版本（可选参考） |
| `README.md` | 项目说明文档，包含环境依赖、运行方式与仓库文件说明 |
| `dataset/` | 数据集存放目录（如本地无数据，代码将自动通过TensorFlow下载） |

---

## 四、代码运行方式
运行完整实验：
```bash
python main.py
```

---

## 五、核心实验结果
最优超参数为初始学习率0.1、隐藏层维度768、批量大小128、L2正则化强度5e-5；测试集最终准确率为90.08%
