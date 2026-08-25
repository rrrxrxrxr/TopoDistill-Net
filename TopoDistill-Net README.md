# TopoDistill-Net

**Topology-Aligned Cross-Modal Distillation from fMRI to EEG for Schizophrenia Detection**

> 一种将 fMRI 功能网络先验通过拓扑对齐跨模态蒸馏迁移到 EEG 模型的精神分裂症检测框架，推理时仅需 EEG 信号。

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Transactions-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)]()

**代码仓库**：https://github.com/rrrxrxrxr/TopoDistill-Net

---

## 目录

- [项目简介](#项目简介)
- [论文引用](#论文引用)
- [模型架构](#模型架构)
- [环境要求与安装](#环境要求与安装)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [输入输出说明](#输入输出说明)
- [关键超参数](#关键超参数)
- [实验结果](#实验结果)
- [项目结构](#项目结构)
- [参考文献](#参考文献)

---

## 项目简介

精神分裂症的临床诊断高度依赖结构化访谈，存在评分者间差异大、诊断延迟等问题。脑电图（EEG）以低成本、高时间分辨率为客观分类提供了可能，但现有深度 EEG 方法存在两个核心局限：

1. **未显式建模节律特异性振荡异常**——精神分裂症的神经生理障碍表现为各频段（delta/theta/alpha/beta/gamma）内的异常活动及频段间的病理性耦合，而通用时空模型无法区分这些节律特异性特征。
2. **缺乏功能网络拓扑先验**——头皮 EEG 受容积传导影响空间特异性有限，而 fMRI 可在脑区层面刻画默认网络、突显网络、额顶控制网络的失连接模式，这些信息无法被纯 EEG 模型获取。

**TopoDistill-Net** 通过两个核心模块解决上述问题：

- **Rhythm-Aware Hierarchical Spectrotemporal Encoder (RHSE)**：并行的节律特异性残差注意力单元（RRAU）捕获频段内异常，跨节律交互融合单元（CRIFU）建模病理性跨频耦合。
- **Topology-Aligned Prototype Distillation (TAPD)**：拓扑对齐层（TAL）将 EEG 电极特征投影到 116 区 AAL 脑区空间，原型引导的跨模态蒸馏（PCMD）从预训练 Graphormer fMRI 教师模型迁移功能网络组织先验。

fMRI 教师仅在训练阶段使用，**推理时 TopoDistill-Net 仅需原始 EEG**（1.459M 参数，单样本 1.2 ms，0.0932 GFLOPs），适合低成本临床筛查部署。

---

## 论文引用

如果您在研究中使用了本代码，请引用以下论文：

```bibtex
@article{liu2025topodistill,
  title   = {TopoDistill-Net: Topology-Aligned Cross-Modal Distillation from fMRI to EEG for Schizophrenia Detection},
  author  = {Liu, Ke and Xu, Rui and Wang, Wenlong and Xiao, Bin and Wu, Wei},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2025},
  note    = {Under Review}
}
```

**作者信息**：
- Ke Liu, Rui Xu, Bin Xiao — 重庆邮电大学计算智能重庆市重点实验室
- Wenlong Wang, Wei Wu — 上海交通大学医学院附属松江医院/上海市情感障碍重点实验室
- 通讯作者：Bin Xiao (xiaobin@cqupt.edu.cn), Wei Wu (weiwuneuro@sjtu.edu.cn)

**基金支持**：国家自然科学基金（62476034, U24A20338）、重庆市自然科学基金（CSTB2025NSCQ-JM0010）、重庆市教委科技项目（KJZD-K202500607）、上海市教委 AI 项目（JWAIZD-4）。

---

## 模型架构

### 整体流程

```
原始EEG [C, T]
    │
    ▼  四阶Butterworth带通滤波（零相位）
五频段信号 [C, 5, T]  (delta/theta/alpha/beta/gamma)
    │
    ▼
┌─────────────────────────────────────────┐
│  RHSE (Rhythm-Aware Hierarchical        │
│        Spectrotemporal Encoder)         │
│                                         │
│  ┌────────┐ ┌────────┐ ... ┌────────┐  │
│  │ RRAU   │ │ RRAU   │     │ RRAU   │  │  ← 每频段独立: Conv1D×3 + BN + GELU + SE
│  │(delta) │ │(theta) │     │(gamma) │  │
│  └───┬────┘ └───┬────┘     └───┬────┘  │
│      └──────────┴──────────────┘        │
│                    │ Concat              │
│                    ▼                     │
│           ┌─────────────────┐            │
│           │  CRIFU          │            │  ← 1×1 Conv + CBAM(通道+空间注意力)×2
│           │(跨节律融合)      │            │
│           └────────┬────────┘            │
└────────────────────┼─────────────────────┘
                     │ F_RHSE [D, T]
                     ▼  时间平均池化
              F_a [C, D]
                     │
                     ▼
┌─────────────────────────────────────────┐
│  TAL (Topology Alignment Layer)         │
│  静态映射 W_s ∈ R^{116×C}               │
│  + 动态MLP映射 (subject-specific)       │
│  → F_map [116, D]                       │
└────────────────────┬─────────────────────┘
                     │ flatten → f_map
                     ▼
              ┌──────────────┐
              │  Classifier  │  → 二分类 logits [2]
              │ (3层MLP)     │
              └──────┬───────┘
                     │
    ┌────────────────┼────────────────┐
    ▼ (训练时)       │                ▼ (推理时)
┌──────────────────┐ │           仅EEG输入
│ fMRI Teacher     │ │           直接输出预测
│ (Graphormer)     │ │
│  116×116 FC矩阵  │ │
│  → 类原型 p_0,p_1│ │
└────────┬─────────┘ │
         │           │
         ▼           ▼
  三损失蒸馏: L_feat + L_contrast + L_prob
```

### 1. RHSE — 节律感知层级时频编码器

#### RRAU (Rhythm-Specific Residual Attention Unit)

每个频段独立通过一个 RRAU，结构为：

```
输入 X_k ∈ R^{C×T}
  → Conv1D(C→D, k=3, p=1) → BN → GELU → SE注意力
  → Conv1D(D→D, k=3, p=1) → BN → GELU → SE注意力
  → Conv1D(D→D, k=3, p=1) → BN
  + 残差投影 Proj(X_k)
  → GELU
输出 F_k ∈ R^{D×T}
```

- **SE (Squeeze-and-Excitation)** 注意力：全局平均池化 → FC → ReLU → FC → Sigmoid，对通道维度自适应加权。
- 残差连接稳定训练。

代码对应：`BCFE_Block` 类（`SENet1D` + 三层 Conv1D + 残差）。

#### CRIFU (Cross-Rhythm Interaction Fusion Unit)

五个 RRAU 输出在通道维拼接后：

```
F_cat ∈ R^{(K·D)×T}, K=5
  → Conv1×1(K·D→D)  → F_r ∈ R^{D×T}
  → CBAM(通道注意力+空间注意力) × 2
输出 F_RHSE ∈ R^{D×T}
```

- **CBAM 通道注意力**：平均池化 + 最大池化 → MLP → Sigmoid 加权。
- **CBAM 空间注意力**：通道维平均/最大池化 → Conv7×1 → Sigmoid 加权。

代码对应：`CFFE_Block` 类（`CBAM1D` + 两层 Conv1D）。

### 2. TAPD — 拓扑对齐原型蒸馏

#### TAL (Topology Alignment Layer)

```
F_RHSE [D, T] → 时间平均池化 → F_a [C, D]

静态映射:  F_static = W_s · F_a,  W_s ∈ R^{116×C} (可学习)
动态映射:  F_flat → Linear(C·D→H) → PReLU → Linear(H→116·D) → F_dynamic

F_map = F_static + F_dynamic  ∈ R^{116×D}
```

- 静态分量提供群体级正则化，动态分量实现受试者自适应。
- 映射后展平 `f_map ∈ R^{116·D}` 送入分类器。

代码对应：`BrainRegionMapper` 类（`mapping_matrix` 静态映射 + `mapper` MLP 动态映射）。

#### PCMD — 原型引导跨模态蒸馏

**fMRI 教师模型 (Graphormer)**：
- 输入：116×116 全脑功能连接矩阵（AAL 图谱，Pearson 相关）
- 节点嵌入：Linear(116→128) + 可学习位置编码
- 3 层 Transformer Encoder（8 头，前馈 512，dropout 0.1）
- 展平 → 2 层 MLP 原型生成器 → 64 维样本级原型向量
- 线性分类头

**类原型计算**：`p_c = mean(teacher_prototype_i), ∀i ∈ class c`（HC/SCZ 两类）。

**三种蒸馏损失**：

| 损失 | 公式 | 作用 |
|------|------|------|
| 特征对齐 L_feat | `MSE(z_i, p_{y_i})` | 约束表示空间，使 EEG 特征逼近对应类 fMRI 原型 |
| 原型对比 L_contrast | 温度缩放 InfoNCE，τ=0.07 | 结构化表示空间，拉近同类、推远异类原型 |
| 概率蒸馏 L_prob | `KL(teacher_q || student_s)` × τ² | 对齐决策行为，迁移教师的概率分布 |

总损失：`L_total = L_CE + α·L_feat + β·L_contrast + γ·L_prob`

代码对应：`GraphormerTeacher`、`DistillationLoss` 类。

---

## 环境要求与安装

### 依赖

```
Python >= 3.8
PyTorch >= 1.10
numpy
scipy
scikit-learn
mne >= 1.0
matplotlib
seaborn
```

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/rrrxrxrxr/TopoDistill-Net.git
cd TopoDistill-Net

# 创建虚拟环境（推荐）
conda create -n topodistill python=3.9
conda activate topodistill

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn mne matplotlib seaborn
```

### 硬件要求

- 推荐：NVIDIA GPU（实验使用 RTX 5090，24GB 显存）
- CPU 也可运行，但训练速度较慢

---

## 数据准备

### 数据集概览

| 数据集 | 类型 | HC | SCZ | 通道/脑区 | 采样率 | 用途 |
|--------|------|----|-----|-----------|--------|------|
| Sch Dataset | EEG | 39 | 45 | 16 通道 | 128 Hz | 学生模型训练/评估 |
| ASZED-153 | EEG | 77 | 76 | 16 通道 | 200/256 Hz | 学生模型训练/评估 |
| RepOD | EEG | 14 | 14 | 19 通道 | 250 Hz | 学生模型训练/评估 |
| COBRE | fMRI | 75 | 72 | 116 脑区 | TR=2s | 教师模型预训练 |

> **注意**：COBRE fMRI 数据仅用于教师模型预训练和类原型提取，**不参与任何 EEG 分类评估**。

### 数据目录结构

```
data/
├── norm_repod/          # 健康对照 EEG (.edf 文件)
│   ├── subject01.edf
│   ├── subject02.edf
│   └── ...
├── sch_repod/           # 精神分裂症 EEG (.edf 文件)
│   ├── patient01.edf
│   ├── patient02.edf
│   └── ...
└── COBRE-2D/            # fMRI 功能连接矩阵 (.npy / .npz)
    ├── control_001.npy  # 116×116 矩阵
    ├── patient_001.npy
    └── ...
```

### 预处理流程

代码中 `EEGDataLoader` 类自动执行以下预处理：

1. **读取 EDF**：使用 MNE 库读取原始 EEG
2. **重采样**：统一到 250 Hz
3. **带通滤波**：1–50 Hz（FIR 滤波器）
4. **通道统一**：自动检测最大通道数，不足补零、过多裁剪
5. **滑动窗口**：2 秒窗口（500 时间点），55% 重叠
6. **标准化**：`StandardScaler` 对所有样本展平后 fit_transform
7. **频段分解**：四阶 Butterworth 零相位带通滤波，提取 5 个频段
   - delta: 1–4 Hz
   - theta: 4–8 Hz
   - alpha: 8–13 Hz
   - beta: 13–30 Hz
   - gamma: 30–50 Hz

### fMRI 教师数据准备

COBRE fMRI 数据需预处理为 116×116 功能连接矩阵：
- 使用 AAL 图谱提取 116 脑区时间序列
- 计算 Pearson 相关矩阵
- 保存为 `.npy` 或 `.npz` 格式
- 文件名包含 `control`/`healthy`/`norm` 标记为 HC，否则标记为 SCZ

---

## 使用方法

### 配置数据路径

在 `TopoDistill-Net.py` 顶部修改路径：

```python
NORM_DIR = r"path/to/norm_repod"      # 健康对照 EEG 目录
SCH_DIR  = r"path/to/sch_repod"       # 精神分裂症 EEG 目录
FMRI_PATH = r"path/to/COBRE-2D"       # fMRI 功能连接矩阵目录
```

### 训练完整模型

```bash
python TopoDistill-Net.py \
    --hidden_dim 64 \
    --batch_size 16 \
    --epochs 50 \
    --lr 0.001 \
    --teacher_epochs 30 \
    --use_distill \
    --use_mapper \
    --use_gnn_tcn \
    --temperature 0.07 \
    --alpha 0.5 \
    --beta 0.3 \
    --gamma 0.2 \
    --augmentations time_shift gaussian_noise
```

由于 `--use_distill`、`--use_mapper`、`--use_gnn_tcn` 默认为 `True`，最简训练命令：

```bash
python TopoDistill-Net.py
```

### 运行消融实验

```bash
python TopoDistill-Net.py --ablation
```

消融配置在 `run_ablation()` 函数中定义，可取消注释以启用更多变体：

| 配置 | use_distill | use_mapper | use_gnn_tcn | 说明 |
|------|-------------|------------|-------------|------|
| Baseline | ✗ | ✗ | ✗ | 仅展平+分类器 |
| Distill Only | ✓ | ✗ | ✗ | 仅蒸馏 |
| Mapper Only | ✗ | ✓ | ✗ | 仅拓扑映射 |
| BCFE+CFFE Only | ✗ | ✗ | ✓ | 仅 RHSE 编码器 |
| Distill + Mapper | ✓ | ✓ | ✗ | 蒸馏+映射 |
| Distill + BCFE+CFFE | ✓ | ✗ | ✓ | 蒸馏+编码器 |
| Mapper + BCFE+CFFE | ✗ | ✓ | ✓ | 映射+编码器 |
| Full Model | ✓ | ✓ | ✓ | 完整 TopoDistill-Net |

### 训练输出

训练过程中每 10 个 epoch 打印：

```
Epoch  10/50: Loss=0.3421, Train Acc=0.8562, Val Acc=0.7834, F1=0.7756, AUC=0.8210, SEN=0.7234, SPE=0.8512
```

5 折交叉验证结束后输出汇总：

```
============================================================
5-Fold Cross Validation Results
Config: Distill=True, Mapper=True, GNN_TCN=True
============================================================
Accuracy:    0.8319 ± 0.0607
F1-Score:    0.8286 ± 0.0619
AUC:         0.8614 ± 0.0573
Sensitivity: 0.7471 ± 0.0752
Specificity: 0.9322 ± 0.0628
```

### 模型调用（推理示例）

```python
import torch
import numpy as np
from TopoDistill_Net import SZAttNetDistill

# 初始化模型
model = SZAttNetDistill(
    eeg_channels=19,      # EEG 通道数
    time_points=500,      # 2秒 × 250Hz
    fmri_regions=116,     # AAL 脑区数
    hidden_dim=64,
    num_classes=2,
    use_distill=True,
    use_mapper=True,
    use_gnn_tcn=True
)

# 加载训练好的权重
model.load_state_dict(torch.load('topodistill_best.pth'))
model.eval()

# 构造输入: [batch, channels, 5_freq_bands, time_points]
# 注意: 输入必须是已经过带通滤波分解为5频段的信号
dummy_input = torch.randn(1, 19, 5, 500)

with torch.no_grad():
    logits = model(dummy_input)              # [1, 2]
    probs = torch.softmax(logits, dim=1)     # [1, 2]
    pred = torch.argmax(logits, dim=1).item() # 0=HC, 1=SCZ

print(f"预测类别: {'精神分裂症' if pred == 1 else '健康对照'}")
print(f"置信度: HC={probs[0,0]:.4f}, SCZ={probs[0,1]:.4f}")
```

### 获取中间特征

```python
with torch.no_grad():
    logits, features = model(dummy_input, return_features=True)
    # features: 蒸馏对齐后的特征向量 [batch, hidden_dim]
```

---

## 输入输出说明

### 输入

| 张量 | 形状 | 说明 |
|------|------|------|
| `freq_signal` | `[B, C, K, T]` | 五频段 EEG 信号 |
| - B | batch size | 批次大小（默认 16） |
| - C | channels | EEG 电极通道数（16 或 19，自动统一） |
| - K | freq_bands | 频段数，固定为 5（delta/theta/alpha/beta/gamma） |
| - T | time_points | 时间点数，固定为 500（2s @ 250Hz） |

**数据类型**：`torch.FloatTensor`

**预处理要求**：
- 已重采样至 250 Hz
- 已带通滤波 1–50 Hz
- 已标准化（StandardScaler）
- 已分解为 5 个频段（代码中 `DistillEEGDataset.extract_frequency_bands()` 自动完成）

### 输出

| 张量 | 形状 | 说明 |
|------|------|------|
| `logits` | `[B, 2]` | 二分类 logits（未归一化） |
| `probs` | `[B, 2]` | Softmax 概率（`[P(HC), P(SCZ)]`） |
| `pred` | `[B]` | 预测类别（0=健康对照 HC，1=精神分裂症 SCZ） |

`return_features=True` 时额外返回：

| 张量 | 形状 | 说明 |
|------|------|------|
| `aligned_features` | `[B, hidden_dim]` | 蒸馏对齐特征（`use_distill=True` 时） |
| `mapped_features` | `[B, 116×D]` | 拓扑对齐后脑区特征（`use_distill=False` 时） |

### 评估指标

代码中 `calculate_metrics()` 计算以下 5 个指标：

| 指标 | 公式 | 含义 |
|------|------|------|
| Accuracy (ACC) | (TP+TN)/(TP+TN+FP+FN) | 总体分类准确率 |
| F1-Score | 2·P·R/(P+R) | 精确率与召回率的调和平均 |
| AUC | ROC 曲线下面积 | 分类器区分能力 |
| Sensitivity (SEN) | TP/(TP+FN) | 召回率/真阳性率 |
| Specificity (SPE) | TN/(TN+FP) | 真阴性率 |

---

## 关键超参数

### 模型结构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_dim` | 64 | RRAU/CRIFU 隐藏维度 D |
| `--fmri_regions` | 116 | AAL 脑区数（自动从 fMRI 数据检测） |
| `--num_classes` | 2 | 分类类别数（HC vs SCZ） |

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 16 | 批次大小 |
| `--epochs` | 50 | 最大训练轮数 |
| `--lr` | 0.001 | 初始学习率（AdamW） |
| `--teacher_epochs` | 30 | fMRI 教师模型预训练轮数 |
| weight_decay | 1e-4 | AdamW 权重衰减（代码内固定） |
| gradient_clip | 1.0 | 梯度裁剪范数（代码内固定） |
| scheduler | ReduceLROnPlateau | factor=0.5, patience=10（代码内固定） |

### 蒸馏损失权重

| 参数 | 默认值 | 说明 | 论文最优值 |
|------|--------|------|-----------|
| `--alpha` | 0.5 | 特征对齐损失权重 L_feat | 0.5 |
| `--beta` | 0.3 | 原型对比损失权重 L_contrast | 0.3 |
| `--gamma` | 0.2 | 概率蒸馏损失权重 L_prob | 0.2 |
| `--temperature` | 0.07 | 对比学习温度 τ | 0.07 |

### 模块开关

| 参数 | 默认值 | 对应模块 |
|------|--------|----------|
| `--use_distill` | True | TAPD 跨模态蒸馏 |
| `--use_mapper` | True | TAL 拓扑对齐层 |
| `--use_gnn_tcn` | True | RHSE 编码器（BCFE+CFFE） |

### 数据增强

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--augmentations` | `time_shift gaussian_noise` | 训练时数据增强方式 |
| time_shift | ±50 样本 | 循环时间偏移 |
| gaussian_noise | σ=5% 信号幅度 | 高斯噪声注入 |

### 验证策略

- **5 折分层交叉验证**（`StratifiedKFold`，`random_state=42`）
- **受试者级划分**：同一受试者的所有滑动窗口样本全部归入训练集或验证集，防止数据泄漏
- 每折保存验证集准确率最高的模型状态

---

## 实验结果

### 与基准方法对比

#### RepOD 数据集

| 方法 | ACC (%) | AUC (%) | F1 (%) | SEN (%) | SPE (%) |
|------|---------|---------|--------|---------|---------|
| SVM | 61.33±11.27 | 65.12±11.56 | 65.76±11.06 | 76.67±20.00 | 46.00±18.98 |
| TSception | 72.89±6.12 | 81.23±4.98 | 73.67±8.45 | 74.23±9.12 | 71.56±7.89 |
| MTCN | 71.56±8.23 | 80.78±6.12 | 73.12±10.45 | 76.45±10.89 | 67.32±10.23 |
| EEGNet | 65.77±7.28 | 67.93±8.05 | 64.59±12.34 | 62.23±16.75 | 66.88±9.42 |
| DeepConvNet | 71.95±9.30 | 74.12±9.87 | 66.64±18.87 | 60.94±23.77 | 70.56±11.23 |
| EEG-Conformer | 71.89±8.01 | 81.01±5.98 | 73.56±10.23 | 76.89±10.67 | 67.65±10.01 |
| LMDA-Net | 67.73±6.99 | 69.87±7.65 | 67.01±10.73 | 64.73±15.33 | 66.92±8.47 |
| MAS-DGAT-Net | 75.44±2.41 | 77.38±1.02 | 74.60±2.84 | 76.20±3.55 | 74.10±3.21 |
| **TopoDistill-Net** | **83.19±6.07** | **86.14±5.73** | **82.86±6.19** | 74.71±7.52 | **93.22±6.28** |

#### Sch Dataset（青少年队列）

| 方法 | ACC (%) | AUC (%) | F1 (%) | SEN (%) | SPE (%) |
|------|---------|---------|--------|---------|---------|
| SVM | 67.50±8.47 | 68.23±8.76 | 68.97±9.12 | 70.56±16.33 | 64.44±4.97 |
| EEG-Conformer | 70.23±4.89 | 78.32±6.23 | 72.01±3.98 | 72.34±5.01 | 68.23±9.56 |
| MAS-DGAT-Net | 70.12±5.23 | 77.89±6.01 | 71.34±4.67 | 72.01±5.34 | 68.23±6.12 |
| **TopoDistill-Net** | **71.56±2.85** | 77.67±2.82 | **72.34±3.59** | 65.67±3.18 | **82.34±3.83** |

#### ASZED-153（异质性多系统队列）

| 方法 | ACC (%) | AUC (%) | F1 (%) | SEN (%) | SPE (%) |
|------|---------|---------|--------|---------|---------|
| SVM | 62.53±6.04 | 63.23±6.32 | 64.09±4.89 | 66.19±4.32 | 58.87±6.57 |
| TSception | 70.89±5.45 | 72.12±5.89 | 72.23±5.12 | 75.34±6.12 | 67.45±5.67 |
| EEG-Conformer | 71.23±6.21 | 74.23±5.45 | 68.45±7.76 | 75.89±12.01 | 67.56±5.21 |
| MAS-DGAT-Net | 71.34±5.12 | 73.78±4.98 | 70.12±5.34 | 74.23±6.12 | 67.89±5.67 |
| **TopoDistill-Net** | **74.56±4.88** | **75.89±4.84** | **73.89±5.85** | 70.12±6.28 | **80.12±4.82** |

> 显著性标记：`* p<0.05`，`** p<0.01`（配对 t 检验 + FDR 校正，相对于 TopoDistill-Net）。

### 消融实验

| 数据集 | 变体 | ACC (%) | AUC (%) | SPE (%) |
|--------|------|---------|---------|---------|
| RepOD | w/o TAL | 81.56±6.42* | 84.72±6.15* | 91.45±6.61* |
| | w/o RHSE | 80.52±6.83** | 83.45±6.59** | 90.90±7.57** |
| | w/o distill | 81.29±6.12* | 84.31±5.86* | 91.73±6.33* |
| | **Full** | **83.19±6.07** | **86.14±5.73** | **93.22±6.28** |
| Sch | w/o TAL | 69.89±3.45* | 76.12±3.02* | 80.67±3.45* |
| | w/o RHSE | 68.45±3.12** | 74.56±2.98** | 79.23±3.67** |
| | w/o distill | 69.23±3.92* | 75.34±2.88* | 80.12±2.87* |
| | **Full** | **71.56±2.85** | **77.67±2.82** | **82.34±3.83** |
| ASZED | w/o TAL | 71.89±5.12** | 74.23±5.01* | 78.34±5.23* |
| | w/o RHSE | 69.34±5.23** | 73.78±5.01** | 77.45±5.89** |
| | w/o distill | 70.12±6.95** | 74.56±5.89* | 78.23±7.88* |
| | **Full** | **74.56±4.88** | **75.89±4.84** | **80.12±4.82** |

### 蒸馏损失权重敏感性（RepOD）

| α | β | γ | ACC (%) | SPE (%) |
|---|---|---|---------|---------|
| 1 | 0 | 0 | 81.67±7.27 | 91.88±7.14 |
| 0 | 1 | 0 | 82.01±6.93 | 92.10±6.80 |
| 0 | 0 | 1 | 81.85±6.71 | 91.96±6.92 |
| 0.33 | 0.33 | 0.33 | 82.54±6.39 | 92.56±6.53 |
| **0.5** | **0.3** | **0.2** | **83.19±6.07** | **93.22±6.28** |
| 0.7 | 0.2 | 0.1 | 82.91±6.24 | 92.84±6.44 |

### 单频段贡献分析（RepOD）

| 频段 | ACC (%) | SEN (%) | SPE (%) |
|------|---------|---------|---------|
| Delta | 80.52±7.33 | 66.04±8.51 | **94.78±5.57** |
| Theta | 79.51±7.86 | 71.23±8.02 | 87.76±6.79 |
| Alpha | 79.78±7.04 | 69.12±7.34 | 89.71±6.04 |
| Beta | 80.71±8.14 | 72.31±8.82 | 88.87±7.09 |
| Gamma | 80.62±7.52 | **78.21±8.29** | 82.81±6.38 |
| All (w/o CRIFU) | 81.47±6.22 | 72.45±7.71 | 91.32±6.43 |
| **All (Full)** | **83.19±6.07** | 74.71±7.52 | **93.22±6.28** |

> Delta 频段特异性最高，Gamma 频段敏感性最高，五频段融合 + CRIFU 取得最优性能。

### fMRI 教师模型独立性能（COBRE，5 折 CV）

| ACC | AUC | F1 | SEN | SPE |
|-----|-----|-----|-----|-----|
| 78.23% | 82.56% | 77.78% | 76.39% | 80.00% |

---

## 项目结构

```
TopoDistill-Net/
├── TopoDistill-Net.py          # 主代码（模型定义、训练、评估）
├── README.md                   # 本文件
├── data/
│   ├── norm_repod/             # 健康对照 EEG (.edf)
│   ├── sch_repod/              # 精神分裂症 EEG (.edf)
│   └── COBRE-2D/               # fMRI 功能连接矩阵 (.npy/.npz)
└── outputs/                    # 训练结果（自动创建）
```

### 代码模块说明

| 类/函数 | 位置 | 功能 |
|---------|------|------|
| `EEGDataAugmentation` | 数据增强 | 时间偏移、高斯噪声 |
| `SENet1D` | 注意力 | 1D 压缩-激励注意力模块 |
| `BCFE_Block` | RHSE-RRAU | 节律特异性残差注意力单元（Conv1D×3 + SE + 残差） |
| `CBAM1D` | 注意力 | 1D CBAM 通道+空间注意力 |
| `CFFE_Block` | RHSE-CRIFU | 跨节律交互融合单元（Conv1D + CBAM×2） |
| `ModifiedGNN_TCN_Backbone` | RHSE | 完整编码器：5 个 BCFE 并行 + CFFE 融合 |
| `GraphAttentionLayer` | 可选 | 简化版 GAT 层（代码中预留，未在主路径使用） |
| `BrainRegionMapper` | TAL | EEG 电极→fMRI 脑区拓扑映射（静态+动态） |
| `GraphormerTeacher` | 教师 | fMRI Graphormer 教师模型 |
| `DistillationLoss` | 蒸馏 | 三损失组合：特征对齐+对比+概率蒸馏 |
| `SZAttNetDistill` | 主模型 | TopoDistill-Net 完整模型 |
| `DistillEEGDataset` | 数据集 | PyTorch Dataset，含频段分解和增强 |
| `EEGDataLoader` | 数据加载 | EDF 读取、预处理、滑动窗口 |
| `SZAttNetTrainer` | 训练器 | 5 折 CV 训练、评估、指标计算 |
| `run_ablation()` | 消融 | 消融实验运行器 |
| `main()` | 入口 | 命令行参数解析和训练启动 |

---

## 参考文献

本工作引用的关键文献：

1. Uhlhaas, P. J., & Singer, W. (2010). Abnormal neural oscillations and synchrony in schizophrenia. *Nature Reviews Neuroscience*, 11(2), 100–113.
2. Friston, K. J., & Frith, C. D. (1995). Schizophrenia: a disconnection syndrome. *Clinical Neuroscience*, 3(2), 89–97.
3. Lynall, M. E., et al. (2010). Functional connectivity and brain networks in schizophrenia. *Journal of Neuroscience*, 30(4), 945–953.
4. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR*, 7132–7141.
5. Woo, S., et al. (2018). CBAM: Convolutional block attention module. *ECCV*.
6. Ding, Y., et al. (2022). TSception: Capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition. *IEEE Transactions on Affective Computing*, 14(3), 2238–2250.
7. Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based BCIs. *Journal of Neural Engineering*, 15(5), 056013.
8. Song, Y., et al. (2022). EEG Conformer: Convolutional transformer for EEG decoding. *IEEE TNSRE*, 31, 710–719.
9. Liu, S., et al. (2024). MAS-DGAT-Net: A dynamic graph attention network for EEG emotion recognition. *Knowledge-Based Systems*, 305, 112599.
10. Mayer, A. R., et al. (2013). Functional imaging of the hemodynamic sensory gating response in schizophrenia. *Human Brain Mapping*, 34(9), 2302–2312. (COBRE 数据集)
11. Olejarczyk, E., & Jernajczyk, W. (2017). EEG in schizophrenia. *RepOD*.
12. Gorbachevskaya, N., & Borisov, S. V. (2019). EEG of healthy adolescents and adolescents with symptoms of schizophrenia. (Sch Dataset)
13. Mosaku, S., et al. (2025). An open-access EEG dataset from indigenous African populations for schizophrenia research. *Data in Brief*, 111934. (ASZED-153)

完整参考文献列表请参见论文原文。

---

## 常见问题

**Q: 推理时需要 fMRI 数据吗？**
A: 不需要。fMRI 教师模型仅在训练阶段用于提取功能网络先验和类原型，推理时 TopoDistill-Net 仅输入 EEG 信号。

**Q: 如何使用自己的 EEG 数据？**
A: 将 EDF 文件放入对应目录（健康对照→`norm_repod`，患者→`sch_repod`），代码会自动处理重采样、滤波、通道统一和滑动窗口。确保修改 `NORM_DIR` 和 `SCH_DIR` 路径。

**Q: 通道数不一致怎么办？**
A: `EEGDataLoader.unify_channels()` 会自动检测数据集中的最大通道数，对通道不足的样本补零、通道过多的样本裁剪。也可通过 `target_channels` 参数手动指定。

**Q: 没有 fMRI 数据能训练吗？**
A: 可以。设置 `--use_distill` 为 False 即可关闭蒸馏，仅使用 RHSE + TAL 训练 EEG 模型。若使用蒸馏但无 fMRI 数据，代码会自动使用随机原型（不推荐，会降低性能）。

**Q: 如何复现论文结果？**
A: 使用默认超参数（`hidden_dim=64, lr=0.001, batch_size=16, epochs=50, α=0.5, β=0.3, γ=0.2`），设置 `SEED=42`，在对应数据集上运行 5 折交叉验证。

---

## 许可证

本项目仅供学术研究使用。

---

## 致谢

感谢 COBRE、RepOD、Sch Dataset、ASZED-153 数据集的开源贡献者。本工作得到国家自然科学基金等项目支持。
