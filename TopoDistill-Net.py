
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, recall_score)
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import signal
import random
import argparse

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
NORM_DIR = r"D:\code\code\SZAtt-Net-main\norm_repod"
SCH_DIR = r"D:\code\code\SZAtt-Net-main\sch_repod"
FMRI_PATH = r"D:\code\code\SZAtt-Net-main\COBRE-2D"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 数据增强模块 ====================
class EEGDataAugmentation:
    """EEG数据增强"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

    def time_shift(self, eeg_data, shift_range=50):
        channels, time_points = eeg_data.shape
        shift = np.random.randint(-shift_range, shift_range)
        if shift == 0:
            return eeg_data.copy()
        shifted_data = np.zeros_like(eeg_data)
        if shift > 0:
            shifted_data[:, shift:] = eeg_data[:, :-shift]
            shifted_data[:, :shift] = eeg_data[:, :shift]
        else:
            shift = abs(shift)
            shifted_data[:, :-shift] = eeg_data[:, shift:]
            shifted_data[:, -shift:] = eeg_data[:, -shift:]
        return shifted_data

    def gaussian_noise(self, eeg_data, noise_level=0.05):
        channels, time_points = eeg_data.shape
        noisy_data = eeg_data.copy()
        for ch in range(channels):
            signal_std = np.std(eeg_data[ch, :])
            noise = np.random.normal(0, signal_std * noise_level, time_points)
            noisy_data[ch, :] = eeg_data[ch, :] + noise
        return noisy_data

    def apply_augmentations(self, eeg_data, augmentations=None, **kwargs):
        if augmentations is None:
            return eeg_data.copy()
        augmented_data = eeg_data.copy()
        for aug in augmentations:
            if aug == 'time_shift':
                shift_range = kwargs.get('time_shift_range', 50)
                augmented_data = self.time_shift(augmented_data, shift_range)
            elif aug == 'gaussian_noise':
                noise_level = kwargs.get('noise_level', 0.05)
                augmented_data = self.gaussian_noise(augmented_data, noise_level)
        return augmented_data


# ==================== 修改 BCFE + CFFE 模块 (替代原GNN+TCN) ====================
class SENet1D(nn.Module):
    """1D Squeeze-and-Excitation Block"""
    def __init__(self, in_channel, reduction=16):
        super(SENet1D, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, L]
        b, c, l = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class BCFE_Block(nn.Module):
    """
    修改的 BCFE (Brain Connectivity Feature Extraction) 模块
    基于原 MAS-DGAT-Net 的 sperchannel，但进行了简化和适配
    输入: [B, C, T] 原始EEG信号或频段特征
    输出: [B, C, hidden_dim]
    """
    def __init__(self, in_channels, hidden_dim=64):
        super(BCFE_Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.se1 = SENet1D(hidden_dim, reduction=4)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.se2 = SENet1D(hidden_dim, reduction=4)

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.activation = nn.PReLU()
        self.residual = nn.Conv1d(in_channels, hidden_dim, kernel_size=1) if in_channels != hidden_dim else nn.Identity()

    def forward(self, x):
        # x: [B, C, T]
        residual = self.residual(x)

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.se1(out)

        out = self.activation(self.bn2(self.conv2(out)))
        out = self.se2(out)

        out = self.bn3(self.conv3(out))
        out = self.activation(out + residual)  # 残差连接

        return out  # [B, hidden_dim, T]


class CBAM1D(nn.Module):
    """1D CBAM 注意力模块"""
    def __init__(self, channels, reduction=8):
        super(CBAM1D, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.conv_spatial = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


class CFFE_Block(nn.Module):
    """
    修改的 CFFE (Cross-Frequency Feature Extraction) 模块
    基于原 MAS-DGAT-Net 的 cbam_pro，适配多频段特征
    输入: [B, C, hidden_dim, T] 或 [B, C, hidden_dim] -> 处理后输出增强特征
    """
    def __init__(self, in_channels, hidden_dim=64):
        super(CFFE_Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.cbam1 = CBAM1D(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.cbam2 = CBAM1D(hidden_dim)

        self.activation = nn.PReLU()

    def forward(self, x):
        # x: [B, C, L] (L可以是时间维度)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.cbam1(out)

        out = self.activation(self.bn2(self.conv2(out)))
        out = self.cbam2(out)

        return out  # [B, hidden_dim, L]


class ModifiedGNN_TCN_Backbone(nn.Module):
    """
    融合 BCFE + CFFE 的主干网络，替代原有的 GNN+TCN
    支持频段特征输入，输出全局特征向量
    """
    def __init__(self, eeg_channels, time_points, num_freq_bands=5, hidden_dim=64):
        super(ModifiedGNN_TCN_Backbone, self).__init__()
        self.num_freq_bands = num_freq_bands
        self.hidden_dim = hidden_dim

        # 每个频段独立的 BCFE 处理
        self.bcfe_blocks = nn.ModuleList([
            BCFE_Block(in_channels=eeg_channels, hidden_dim=hidden_dim)
            for _ in range(num_freq_bands)
        ])

        # 频段融合后的 CFFE 处理
        self.cffe_block = CFFE_Block(in_channels=hidden_dim * num_freq_bands, hidden_dim=hidden_dim)

        # 全局池化后输出维度
        self.output_dim = hidden_dim * time_points  # 经过CFFE后时间维度不变

        # 可选的图卷积层（保留图结构信息，但简化）
        self.gat_layer = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=0.1, alpha=0.2, concat=False)

    def forward(self, freq_signal, adj=None):
        """
        freq_signal: [B, C, 5, T]
        adj: 可选的功能连接矩阵 [B, C, C]
        """
        batch_size, channels, num_bands, time_points = freq_signal.shape

        band_outputs = []
        for band_idx in range(num_bands):
            band_data = freq_signal[:, :, band_idx, :]  # [B, C, T]
            bcfe_out = self.bcfe_blocks[band_idx](band_data)  # [B, hidden_dim, T]
            band_outputs.append(bcfe_out)

        # 拼接所有频段特征 [B, hidden_dim*5, T]
        concat_features = torch.cat(band_outputs, dim=1)

        # CFFE 进一步提取跨频段特征
        cffe_out = self.cffe_block(concat_features)  # [B, hidden_dim, T]

        # 可选的图卷积 (如果提供了邻接矩阵)
        if adj is not None:
            # 将特征重塑为 [B, C, hidden_dim*T] 进行GAT? 这里简化为对通道维度进行注意力
            # 为了适配图注意力，我们先将cffe_out重塑为 [B, channels, hidden_dim*T]
            # 但这里 channels 与 hidden_dim 不一致，所以我们采用另一种方式：
            # 直接对每个时间步做平均池化后应用GAT
            pooled = cffe_out.mean(dim=2)  # [B, hidden_dim]
            # 这里不再深入图卷积以保持简洁，因为主要创新已体现在BCFE+CFFE
            # 如果需要图卷积，可以扩展
            pass

        # 展平作为全局特征
        global_feat = cffe_out.reshape(batch_size, -1)  # [B, hidden_dim * T]

        return global_feat, cffe_out


class GraphAttentionLayer(nn.Module):
    """简化版GAT层，用于可能的图增强"""
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.matmul(inp, self.W)
        N = h.size(1)
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features),
                             h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


# ==================== 脑区映射模块 ====================
class BrainRegionMapper(nn.Module):
    """EEG通道到fMRI脑区的映射模块"""
    def __init__(self, eeg_channels, fmri_regions, hidden_dim=64):
        super(BrainRegionMapper, self).__init__()
        self.eeg_channels = eeg_channels
        self.fmri_regions = fmri_regions
        self.mapping_matrix = nn.Parameter(torch.randn(eeg_channels, fmri_regions) * 0.1)
        self.mapper = nn.Sequential(
            nn.Linear(eeg_channels, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, fmri_regions),
            nn.Softmax(dim=1)
        )

    def forward(self, eeg_features, return_attention=False):
        # eeg_features: [B, C, D]
        batch_size, eeg_channels, feature_dim = eeg_features.shape
        mapping_weights = F.softmax(self.mapping_matrix, dim=1)
        pooled = eeg_features.mean(dim=2)
        dynamic_weights = self.mapper(pooled)
        static_mapped = torch.einsum('bcf,cr->brf', eeg_features, mapping_weights)
        dynamic_mapped = torch.einsum('bcf,bcr->brf', eeg_features,
                                      dynamic_weights.unsqueeze(1).expand(-1, feature_dim, -1))
        mapped_features = static_mapped + dynamic_mapped
        if return_attention:
            return mapped_features, mapping_weights
        return mapped_features


# ==================== fMRI蒸馏模块 ====================
class GraphormerTeacher(nn.Module):
    """Graphormer教师模型"""
    def __init__(self, num_regions=116, hidden_dim=128, num_heads=8, num_layers=3):
        super(GraphormerTeacher, self).__init__()
        self.num_regions = num_regions
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Linear(num_regions, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_regions, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.prototype_generator = nn.Sequential(
            nn.Linear(hidden_dim * num_regions, hidden_dim * 2),
            nn.PReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)

    def forward(self, fmri_matrix, return_prototypes=False):
        batch_size = fmri_matrix.size(0)
        node_features = self.node_embedding(fmri_matrix) + self.pos_encoding
        transformer_out = self.transformer(node_features)
        flattened = transformer_out.reshape(batch_size, -1)
        prototypes = self.prototype_generator(flattened)
        logits = self.classifier(prototypes)
        if return_prototypes:
            return logits, prototypes
        return logits


class DistillationLoss(nn.Module):
    """蒸馏损失（特征对齐+对比学习+概率蒸馏）"""
    def __init__(self, temperature=0.07, alpha=0.5, beta=0.3, gamma=0.2):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, eeg_logits, eeg_features, eeg_labels, fmri_prototypes, fmri_probs):
        device = eeg_features.device
        classification_loss = self.ce_loss(eeg_logits, eeg_labels)
        target_prototypes = fmri_prototypes[eeg_labels]
        feature_align_loss = self.mse_loss(eeg_features, target_prototypes)

        # Contrastive
        eeg_norm = F.normalize(eeg_features, dim=1)
        proto_norm = F.normalize(fmri_prototypes, dim=1)
        sim_matrix = torch.matmul(eeg_norm, proto_norm.T) / self.temperature
        contrastive_loss = self.ce_loss(sim_matrix, eeg_labels)

        # Probability distillation
        target_probs = fmri_probs[eeg_labels]
        eeg_probs = F.log_softmax(eeg_logits / self.temperature, dim=1)
        target_probs_scaled = F.softmax(target_probs / self.temperature, dim=1)
        prob_distill_loss = self.kl_loss(eeg_probs, target_probs_scaled) * (self.temperature ** 2)

        total_loss = (classification_loss + self.alpha * feature_align_loss +
                      self.beta * contrastive_loss + self.gamma * prob_distill_loss)
        return total_loss, classification_loss, feature_align_loss, contrastive_loss, prob_distill_loss


# ==================== 主模型 ====================
class SZAttNetDistill(nn.Module):
    def __init__(self, eeg_channels, time_points, fmri_regions=116,
                 hidden_dim=64, num_classes=2,
                 use_distill=False, use_mapper=True, use_gnn_tcn=True):
        super(SZAttNetDistill, self).__init__()
        self.eeg_channels = eeg_channels
        self.time_points = time_points
        self.fmri_regions = fmri_regions
        self.hidden_dim = hidden_dim
        self.use_distill = use_distill
        self.use_mapper = use_mapper
        self.use_gnn_tcn = use_gnn_tcn

        if use_gnn_tcn:
            self.backbone = ModifiedGNN_TCN_Backbone(eeg_channels, time_points, hidden_dim=hidden_dim)
            backbone_out_dim = hidden_dim * time_points
        else:
            # 简单基线：直接展平频段特征
            backbone_out_dim = eeg_channels * 5 * time_points
            self.backbone = None

        if use_mapper:
            self.mapper = BrainRegionMapper(eeg_channels, fmri_regions, hidden_dim)
            # 添加投影层：将backbone展平输出映射到eeg_channels*hidden_dim，以适配mapper
            self.mapper_proj = nn.Linear(hidden_dim, eeg_channels * hidden_dim)
            # 映射后特征维度变为 fmri_regions * hidden_dim
            self.mapped_feat_dim = fmri_regions * hidden_dim
        else:
            self.mapped_feat_dim = backbone_out_dim

        # 分类器输入维度
        classifier_in_dim = self.mapped_feat_dim if use_mapper else backbone_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim * 2),
            nn.PReLU(), nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        if use_distill:
            self.feature_projector = nn.Linear(classifier_in_dim, hidden_dim)  # 输出维度与fMRI原型匹配

    def forward(self, freq_signal, raw_signal=None, return_features=False):
        batch_size = freq_signal.size(0)

        if self.use_gnn_tcn:
            global_feat, cffe_out = self.backbone(freq_signal)  # [B, D]
        else:
            global_feat = freq_signal.reshape(batch_size, -1)

        if self.use_mapper and self.use_gnn_tcn:
            if hasattr(self, 'backbone') and self.backbone is not None:
                # 从backbone获取cffe_out: [B, hidden_dim, T]
                _, cffe_out = self.backbone(freq_signal)
                # 对时间维平均池化得到通道级特征: [B, hidden_dim]
                channel_feat = cffe_out.mean(dim=2)  # [B, hidden_dim]
                # 通过投影层映射到 eeg_channels 空间
                projected = self.mapper_proj(channel_feat)  # [B, eeg_channels * hidden_dim]
                projected = projected.reshape(batch_size, self.eeg_channels, self.hidden_dim)  # [B, C, D]
                # 使用mapper的静态映射矩阵投影到脑区空间
                mapping_weights = F.softmax(self.mapper.mapping_matrix, dim=1)  # [C, R]
                mapped = torch.einsum('bcd,cr->brd', projected, mapping_weights)  # [B, R, D]
                mapped_features = mapped.reshape(batch_size, -1)  # [B, R * D]
            else:
                mapped_features = global_feat
        else:
            mapped_features = global_feat

        logits = self.classifier(mapped_features)

        if return_features and self.use_distill:
            aligned_features = self.feature_projector(mapped_features)
            return logits, aligned_features
        elif return_features:
            return logits, mapped_features
        return logits


# ==================== 数据集 ====================
class DistillEEGDataset(Dataset):
    def __init__(self, signals, labels, subjects, augmentations=None, is_training=False):
        self.signals = signals
        self.labels = labels
        self.subjects = subjects
        self.is_training = is_training
        self.augmenter = EEGDataAugmentation() if augmentations and is_training else None
        self.augmentations = augmentations

    def extract_frequency_bands(self, eeg_signal, sfreq=250):
        channels, time_points = eeg_signal.shape
        freq_bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
        nyquist = sfreq / 2
        freq_features = np.zeros((channels, len(freq_bands), time_points))
        for ch in range(channels):
            for band_idx, (low, high) in enumerate(freq_bands):
                if high < nyquist:
                    try:
                        b, a = signal.butter(4, [low/nyquist, high/nyquist], btype='band')
                        filtered = signal.filtfilt(b, a, eeg_signal[ch])
                        freq_features[ch, band_idx, :] = filtered
                    except:
                        freq_features[ch, band_idx, :] = eeg_signal[ch]
        return freq_features

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        if self.is_training and self.augmenter is not None:
            signal = self.augmenter.apply_augmentations(signal, self.augmentations,
                                                        time_shift_range=50, noise_level=0.05)
        freq_signal = self.extract_frequency_bands(signal)
        return {
            'raw_signal': torch.FloatTensor(signal),
            'freq_signal': torch.FloatTensor(freq_signal),
            'label': torch.LongTensor([self.labels[idx]]),
            'subject': self.subjects[idx]
        }


# ==================== 数据加载器 ====================
class EEGDataLoader:
    def __init__(self, norm_dir, sch_dir, target_sfreq=250, target_channels=None, window_size=None):
        self.norm_dir = Path(norm_dir)
        self.sch_dir = Path(sch_dir)
        self.target_sfreq = target_sfreq
        self.target_channels = target_channels  # 统一的目标通道数
        self.window_size = window_size or int(2 * target_sfreq)
        self.overlap = 0.55

    def bandpass_filter(self, raw, l_freq=1.0, h_freq=50.0):
        """带通滤波"""
        return raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

    def unify_channels(self, eeg_data):
        """
        统一通道数：裁剪或补零
        eeg_data: [channels, time_points]
        """
        if self.target_channels is None:
            return eeg_data

        current_channels = eeg_data.shape[0]

        if current_channels == self.target_channels:
            return eeg_data
        elif current_channels > self.target_channels:
            # 裁剪多余的通道
            return eeg_data[:self.target_channels, :]
        else:
            # 补零
            pad = np.zeros((self.target_channels - current_channels, eeg_data.shape[1]))
            return np.vstack([eeg_data, pad])

    def load_data(self, folder_path, label):
        """加载单个文件夹的数据"""
        data = []
        labels = []
        subjects = []

        folder_path = Path(folder_path)
        edf_files = list(folder_path.glob("*.edf"))

        if not edf_files:
            print(f"No EDF files found in {folder_path}")
            return np.array([]), np.array([]), np.array([])

        print(f"Loading from {folder_path}: found {len(edf_files)} files")

        for file in edf_files:
            try:
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                subject_id = file.stem

                # 获取原始信息
                original_sfreq = raw.info['sfreq']
                original_channels = len(raw.ch_names)
                print(f"  {subject_id}: {original_channels} channels, {original_sfreq} Hz")

                # 重采样
                if original_sfreq != self.target_sfreq:
                    raw.resample(self.target_sfreq, npad='auto', verbose=False)

                # 滤波
                raw = self.bandpass_filter(raw)

                # 获取数据
                eeg_data = raw.get_data()

                # 统一通道数
                if self.target_channels is None:
                    self.target_channels = eeg_data.shape[0]
                    print(f"  Set target channels to {self.target_channels}")

                eeg_data = self.unify_channels(eeg_data)

                # 滑动窗口
                step = int(self.window_size * (1 - self.overlap))
                for start in range(0, eeg_data.shape[1] - self.window_size + 1, step):
                    window = eeg_data[:, start:start + self.window_size]
                    data.append(window)
                    labels.append(label)
                    subjects.append(subject_id)

            except Exception as e:
                print(f"  Error loading {file}: {e}")
                continue

        return np.array(data), np.array(labels), np.array(subjects)

    def load_all_data(self):
        """加载所有数据"""
        # 先扫描所有文件确定最大通道数
        print("Scanning for channel counts...")
        all_channels = []

        for folder in [self.norm_dir, self.sch_dir]:
            for file in folder.glob("*.edf"):
                try:
                    raw = mne.io.read_raw_edf(file, preload=False, verbose=False)
                    all_channels.append(len(raw.ch_names))
                except:
                    continue

        if all_channels:
            # 使用最大通道数作为统一标准
            self.target_channels = max(all_channels)
            print(f"Detected channel counts: {set(all_channels)}")
            print(f"Will unify to {self.target_channels} channels")
        else:
            self.target_channels = 24  # 默认值
            print(f"No files found, using default {self.target_channels} channels")

        print("\nLoading healthy controls...")
        norm_data, norm_labels, norm_subjects = self.load_data(self.norm_dir, 0)
        print(f"Controls: {len(norm_data)} windows")

        print("\nLoading schizophrenia patients...")
        sch_data, sch_labels, sch_subjects = self.load_data(self.sch_dir, 1)
        print(f"Patients: {len(sch_data)} windows")

        # 检查是否有数据
        if len(norm_data) == 0 and len(sch_data) == 0:
            raise ValueError("No data loaded! Check the data paths.")
        elif len(norm_data) == 0:
            print("Warning: No control data found!")
            X = sch_data
            y = sch_labels
            subjects = sch_subjects
        elif len(sch_data) == 0:
            print("Warning: No patient data found!")
            X = norm_data
            y = norm_labels
            subjects = norm_subjects
        else:
            X = np.concatenate([norm_data, sch_data], axis=0)
            y = np.concatenate([norm_labels, sch_labels], axis=0)
            subjects = np.concatenate([norm_subjects, sch_subjects], axis=0)

        print(f"\nTotal samples: {len(X)}")
        print(f"Controls: {len(norm_data)}, Patients: {len(sch_data)}")
        print(f"Channels: {X.shape[1]}, Time points: {X.shape[2]}")

        return X, y, subjects, X.shape[1], X.shape[2]


# ==================== 训练器 ====================
class SZAttNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = device
        self.load_eeg_data()

        # 始终初始化 fmri_regions（即使不使用蒸馏）
        self.init_fmri_regions()

        if args.use_distill:
            self.init_teacher_model()

        self.history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [], 'val_sen': [], 'val_spe': []}

    def init_fmri_regions(self):
        """初始化fMRI脑区数量（无论是否使用蒸馏都需要）"""
        fmri_path = Path(FMRI_PATH)

        # 尝试从fMRI数据中获取脑区数量
        fmri_files = list(fmri_path.glob("*.npy")) + list(fmri_path.glob("*.npz"))
        self.fmri_regions = 116  # 默认值

        for file in fmri_files:
            try:
                mat = np.load(file, allow_pickle=True)
                if isinstance(mat, np.lib.npyio.NpzFile):
                    mat = mat[list(mat.keys())[0]]
                if len(mat.shape) >= 2:
                    self.fmri_regions = mat.shape[0]  # 假设是 [regions, regions]
                    print(f"Detected fMRI regions: {self.fmri_regions} from {file.name}")
                    break
            except Exception as e:
                continue

        print(f"Using {self.fmri_regions} fMRI regions")

    def load_eeg_data(self):
        loader = EEGDataLoader(NORM_DIR, SCH_DIR)
        self.X, self.y, self.subjects, self.eeg_channels, self.time_points = loader.load_all_data()
        original_shape = self.X.shape
        X_flat = self.X.reshape(original_shape[0], -1)
        self.scaler = StandardScaler()
        X_normalized = self.scaler.fit_transform(X_flat)
        self.X = X_normalized.reshape(original_shape)
        print(f"EEG data shape: {self.X.shape}")

    def init_teacher_model(self):
        """初始化并训练教师模型（仅在 use_distill=True 时调用）"""
        fmri_path = Path(FMRI_PATH)
        fmri_files = list(fmri_path.glob("*.npy")) + list(fmri_path.glob("*.npz"))
        data, labels = [], []

        for file in fmri_files:
            try:
                mat = np.load(file, allow_pickle=True)
                if isinstance(mat, np.lib.npyio.NpzFile):
                    mat = mat[list(mat.keys())[0]]
                fname = file.stem.lower()
                label = 0 if ('control' in fname or 'healthy' in fname or 'norm' in fname) else 1
                data.append(mat)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

        if data:
            data = np.array(data)
            labels = np.array(labels)
            self.fmri_regions = data.shape[1]  # 更新为实际值
            print(f"Loaded {len(data)} fMRI samples, regions: {self.fmri_regions}")

            # 训练教师模型
            teacher = GraphormerTeacher(num_regions=self.fmri_regions).to(device)
            optimizer = optim.Adam(teacher.parameters(), lr=0.001, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
            loader = DataLoader(dataset, batch_size=16, shuffle=True)

            teacher.train()
            for epoch in range(self.args.teacher_epochs):
                total_loss = 0
                for batch_data, batch_labels in loader:
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    optimizer.zero_grad()
                    logits = teacher(batch_data)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    print(
                        f"  Teacher Epoch {epoch + 1}/{self.args.teacher_epochs}, Loss: {total_loss / len(loader):.4f}")

            # 提取原型
            teacher.eval()
            health_protos, patient_protos = [], []
            health_probs, patient_probs = [], []

            with torch.no_grad():
                for batch_data, batch_labels in loader:
                    batch_data = batch_data.to(device)
                    logits, proto = teacher(batch_data, return_prototypes=True)
                    probs = F.softmax(logits, dim=1)

                    for i, lab in enumerate(batch_labels):
                        if lab == 0:
                            health_protos.append(proto[i].cpu())
                            health_probs.append(probs[i].cpu())
                        else:
                            patient_protos.append(proto[i].cpu())
                            patient_probs.append(probs[i].cpu())

            if health_protos:
                self.health_proto = torch.stack(health_protos).mean(dim=0)
                self.health_prob = torch.stack(health_probs).mean(dim=0)
            else:
                self.health_proto = torch.randn(64)
                self.health_prob = torch.tensor([0.8, 0.2])

            if patient_protos:
                self.patient_proto = torch.stack(patient_protos).mean(dim=0)
                self.patient_prob = torch.stack(patient_probs).mean(dim=0)
            else:
                self.patient_proto = torch.randn(64)
                self.patient_prob = torch.tensor([0.2, 0.8])

            self.fmri_prototypes = torch.stack([self.health_proto, self.patient_proto]).to(device)
            self.fmri_probs = torch.stack([self.health_prob, self.patient_prob]).to(device)

            print(f"Prototypes extracted: health={self.health_proto.shape}, patient={self.patient_proto.shape}")
        else:
            print("No fMRI data found, using random prototypes")
            self.fmri_regions = 116
            proto_dim = 64
            self.fmri_prototypes = torch.randn(2, proto_dim).to(device)
            self.fmri_probs = torch.softmax(torch.randn(2, 2), dim=1).to(device)

    def calculate_metrics(self, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')

        try:
            auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
        except:
            auc = 0.5

        # 计算混淆矩阵以获取 SEN 和 SPE
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        else:
            sen = recall_score(y_true, y_pred, average='binary')
            spe = 0.0

        return acc, f1, auc, sen, spe

    def train_fold(self, train_idx, val_idx, fold):
        print(f"\n{'=' * 60}\nFold {fold}/5")
        print(
            f"Config: Distill={self.args.use_distill}, Mapper={self.args.use_mapper}, GNN_TCN={self.args.use_gnn_tcn}")
        print('=' * 60)

        X_train, X_val = self.X[train_idx], self.X[val_idx]
        y_train, y_val = self.y[train_idx], self.y[val_idx]

        train_dataset = DistillEEGDataset(X_train, y_train, self.subjects[train_idx],
                                          augmentations=self.args.augmentations, is_training=True)
        val_dataset = DistillEEGDataset(X_val, y_val, self.subjects[val_idx], is_training=False)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)

        model = SZAttNetDistill(
            eeg_channels=self.eeg_channels, time_points=self.time_points,
            fmri_regions=self.fmri_regions, hidden_dim=self.args.hidden_dim,
            use_distill=self.args.use_distill, use_mapper=self.args.use_mapper,
            use_gnn_tcn=self.args.use_gnn_tcn
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        ce_criterion = nn.CrossEntropyLoss()
        distill_criterion = DistillationLoss(temperature=self.args.temperature, alpha=self.args.alpha,
                                             beta=self.args.beta,
                                             gamma=self.args.gamma) if self.args.use_distill else None

        best_val_acc = 0
        best_state = None

        for epoch in range(self.args.epochs):
            model.train()
            total_loss = 0
            train_preds, train_labels = [], []

            for batch in train_loader:
                freq_signal = batch['freq_signal'].to(device)
                labels = batch['label'].squeeze().to(device)

                optimizer.zero_grad()

                if self.args.use_distill:
                    logits, aligned_feat = model(freq_signal, return_features=True)
                    loss, _, _, _, _ = distill_criterion(logits, aligned_feat, labels,
                                                         self.fmri_prototypes, self.fmri_probs)
                else:
                    logits = model(freq_signal)
                    loss = ce_criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(train_labels, train_preds)

            # Validation
            model.eval()
            val_preds, val_probs, val_labels = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    freq_signal = batch['freq_signal'].to(device)
                    labels = batch['label'].squeeze().to(device)
                    logits = model(freq_signal)
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_probs.extend(probs[:, 1].cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            acc, f1, auc, sen, spe = self.calculate_metrics(val_labels, val_preds, val_probs)
            scheduler.step(acc)

            self.history['train_loss'].append(total_loss / len(train_loader))
            self.history['val_acc'].append(acc)
            self.history['val_f1'].append(f1)
            self.history['val_auc'].append(auc)
            self.history['val_sen'].append(sen)
            self.history['val_spe'].append(spe)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}/{self.args.epochs}: "
                      f"Loss={total_loss / len(train_loader):.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, SEN={sen:.4f}, SPE={spe:.4f}")

            if acc > best_val_acc:
                best_val_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Fold {fold} Best Val Acc: {best_val_acc:.4f}")
        return best_val_acc, best_state

    def train(self):
        unique_subjects = np.unique(self.subjects)
        subject_labels = np.array([self.y[self.subjects == sub][0] for sub in unique_subjects])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        fold_accs, fold_f1s, fold_aucs, fold_sens, fold_spes = [], [], [], [], []
        all_best_states = []

        for fold, (train_subj_idx, val_subj_idx) in enumerate(skf.split(unique_subjects, subject_labels)):
            train_subjs = unique_subjects[train_subj_idx]
            val_subjs = unique_subjects[val_subj_idx]
            train_idx = np.isin(self.subjects, train_subjs)
            val_idx = np.isin(self.subjects, val_subjs)

            best_acc, best_state = self.train_fold(train_idx, val_idx, fold + 1)
            all_best_states.append(best_state)

            # Evaluate final metrics on this fold's val set
            val_dataset = DistillEEGDataset(self.X[val_idx], self.y[val_idx], self.subjects[val_idx], is_training=False)
            val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)

            model = SZAttNetDistill(
                eeg_channels=self.eeg_channels, time_points=self.time_points,
                fmri_regions=self.fmri_regions, hidden_dim=self.args.hidden_dim,
                use_distill=self.args.use_distill, use_mapper=self.args.use_mapper,
                use_gnn_tcn=self.args.use_gnn_tcn
            ).to(device)
            model.load_state_dict(best_state)
            model.eval()

            preds, probs, labels = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    freq_signal = batch['freq_signal'].to(device)
                    labs = batch['label'].squeeze().to(device)
                    logits = model(freq_signal)
                    prob = F.softmax(logits, dim=1)
                    pred = torch.argmax(logits, dim=1)
                    preds.extend(pred.cpu().numpy())
                    probs.extend(prob[:, 1].cpu().numpy())
                    labels.extend(labs.cpu().numpy())

            acc, f1, auc, sen, spe = self.calculate_metrics(labels, preds, probs)
            fold_accs.append(acc)
            fold_f1s.append(f1)
            fold_aucs.append(auc)
            fold_sens.append(sen)
            fold_spes.append(spe)

            print(f"Fold {fold + 1} Final - Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, SEN={sen:.4f}, SPE={spe:.4f}")

        # 打印最终结果
        print("\n" + "=" * 60)
        print("5-Fold Cross Validation Results")
        print(
            f"Config: Distill={self.args.use_distill}, Mapper={self.args.use_mapper}, GNN_TCN={self.args.use_gnn_tcn}")
        print("=" * 60)
        print(f"Accuracy:    {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        print(f"F1-Score:    {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        print(f"AUC:         {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"Sensitivity: {np.mean(fold_sens):.4f} ± {np.std(fold_sens):.4f}")
        print(f"Specificity: {np.mean(fold_spes):.4f} ± {np.std(fold_spes):.4f}")

        return fold_accs, fold_f1s, fold_aucs, fold_sens, fold_spes


# ==================== 消融实验运行器 ====================
def run_ablation(args):
    configs = [
        #{'use_distill': False, 'use_mapper': False, 'use_gnn_tcn': False, 'name': 'Baseline (None)'},
        #{'use_distill': True,  'use_mapper': False, 'use_gnn_tcn': False, 'name': 'Distill Only'},
        #{'use_distill': False, 'use_mapper': True,  'use_gnn_tcn': False, 'name': 'Mapper Only'},
        #{'use_distill': False, 'use_mapper': False, 'use_gnn_tcn': True,  'name': 'BCFE+CFFE Only'},
        #{'use_distill': True,  'use_mapper': True,  'use_gnn_tcn': False, 'name': 'Distill + Mapper'},
        #{'use_distill': True,  'use_mapper': False, 'use_gnn_tcn': True,  'name': 'Distill + BCFE+CFFE'},
        #{'use_distill': False, 'use_mapper': True,  'use_gnn_tcn': True,  'name': 'Mapper + BCFE+CFFE'},
        {'use_distill': False,  'use_mapper': True,  'use_gnn_tcn': True,  'name': 'Full Model'},
    ]
    results = []
    for cfg in configs:
        print("\n" + "="*60)
        print(f"Running Ablation: {cfg['name']}")
        print("="*60)
        # 更新args
        args.use_distill = cfg['use_distill']
        args.use_mapper = cfg['use_mapper']
        args.use_gnn_tcn = cfg['use_gnn_tcn']
        trainer = SZAttNetTrainer(args)
        trainer.train()
        # 由于train()内部已经打印了5-fold结果，这里不重复计算，但可以记录最终平均值
        # 实际实现中可以将结果返回，这里简化
        results.append({'config': cfg['name'], 'trainer': trainer})
    # 汇总对比
    print("\n" + "="*60)
    print("Ablation Study Summary")
    print("="*60)
    for res in results:
        print(f"{res['config']}: (see above for detailed metrics)")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='SZAttNet with fMRI-EEG Distillation & Ablation')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--teacher_epochs', type=int, default=30)
    parser.add_argument('--use_distill', action='store_true', default=True)
    parser.add_argument('--use_mapper', action='store_true', default=True)
    parser.add_argument('--use_gnn_tcn', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--augmentations', nargs='+', default=['time_shift', 'gaussian_noise'])
    parser.add_argument('--ablation', action='store_true', help='Run ablation study over all modules')
    args = parser.parse_args()

    if args.ablation:
        run_ablation(args)
    else:
        trainer = SZAttNetTrainer(args)
        trainer.train()


if __name__ == "__main__":
    main()