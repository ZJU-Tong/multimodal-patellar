import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import argparse
from collections import defaultdict
import json
from datetime import datetime
import random
import sys
sys.path.append('..')
from model import MODEL_DICT
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重现性
def set_seed(seed=42):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class DemographicDataset(Dataset):
    """人口统计学数据集（性别+年龄）"""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 处理性别：男=1, 女=0
        gender = sample.get('gender', '')
        gender_encoded = 1.0 if gender == '男' else 0.0
        
        # 处理年龄：标准化到0-1范围
        age = sample.get('age', 0)
        try:
            age_float = float(age) if age != '' else 0.0
            # 假设年龄范围在10-80岁之间，进行标准化
            age_normalized = (age_float - 10.0) / 70.0
            age_normalized = max(0.0, min(1.0, age_normalized))  # 限制在0-1范围
        except (ValueError, TypeError):
            age_normalized = 0.0
        
        # 组合特征向量
        features = torch.tensor([gender_encoded, age_normalized], dtype=torch.float32)
        label = sample['label']
        patient_id = sample['patient_id']
        
        return features, label, patient_id

class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 优先使用tokenized_text（英文分词后文本），否则回退到text
        text = sample.get('text_tokenized', None)
        if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() == '':
            text = sample.get('text', '')
        label = sample['label']
        patient_id = sample['patient_id']
        
        # 文本编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label,
            'patient_id': patient_id
        }

class BinaryFocalLossWithLogits(nn.Module):
    """二分类Focal Loss（logits版）
    loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    其中 p_t = sigmoid(logit) 若 y=1，否则 1 - sigmoid(logit)
    alpha_t = alpha 若 y=1，否则 1 - alpha
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N,), targets: (N,) in {0,1}
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating = (1 - p_t) ** self.gamma
        loss = alpha_t * modulating * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class CTDataset(Dataset):
    """CT图像数据集"""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            # 返回一个默认的黑色图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample['patient_id']

class DemographicClassifier(nn.Module):
    """基于人口统计学特征的简化MLP分类器"""
    def __init__(self, input_dim=2, hidden_dim=16, num_classes=1, dropout=0.2):
        super(DemographicClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # 移除BatchNorm，对于小数据集可能不稳定
    
    def forward(self, x):
        # 第一层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        return x

class TextClassifier(nn.Module):
    """基于BERT的文本分类器"""
    def __init__(self, num_classes=1, dropout=0.3):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 冻结BERT的前几层，只训练最后几层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[:8].parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

def get_transforms(is_training=True):
    """获取图像变换"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_multimodal_data(data_dir):
    """加载多模态数据"""
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"找不到metadata.csv文件: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    samples = []
    
    for _, row in df.iterrows():
        patient_id = str(row['patient_id'])
        label = int(row['label'])
        # 读取tokenized_text优先（英文版分词），无则回退text
        if 'tokenized_text' in df.columns and not pd.isna(row['tokenized_text']):
            text = str(row['tokenized_text'])
        else:
            text = str(row['text']) if 'text' in df.columns and not pd.isna(row['text']) else ''
        
        # 读取性别和年龄信息
        gender = str(row['gender']) if 'gender' in df.columns and not pd.isna(row['gender']) else ''
        age = str(row['age']) if 'age' in df.columns and not pd.isna(row['age']) else ''
        
        # 解析图像路径
        images_str = row['images']
        if pd.isna(images_str):
            continue
            
        image_paths = images_str.split(';')
        for img_path in image_paths:
            img_path = img_path.strip()
            if img_path and os.path.exists(os.path.join(data_dir, '..', img_path)):
                samples.append({
                    'patient_id': patient_id,
                    'label': label,
                    'text': text,
                    'tokenized_text': text,
                    'gender': gender,
                    'age': age,
                    'path': os.path.join(data_dir, '..', img_path)
                })
    
    return samples

def split_data_by_patient(samples, test_size=0.2, random_state=42):
    """按病人ID划分数据"""
    # 按病人ID分组
    patient_data = defaultdict(list)
    for sample in samples:
        patient_data[sample['patient_id']].append(sample)
    
    # 获取所有病人ID和对应的标签
    patient_ids = list(patient_data.keys())
    patient_labels = []
    
    for pid in patient_ids:
        # 同一病人的所有图像标签应该相同，取第一个
        patient_labels.append(patient_data[pid][0]['label'])
    
    # 按病人ID进行分层划分
    train_pids, val_pids = train_test_split(
        patient_ids, 
        test_size=test_size, 
        stratify=patient_labels,
        random_state=random_state
    )
    
    # 根据病人ID划分样本
    train_samples = []
    val_samples = []
    
    for pid in train_pids:
        train_samples.extend(patient_data[pid])
    
    for pid in val_pids:
        val_samples.extend(patient_data[pid])
    
    return train_samples, val_samples

def get_kfold_splits(samples, n_splits=5, random_state=42):
    """按病人ID进行K折交叉验证划分"""
    # 按病人ID分组
    patient_data = defaultdict(list)
    for sample in samples:
        patient_data[sample['patient_id']].append(sample)
    
    # 获取所有病人ID和对应的标签
    patient_ids = list(patient_data.keys())
    patient_labels = []
    
    for pid in patient_ids:
        # 同一病人的所有图像标签应该相同，取第一个
        patient_labels.append(patient_data[pid][0]['label'])
    
    # 创建分层K折
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_splits = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids = [patient_ids[i] for i in val_idx]
        
        # 根据病人ID划分样本
        train_samples = []
        val_samples = []
        
        for pid in train_pids:
            train_samples.extend(patient_data[pid])
        
        for pid in val_pids:
            val_samples.extend(patient_data[pid])
        
        fold_splits.append((train_samples, val_samples))
    
    return fold_splits

def calculate_metrics(y_true, y_pred, y_prob):
    """计算各种评估指标"""
    # 计算正负样本的准确率
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_p = accuracy_score(y_true[pos_mask], y_pred[pos_mask]) if pos_mask.sum() > 0 else 0
    accuracy_n = accuracy_score(y_true[neg_mask], y_pred[neg_mask]) if neg_mask.sum() > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    
    return {
        'accuracy': accuracy,
        'accuracy_p': accuracy_p,
        'accuracy_n': accuracy_n,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

def patient_level_voting(predictions, labels, patient_ids):
    """对同一病人的多张CT图像进行投票"""
    patient_data = defaultdict(lambda: {'predictions': [], 'labels': []})
    
    # 按病人ID分组，同时收集预测和标签
    for pred, label, pid in zip(predictions, labels, patient_ids):
        patient_data[pid]['predictions'].append(pred)
        patient_data[pid]['labels'].append(label)
    
    # 对每个病人进行投票
    final_predictions = []
    final_labels = []
    final_patient_ids = []
    
    for pid, data in patient_data.items():
        # 平均投票
        avg_prediction = np.mean(data['predictions'])
        # 标签应该都相同，取第一个
        patient_label = data['labels'][0]
        
        final_predictions.append(avg_prediction)
        final_labels.append(patient_label)
        final_patient_ids.append(pid)
    
    return np.array(final_predictions), np.array(final_labels), final_patient_ids

def train_demographic_model(train_samples, val_samples, epochs=30, batch_size=64, lr=0.01, patience=10, device='cuda'):
    """训练人口统计学模型"""
    print(f"\n{'='*60}")
    print("开始训练人口统计学模型")
    print(f"{'='*60}")
    
    # 创建数据集
    train_dataset = DemographicDataset(train_samples)
    val_dataset = DemographicDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 统计信息
    train_patients = len(set([s['patient_id'] for s in train_samples]))
    val_patients = len(set([s['patient_id'] for s in val_samples]))
    train_pos = sum([1 for s in train_samples if s['label'] == 1])
    train_neg = len(train_samples) - train_pos
    val_pos = sum([1 for s in val_samples if s['label'] == 1])
    val_neg = len(val_samples) - val_pos
    
    print(f"训练集: {len(train_dataset)}个样本, {train_patients}个病人 (阳性:{train_pos}, 阴性:{train_neg})")
    print(f"验证集: {len(val_dataset)}个样本, {val_patients}个病人 (阳性:{val_pos}, 阴性:{val_neg})")
    
    # 创建简化模型
    model = DemographicClassifier(input_dim=2, hidden_dim=16, dropout=0.2)
    model = model.to(device)
    
    # 计算类别权重 (pos_weight)
    pos_weight = torch.tensor([train_neg / train_pos]) if train_pos > 0 else torch.tensor([1.0])
    pos_weight = pos_weight.to(device)

    # 损失函数和优化器 - 使用更简单的配置
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)  # 增加权重衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
    
    # 早停策略
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    print("开始训练人口统计学模型...")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        all_predictions = []
        all_labels = []
        all_patient_ids = []
        
        for features, labels, patient_ids in train_loader:
            features = features.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 收集预测结果
            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            all_predictions.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_patient_ids.extend(patient_ids)
        
        # 病人级别投票
        patient_probs, patient_labels, patient_ids = patient_level_voting(all_predictions, all_labels, all_patient_ids)
        patient_preds = (patient_probs > 0.5).astype(int)
        train_metrics = calculate_metrics(patient_labels, patient_preds, patient_probs)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        all_patient_ids = []
        
        with torch.no_grad():
            for features, labels, patient_ids in val_loader:
                features = features.to(device)
                labels = labels.float().to(device)
                
                outputs = model(features).squeeze(-1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # 收集预测结果
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                all_patient_ids.extend(patient_ids)
        
        # 病人级别投票
        patient_probs, patient_labels, patient_ids = patient_level_voting(all_predictions, all_labels, all_patient_ids)
        patient_preds = (patient_probs > 0.5).astype(int)
        val_metrics = calculate_metrics(patient_labels, patient_preds, patient_probs)
        
        # 更新学习率
        scheduler.step(val_metrics['auc'])
        
        # 打印结果
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        print("-" * 50)
        
        # 早停检查
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停触发！最佳AUC: {best_auc:.4f}")
            break
    
    print(f"人口统计学模型训练完成！最佳AUC: {best_auc:.4f}")
    
    return best_model_state, best_auc, val_metrics

def train_text_model(train_samples, val_samples, epochs=20, batch_size=16, lr=2e-5, patience=10, device='cuda'):
    """训练文本模型"""
    print(f"\n{'='*60}")
    print("开始训练文本模型")
    print(f"{'='*60}")
    
    # 初始化英文BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    
    # 创建数据集
    train_dataset = TextDataset(train_samples, tokenizer)
    val_dataset = TextDataset(val_samples, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 统计信息
    train_patients = len(set([s['patient_id'] for s in train_samples]))
    val_patients = len(set([s['patient_id'] for s in val_samples]))
    train_pos = sum([1 for s in train_samples if s['label'] == 1])
    train_neg = len(train_samples) - train_pos
    val_pos = sum([1 for s in val_samples if s['label'] == 1])
    val_neg = len(val_samples) - val_pos
    
    print(f"训练集: {len(train_dataset)}个样本, {train_patients}个病人 (阳性:{train_pos}, 阴性:{train_neg})")
    print(f"验证集: {len(val_dataset)}个样本, {val_patients}个病人 (阳性:{val_pos}, 阴性:{val_neg})")
    
    # 创建模型
    model = TextClassifier()
    model = model.to(device)
    
    # 计算类别权重 (pos_weight)
    pos_weight = torch.tensor([train_neg / train_pos]) if train_pos > 0 else torch.tensor([1.0])
    pos_weight = pos_weight.to(device)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 早停策略
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    print("开始训练文本模型...")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        all_predictions = []
        all_labels = []
        all_patient_ids = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)
            patient_ids = batch['patient_id']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).squeeze(-1)  # 只压缩最后一个维度
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 收集预测结果
            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            all_predictions.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_patient_ids.extend(patient_ids)
        
        # 病人级别投票
        patient_probs, patient_labels, patient_ids = patient_level_voting(all_predictions, all_labels, all_patient_ids)
        patient_preds = (patient_probs > 0.5).astype(int)
        train_metrics = calculate_metrics(patient_labels, patient_preds, patient_probs)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].float().to(device)
                patient_ids = batch['patient_id']
                
                outputs = model(input_ids, attention_mask).squeeze(-1)  # 只压缩最后一个维度
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # 收集预测结果
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                all_patient_ids.extend(patient_ids)
        
        # 病人级别投票
        patient_probs, patient_labels, patient_ids = patient_level_voting(all_predictions, all_labels, all_patient_ids)
        patient_preds = (patient_probs > 0.5).astype(int)
        val_metrics = calculate_metrics(patient_labels, patient_preds, patient_probs)
        
        # 更新学习率
        scheduler.step(val_metrics['auc'])
        
        # 打印结果
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        print("-" * 50)
        
        # 早停检查
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停触发！最佳AUC: {best_auc:.4f}")
            break
    
    print(f"文本模型训练完成！最佳AUC: {best_auc:.4f}")
    
    return best_model_state, best_auc, val_metrics

def train_image_model(model_name, train_samples, val_samples, epochs=30, batch_size=32, lr=0.0001, patience=10, device='cuda'):
    """训练图像模型"""
    print(f"\n{'='*60}")
    print(f"开始训练图像模型: {model_name}")
    print(f"{'='*60}")
    
    # 创建数据集
    train_dataset = CTDataset(train_samples, transform=get_transforms(True))
    val_dataset = CTDataset(val_samples, transform=get_transforms(False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 统计信息
    train_patients = len(set([s['patient_id'] for s in train_samples]))
    val_patients = len(set([s['patient_id'] for s in val_samples]))
    train_pos = sum([1 for s in train_samples if s['label'] == 1])
    train_neg = len(train_samples) - train_pos
    val_pos = sum([1 for s in val_samples if s['label'] == 1])
    val_neg = len(val_samples) - val_pos
    
    print(f"训练集: {len(train_dataset)}张图像, {train_patients}个病人 (阳性:{train_pos}, 阴性:{train_neg})")
    print(f"验证集: {len(val_dataset)}张图像, {val_patients}个病人 (阳性:{val_pos}, 阴性:{val_neg})")
    
    # 创建模型
    model = MODEL_DICT[model_name]()
    model = model.to(device)
    
    # 计算类别权重 (pos_weight)
    pos_weight = torch.tensor([train_neg / train_pos]) if train_pos > 0 else torch.tensor([1.0])
    pos_weight = pos_weight.to(device)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 早停策略
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    print(f"开始训练 {model_name} 模型...")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        all_predictions = []
        all_labels = []
        all_patient_ids = []
        
        for images, labels, patient_ids in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 收集预测结果
            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            all_predictions.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_patient_ids.extend(patient_ids)
        
        # 病人级别投票
        patient_probs, patient_labels, patient_ids = patient_level_voting(all_predictions, all_labels, all_patient_ids)
        patient_preds = (patient_probs > 0.5).astype(int)
        train_metrics = calculate_metrics(patient_labels, patient_preds, patient_probs)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        all_patient_ids = []
        
        with torch.no_grad():
            for images, labels, patient_ids in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # 收集预测结果
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                all_patient_ids.extend(patient_ids)
        
        # 病人级别投票
        patient_probs, patient_labels, patient_ids = patient_level_voting(all_predictions, all_labels, all_patient_ids)
        patient_preds = (patient_probs > 0.5).astype(int)
        val_metrics = calculate_metrics(patient_labels, patient_preds, patient_probs)
        
        # 更新学习率
        scheduler.step(val_metrics['auc'])
        
        # 打印结果
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        print("-" * 50)
        
        # 早停检查
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停触发！最佳AUC: {best_auc:.4f}")
            break
    
    print(f"{model_name} 图像模型训练完成！最佳AUC: {best_auc:.4f}")
    
    return best_model_state, best_auc, val_metrics

def train_late_fusion_models(model_name, train_dir, epochs=30, batch_size=32, lr=0.0001, patience=10, n_folds=5, training_timestamp=None, seed=42, train_mode='both', external_mode='separate', external_dir='data/外部验证集', external_ratio=0.2):
    """训练晚期融合模型"""
    # 设置随机种子
    set_seed(seed)
    print(f"设置随机种子: {seed}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载多模态数据
    print("加载多模态数据...")
    base_samples = load_multimodal_data(train_dir)
    print(f"训练集样本数: {len(base_samples)}")

    # 处理外部测试集模式
    external_info = {
        'mode': external_mode,
        'external_dir': external_dir,
        'external_ratio': external_ratio,
        'external_patients': 0,
        'external_pos_images': 0,
        'external_neg_images': 0,
        'mixed_manifest': None
    }

    external_samples = None
    if external_mode == 'mixed':
        # 合并训练集与外部验证集，然后从中划分外部测试集（按病人分层）
        print(f"外部模式: mixed，加载外部验证集用于混合: {external_dir}")
        ext_samples = load_multimodal_data(external_dir)
        print(f"外部验证集样本数: {len(ext_samples)}")
        combined = base_samples + ext_samples

        # 按病人分层划分 external test
        patient_data = defaultdict(list)
        for s in combined:
            patient_data[s['patient_id']].append(s)
        patient_ids = list(patient_data.keys())
        patient_labels = [patient_data[pid][0]['label'] for pid in patient_ids]

        test_pids, train_pids = train_test_split(
            patient_ids,
            test_size=1 - external_ratio,
            stratify=patient_labels,
            random_state=seed
        )

        # 统计外部测试集信息
        external_samples = []
        train_pool_samples = []
        for pid in patient_ids:
            if pid in test_pids:
                external_samples.extend(patient_data[pid])
            else:
                train_pool_samples.extend(patient_data[pid])

        external_info['external_patients'] = len(set([s['patient_id'] for s in external_samples]))
        external_info['external_pos_images'] = sum(1 for s in external_samples if s['label'] == 1)
        external_info['external_neg_images'] = sum(1 for s in external_samples if s['label'] == 0)

        print(f"混合后外部测试集: {len(external_samples)} 张图像, {external_info['external_patients']} 个病人 (阳性:{external_info['external_pos_images']}, 阴性:{external_info['external_neg_images']})")
        all_samples = train_pool_samples
        print(f"用于K折训练的样本数: {len(all_samples)}")
    elif external_mode == 'partial':
        # 外部验证集按比例加入训练，其余保留为外部测试
        print(f"外部模式: partial，将外部验证集按比例加入训练，其余保留为外部测试: {external_dir}")
        ext_samples = load_multimodal_data(external_dir)
        print(f"外部验证集样本数: {len(ext_samples)}")

        patient_data = defaultdict(list)
        for s in ext_samples:
            patient_data[s['patient_id']].append(s)
        patient_ids = list(patient_data.keys())
        patient_labels = [patient_data[pid][0]['label'] for pid in patient_ids]

        ext_train_pids, ext_test_pids = train_test_split(
            patient_ids,
            test_size=max(0.0, min(1.0, 1 - external_ratio)),
            stratify=patient_labels if len(set(patient_labels)) > 1 else None,
            random_state=seed
        )

        ext_train_samples = []
        external_samples = []
        for pid in patient_ids:
            if pid in ext_train_pids:
                ext_train_samples.extend(patient_data[pid])
            else:
                external_samples.extend(patient_data[pid])

        external_info['external_patients'] = len(set([s['patient_id'] for s in external_samples]))
        external_info['external_pos_images'] = sum(1 for s in external_samples if s['label'] == 1)
        external_info['external_neg_images'] = sum(1 for s in external_samples if s['label'] == 0)
        external_info['mixed_into_train_patients'] = len(set([s['patient_id'] for s in ext_train_samples]))
        external_info['mixed_into_train_pos_images'] = sum(1 for s in ext_train_samples if s['label'] == 1)
        external_info['mixed_into_train_neg_images'] = sum(1 for s in ext_train_samples if s['label'] == 0)

        print(f"外部测试保留: {len(external_samples)} 张图像, {external_info['external_patients']} 个病人 (阳性:{external_info['external_pos_images']}, 阴性:{external_info['external_neg_images']})")
        print(f"外部集混入训练: {len(ext_train_samples)} 张图像, {external_info['mixed_into_train_patients']} 个病人 (阳性:{external_info['mixed_into_train_pos_images']}, 阴性:{external_info['mixed_into_train_neg_images']})")

        all_samples = base_samples + ext_train_samples
        print(f"用于K折训练的样本数: {len(all_samples)}")
    else:
        # 使用原有训练集进行K折，外部验证集在测试脚本中单独评估
        print("外部模式: separate，训练仅使用训练集；外部验证在测试阶段进行")
        all_samples = base_samples
    
    # 获取K折划分
    print(f"按病人ID进行{n_folds}折交叉验证划分...")
    fold_splits = get_kfold_splits(all_samples, n_splits=n_folds)
    
    # 训练所有折
    all_fold_results = []
    best_fold_auc = 0.0
    best_fold_idx = 0
    
    for fold, (train_samples, val_samples) in enumerate(fold_splits):
        print(f"\n{'='*80}")
        print(f"开始训练 Fold {fold+1}")
        print(f"{'='*80}")
        
        # 训练人口统计学模型（可选）
        demographic_model_state, demographic_auc, demographic_metrics = None, 0.0, None
        if train_mode in ['both', 'demographic']:
            demographic_model_state, demographic_auc, demographic_metrics = train_demographic_model(
                train_samples, val_samples, 
                epochs=30, batch_size=64, lr=0.01, patience=10, device=device
            )
        
        # 训练文本模型（可选）
        text_model_state, text_auc, text_metrics = None, 0.0, None
        if train_mode in ['both', 'text']:
            text_model_state, text_auc, text_metrics = train_text_model(
                train_samples, val_samples, 
                epochs=20, batch_size=16, lr=2e-5, patience=10, device=device
            )
        
        # 训练图像模型（可选）
        image_model_state, image_auc, image_metrics = None, 0.0, None
        if train_mode in ['both', 'image']:
            image_model_state, image_auc, image_metrics = train_image_model(
                model_name, train_samples, val_samples,
                epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, device=device
            )
        
        # 保存折结果
        fold_result = {
            'fold': fold,
            'demographic_model_state': demographic_model_state,
            'text_model_state': text_model_state,
            'image_model_state': image_model_state,
            'demographic_auc': float(demographic_auc) if demographic_auc is not None else None,
            'text_auc': float(text_auc) if text_auc is not None else None,
            'image_auc': float(image_auc) if image_auc is not None else None,
            'demographic_metrics': ({k: float(v) for k, v in demographic_metrics.items()} if demographic_metrics is not None else None),
            'text_metrics': ({k: float(v) for k, v in text_metrics.items()} if text_metrics is not None else None),
            'image_metrics': ({k: float(v) for k, v in image_metrics.items()} if image_metrics is not None else None)
        }
        
        all_fold_results.append(fold_result)
        
        # 记录最佳折（依据所训练的模型）
        if train_mode == 'demographic':
            select_auc = demographic_auc
        elif train_mode == 'text':
            select_auc = text_auc
        elif train_mode == 'image':
            select_auc = image_auc
        else:  # both模式，选择最高的AUC
            select_auc = max([auc for auc in [demographic_auc, text_auc, image_auc] if auc is not None and auc > 0])
        
        if select_auc > best_fold_auc:
            best_fold_auc = select_auc
            best_fold_idx = fold
    
    # 保存结果
    if training_timestamp is None:
        training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = os.path.join('late_fusion_results', training_timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存每一折的结果
    for i, fold_result in enumerate(all_fold_results):
        fold_path = os.path.join(results_dir, f'late_fusion_fold_{i+1}_results.json')
        text_metrics_serialized = ({k: float(v) for k, v in fold_result['text_metrics'].items()}
                                   if fold_result['text_metrics'] is not None else None)
        image_metrics_serialized = ({k: float(v) for k, v in fold_result['image_metrics'].items()}
                                    if fold_result['image_metrics'] is not None else None)
        fold_data = {
            'model_name': model_name,
            'fold': i + 1,
            'training_timestamp': training_timestamp,
            'text_auc': fold_result['text_auc'],
            'image_auc': fold_result['image_auc'],
            'text_metrics': text_metrics_serialized,
            'image_metrics': image_metrics_serialized,
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'patience': patience,
                'n_folds': n_folds,
                'train_mode': train_mode,
                'external_mode': external_mode,
                'external_dir': external_dir,
                'external_ratio': external_ratio
            }
        }
        
        with open(fold_path, 'w', encoding='utf-8') as f:
            json.dump(fold_data, f, indent=2, ensure_ascii=False)

    # 若为mixed模式，保存外部测试集清单
    if external_mode == 'mixed' and external_samples is not None:
        manifest_path = os.path.join(results_dir, 'mixed_external_manifest.json')
        # 仅保留必要字段
        serializable = []
        for s in external_samples:
            serializable.append({
                'patient_id': s['patient_id'],
                'label': int(s['label']),
                'path': s['path'],
                'text': s.get('text', ''),
                'tokenized_text': s.get('tokenized_text', s.get('text', ''))
            })
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({'samples': serializable}, f, indent=2, ensure_ascii=False)
        external_info['mixed_manifest'] = manifest_path
    
    # 保存最佳模型（依据训练模式）
    os.makedirs('saved_models', exist_ok=True)
    best_demographic_model_path = None
    best_text_model_path = None
    best_image_model_path = None
    
    if train_mode in ['both', 'demographic']:
        best_demographic_model_path = os.path.join('saved_models', f'best_demographic_model_{model_name}.pth')
        torch.save({
            'model_state_dict': all_fold_results[best_fold_idx]['demographic_model_state'],
            'model_type': 'demographic',
            'fold': best_fold_idx + 1,
            'best_auc': all_fold_results[best_fold_idx]['demographic_auc'],
            'metrics': ({k: float(v) for k, v in all_fold_results[best_fold_idx]['demographic_metrics'].items()}
                        if all_fold_results[best_fold_idx]['demographic_metrics'] is not None else None)
        }, best_demographic_model_path)
    
    if train_mode in ['both', 'text']:
        best_text_model_path = os.path.join('saved_models', f'best_text_model_{model_name}.pth')
        torch.save({
            'model_state_dict': all_fold_results[best_fold_idx]['text_model_state'],
            'model_type': 'text',
            'fold': best_fold_idx + 1,
            'best_auc': all_fold_results[best_fold_idx]['text_auc'],
            'metrics': ({k: float(v) for k, v in all_fold_results[best_fold_idx]['text_metrics'].items()}
                        if all_fold_results[best_fold_idx]['text_metrics'] is not None else None)
        }, best_text_model_path)
    if train_mode in ['both', 'image']:
        best_image_model_path = os.path.join('saved_models', f'best_image_model_{model_name}.pth')
        torch.save({
            'model_state_dict': all_fold_results[best_fold_idx]['image_model_state'],
            'model_type': 'image',
            'model_name': model_name,
            'fold': best_fold_idx + 1,
            'best_auc': all_fold_results[best_fold_idx]['image_auc'],
            'metrics': ({k: float(v) for k, v in all_fold_results[best_fold_idx]['image_metrics'].items()}
                        if all_fold_results[best_fold_idx]['image_metrics'] is not None else None)
        }, best_image_model_path)
    
    # 计算统计结果
    demographic_aucs = [fold['demographic_auc'] for fold in all_fold_results if fold['demographic_auc'] is not None]
    text_aucs = [fold['text_auc'] for fold in all_fold_results if fold['text_auc'] is not None]
    image_aucs = [fold['image_auc'] for fold in all_fold_results if fold['image_auc'] is not None]
    
    final_results = {
        'model_name': model_name,
        'training_timestamp': training_timestamp,
        'n_folds': n_folds,
        'best_fold': best_fold_idx + 1,
        'train_mode': train_mode,
        'external_info': external_info,
        'demographic_auc_mean': float(np.mean(demographic_aucs)) if len(demographic_aucs) > 0 else None,
        'demographic_auc_std': float(np.std(demographic_aucs)) if len(demographic_aucs) > 0 else None,
        'text_auc_mean': float(np.mean(text_aucs)) if len(text_aucs) > 0 else None,
        'text_auc_std': float(np.std(text_aucs)) if len(text_aucs) > 0 else None,
        'image_auc_mean': float(np.mean(image_aucs)) if len(image_aucs) > 0 else None,
        'image_auc_std': float(np.std(image_aucs)) if len(image_aucs) > 0 else None,
        'best_demographic_auc': float(max(demographic_aucs)) if len(demographic_aucs) > 0 else None,
        'best_text_auc': float(max(text_aucs)) if len(text_aucs) > 0 else None,
        'best_image_auc': float(max(image_aucs)) if len(image_aucs) > 0 else None,
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'patience': patience,
            'n_folds': n_folds,
            'train_mode': train_mode,
            'external_mode': external_mode,
            'external_dir': external_dir,
            'external_ratio': external_ratio
        }
    }
    
    final_path = os.path.join(results_dir, f'late_fusion_cross_validation_results.json')
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印结果总结
    print(f"\n{'='*80}")
    print(f"晚期融合模型训练完成")
    print(f"{'='*80}")
    print(f"最佳折: Fold {best_fold_idx + 1}")
    if len(demographic_aucs) > 0:
        print(f"人口统计学模型 - 平均AUC: {np.mean(demographic_aucs):.4f} ± {np.std(demographic_aucs):.4f}, 最佳AUC: {max(demographic_aucs):.4f}")
    if len(text_aucs) > 0:
        print(f"文本模型 - 平均AUC: {np.mean(text_aucs):.4f} ± {np.std(text_aucs):.4f}, 最佳AUC: {max(text_aucs):.4f}")
    if len(image_aucs) > 0:
        print(f"图像模型 - 平均AUC: {np.mean(image_aucs):.4f} ± {np.std(image_aucs):.4f}, 最佳AUC: {max(image_aucs):.4f}")
    if best_demographic_model_path:
        print(f"最佳人口统计学模型保存到: {best_demographic_model_path}")
    if best_text_model_path:
        print(f"最佳文本模型保存到: {best_text_model_path}")
    if best_image_model_path:
        print(f"最佳图像模型保存到: {best_image_model_path}")
    print(f"详细结果保存到: {final_path}")
    
    return final_results, all_fold_results

def main():
    parser = argparse.ArgumentParser(description='训练晚期融合模型（图像+文本）')
    parser.add_argument('--model', type=str, default='efficientnet_b0', 
                       choices=list(MODEL_DICT.keys()),
                       help='选择图像模型')
    parser.add_argument('--train_dir', type=str, default='data/训练集',
                       help='训练数据目录')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--train_mode', type=str, default='both', choices=['both', 'image', 'text', 'demographic'],
                       help='训练模式：both=同时训练，image=仅图像，text=仅文本，demographic=仅人口统计学')
    parser.add_argument('--external_mode', type=str, default='separate', choices=['separate', 'mixed', 'partial'],
                       help='外部测试集使用模式：separate=单独外部集，mixed=与训练集合并后再划分外部测试，partial=外部集按比例加入训练，其余保留为外部测试')
    parser.add_argument('--external_dir', type=str, default='data/外部验证集',
                       help='外部验证集目录（当external_mode=mixed时会被加载）')
    parser.add_argument('--external_ratio', type=float, default=0.25,
                       help='mixed模式下从合并数据中划分为外部测试集的比例（按病人）')
    
    args = parser.parse_args()
    
    # 训练模型
    final_results, all_fold_results = train_late_fusion_models(
        model_name=args.model,
        train_dir=args.train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        n_folds=args.n_folds,
        seed=args.seed,
        train_mode=args.train_mode,
        external_mode=args.external_mode,
        external_dir=args.external_dir,
        external_ratio=args.external_ratio
    )

if __name__ == '__main__':
    main()
