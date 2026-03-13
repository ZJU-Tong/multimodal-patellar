import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
        text = sample.get('tokenized_text', None)
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
    def __init__(self, num_classes=1, dropout=0.3, bert_model='./bert-base-uncased'):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
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

def select_bert_model_for_checkpoint(checkpoint):
    """根据checkpoint选择与之匹配的BERT模型/词表"""
    if isinstance(checkpoint, dict):
        bert_model = checkpoint.get('bert_model', None)
        if bert_model:
            return bert_model
        state = checkpoint.get('model_state_dict', {})
    else:
        state = {}
    embed_key = 'bert.embeddings.word_embeddings.weight'
    if embed_key in state:
        vocab_size = int(state[embed_key].shape[0])
        if vocab_size == 21128:
            return './bert-base-chinese' if os.path.isdir('./bert-base-chinese') else 'bert-base-chinese'
        if vocab_size == 30522:
            return './bert-base-uncased' if os.path.isdir('./bert-base-uncased') else 'bert-base-uncased'
    return './bert-base-uncased' if os.path.isdir('./bert-base-uncased') else 'bert-base-uncased'

def get_transforms():
    """获取图像变换"""
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
                    'gender': gender,
                    'age': age,
                    'path': os.path.join(data_dir, '..', img_path)
                })
    
    return samples

def load_samples_from_manifest(manifest_path):
    """从mixed模式保存的清单加载样本"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    # 过滤掉不存在路径的条目
    valid = []
    for s in samples:
        if s.get('path') and os.path.exists(s['path']):
            valid.append(s)
    return valid

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

def load_model(model_path, model_type, model_name=None):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if model_type == 'demographic':
        model = DemographicClassifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
    elif model_type == 'text':
        bert_model = select_bert_model_for_checkpoint(checkpoint)
        if isinstance(checkpoint, dict):
            checkpoint['bert_model'] = bert_model
        model = TextClassifier(bert_model=bert_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
    elif model_type == 'image':
        if model_name is None:
            raise ValueError("图像模型需要指定model_name")
        model = MODEL_DICT[model_name]()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

def predict_demographic_model(model, dataloader, device):
    """使用人口统计学模型进行预测"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for features, labels, patient_ids in dataloader:
            features = features.to(device)
            labels = labels
            
            outputs = model(features).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_predictions.extend(probs)
            all_labels.extend(labels.numpy())
            all_patient_ids.extend(patient_ids)
    
    return all_predictions, all_labels, all_patient_ids

def predict_text_model(model, dataloader, device):
    """使用文本模型进行预测"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            patient_ids = batch['patient_id']
            
            outputs = model(input_ids, attention_mask).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_predictions.extend(probs)
            all_labels.extend(labels.numpy())
            all_patient_ids.extend(patient_ids)
    
    return all_predictions, all_labels, all_patient_ids

def predict_image_model(model, dataloader, device):
    """使用图像模型进行预测"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for images, labels, patient_ids in dataloader:
            images = images.to(device)
            labels = labels
            
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_predictions.extend(probs)
            all_labels.extend(labels.numpy())
            all_patient_ids.extend(patient_ids)
    
    return all_predictions, all_labels, all_patient_ids

def learn_fusion_weights(demographic_probs, text_probs, image_probs, labels, method='logistic'):
    """学习融合权重"""
    if method == 'logistic':
        # 使用逻辑回归学习融合权重
        X = np.column_stack([demographic_probs, text_probs, image_probs])
        clf = LogisticRegression(random_state=42)
        clf.fit(X, labels)
        
        # 返回融合后的概率
        fusion_probs = clf.predict_proba(X)[:, 1]
        return fusion_probs, clf
    elif method == 'weighted_average':
        # 简单的加权平均，权重通过网格搜索确定
        best_auc = 0
        best_weights = (1/3, 1/3, 1/3)
        best_probs = None
        
        for w_demo in np.arange(0.1, 0.8, 0.1):
            for w_text in np.arange(0.1, 0.9 - w_demo, 0.1):
                w_image = 1.0 - w_demo - w_text
                if w_image < 0.1:
                    continue
                    
                fusion_probs = w_demo * demographic_probs + w_text * text_probs + w_image * image_probs
                auc = roc_auc_score(labels, fusion_probs)
                
                if auc > best_auc:
                    best_auc = auc
                    best_weights = (w_demo, w_text, w_image)
                    best_probs = fusion_probs
        
        return best_probs, best_weights
    else:
        raise ValueError(f"未知的融合方法: {method}")

def test_late_fusion_models(test_dir, demographic_model_path=None, text_model_path=None, image_model_path=None, batch_size=32, results_timestamp=None, seed=42, mixed_manifest=None):
    """测试晚期融合模型"""
    # 设置随机种子
    set_seed(seed)
    print(f"设置随机种子: {seed}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据
    print("加载测试数据...")
    if mixed_manifest is not None:
        print(f"从清单加载混合外部测试集: {mixed_manifest}")
        test_samples = load_samples_from_manifest(mixed_manifest)
    else:
        test_samples = load_multimodal_data(test_dir)
    print(f"总共加载了 {len(test_samples)} 个测试样本")
    
    # 统计信息
    test_patients = len(set([s['patient_id'] for s in test_samples]))
    test_pos = sum([1 for s in test_samples if s['label'] == 1])
    test_neg = len(test_samples) - test_pos
    
    # 统计病人级别的阳性和阴性数量
    test_patient_labels = {}
    for sample in test_samples:
        pid = sample['patient_id']
        if pid not in test_patient_labels:
            test_patient_labels[pid] = sample['label']
    
    test_pos_patients = sum(1 for label in test_patient_labels.values() if label == 1)
    test_neg_patients = len(test_patient_labels) - test_pos_patients
    
    print(f"测试集: {len(test_samples)}张图像, {test_patients}个病人 (阳性:{test_pos}张/{test_pos_patients}人, 阴性:{test_neg}张/{test_neg_patients}人)")
    print(f"测试集阳性比例: 图像{test_pos/len(test_samples):.3f}, 病人{test_pos_patients/len(test_patient_labels):.3f}")
    
    # 加载模型
    demographic_model = None
    demographic_checkpoint = None
    if demographic_model_path:
        print("加载人口统计学模型...")
        demographic_model, demographic_checkpoint = load_model(demographic_model_path, 'demographic')
        demographic_model = demographic_model.to(device)
    
    text_model = None
    text_checkpoint = None
    if text_model_path:
        print("加载文本模型...")
        text_model, text_checkpoint = load_model(text_model_path, 'text')
        text_model = text_model.to(device)
    
    image_model = None
    image_checkpoint = None
    if image_model_path:
        print("加载图像模型...")
        # 尝试从不同的checkpoint获取模型名称
        image_model_name = 'efficientnet_b0'  # 默认值
        if text_checkpoint and 'model_name' in text_checkpoint:
            image_model_name = text_checkpoint['model_name']
        elif demographic_checkpoint and 'model_name' in demographic_checkpoint:
            image_model_name = demographic_checkpoint['model_name']
        
        image_model, image_checkpoint = load_model(image_model_path, 'image', image_model_name)
        image_model = image_model.to(device)
    
    # 打印模型信息
    if demographic_checkpoint:
        print(f"人口统计学模型来自第 {demographic_checkpoint.get('fold', 'unknown')} 折")
    if text_checkpoint:
        print(f"文本模型来自第 {text_checkpoint.get('fold', 'unknown')} 折")
    if image_checkpoint:
        print(f"图像模型来自第 {image_checkpoint.get('fold', 'unknown')} 折")
    
    # 创建数据集和数据加载器
    demographic_dataset = None
    demographic_loader = None
    if demographic_model:
        demographic_dataset = DemographicDataset(test_samples)
        demographic_loader = DataLoader(demographic_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    text_dataset = None
    text_loader = None
    if text_model:
        tokenizer = BertTokenizer.from_pretrained(text_checkpoint.get('bert_model', './bert-base-uncased') if text_checkpoint else './bert-base-uncased')
        text_dataset = TextDataset(test_samples, tokenizer)
        text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    image_dataset = None
    image_loader = None
    if image_model:
        image_dataset = CTDataset(test_samples, transform=get_transforms())
        image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 进行预测
    demographic_predictions = None
    demographic_labels = None
    demographic_patient_ids = None
    if demographic_model and demographic_loader:
        print("使用人口统计学模型进行预测...")
        demographic_predictions, demographic_labels, demographic_patient_ids = predict_demographic_model(demographic_model, demographic_loader, device)
    
    text_predictions = None
    text_labels = None
    text_patient_ids = None
    if text_model and text_loader:
        print("使用文本模型进行预测...")
        text_predictions, text_labels, text_patient_ids = predict_text_model(text_model, text_loader, device)
    
    image_predictions = None
    image_labels = None
    image_patient_ids = None
    if image_model and image_loader:
        print("使用图像模型进行预测...")
        image_predictions, image_labels, image_patient_ids = predict_image_model(image_model, image_loader, device)
    
    # 病人级别投票
    print("进行病人级别投票...")
    demographic_patient_probs = None
    demographic_patient_labels = None
    demographic_patient_ids_final = None
    if demographic_predictions is not None:
        demographic_patient_probs, demographic_patient_labels, demographic_patient_ids_final = patient_level_voting(demographic_predictions, demographic_labels, demographic_patient_ids)
    
    text_patient_probs = None
    text_patient_labels = None
    text_patient_ids_final = None
    if text_predictions is not None:
        text_patient_probs, text_patient_labels, text_patient_ids_final = patient_level_voting(text_predictions, text_labels, text_patient_ids)
    
    image_patient_probs = None
    image_patient_labels = None
    image_patient_ids_final = None
    if image_predictions is not None:
        image_patient_probs, image_patient_labels, image_patient_ids_final = patient_level_voting(image_predictions, image_labels, image_patient_ids)
    
    # 确保所有模型的病人ID顺序一致
    patient_ids_list = [ids for ids in [demographic_patient_ids_final, text_patient_ids_final, image_patient_ids_final] if ids is not None]
    patient_labels_list = [labels for labels in [demographic_patient_labels, text_patient_labels, image_patient_labels] if labels is not None]
    
    if len(patient_ids_list) > 1:
        for i in range(1, len(patient_ids_list)):
            assert patient_ids_list[0] == patient_ids_list[i], f"模型{i}的病人ID顺序与模型0不一致"
        for i in range(1, len(patient_labels_list)):
            assert np.array_equal(patient_labels_list[0], patient_labels_list[i]), f"模型{i}的标签与模型0不一致"
    
    final_patient_labels = patient_labels_list[0] if patient_labels_list else None
    final_patient_ids = patient_ids_list[0] if patient_ids_list else None
    
    # 学习融合权重和计算指标
    fusion_probs = None
    fusion_weights = None
    fusion_metrics = None
    
    # 收集可用的概率
    available_probs = []
    prob_names = []
    if demographic_patient_probs is not None:
        available_probs.append(demographic_patient_probs)
        prob_names.append('demographic')
    if text_patient_probs is not None:
        available_probs.append(text_patient_probs)
        prob_names.append('text')
    if image_patient_probs is not None:
        available_probs.append(image_patient_probs)
        prob_names.append('image')
    
    if len(available_probs) >= 2:
        print("学习融合权重...")
        if len(available_probs) == 3:
            fusion_probs, fusion_weights = learn_fusion_weights(
                demographic_patient_probs, text_patient_probs, image_patient_probs, final_patient_labels, method='logistic'
            )
        elif len(available_probs) == 2:
            # 对于两个模型的情况，使用简化的融合
            X = np.column_stack(available_probs)
            clf = LogisticRegression(random_state=42)
            clf.fit(X, final_patient_labels)
            fusion_probs = clf.predict_proba(X)[:, 1]
            fusion_weights = clf
        
        # 计算融合后的指标
        fusion_preds = (fusion_probs > 0.5).astype(int)
        fusion_metrics = calculate_metrics(final_patient_labels, fusion_preds, fusion_probs)
    
    # 计算单独模型的指标
    demographic_metrics = None
    if demographic_patient_probs is not None:
        demographic_preds = (demographic_patient_probs > 0.5).astype(int)
        demographic_metrics = calculate_metrics(final_patient_labels, demographic_preds, demographic_patient_probs)
    
    text_metrics = None
    if text_patient_probs is not None:
        text_preds = (text_patient_probs > 0.5).astype(int)
        text_metrics = calculate_metrics(final_patient_labels, text_preds, text_patient_probs)
    
    image_metrics = None
    if image_patient_probs is not None:
        image_preds = (image_patient_probs > 0.5).astype(int)
        image_metrics = calculate_metrics(final_patient_labels, image_preds, image_patient_probs)
    
    # 保存结果
    if results_timestamp is None:
        results_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = os.path.join('test_results', results_timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备结果数据
    results = {
        'test_timestamp': results_timestamp,
        'model_type': 'late_fusion_multimodal',
        'demographic_model_path': demographic_model_path,
        'text_model_path': text_model_path,
        'image_model_path': image_model_path,
        'fusion_method': 'logistic_regression',
        'fusion_weights': fusion_weights if isinstance(fusion_weights, tuple) else 'logistic_coefficients',
        'test_data_info': {
            'total_samples': len(test_samples),
            'total_patients': test_patients,
            'positive_samples': test_pos,
            'negative_samples': test_neg,
            'positive_patients': test_pos_patients,
            'negative_patients': test_neg_patients,
            'positive_ratio_samples': test_pos/len(test_samples),
            'positive_ratio_patients': test_pos_patients/len(test_patient_labels)
        },
        'results': {},
        'patient_level_predictions': {
            'patient_ids': final_patient_ids,
            'true_labels': [int(x) for x in final_patient_labels] if final_patient_labels is not None else None
        }
    }
    
    # 添加各模型结果
    if demographic_metrics:
        results['results']['demographic_model'] = {k: float(v) for k, v in demographic_metrics.items()}
        results['patient_level_predictions']['demographic_probabilities'] = [float(x) for x in demographic_patient_probs]
        results['patient_level_predictions']['demographic_predictions'] = [int(x) for x in (demographic_patient_probs > 0.5).astype(int)]
    
    if text_metrics:
        results['results']['text_model'] = {k: float(v) for k, v in text_metrics.items()}
        results['patient_level_predictions']['text_probabilities'] = [float(x) for x in text_patient_probs]
        results['patient_level_predictions']['text_predictions'] = [int(x) for x in (text_patient_probs > 0.5).astype(int)]
    
    if image_metrics:
        results['results']['image_model'] = {k: float(v) for k, v in image_metrics.items()}
        results['patient_level_predictions']['image_probabilities'] = [float(x) for x in image_patient_probs]
        results['patient_level_predictions']['image_predictions'] = [int(x) for x in (image_patient_probs > 0.5).astype(int)]
    
    if fusion_metrics:
        results['results']['late_fusion'] = {k: float(v) for k, v in fusion_metrics.items()}
        results['patient_level_predictions']['fusion_probabilities'] = [float(x) for x in fusion_probs]
        results['patient_level_predictions']['fusion_predictions'] = [int(x) for x in (fusion_probs > 0.5).astype(int)]
    
    # 保存结果
    results_path = os.path.join(results_dir, 'late_fusion_test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print(f"\n{'='*80}")
    print("晚期融合模型测试结果")
    print(f"{'='*80}")
    print(f"测试数据: {test_patients}个病人 (阳性:{test_pos_patients}人, 阴性:{test_neg_patients}人)")
    print(f"融合方法: 逻辑回归")
    print(f"\n单独模型结果:")
    
    if demographic_metrics:
        print(f"人口统计学模型 - Accuracy: {demographic_metrics['accuracy']:.4f}, Precision: {demographic_metrics['precision']:.4f}, Recall: {demographic_metrics['recall']:.4f}, F1: {demographic_metrics['f1_score']:.4f}, AUC: {demographic_metrics['auc']:.4f}")
    if text_metrics:
        print(f"文本模型 - Accuracy: {text_metrics['accuracy']:.4f}, Precision: {text_metrics['precision']:.4f}, Recall: {text_metrics['recall']:.4f}, F1: {text_metrics['f1_score']:.4f}, AUC: {text_metrics['auc']:.4f}")
    if image_metrics:
        print(f"图像模型 - Accuracy: {image_metrics['accuracy']:.4f}, Precision: {image_metrics['precision']:.4f}, Recall: {image_metrics['recall']:.4f}, F1: {image_metrics['f1_score']:.4f}, AUC: {image_metrics['auc']:.4f}")
    
    if fusion_metrics:
        print(f"\n晚期融合结果:")
        print(f"Accuracy: {fusion_metrics['accuracy']:.4f}")
        print(f"Accuracy-P: {fusion_metrics['accuracy_p']:.4f}")
        print(f"Accuracy-N: {fusion_metrics['accuracy_n']:.4f}")
        print(f"Precision: {fusion_metrics['precision']:.4f}")
        print(f"Recall: {fusion_metrics['recall']:.4f}")
        print(f"F1-score: {fusion_metrics['f1_score']:.4f}")
        print(f"AUC: {fusion_metrics['auc']:.4f}")
    
    print(f"\n结果保存到: {results_path}")
    
    return results
    """测试晚期融合模型"""
    # 设置随机种子
    set_seed(seed)
    print(f"设置随机种子: {seed}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据
    print("加载测试数据...")
    if mixed_manifest is not None:
        print(f"从清单加载混合外部测试集: {mixed_manifest}")
        test_samples = load_samples_from_manifest(mixed_manifest)
    else:
        test_samples = load_multimodal_data(test_dir)
    print(f"总共加载了 {len(test_samples)} 个测试样本")
    
    # 统计信息
    test_patients = len(set([s['patient_id'] for s in test_samples]))
    test_pos = sum([1 for s in test_samples if s['label'] == 1])
    test_neg = len(test_samples) - test_pos
    
    # 统计病人级别的阳性和阴性数量
    test_patient_labels = {}
    for sample in test_samples:
        pid = sample['patient_id']
        if pid not in test_patient_labels:
            test_patient_labels[pid] = sample['label']
    
    test_pos_patients = sum(1 for label in test_patient_labels.values() if label == 1)
    test_neg_patients = len(test_patient_labels) - test_pos_patients
    
    print(f"测试集: {len(test_samples)}张图像, {test_patients}个病人 (阳性:{test_pos}张/{test_pos_patients}人, 阴性:{test_neg}张/{test_neg_patients}人)")
    print(f"测试集阳性比例: 图像{test_pos/len(test_samples):.3f}, 病人{test_pos_patients/len(test_patient_labels):.3f}")
    
    # 加载模型
    print("加载文本模型...")
    text_model, text_checkpoint = load_model(text_model_path, 'text')
    text_model = text_model.to(device)
    
    print("加载图像模型...")
    image_model_name = text_checkpoint.get('model_name', 'efficientnet_b0')  # 从文本模型checkpoint获取图像模型名称
    image_model, image_checkpoint = load_model(image_model_path, 'image', image_model_name)
    image_model = image_model.to(device)
    
    print(f"文本模型来自第 {text_checkpoint.get('fold', 'unknown')} 折")
    print(f"图像模型来自第 {image_checkpoint.get('fold', 'unknown')} 折")
    
    tokenizer = BertTokenizer.from_pretrained(text_checkpoint.get('bert_model', './bert-base-uncased'))
    
    # 创建数据集
    text_dataset = TextDataset(test_samples, tokenizer)
    image_dataset = CTDataset(test_samples, transform=get_transforms())
    
    text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 进行预测
    print("使用文本模型进行预测...")
    text_predictions, text_labels, text_patient_ids = predict_text_model(text_model, text_loader, device)
    
    print("使用图像模型进行预测...")
    image_predictions, image_labels, image_patient_ids = predict_image_model(image_model, image_loader, device)
    
    # 病人级别投票
    print("进行病人级别投票...")
    text_patient_probs, text_patient_labels, text_patient_ids = patient_level_voting(text_predictions, text_labels, text_patient_ids)
    image_patient_probs, image_patient_labels, image_patient_ids = patient_level_voting(image_predictions, image_labels, image_patient_ids)
    
    # 确保两个模型的病人ID顺序一致
    assert text_patient_ids == image_patient_ids, "文本模型和图像模型的病人ID顺序不一致"
    assert np.array_equal(text_patient_labels, image_patient_labels), "文本模型和图像模型的标签不一致"
    
    # 学习融合权重
    print("学习融合权重...")
    fusion_probs, fusion_weights = learn_fusion_weights(
        text_patient_probs, image_patient_probs, text_patient_labels, method='logistic'
    )
    
    # 计算融合后的指标
    fusion_preds = (fusion_probs > 0.5).astype(int)
    fusion_metrics = calculate_metrics(text_patient_labels, fusion_preds, fusion_probs)
    
    # 计算单独模型的指标
    text_preds = (text_patient_probs > 0.5).astype(int)
    text_metrics = calculate_metrics(text_patient_labels, text_preds, text_patient_probs)
    
    image_preds = (image_patient_probs > 0.5).astype(int)
    image_metrics = calculate_metrics(image_patient_labels, image_preds, image_patient_probs)
    
    # 保存结果
    if results_timestamp is None:
        results_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = os.path.join('test_results', results_timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备结果数据
    results = {
        'test_timestamp': results_timestamp,
        'model_type': 'late_fusion',
        'text_model_path': text_model_path,
        'image_model_path': image_model_path,
        'image_model_name': image_model_name,
        'fusion_method': 'logistic_regression',
        'fusion_weights': fusion_weights if isinstance(fusion_weights, tuple) else 'logistic_coefficients',
        'test_data_info': {
            'total_samples': len(test_samples),
            'total_patients': test_patients,
            'positive_samples': test_pos,
            'negative_samples': test_neg,
            'positive_patients': test_pos_patients,
            'negative_patients': test_neg_patients,
            'positive_ratio_samples': test_pos/len(test_samples),
            'positive_ratio_patients': test_pos_patients/len(test_patient_labels)
        },
        'results': {
            'text_model': {k: float(v) for k, v in text_metrics.items()},
            'image_model': {k: float(v) for k, v in image_metrics.items()},
            'late_fusion': {k: float(v) for k, v in fusion_metrics.items()}
        },
        'patient_level_predictions': {
            'patient_ids': text_patient_ids,
            'true_labels': [int(x) for x in text_patient_labels],
            'text_probabilities': [float(x) for x in text_patient_probs],
            'image_probabilities': [float(x) for x in image_patient_probs],
            'fusion_probabilities': [float(x) for x in fusion_probs],
            'text_predictions': [int(x) for x in text_preds],
            'image_predictions': [int(x) for x in image_preds],
            'fusion_predictions': [int(x) for x in fusion_preds]
        }
    }
    
    # 保存结果
    results_path = os.path.join(results_dir, 'late_fusion_test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print(f"\n{'='*80}")
    print("晚期融合模型测试结果")
    print(f"{'='*80}")
    print(f"测试数据: {test_patients}个病人 (阳性:{test_pos_patients}人, 阴性:{test_neg_patients}人)")
    print(f"融合方法: 逻辑回归")
    print(f"\n单独模型结果:")
    print(f"文本模型 - Accuracy: {text_metrics['accuracy']:.4f}, Precision: {text_metrics['precision']:.4f}, Recall: {text_metrics['recall']:.4f}, F1: {text_metrics['f1_score']:.4f}, AUC: {text_metrics['auc']:.4f}")
    print(f"图像模型 - Accuracy: {image_metrics['accuracy']:.4f}, Precision: {image_metrics['precision']:.4f}, Recall: {image_metrics['recall']:.4f}, F1: {image_metrics['f1_score']:.4f}, AUC: {image_metrics['auc']:.4f}")
    print(f"\n晚期融合结果:")
    print(f"Accuracy: {fusion_metrics['accuracy']:.4f}")
    print(f"Accuracy-P: {fusion_metrics['accuracy_p']:.4f}")
    print(f"Accuracy-N: {fusion_metrics['accuracy_n']:.4f}")
    print(f"Precision: {fusion_metrics['precision']:.4f}")
    print(f"Recall: {fusion_metrics['recall']:.4f}")
    print(f"F1-score: {fusion_metrics['f1_score']:.4f}")
    print(f"AUC: {fusion_metrics['auc']:.4f}")
    print(f"\n结果保存到: {results_path}")
    
    return results

def test_all_late_fusion_models(test_dir, results_timestamp=None, seed=42, mixed_manifest=None):
    """测试所有晚期融合模型"""
    # 设置随机种子
    set_seed(seed)
    
    # 查找所有保存的模型
    saved_models_dir = 'saved_models'
    if not os.path.exists(saved_models_dir):
        print(f"找不到保存的模型目录: {saved_models_dir}")
        return
    
    # 查找各类模型
    demographic_models = [f for f in os.listdir(saved_models_dir) if f.startswith('best_demographic_model_')]
    text_models = [f for f in os.listdir(saved_models_dir) if f.startswith('best_text_model_')]
    image_models = [f for f in os.listdir(saved_models_dir) if f.startswith('best_image_model_')]
    
    print(f"找到 {len(demographic_models)} 个人口统计学模型, {len(text_models)} 个文本模型和 {len(image_models)} 个图像模型")
    
    # 测试各种模型组合
    model_combinations = []
    
    # 单模型测试
    for demo_model in demographic_models:
        model_name = demo_model.replace('best_demographic_model_', '').replace('.pth', '')
        model_combinations.append({
            'name': f'demographic_only_{model_name}',
            'demographic': os.path.join(saved_models_dir, demo_model),
            'text': None,
            'image': None
        })
    
    for text_model in text_models:
        model_name = text_model.replace('best_text_model_', '').replace('.pth', '')
        model_combinations.append({
            'name': f'text_only_{model_name}',
            'demographic': None,
            'text': os.path.join(saved_models_dir, text_model),
            'image': None
        })
    
    for image_model in image_models:
        model_name = image_model.replace('best_image_model_', '').replace('.pth', '')
        model_combinations.append({
            'name': f'image_only_{model_name}',
            'demographic': None,
            'text': None,
            'image': os.path.join(saved_models_dir, image_model)
        })
    
    # 双模型组合测试
    for demo_model in demographic_models:
        for text_model in text_models:
            demo_name = demo_model.replace('best_demographic_model_', '').replace('.pth', '')
            text_name = text_model.replace('best_text_model_', '').replace('.pth', '')
            if demo_name == text_name:  # 只测试匹配的模型对
                model_combinations.append({
                    'name': f'demographic_text_{demo_name}',
                    'demographic': os.path.join(saved_models_dir, demo_model),
                    'text': os.path.join(saved_models_dir, text_model),
                    'image': None
                })
    
    for demo_model in demographic_models:
        for image_model in image_models:
            demo_name = demo_model.replace('best_demographic_model_', '').replace('.pth', '')
            image_name = image_model.replace('best_image_model_', '').replace('.pth', '')
            if demo_name == image_name:  # 只测试匹配的模型对
                model_combinations.append({
                    'name': f'demographic_image_{demo_name}',
                    'demographic': os.path.join(saved_models_dir, demo_model),
                    'text': None,
                    'image': os.path.join(saved_models_dir, image_model)
                })
    
    for text_model in text_models:
        for image_model in image_models:
            text_name = text_model.replace('best_text_model_', '').replace('.pth', '')
            image_name = image_model.replace('best_image_model_', '').replace('.pth', '')
            if text_name == image_name:  # 只测试匹配的模型对
                model_combinations.append({
                    'name': f'text_image_{text_name}',
                    'demographic': None,
                    'text': os.path.join(saved_models_dir, text_model),
                    'image': os.path.join(saved_models_dir, image_model)
                })
    
    # 三模型组合测试
    for demo_model in demographic_models:
        for text_model in text_models:
            for image_model in image_models:
                demo_name = demo_model.replace('best_demographic_model_', '').replace('.pth', '')
                text_name = text_model.replace('best_text_model_', '').replace('.pth', '')
                image_name = image_model.replace('best_image_model_', '').replace('.pth', '')
                if demo_name == text_name == image_name:  # 只测试匹配的模型组合
                    model_combinations.append({
                        'name': f'all_modalities_{demo_name}',
                        'demographic': os.path.join(saved_models_dir, demo_model),
                        'text': os.path.join(saved_models_dir, text_model),
                        'image': os.path.join(saved_models_dir, image_model)
                    })
    
    print(f"将测试 {len(model_combinations)} 种模型组合")
    
    # 测试每个模型组合
    for combination in model_combinations:
        print(f"\n测试模型组合: {combination['name']}")
        try:
            test_late_fusion_models(
                test_dir=test_dir,
                demographic_model_path=combination['demographic'],
                text_model_path=combination['text'],
                image_model_path=combination['image'],
                results_timestamp=results_timestamp,
                seed=seed,
                mixed_manifest=mixed_manifest
            )
        except Exception as e:
            print(f"测试模型组合 {combination['name']} 时出错: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='测试晚期融合模型')
    parser.add_argument('--test_dir', type=str, default='data/外部验证集',
                       help='测试数据目录')
    parser.add_argument('--demographic_model', type=str, default=None,
                       help='人口统计学模型路径')
    parser.add_argument('--text_model', type=str, default=None,
                       help='文本模型路径')
    parser.add_argument('--image_model', type=str, default=None,
                       help='图像模型路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--test_all', action='store_true',
                       help='测试所有模型')
    parser.add_argument('--mixed_manifest', type=str, default=None,
                       help='mixed模式下由训练脚本保存的外部测试集清单路径')
    
    args = parser.parse_args()
    
    if args.test_all:
        # 测试所有模型
        test_all_late_fusion_models(
            test_dir=args.test_dir,
            seed=args.seed,
            mixed_manifest=args.mixed_manifest
        )
    else:
        # 测试指定模型
        if not any([args.demographic_model, args.text_model, args.image_model]):
            print("请至少指定一个模型路径（demographic_model, text_model, image_model），或使用 --test_all 测试所有模型")
            return
        
        test_late_fusion_models(
            test_dir=args.test_dir,
            demographic_model_path=args.demographic_model,
            text_model_path=args.text_model,
            image_model_path=args.image_model,
            batch_size=args.batch_size,
            seed=args.seed,
            mixed_manifest=args.mixed_manifest
        )

if __name__ == '__main__':
    main()
