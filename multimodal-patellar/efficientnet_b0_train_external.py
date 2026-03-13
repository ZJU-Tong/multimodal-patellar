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
from sklearn.model_selection import train_test_split
import argparse
from collections import defaultdict
import json
from datetime import datetime
import random

from model import MODEL_DICT


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
        # targets: labels
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # alpha 处理
        if self.alpha is not None:
            # targets 为 1 时 alpha，为 0 时 1-alpha
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        else:
            loss = (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BCELossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits, targets):
        return self.criterion(self.sigmoid(logits), targets)


class CTDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, sample['label'], sample['patient_id']


def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_multimodal_data(data_dir):
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"找不到metadata.csv文件: {metadata_path}")

    df = pd.read_csv(metadata_path)
    samples = []

    for _, row in df.iterrows():
        patient_id = str(row['patient_id'])
        label = int(row['label'])
        images_str = row['images']
        if pd.isna(images_str):
            continue

        image_paths = images_str.split(';')
        for img_path in image_paths:
            img_path = img_path.strip()
            full_path = os.path.join(data_dir, '..', img_path)
            if img_path and os.path.exists(full_path):
                samples.append({
                    'patient_id': patient_id,
                    'label': label,
                    'path': full_path
                })

    return samples


def split_data_by_patient(samples, test_size=0.2, random_state=42):
    patient_data = defaultdict(list)
    for sample in samples:
        patient_data[sample['patient_id']].append(sample)

    patient_ids = list(patient_data.keys())
    patient_labels = []

    for pid in patient_ids:
        patient_labels.append(patient_data[pid][0]['label'])

    train_pids, val_pids = train_test_split(
        patient_ids,
        test_size=test_size,
        stratify=patient_labels if len(set(patient_labels)) > 1 else None,
        random_state=random_state
    )

    train_samples = []
    val_samples = []

    for pid in train_pids:
        train_samples.extend(patient_data[pid])

    for pid in val_pids:
        val_samples.extend(patient_data[pid])

    return train_samples, val_samples


def balance_external_samples(samples, seed=42):
    patient_data = defaultdict(list)
    for sample in samples:
        patient_data[sample['patient_id']].append(sample)

    pos_patients = [pid for pid, items in patient_data.items() if items[0]['label'] == 1]
    neg_patients = [pid for pid, items in patient_data.items() if items[0]['label'] == 0]

    if len(pos_patients) == 0 or len(neg_patients) == 0:
        return samples

    target_count = min(len(pos_patients), len(neg_patients))
    rng = random.Random(seed)
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)
    selected_pids = set(pos_patients[:target_count] + neg_patients[:target_count])

    balanced = []
    for pid in selected_pids:
        balanced.extend(patient_data[pid])

    return balanced


def calculate_metrics(y_true, y_pred, y_prob):
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
    patient_data = defaultdict(lambda: {'predictions': [], 'labels': []})

    for pred, label, pid in zip(predictions, labels, patient_ids):
        patient_data[pid]['predictions'].append(pred)
        patient_data[pid]['labels'].append(label)

    final_predictions = []
    final_labels = []
    final_patient_ids = []

    for pid, data in patient_data.items():
        avg_prediction = np.mean(data['predictions'])
        patient_label = data['labels'][0]

        final_predictions.append(avg_prediction)
        final_labels.append(patient_label)
        final_patient_ids.append(pid)

    return np.array(final_predictions), np.array(final_labels), final_patient_ids


def predict_image_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_patient_ids = []

    with torch.no_grad():
        for images, labels, patient_ids in dataloader:
            images = images.to(device)
            outputs = model(images).squeeze(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(probs)
            all_labels.extend(labels.numpy())
            all_patient_ids.extend(patient_ids)

    return all_predictions, all_labels, all_patient_ids


def train_image_model(train_samples, val_samples, epochs=30, batch_size=32, lr=0.0001, patience=10, device='cuda', num_workers=4, loss_type='BCEWithLogitsLoss'):
    print(f"\n{'='*60}")
    print(f"开始训练图像模型: efficientnet_b0 (Loss: {loss_type})")
    print(f"{'='*60}")

    train_dataset = CTDataset(train_samples, transform=get_transforms(True))
    val_dataset = CTDataset(val_samples, transform=get_transforms(False))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_patients = len(set([s['patient_id'] for s in train_samples]))
    val_patients = len(set([s['patient_id'] for s in val_samples]))
    train_pos = sum([1 for s in train_samples if s['label'] == 1])
    train_neg = len(train_samples) - train_pos
    val_pos = sum([1 for s in val_samples if s['label'] == 1])
    val_neg = len(val_samples) - val_pos

    print(f"训练集: {len(train_dataset)}张图像, {train_patients}个病人 (阳性:{train_pos}, 阴性:{train_neg})")
    print(f"验证集: {len(val_dataset)}张图像, {val_patients}个病人 (阳性:{val_pos}, 阴性:{val_neg})")

    model = MODEL_DICT['efficientnet_b0'](dropout=0.6)
    model = model.to(device)

    pos_weight = torch.tensor([train_neg / train_pos]) if train_pos > 0 else torch.tensor([1.0])
    pos_weight = pos_weight.to(device)

    print(f"使用损失函数: {loss_type}")
    if loss_type == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == 'BCELoss':
        criterion = BCELossWrapper()
    elif loss_type == 'FocalLoss':
        criterion = FocalLoss(alpha=0.25, gamma=2)
    else:
        raise ValueError(f"不支持的损失函数: {loss_type}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_auc = 0.0
    patience_counter = 0
    best_model_state = None
    last_train_metrics = None
    best_val_metrics = None

    print("开始训练 efficientnet_b0 模型...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_predictions = []
        all_labels = []
        all_patient_ids = []

        for images, labels, patient_ids in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            all_predictions.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_patient_ids.extend(patient_ids)

        patient_probs, patient_labels, _ = patient_level_voting(all_predictions, all_labels, all_patient_ids)
        patient_preds = (patient_probs > 0.5).astype(int)
        train_metrics = calculate_metrics(patient_labels, patient_preds, patient_probs)
        last_train_metrics = train_metrics

        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        all_patient_ids = []

        with torch.no_grad():
            for images, labels, patient_ids in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                outputs = model(images).squeeze(-1)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                all_patient_ids.extend(patient_ids)

        val_patient_probs, val_patient_labels, _ = patient_level_voting(all_predictions, all_labels, all_patient_ids)
        val_patient_preds = (val_patient_probs > 0.5).astype(int)
        val_metrics = calculate_metrics(val_patient_labels, val_patient_preds, val_patient_probs)

        scheduler.step(val_metrics['auc'])

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        print("-" * 50)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_val_metrics = val_metrics
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停触发！最佳AUC: {best_auc:.4f}")
            break

    print(f"efficientnet_b0 图像模型训练完成！最佳AUC: {best_auc:.4f}")
    return best_model_state, best_auc, last_train_metrics, best_val_metrics


def evaluate_image_model(model_state, test_samples, batch_size=32, device='cuda', num_workers=4):
    print(f"\n{'='*60}")
    print("开始外部验证集测试: efficientnet_b0")
    print(f"{'='*60}")

    test_dataset = CTDataset(test_samples, transform=get_transforms(False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_patients = len(set([s['patient_id'] for s in test_samples]))
    test_pos = sum([1 for s in test_samples if s['label'] == 1])
    test_neg = len(test_samples) - test_pos

    print(f"外部验证集: {len(test_dataset)}张图像, {test_patients}个病人 (阳性:{test_pos}, 阴性:{test_neg})")

    model = MODEL_DICT['efficientnet_b0'](dropout=0.5)
    model.load_state_dict(model_state)
    model = model.to(device)

    predictions, labels, patient_ids = predict_image_model(model, test_loader, device)
    patient_probs, patient_labels, patient_ids = patient_level_voting(predictions, labels, patient_ids)
    patient_preds = (patient_probs > 0.5).astype(int)
    metrics = calculate_metrics(patient_labels, patient_preds, patient_probs)

    print(f"外部验证结果 - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc']:.4f}")
    return metrics, patient_ids, patient_labels, patient_probs


def main():
    parser = argparse.ArgumentParser(description='训练EfficientNet-b0并在外部验证集测试')
    parser.add_argument('--train_dir', type=str, default='data/data_mixed/训练集', help='训练数据目录')
    parser.add_argument('--external_dir', type=str, default='data/data_mixed/外部验证集', help='外部验证数据目录')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader线程数')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='训练集内部分层验证比例')
    parser.add_argument('--loss', type=str, default='BCEWithLogitsLoss', choices=['BCEWithLogitsLoss', 'BCELoss', 'FocalLoss'], help='损失函数')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    all_train_samples = load_multimodal_data(args.train_dir)
    train_samples, val_samples = split_data_by_patient(
        all_train_samples,
        test_size=args.val_ratio,
        random_state=args.seed
    )

    external_samples = load_multimodal_data(args.external_dir)
    external_samples = balance_external_samples(external_samples, seed=args.seed)

    best_model_state, best_auc, train_metrics, val_metrics = train_image_model(
        train_samples,
        val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
        num_workers=args.num_workers,
        loss_type=args.loss
    )

    test_metrics, patient_ids, patient_labels, patient_probs = evaluate_image_model(
        best_model_state,
        external_samples,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results/efficientnet_b0_results', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    os.makedirs('saved_models', exist_ok=True)
    model_path = os.path.join('saved_models', f'efficientnet_b0_external_{timestamp}.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'model_name': 'efficientnet_b0',
        'best_val_auc': best_auc,
        'train_metrics': ({k: float(v) for k, v in train_metrics.items()} if train_metrics else None),
        'val_metrics': ({k: float(v) for k, v in val_metrics.items()} if val_metrics else None),
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'patience': args.patience,
            'seed': args.seed,
            'val_ratio': args.val_ratio,
            'loss': args.loss
        }
    }, model_path)

    results = {
        'model_name': 'efficientnet_b0',
        'train_dir': args.train_dir,
        'external_dir': args.external_dir,
        'timestamp': timestamp,
        'train_metrics': ({k: float(v) for k, v in train_metrics.items()} if train_metrics else None),
        'val_metrics': ({k: float(v) for k, v in val_metrics.items()} if val_metrics else None),
        'external_metrics': {k: float(v) for k, v in test_metrics.items()},
        'model_path': model_path,
        'patient_level_predictions': {
            'patient_ids': patient_ids,
            'true_labels': [int(x) for x in patient_labels],
            'probabilities': [float(x) for x in patient_probs],
            'predictions': [int(x) for x in (patient_probs > 0.5).astype(int)]
        }
    }

    results_path = os.path.join(results_dir, 'efficientnet_b0_external_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"模型已保存: {model_path}")
    print(f"结果已保存: {results_path}")


if __name__ == '__main__':
    main()
