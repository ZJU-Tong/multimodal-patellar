import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from loss import WeightedBCEWithLogitsLoss, FocalLoss
from model import CTTransformer, CTTransformer_text
from torchmetrics import AUROC, Accuracy, Recall
import matplotlib.pyplot as plt
import random
import os
import re
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, f1_score, precision_score
import pandas as pd
import seaborn as sns

from util.tokenizer import Tokenizer


# 添加随机种子设置
def set_seed(seed):
    random.seed(seed)  # Python的随机种子
    np.random.seed(seed)  # Numpy的随机种子
    torch.manual_seed(seed)  # PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # PyTorch的GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 多GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的随机性

seed = 42
# 在开始训练前设置随机种子
set_seed(seed)  # 你可以选择任意整数作为种子

batch_size = 128
learning_rate = 1e-4
epochs = 150

# 添加配置变量
use_text = True  # 控制是否使用文本数据
model_type = 'CTTransformer_text' if use_text else 'CTTransformer'

class PatientDataset(Dataset):
    def __init__(self, patient_data, patient_ids, transform=None):
        self.patient_data = patient_data
        self.patient_ids = patient_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_info = self.patient_data[patient_id]
        
        # 读取该病人的所有CT图像
        images = []
        for img_path in patient_info['images']:
            image = torchvision.io.read_image(img_path)
            
            # 处理图像通道
            if image.shape[0] == 4:
                image = image[:3, :, :]
            elif image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
                
            if self.transform:
                image = self.transform(image)
            images.append(image)
            
        # 将图像堆叠成一个张量
        images = torch.stack(images)  # (num_images, C, H, W)
        
        if use_text:
            # 使用分词后的文本
            return images, patient_info['text'], patient_info['label']
        else:
            return images, patient_info['label']

def prepare_data(base_image_dir, col_name='查体_分词', positive_csv_path=None, negative_csv_path=None):
    """准备数据，加载分词后的文本数据"""
    patient_data = {}
    
    # 如果使用文本数据，先读取Excel文件
    text_map = {}
    gender_map = {}  # 存储性别信息
    age_map = {}     # 存储年龄信息
    
    if use_text:
        if positive_csv_path and negative_csv_path:
            dtype_dict = {'病历号': str}
            df_positive = pd.read_excel(positive_csv_path, dtype=dtype_dict)
            df_negative = pd.read_excel(negative_csv_path, dtype=dtype_dict)
            
            # 处理缺失值
            df_positive[col_name] = df_positive[col_name].fillna('')
            df_negative[col_name] = df_negative[col_name].fillna('')
            
            # 合并数据框以计算性别和年龄的统计值
            all_df = pd.concat([df_positive, df_negative])
            
            # # 处理性别缺失值 - 使用众数填充
            # gender_mode = all_df['性别'].mode()[0]  # 获取性别的众数
            # df_positive['性别'] = df_positive['性别'].fillna(gender_mode)
            # df_negative['性别'] = df_negative['性别'].fillna(gender_mode)
            
            # # 处理年龄缺失值 - 使用平均数填充
            # age_mean = all_df['年龄'].mean()  # 获取年龄的平均值
            # df_positive['年龄'] = df_positive['年龄'].fillna(age_mean)
            # df_negative['年龄'] = df_negative['年龄'].fillna(age_mean)
            
            # 创建病历号到各特征的映射
            for _, row in df_positive.iterrows():
                text_map[str(row['病历号'])] = row[col_name]
                gender_map[str(row['病历号'])] = row['性别']
                age_map[str(row['病历号'])] = row['年龄']
            for _, row in df_negative.iterrows():
                text_map[str(row['病历号'])] = row[col_name]
                gender_map[str(row['病历号'])] = row['性别']
                age_map[str(row['病历号'])] = row['年龄']
        else:
            print("警告：use_text=True但未提供文本数据路径")
    
    # 临时存储所有找到的患者ID及其数据
    temp_patient_data = {}
    
    # 处理negative文件夹
    negative_dir = os.path.join(base_image_dir, 'negative')
    for filename in os.listdir(negative_dir):
        if filename.endswith(('.png')):
            match = re.search(r'(\d+)', filename)
            if match:
                patient_id = match.group(1)
                if patient_id not in temp_patient_data:
                    temp_patient_data[patient_id] = {
                        'images': [],
                        'label': 0,
                    }
                temp_patient_data[patient_id]['images'].append(os.path.join(negative_dir, filename))
            else:
                print(f"警告：文件名中未找到病人ID: {filename}")
    
    # 处理positive文件夹
    positive_dir = os.path.join(base_image_dir, 'positive')
    for filename in os.listdir(positive_dir):
        if filename.endswith(('.png')):
            match = re.search(r'(\d+)', filename)
            if match:
                patient_id = match.group(1)
                if patient_id not in temp_patient_data:
                    temp_patient_data[patient_id] = {
                        'images': [],
                        'label': 1,
                    }
                temp_patient_data[patient_id]['images'].append(os.path.join(positive_dir, filename))
            else:
                print(f"警告：文件名中未找到病人ID: {filename}")
    
    # 过滤统计
    filtered_no_text = 0
    filtered_no_images = 0
    
    # 如果使用文本数据，检查表格中的病人是否有对应的CT图像
    if use_text:
        # 查找表格中有多少病人在图像目录中找不到
        text_patients_without_images = set(text_map.keys()) - set(temp_patient_data.keys())
        filtered_no_images = len(text_patients_without_images)
        if filtered_no_images > 0:
            print(f"警告：表格中有 {filtered_no_images} 名病人在图像目录中找不到")
    
    # 过滤患者: 如果使用文本数据，则只保留在表格中存在的患者
    for patient_id, patient_info in temp_patient_data.items():
        if use_text:
            # 检查患者是否在病历表格中
            if patient_id not in text_map:
                filtered_no_text += 1
                continue
            # 添加文本数据
            patient_info['text'] = text_map[patient_id]
            # # 添加性别和年龄数据
            # patient_info['gender'] = gender_map.get(patient_id, gender_mode)  # 如果没有性别信息，使用众数
            # patient_info['age'] = age_map.get(patient_id, age_mean)  # 如果没有年龄信息，使用平均数
        else:
            # 不使用文本数据时，添加空值
            patient_info['text'] = ''
            patient_info['gender'] = None
            patient_info['age'] = None
        
        # 将有效患者添加到最终数据集
        patient_data[patient_id] = patient_info
    
    # 将病人数据转换为列表
    patient_ids = list(patient_data.keys())
    
    # 打印过滤信息
    if use_text:
        print(f"过滤掉了 {filtered_no_text} 名在病历表格中不存在的患者")
        print(f"过滤掉了 {filtered_no_images} 名在图像目录中找不到的患者")
        print(f"最终使用 {len(patient_ids)} 名患者的数据")
    
    return patient_data, patient_ids

# 根据use_text决定是否传入文本数据路径
if use_text:
    patient_data, patient_ids = prepare_data(
        base_image_dir='./data/new_data/',
        col_name='查体_分词',
        positive_csv_path='./data/positive_processed2.xlsx',
        negative_csv_path='./data/negative_processed2.xlsx'
    )
else:
    patient_data, patient_ids = prepare_data(
        base_image_dir='./data/最终训练集/'
    )

# 设置随机种子
random.seed(seed)

# 按病人ID进行划分
total_patients = len(patient_ids)
train_size = int(total_patients * 0.5)
val_size = int(total_patients * 0.1)
test_size = total_patients - train_size - val_size

# 随机打乱病人ID列表
random.shuffle(patient_ids)

# 划分病人ID
train_patient_ids = patient_ids[:train_size]
val_patient_ids = patient_ids[train_size:train_size + val_size]
test_patient_ids = patient_ids[train_size + val_size:]

# 根据病人ID生成最终的数据集
def create_dataset_from_patient_ids(patient_ids, patient_data):
    all_image_paths = []
    all_labels = []
    for pid in patient_ids:
        for img_path in patient_data[pid]['images']:
            all_image_paths.append(img_path)
            all_labels.append(patient_data[pid]['label'])
    return all_image_paths, all_labels

# 创建训练集、验证集和测试集
train_images, train_labels = create_dataset_from_patient_ids(train_patient_ids, patient_data)
val_images, val_labels = create_dataset_from_patient_ids(val_patient_ids, patient_data)
test_images, test_labels = create_dataset_from_patient_ids(test_patient_ids, patient_data)

# 打印病人ID分布
print("\n各集合病人数量:")
print(f"训练集病人数: {len(train_patient_ids)}")
print(f"验证集病人数: {len(val_patient_ids)}")
print(f"测试集病人数: {len(test_patient_ids)}")
print("="*50)

# 增强数据增强
aug = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),  # 添加垂直翻转
    transforms.RandomRotation(15),         # 增加旋转角度
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.15, 0.15),    # 增加平移范围
        scale=(0.85, 1.15)         # 增加缩放范围
    ),
    transforms.ColorJitter(
        brightness=0.3,            # 增加亮度调整
        contrast=0.3,              # 增加对比度调整
        saturation=0.2             # 增加饱和度调整
    ),
    transforms.RandomErasing(p=0.2),  # 添加随机擦除
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 验证和测试集使用基础变换
base_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    """处理不同病人的CT图像数量不同的情况"""
    if use_text:
        # 解包带有文本的batch
        images, texts, labels = zip(*batch)
        
        # 直接使用分词后的文本列表
        processed_texts = []
        for text in texts:
            # 过滤掉空字符串
            tokens = [token for token in text.split('/') if token.strip()]
            processed_texts.append(tokens)
        
        # 使用tokenizer处理分词后的文本
        tokenizer = Tokenizer()
        text_encoding = tokenizer.tokenize(processed_texts)
    else:
        # 解包只有图像的batch
        images, labels = zip(*batch)
    
    # 找出这个batch中最大的CT图像数量
    max_images = max(x.size(0) for x in images)
    
    # 填充每个病人的CT图像序列到相同长度
    padded_images = []
    for imgs in images:
        num_images = imgs.size(0)
        if num_images < max_images:
            # 填充零张量
            padding = torch.zeros((max_images - num_images,) + imgs.size()[1:])
            imgs = torch.cat([imgs, padding], dim=0)
        padded_images.append(imgs)
    
    if use_text:
        return {
            'image': torch.stack(padded_images),
            'input_ids': text_encoding['input_ids'],
            'attention_mask': text_encoding['attention_mask'],
            'label': torch.tensor(labels)
        }
    else:
        return {
            'image': torch.stack(padded_images),
            'label': torch.tensor(labels)
        }

# 创建数据加载器
train_dataset = PatientDataset(patient_data, train_patient_ids, transform=aug)
val_dataset = PatientDataset(patient_data, val_patient_ids, transform=base_transform)
test_dataset = PatientDataset(patient_data, test_patient_ids, transform=base_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # 减小batch_size因为每个样本包含多张图像
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# 根据配置选择模型
model = CTTransformer_text(use_text=use_text)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 替换原有的损失函数
loss_nn = WeightedBCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Initialize lists to store metrics
train_losses = []
val_losses = []
train_aucs = []
val_aucs = []
train_accs = []
val_accs = []
total_train_step = 0
total_test_step = 0

# 在模型训练前初始化metrics
train_auroc = AUROC(task="binary").to(device)
val_auroc = AUROC(task="binary").to(device)
train_acc = Accuracy(task="binary").to(device)
val_acc = Accuracy(task="binary").to(device)
train_recall = Recall(task="binary").to(device)
val_recall = Recall(task="binary").to(device)

# 初始化存储recall的列表
train_recalls = []
val_recalls = []

# 在训练循环前添加
best_val_acc = 0.0
patience = 50
patience_counter = 50
best_model_path = 'model_save/best_multi_model.pth'

# Training loop
for epoch in range(epochs):
    # Training
    model.train()
    epoch_train_loss = 0
    
    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        if use_text:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(images, input_ids, attention_mask)
        else:
            outputs = model(images)
            
        loss = loss_nn(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        
        # 计算训练指标
        # 确保预测值和目标值的形状一致
        pred_probs = outputs.sigmoid().view(-1)  # 展平为一维
        labels = labels.view(-1)  # 展平为一维
        
        train_auroc.update(pred_probs, labels)
        train_acc.update(pred_probs, labels)
        train_recall.update(pred_probs, labels)
        
        total_train_step += 1
        epoch_train_loss += loss.item()
        
    train_auc = train_auroc.compute()
    train_accuracy = train_acc.compute()
    train_recall_value = train_recall.compute()
    train_losses.append(epoch_train_loss / len(train_loader))
    train_aucs.append(train_auc.item())
    train_accs.append(train_accuracy.item())
    train_recalls.append(train_recall_value.item())
    
    # 重置metrics
    train_auroc.reset()
    train_acc.reset()
    train_recall.reset()
    
    # Validation
    model.eval()
    epoch_test_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if use_text:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(images, input_ids, attention_mask)
            else:
                outputs = model(images)
                
            loss = loss_nn(outputs, labels.unsqueeze(1).float())
            epoch_test_loss += loss.item() * images.size(0)  # 乘以batch size来获得总loss
            
            # 计算验证指标
            pred_probs = outputs.sigmoid().view(-1)  # 展平为一维
            labels = labels.view(-1)  # 展平为一维
            
            val_auroc.update(pred_probs, labels)
            val_acc.update(pred_probs, labels)
            val_recall.update(pred_probs, labels)
            
            total_test_step += 1
            
    val_auc = val_auroc.compute()
    val_accuracy = val_acc.compute()
    val_recall_value = val_recall.compute()
    val_losses.append(epoch_test_loss / len(val_dataset))  # 除以总样本数得到平均loss
    val_aucs.append(val_auc.item())
    val_accs.append(val_accuracy.item())
    val_recalls.append(val_recall_value.item())
    
    # 在验证循环后添加
    if val_accuracy.item() > best_val_acc:
        best_val_acc = val_accuracy.item()
        patience_counter = 0
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1],
                'val_auc': val_auc.item(),
                'val_acc': val_accuracy.item(),
            }, best_model_path)
            print(f"保存最佳模型，验证集ACC: {val_accuracy.item():.4f}")
        except Exception as e:
            print(f"保存模型时发生错误: {str(e)}")
    else:
        patience_counter += 1
        
    # 更新学习率
    scheduler.step(val_auc)
    
    # 早停检查
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    # 重置metrics
    val_auroc.reset()
    val_acc.reset()
    val_recall.reset()
    
    print(f"Epoch: {epoch}")
    print(f"Train - Loss: {train_losses[-1]:.4f}, AUC: {train_aucs[-1]:.4f}, Acc: {train_accs[-1]:.4f}, Recall: {train_recalls[-1]:.4f}")
    print(f"Val   - Loss: {val_losses[-1]:.4f}, AUC: {val_aucs[-1]:.4f}, Acc: {val_accs[-1]:.4f}, Recall: {val_recalls[-1]:.4f}")
    print("-" * 60)

# 在训练循环结束后添加测试集评估代码
print("开始在测试集上进行评估...")
model.eval()
test_loss = 0

# 创建字典来存储每个病人的预测结果
patient_predictions = {}
patient_true_labels = {}
processed_count = 0  # 用于追踪已处理的样本数

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        if use_text:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(images, input_ids, attention_mask)
        else:
            outputs = model(images)
            
        loss = loss_nn(outputs, labels.unsqueeze(1).float())
        test_loss += loss.item()
        
        # 获取当前batch的预测结果
        batch_size = images.size(0)
        batch_predictions = outputs.sigmoid().squeeze().cpu().numpy()
        batch_targets = labels.cpu().numpy()
        
        # 获取当前batch中病人的ID
        for i in range(batch_size):
            # 使用processed_count来正确获取病人ID
            current_idx = processed_count + i
            if current_idx >= len(test_dataset.patient_ids):
                break
                
            patient_id = test_dataset.patient_ids[current_idx]
            if patient_id not in patient_predictions:
                patient_predictions[patient_id] = []
                patient_true_labels[patient_id] = batch_targets[i]
            patient_predictions[patient_id].append(batch_predictions[i])
        
        processed_count += batch_size

# 添加验证
print(f"\n数据处理统计:")
print(f"测试集总样本数: {len(test_dataset)}")
print(f"处理的样本总数: {processed_count}")
print(f"收集到的病人数: {len(patient_predictions)}")

# 使用平均值进行最终预测
final_predictions = []
final_targets = []
for patient_id in patient_predictions:
    patient_preds = np.array(patient_predictions[patient_id])
    final_pred = np.mean(patient_preds) > 0.5
    final_predictions.append(final_pred)
    final_targets.append(patient_true_labels[patient_id])

# 计算病人级别的指标
test_accuracy = accuracy_score(final_targets, final_predictions)
test_auc = roc_auc_score(final_targets, final_predictions)
test_recall = recall_score(final_targets, final_predictions)
test_precision = precision_score(final_targets, final_predictions)

print("-" * 60)
print("测试集评估结果")
print(f"Test Loss: {test_loss/len(test_loader):.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test Precision: {test_precision:.4f}")

# 分别计算阳性和阴性样本的准确率
positive_indices = [i for i, label in enumerate(final_targets) if label == 1]
negative_indices = [i for i, label in enumerate(final_targets) if label == 0]

if positive_indices:
    positive_predictions = [final_predictions[i] for i in positive_indices]
    positive_targets = [final_targets[i] for i in positive_indices]
    positive_accuracy = accuracy_score(positive_targets, positive_predictions)
    print(f"阳性样本准确率: {positive_accuracy:.4f} ({sum(positive_predictions)}/{len(positive_predictions)})")
else:
    print("测试集中没有阳性样本")

if negative_indices:
    negative_predictions = [final_predictions[i] for i in negative_indices]
    negative_targets = [final_targets[i] for i in negative_indices]
    negative_accuracy = accuracy_score(negative_targets, negative_predictions)
    print(f"阴性样本准确率: {negative_accuracy:.4f} ({sum([1 for p in negative_predictions if p == 0])}/{len(negative_predictions)})")
else:
    print("测试集中没有阴性样本")

# 计算F1分数
test_f1 = f1_score(final_targets, final_predictions)
print(f"Test F1 Score: {test_f1:.4f}")

# 计算混淆矩阵
cm = confusion_matrix(final_targets, final_predictions)

# 创建混淆矩阵的可视化
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])

plt.title('Confusion Matrix', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

# 保存混淆矩阵图
if not os.path.exists('train_result'):
    os.makedirs('train_result')
plt.savefig(f'train_result/{model.__class__.__name__}_seed{seed}_confusion_matrix.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 将F1分数也添加到checkpoint中
checkpoint = torch.load(best_model_path)
checkpoint.update({
    'test_loss': test_loss/len(test_loader),
    'test_auc': test_auc,
    'test_acc': test_accuracy,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_precision': test_precision
})
torch.save(checkpoint, best_model_path)

def plot_metrics(train_metrics, val_metrics, metric_name):
    """绘制训练和验证指标的趋势图"""
    # 设置风格
    plt.style.use('ggplot')
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    
    # 创建x轴数据点
    x = np.arange(len(train_metrics))
    
    # 绘制曲线，采用平滑渐变的线条和填充效果
    # 训练曲线 - 使用蓝色调
    plt.plot(x, train_metrics, linestyle='-', 
             linewidth=2.5, color='#3366cc', label=f'Train {metric_name}')
    
    # 添加训练曲线下方的半透明填充
    plt.fill_between(x, train_metrics, alpha=0.15, color='#3366cc')
    
    # 验证曲线 - 使用橙色调
    plt.plot(x, val_metrics, linestyle='-', 
             linewidth=2.5, color='#ff9900', label=f'Val {metric_name}')
    
    # 添加验证曲线下方的半透明填充
    plt.fill_between(x, val_metrics, alpha=0.15, color='#ff9900')
    
    # 优化坐标轴
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=14, fontweight='bold')
    plt.title(f'Training and Validation {metric_name}', fontsize=16, fontweight='bold', pad=20)
    
    # 设置网格线样式
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7, color='#CCCCCC')
    
    # 美化坐标轴刻度
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 设置坐标轴范围
    y_min = min(min(train_metrics), min(val_metrics))
    y_max = max(max(train_metrics), max(val_metrics))
    y_margin = (y_max - y_min) * 0.15  # 增加边距
    plt.ylim([max(0, y_min - y_margin), min(1.0, y_max + y_margin)])
    
    # 美化图例
    legend = plt.legend(loc='best', frameon=True, fontsize=12, 
                      edgecolor='gray', facecolor='white', framealpha=1,
                      fancybox=True, shadow=True)
    
    # 为图例项添加圆角边框
    frame = legend.get_frame()
    frame.set_linewidth(1.0)
    
    # 添加图表边框，去除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # 设置紧凑布局
    plt.tight_layout()
    
    # 保存为高质量PNG图像
    if not os.path.exists('train_result'):
        os.makedirs('train_result')
    
    plt.savefig('train_result/' + f'{model.__class__.__name__}_seed{seed}_{metric_name.lower()}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# 绘制ACC趋势图
plot_metrics(train_accs, val_accs, 'Accuracy')

# 绘制AUC趋势图
plot_metrics(train_aucs, val_aucs, 'AUC')

# 绘制Loss趋势图
plot_metrics(train_losses, val_losses, 'Loss')
