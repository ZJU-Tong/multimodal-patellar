"""
单图像训练脚本 - 支持内部数据医生诊断对比

新增功能：
- 在训练完成后的内部测试阶段，可以与医生诊断结果进行对比分析
- 通过设置 compare_with_doctor = True 启用此功能
- 医生诊断结果文件路径: ./data/内部数据_医生诊断结果.xlsx
- 会生成详细的对比报告，包括一致性分析和准确性对比

对比分析内容：
1. 模型与医生诊断一致性分析（Kappa系数）
2. 模型vs医生相对于真实标签的准确性对比
3. 详细的患者级别对比表格
4. 统计结果保存至 train_result/internal_doctor_comparison/ 和 train_result/internal_accuracy_comparison/

使用方法：
- 确保医生诊断结果文件存在
- 设置 compare_with_doctor = True
- 运行训练脚本即可
"""

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import SingleImageNet
from torchmetrics import AUROC, Accuracy, Recall
import matplotlib.pyplot as plt
import random
import os
import re
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, f1_score, precision_score
import pandas as pd
import seaborn as sns
import json
import datetime
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

# 设置随机种子
seed = 42
set_seed(seed)

# 训练参数
batch_size = 64
learning_rate = 5e-5
epochs = 100
weight_decay = 5e-4
dropout_rate = 0.6

# 添加配置变量
use_text = False  # 控制是否使用文本数据
model_type = 'SingleImageNet'
compare_with_doctor = True  # 控制是否在内部测试时与医生诊断结果比对

# 医生诊断结果文件配置 - 支持多种级别的医生
doctor_files_config = {
    '低年资普通骨科医师': './data/内部数据集 -zwj-低年资普通骨科医师.xlsx',
    '高年资普通骨科医师': './data/内部数据集 -zwj-高年资普通骨科医师.xlsx',
    '关节外科专科医师': './data/内部数据集 -zwj-关节外科专科医师.xlsx',
    # 可以根据需要继续添加更多医生级别
    # '其他级别医师': './data/其他医生诊断结果.xlsx',
}

# 单图像数据集类
class SingleImageDataset(Dataset):
    def __init__(self, image_paths, labels, patient_ids, texts=None, genders=None, ages=None, transform=None):
        """
        每个图像作为一个单独的样本
        
        参数:
        - image_paths: 图像路径列表
        - labels: 对应的标签列表
        - patient_ids: 每个图像对应的病人ID列表
        - texts: 病人ID到文本描述的字典
        - genders: 病人ID到性别的字典
        - ages: 病人ID到年龄的字典
        - transform: 图像变换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.texts = texts
        self.genders = genders
        self.ages = ages
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个图像和对应特征"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        
        # 读取图像
        image = torchvision.io.read_image(img_path)
        
        # 处理图像通道
        if image.shape[0] == 4:
            image = image[:3, :, :]
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
            
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 如果使用文本数据，返回文本和其他特征
        if self.texts is not None and use_text:
            text = self.texts.get(patient_id, "")
            gender = self.genders.get(patient_id, 0)  # 默认未知性别
            age = self.ages.get(patient_id, 0)  # 默认年龄0
            return image, text, gender, age, label, patient_id
        else:
            return image, label, patient_id

def prepare_data(base_image_dir, col_name='查体_分词', positive_csv_path=None, negative_csv_path=None):
    """准备数据，加载分词后的文本数据"""
    # 用于存储所有图像路径、标签和对应的病人ID
    all_image_paths = []
    all_labels = []
    all_patient_ids = []
    
    # 存储病人特征信息的字典
    text_dict = {}
    gender_dict = {}
    age_dict = {}
    
    # 如果使用文本数据，先读取Excel文件
    if use_text:
        if positive_csv_path and negative_csv_path:
            # 读取数据
            dtype_dict = {'病历号': str}
            df_positive = pd.read_excel(positive_csv_path, dtype=dtype_dict)
            df_negative = pd.read_excel(negative_csv_path, dtype=dtype_dict)
            
            # 处理缺失值
            df_positive[col_name] = df_positive[col_name].fillna('')
            df_negative[col_name] = df_negative[col_name].fillna('')
            
            # 合并数据框以计算性别和年龄的统计值
            all_df = pd.concat([df_positive, df_negative])
            
            # 处理性别缺失值 - 使用众数填充
            gender_mode = all_df['性别'].mode()[0]  # 获取性别的众数
            df_positive['性别'] = df_positive['性别'].fillna(gender_mode)
            df_negative['性别'] = df_negative['性别'].fillna(gender_mode)
            
            # 性别映射：将文本性别转换为数字编码（0:未知，1:男，2:女）
            gender_mapping = {'男': 1, '女': 2}
            df_positive['性别'] = df_positive['性别'].map(lambda x: gender_mapping.get(x, 0))
            df_negative['性别'] = df_negative['性别'].map(lambda x: gender_mapping.get(x, 0))
            
            # 处理年龄缺失值和数据类型问题 - 使用平均数填充
            # 先将年龄转换为数值类型，无法转换的变为NaN
            all_df['年龄'] = pd.to_numeric(all_df['年龄'], errors='coerce')
            df_positive['年龄'] = pd.to_numeric(df_positive['年龄'], errors='coerce')
            df_negative['年龄'] = pd.to_numeric(df_negative['年龄'], errors='coerce')
            
            # 计算平均年龄并填充缺失值
            age_mean = all_df['年龄'].mean()  # 获取年龄的平均值
            df_positive['年龄'] = df_positive['年龄'].fillna(age_mean)
            df_negative['年龄'] = df_negative['年龄'].fillna(age_mean)
            
            # 创建病历号到各特征的映射
            for _, row in df_positive.iterrows():
                # 确保病历号没有前导0
                patient_id = str(int(str(row['病历号']).strip()))
                text_dict[patient_id] = row[col_name]
                gender_dict[patient_id] = row['性别']
                age_dict[patient_id] = row['年龄']
            for _, row in df_negative.iterrows():
                # 确保病历号没有前导0
                patient_id = str(int(str(row['病历号']).strip()))
                text_dict[patient_id] = row[col_name]
                gender_dict[patient_id] = row['性别']
                age_dict[patient_id] = row['年龄']
        else:
            print("警告：use_text=True但未提供文本数据路径")
    
    # 处理negative文件夹
    negative_dir = os.path.join(base_image_dir, 'negative')
    for filename in os.listdir(negative_dir):
        if filename.endswith(('.png')):
            match = re.search(r'(\d+)', filename)
            if match:
                # 去除前导0，将提取的数字转为整数再转回字符串
                patient_id = str(int(match.group(1)))
                img_path = os.path.join(negative_dir, filename)
                
                # 添加到列表
                all_image_paths.append(img_path)
                all_labels.append(0)  # 阴性标签
                all_patient_ids.append(patient_id)
            else:
                print(f"警告：文件名中未找到病人ID: {filename}")
    
    # 处理positive文件夹
    positive_dir = os.path.join(base_image_dir, 'positive')
    for filename in os.listdir(positive_dir):
        if filename.endswith(('.png')):
            match = re.search(r'(\d+)', filename)
            if match:
                # 去除前导0，将提取的数字转为整数再转回字符串
                patient_id = str(int(match.group(1)))
                img_path = os.path.join(positive_dir, filename)
                
                # 添加到列表
                all_image_paths.append(img_path)
                all_labels.append(1)  # 阳性标签
                all_patient_ids.append(patient_id)
            else:
                print(f"警告：文件名中未找到病人ID: {filename}")
    
    # 过滤统计
    filtered_no_text = 0
    patients_not_in_text_map = []
    
    # 如果使用文本数据，过滤掉没有文本的图像
    if use_text:
        filtered_image_paths = []
        filtered_labels = []
        filtered_patient_ids = []
        
        for img_path, label, patient_id in zip(all_image_paths, all_labels, all_patient_ids):
            if patient_id in text_dict:
                filtered_image_paths.append(img_path)
                filtered_labels.append(label)
                filtered_patient_ids.append(patient_id)
            else:
                filtered_no_text += 1
                if patient_id not in patients_not_in_text_map:
                    patients_not_in_text_map.append(patient_id)
        
        all_image_paths = filtered_image_paths
        all_labels = filtered_labels
        all_patient_ids = filtered_patient_ids
    
    # 获取唯一的病人ID列表
    unique_patient_ids = list(set(all_patient_ids))
    
    # 打印统计信息
    print(f"总共找到 {len(unique_patient_ids)} 名患者的 {len(all_image_paths)} 张图像")
    if use_text:
        print(f"过滤掉了 {filtered_no_text} 张在病历表格中不存在的患者图像")
        print(f"过滤掉的患者ID数量: {len(patients_not_in_text_map)}")
    
    return all_image_paths, all_labels, all_patient_ids, unique_patient_ids, text_dict, gender_dict, age_dict

def compute_mean_std(image_paths, max_images=1000):
    """计算图像数据集的均值和标准差"""
    # 创建一个简单的transform来将图像转换为张量
    to_tensor = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ConvertImageDtype(torch.float)
    ])
    
    # 限制处理的图像数量
    if max_images and max_images < len(image_paths):
        sampled_paths = random.sample(image_paths, max_images)
    else:
        sampled_paths = image_paths
    
    # 初始化均值和方差累加器
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    num_images = 0
    
    # 第一遍：计算均值
    for img_path in sampled_paths:
        try:
            img = torchvision.io.read_image(img_path)
            # 处理图像通道
            if img.shape[0] == 4:
                img = img[:3, :, :]
            elif img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
                
            img = to_tensor(img)
            mean_sum += torch.mean(img, dim=(1, 2))
            num_images += 1
            
            if num_images % 100 == 0:
                print(f"已处理 {num_images} 张图像")
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {str(e)}")
    
    mean = mean_sum / num_images
    
    # 第二遍：计算标准差
    for img_path in sampled_paths:
        try:
            img = torchvision.io.read_image(img_path)
            # 处理图像通道
            if img.shape[0] == 4:
                img = img[:3, :, :]
            elif img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
                
            img = to_tensor(img)
            std_sum += torch.std(img, dim=(1, 2))
        except Exception:
            pass
    
    std = std_sum / num_images
    
    return mean.tolist(), std.tolist()

def collate_fn(batch):
    """处理单张图像和文本的批处理"""
    if use_text:
        # 解包带有文本的batch
        images, texts, genders, ages, labels, patient_ids = zip(*batch)
        
        # 处理文本
        processed_texts = []
        for text in texts:
            tokens = [token for token in text.split('/') if token.strip()]
            processed_texts.append(tokens)
        
        tokenizer = Tokenizer()
        text_encoding = tokenizer.tokenize(processed_texts)
        
        # 返回批处理结果
        return {
            'image': torch.stack(images),  # [batch_size, 3, H, W]
            'input_ids': text_encoding['input_ids'],
            'attention_mask': text_encoding['attention_mask'],
            'gender': torch.tensor(genders),
            'age': torch.tensor(ages, dtype=torch.float),
            'label': torch.tensor(labels),
            'patient_id': list(patient_ids)  # 保持为字符串列表
        }
    else:
        # 解包只有图像的batch
        images, labels, patient_ids = zip(*batch)
        
        # 返回批处理结果
        return {
            'image': torch.stack(images),  # [batch_size, 3, H, W]
            'label': torch.tensor(labels),
            'patient_id': list(patient_ids)  # 保持为字符串列表
        }

def evaluate_model(model, test_loader, device, best_model_path, batch_size, learning_rate, seed, train_losses, val_losses, train_aucs, val_aucs, train_accs, val_accs):
    """在测试集上评估模型"""
    print("\n开始在测试集上进行评估...")

    # 检查模型文件是否存在
    if os.path.exists(best_model_path):
        # 加载最佳模型
        try:
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型: {best_model_path}")
            print(f"最佳模型轮次: {checkpoint['epoch']}, 验证集 ACC: {checkpoint['val_acc']:.4f}, AUC: {checkpoint['val_auc']:.4f}")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            print("使用当前模型进行评估")
    else:
        print(f"警告: 模型文件 {best_model_path} 不存在，使用当前模型进行评估")

    model.eval()
    test_loss = 0
    
    # 用于存储每张图像的预测结果和真实标签
    all_predictions = []
    all_targets = []
    all_patient_ids = []  # 记录每个样本对应的病人ID，用于后续分析
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 提取当前batch的病人ID
            if 'patient_id' in batch:
                batch_patient_ids = batch['patient_id']
            else:
                # 如果数据加载器中没有提供病人ID，使用空列表
                batch_patient_ids = ["unknown"] * images.size(0)
            
            if use_text:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                gender = batch['gender'].to(device)
                age = batch['age'].to(device)
                outputs = model(images, input_ids, attention_mask, gender, age)
            else:
                outputs = model(images)
                
            loss = nn.BCEWithLogitsLoss()(outputs, labels.unsqueeze(1).float())
            test_loss += loss.item() * images.size(0)
            
            # 记录预测概率和真实标签
            batch_predictions = outputs.sigmoid().squeeze().cpu().numpy()
            batch_targets = labels.cpu().numpy()
            
            # 如果只有一个样本，确保是数组
            if isinstance(batch_predictions, (float, np.float32, np.float64)):
                batch_predictions = np.array([batch_predictions])
                batch_targets = np.array([batch_targets])
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)
            all_patient_ids.extend(batch_patient_ids)
    
    # 将列表转换为NumPy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    print(f"\n数据处理统计:")
    print(f"测试集总样本数: {len(test_loader.dataset)}")
    print(f"处理的样本总数: {len(all_predictions)}")
    
    # 使用 0.5 作为阈值进行预测
    final_predictions = (all_predictions >= 0.5).astype(int)
    
    # 计算各种评估指标 (图像级别)
    test_accuracy = accuracy_score(all_targets, final_predictions)
    test_auc = roc_auc_score(all_targets, all_predictions)  # AUC使用原始预测概率
    test_recall = recall_score(all_targets, final_predictions)
    test_precision = precision_score(all_targets, final_predictions)
    test_f1 = f1_score(all_targets, final_predictions)
    
    print("-" * 60)
    print("测试集评估结果 (图像级别)")
    print(f"Test Loss: {test_loss/len(test_loader.dataset):.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # 分别计算阳性和阴性样本的准确率
    positive_indices = np.where(all_targets == 1)[0]
    negative_indices = np.where(all_targets == 0)[0]
    
    if len(positive_indices) > 0:
        positive_predictions = final_predictions[positive_indices]
        positive_targets = all_targets[positive_indices]
        positive_accuracy = accuracy_score(positive_targets, positive_predictions)
        print(f"阳性样本准确率: {positive_accuracy:.4f} ({np.sum(positive_predictions)}/{len(positive_predictions)})")
    else:
        print("测试集中没有阳性样本")
    
    if len(negative_indices) > 0:
        negative_predictions = final_predictions[negative_indices]
        negative_targets = all_targets[negative_indices]
        negative_accuracy = accuracy_score(negative_targets, negative_predictions)
        print(f"阴性样本准确率: {negative_accuracy:.4f} ({np.sum(1-negative_predictions)}/{len(negative_predictions)})")
    else:
        print("测试集中没有阴性样本")
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, final_predictions)
    print("\n混淆矩阵:")
    print(cm)
    
    # 进行病人级别的评估
    if len(set(all_patient_ids)) > 1:  # 确保有多个病人ID
        print("\n计算病人级别的评估指标...")
        
        # 创建病人ID到预测和标签的映射
        patient_predictions = {}
        patient_targets = {}
        
        for pid, pred, target in zip(all_patient_ids, all_predictions, all_targets):
            if pid not in patient_predictions:
                patient_predictions[pid] = []
                patient_targets[pid] = target  # 同一病人的标签应该是相同的
            patient_predictions[pid].append(pred)
        
        # 计算每个病人的平均预测概率
        patient_mean_probs = {pid: np.mean(preds) for pid, preds in patient_predictions.items()}
        
        # 将字典转换为列表
        patient_probs = np.array(list(patient_mean_probs.values()))
        patient_true_labels = np.array(list(patient_targets.values()))
        
        # 使用 0.5 作为阈值进行预测
        patient_final_preds = (patient_probs >= 0.5).astype(int)
        
        # 计算病人级别的评估指标
        patient_accuracy = accuracy_score(patient_true_labels, patient_final_preds)
        patient_auc = roc_auc_score(patient_true_labels, patient_probs)
        patient_recall = recall_score(patient_true_labels, patient_final_preds)
        patient_precision = precision_score(patient_true_labels, patient_final_preds)
        patient_f1 = f1_score(patient_true_labels, patient_final_preds)
        
        print("\n病人级别评估结果:")
        print(f"Patient AUC: {patient_auc:.4f}")
        print(f"Patient Accuracy: {patient_accuracy:.4f}")
        print(f"Patient Recall: {patient_recall:.4f}")
        print(f"Patient Precision: {patient_precision:.4f}")
        print(f"Patient F1 Score: {patient_f1:.4f}")
        
        # 分别计算阳性和阴性患者的准确率
        patient_pos_indices = np.where(patient_true_labels == 1)[0]
        patient_neg_indices = np.where(patient_true_labels == 0)[0]
        
        if len(patient_pos_indices) > 0:
            patient_pos_preds = patient_final_preds[patient_pos_indices]
            patient_pos_acc = np.mean(patient_pos_preds == 1)
            print(f"阳性患者准确率: {patient_pos_acc:.4f} ({np.sum(patient_pos_preds)}/{len(patient_pos_preds)})")
        
        if len(patient_neg_indices) > 0:
            patient_neg_preds = patient_final_preds[patient_neg_indices]
            patient_neg_acc = np.mean(patient_neg_preds == 0)
            print(f"阴性患者准确率: {patient_neg_acc:.4f} ({np.sum(1-patient_neg_preds)}/{len(patient_neg_preds)})")
        
        # 病人级别的混淆矩阵
        patient_cm = confusion_matrix(patient_true_labels, patient_final_preds)
        print("\n病人级别混淆矩阵:")
        print(patient_cm)
    
    # 绘制并保存可视化结果
    plot_metrics(train_aucs, val_aucs, "AUC", model_type, batch_size, learning_rate, seed)
    plot_metrics(train_accs, val_accs, "Accuracy", model_type, batch_size, learning_rate, seed)
    plot_metrics(train_losses, val_losses, "Loss", model_type, batch_size, learning_rate, seed)
    
    # 获取训练信息
    epochs_trained = len(train_losses) if train_losses else 'Unknown'
    best_val_acc = max(val_accs) if val_accs else 'Unknown'
    best_val_auc = max(val_aucs) if val_aucs else 'Unknown'
    
    # 保存测试结果
    save_results(test_accuracy, test_auc, test_recall, test_precision, test_f1, 
                batch_size, learning_rate, seed, model_type, use_text,
                epochs_trained=epochs_trained, best_val_acc=best_val_acc, best_val_auc=best_val_auc)

    # 与医生诊断结果对比（如果启用）
    if compare_with_doctor:
        print("\n" + "="*60)
        print("🩺 与内部数据医生诊断结果对比分析")
        print("="*60)
        
        # 计算病人级别的平均预测概率（用于与医生对比）
        if len(set(all_patient_ids)) > 1:  # 确保有多个病人ID
            patient_predictions_for_doctor = {}
            patient_targets_for_doctor = {}
            
            for pid, pred, target in zip(all_patient_ids, all_predictions, all_targets):
                if pid not in patient_predictions_for_doctor:
                    patient_predictions_for_doctor[pid] = []
                    patient_targets_for_doctor[pid] = target
                patient_predictions_for_doctor[pid].append(pred)
            
            # 计算每个病人的平均预测概率
            patient_mean_probs_for_doctor = {pid: np.mean(preds) for pid, preds in patient_predictions_for_doctor.items()}
            
            # 与多种级别医生诊断结果对比
            try:
                print(f"\n   配置的医生级别: {list(doctor_files_config.keys())}")
                # 创建患者ID到真实标签的映射
                patient_true_labels_list = [(pid, patient_targets_for_doctor[pid]) for pid in patient_targets_for_doctor.keys()]
                
                # 循环对比每种级别的医生
                for doctor_level, doctor_file_path in doctor_files_config.items():
                    print(f"\n   {'='*50}")
                    print(f"   🩺 开始与 {doctor_level} 的诊断结果对比")
                    print(f"   {'='*50}")
                    
                    doctor_comparison = compare_with_single_doctor_internal(
                        patient_mean_probs_for_doctor, best_model_path, doctor_level, doctor_file_path
                    )
                    
                    if doctor_comparison['num_common_patients'] > 0:
                        print(f"   与{doctor_level}诊断一致性: {doctor_comparison['agreement_rate']:.4f}")
                        print(f"   Kappa系数: {doctor_comparison['kappa']:.4f}")
                        
                        # 分析医生和模型相对于真实标签的准确性
                        print(f"   正在分析{doctor_level}和模型相对于内部测试集真实标签的准确性...")
                        accuracy_comparison = analyze_single_doctor_vs_model_accuracy_internal(
                            patient_mean_probs_for_doctor, patient_true_labels_list, best_model_path, doctor_level, doctor_file_path
                        )
                    else:
                        print(f"   ⚠️  没有找到与{doctor_level}都有结果的患者")
            except Exception as e:
                print(f"   ❌ 与医生诊断对比时出错: {str(e)}")
        else:
            print(f"   ⚠️  测试集中没有足够的患者进行医生诊断对比")
    else:
        print(f"\n💡 提示: 如需启用与医生诊断结果对比，请将 compare_with_doctor 设置为 True")

def plot_metrics(train_metrics, val_metrics, metric_name, model_type, batch_size, learning_rate, seed):
    """绘制训练和验证指标的趋势图"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_metrics) + 1)
    
    plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    
    plt.title(f'{metric_name} vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    # 确保目录存在
    os.makedirs('train_result/fig', exist_ok=True)
    
    plt.savefig(f'train_result/fig/{model_type}_bs{batch_size}_lr{learning_rate:.0e}_seed{seed}_{metric_name.lower()}.png', dpi=300)
    plt.close()

def save_results(accuracy, auc, recall, precision, f1, batch_size, learning_rate, seed, model_type, use_text, 
                epochs_trained=None, best_val_acc=None, best_val_auc=None):
    """保存测试结果到文件"""
    # 确保目录存在
    os.makedirs('train_result/log', exist_ok=True)
    
    # 收集结果
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # 收集训练参数
    training_params = {
        'timestamp': timestamp,
        'model_type': model_type,
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'weight_decay': float(weight_decay),
        'dropout_rate': float(dropout_rate),
        'epochs': int(epochs),
        'early_stopping_patience': 20,
        'seed': int(seed),
        'use_text': bool(use_text),
        'optimizer': 'AdamW',
        'loss_function': 'BCEWithLogitsLoss',
        'scheduler': 'CosineAnnealingLR',
        'image_size': 224,
        'data_path': './data/最终训练集/',
        'augmentation': 'Enhanced (包含RandomHorizontalFlip, RandomVerticalFlip, RandomRotation等)'
    }
    
    # 收集测试结果
    test_results = {
        'test_auc': float(auc),
        'test_acc': float(accuracy),
        'test_recall': float(recall),
        'test_precision': float(precision),
        'test_f1': float(f1),
        'epochs_trained': epochs_trained if epochs_trained else 'Unknown',
        'best_val_acc': float(best_val_acc) if best_val_acc else 'Unknown',
        'best_val_auc': float(best_val_auc) if best_val_auc else 'Unknown'
    }
    
    # 合并参数和结果
    log_data = {
        'training_params': training_params,
        'test_results': test_results
    }
    
    # 保存为JSON
    json_file = f'train_result/log/{model_type}_bs{batch_size}_lr{learning_rate:.0e}_seed{seed}_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    # 保存到CSV方便对比
    csv_file = 'train_result/log/test_results_history.csv'
    csv_exists = os.path.exists(csv_file)
    
    csv_row = {
        'timestamp': timestamp,
        'model': model_type,
        'seed': seed,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'epochs_trained': epochs_trained if epochs_trained else 'Unknown',
        'test_acc': accuracy,
        'test_auc': auc,
        'test_recall': recall,
        'test_precision': precision,
        'test_f1': f1,
        'use_text': use_text,
        'best_val_acc': best_val_acc if best_val_acc else 'Unknown',
        'json_file': f'{model_type}_bs{batch_size}_lr{learning_rate:.0e}_seed{seed}_{timestamp}.json'
    }
    
    # 将单行数据转换为DataFrame
    df_row = pd.DataFrame([csv_row])
    
    # 追加或创建CSV文件
    if csv_exists:
        df_row.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df_row.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8')
    
    print(f"\n已将测试结果和训练参数保存至: {json_file}")
    print(f"测试结果已添加到历史记录: {csv_file}")

def compare_with_single_doctor_internal(patient_predictions, model_path, doctor_level, doctor_file_path):
    """与单个医生的诊断结果进行对比"""
    from sklearn.metrics import cohen_kappa_score
    
    if not os.path.exists(doctor_file_path):
        print(f"   ❌ 未找到{doctor_level}诊断结果文件: {doctor_file_path}")
        return {
            'agreement_rate': 0.0,
            'kappa': 0.0,
            'num_common_patients': 0,
            'agreements': 0,
            'disagreements': 0
        }
    
    try:
        doctor_df = pd.read_excel(doctor_file_path, dtype={'病历号': str})
    except Exception as e:
        print(f"   ❌ 读取{doctor_level}诊断结果文件时出错: {str(e)}")
        return {
            'agreement_rate': 0.0,
            'kappa': 0.0,
            'num_common_patients': 0,
            'agreements': 0,
            'disagreements': 0
        }
    
    # 处理医生诊断结果
    doctor_results = {}
    valid_diagnoses = 0
    
    for _, row in doctor_df.iterrows():
        try:
            patient_id_raw = str(row['病历号']).strip()
            diagnosis = row['结果']
            
            # 跳过空诊断
            if pd.isna(diagnosis) or diagnosis == '':
                continue
                
            # 标准化病历号（去除前导0）
            try:
                patient_id = str(int(patient_id_raw))
            except (ValueError, TypeError):
                continue
                
            # 转换医生诊断为数字
            if diagnosis == '是':
                doctor_results[patient_id] = 1  # 阳性
                valid_diagnoses += 1
            elif diagnosis == '否':
                doctor_results[patient_id] = 0  # 阴性
                valid_diagnoses += 1
        except Exception as e:
            continue
    
    print(f"   {doctor_level}诊断统计:")
    print(f"   总诊断数: {valid_diagnoses} 个有效诊断")
    print(f"   其中阳性: {sum(doctor_results.values())} 个")
    print(f"   其中阴性: {valid_diagnoses - sum(doctor_results.values())} 个")
    
    # 找到模型和医生都有结果的患者
    common_patients = []
    model_preds = []
    doctor_preds = []
    
    for patient_id in patient_predictions.keys():
        if patient_id in doctor_results:
            common_patients.append(patient_id)
            # 模型预测（使用0.5阈值）
            model_pred = 1 if patient_predictions[patient_id] >= 0.5 else 0
            model_preds.append(model_pred)
            doctor_preds.append(doctor_results[patient_id])
    
    if len(common_patients) == 0:
        print(f"   ⚠️  没有找到模型和{doctor_level}都有结果的患者")
        return {
            'agreement_rate': 0.0,
            'kappa': 0.0,
            'num_common_patients': 0,
            'agreements': 0,
            'disagreements': 0
        }
    
    print(f"   共同患者数: {len(common_patients)} 个")
    
    # 计算一致性
    model_preds = np.array(model_preds)
    doctor_preds = np.array(doctor_preds)
    
    agreements = np.sum(model_preds == doctor_preds)
    disagreements = len(model_preds) - agreements
    agreement_rate = agreements / len(model_preds)
    
    # 计算Kappa系数
    kappa = cohen_kappa_score(doctor_preds, model_preds) if len(set(doctor_preds)) > 1 and len(set(model_preds)) > 1 else 0.0
    
    # 详细分析
    print(f"   一致病例: {agreements} 个")
    print(f"   不一致病例: {disagreements} 个")
    print(f"   一致性: {agreement_rate:.4f}")
    print(f"   Kappa系数: {kappa:.4f}")
    
    # 分析不一致的情况
    model_pos_doctor_neg = np.sum((model_preds == 1) & (doctor_preds == 0))
    model_neg_doctor_pos = np.sum((model_preds == 0) & (doctor_preds == 1))
    
    print(f"   模型阳性-医生阴性: {model_pos_doctor_neg} 个")
    print(f"   模型阴性-医生阳性: {model_neg_doctor_pos} 个")
    
    # 详细的混淆矩阵对比
    print(f"   📊 模型vs{doctor_level}混淆矩阵:")
    print(f"                {doctor_level}诊断")
    print(f"              阴性  阳性")
    print(f"   模型阴性   {np.sum((model_preds == 0) & (doctor_preds == 0)):4d}  {model_neg_doctor_pos:4d}")
    print(f"   模型阳性   {model_pos_doctor_neg:4d}  {np.sum((model_preds == 1) & (doctor_preds == 1)):4d}")
    
    # 保存对比结果
    save_doctor_comparison_results_internal(
        common_patients, model_preds, doctor_preds, patient_predictions, 
        doctor_results, agreement_rate, kappa, model_path, doctor_level
    )
    
    return {
        'agreement_rate': agreement_rate,
        'kappa': kappa,
        'num_common_patients': len(common_patients),
        'agreements': agreements,
        'disagreements': disagreements,
        'model_pos_doctor_neg': model_pos_doctor_neg,
        'model_neg_doctor_pos': model_neg_doctor_pos
    }

def analyze_single_doctor_vs_model_accuracy_internal(patient_predictions, patient_true_labels, model_path, doctor_level, doctor_file_path):
    """分析单个医生和模型相对于内部数据真实标签的准确性"""
    from sklearn.metrics import cohen_kappa_score, classification_report
    
    if not os.path.exists(doctor_file_path):
        print(f"   ❌ 未找到{doctor_level}诊断结果文件: {doctor_file_path}")
        return {}
    
    try:
        doctor_df = pd.read_excel(doctor_file_path, dtype={'病历号': str})
    except Exception as e:
        print(f"   ❌ 读取{doctor_level}诊断结果文件时出错: {str(e)}")
        return {}
    
    # 处理医生诊断结果
    doctor_results = {}
    for _, row in doctor_df.iterrows():
        try:
            patient_id_raw = str(row['病历号']).strip()
            diagnosis = row['结果']
            
            # 跳过空诊断
            if pd.isna(diagnosis) or diagnosis == '':
                continue
                
            # 标准化病历号
            try:
                patient_id = str(int(patient_id_raw))
            except (ValueError, TypeError):
                continue
                
            # 转换医生诊断为数字
            if diagnosis == '是':
                doctor_results[patient_id] = 1
            elif diagnosis == '否':
                doctor_results[patient_id] = 0
        except Exception as e:
            continue
    
    # 创建患者ID到真实标签的映射
    patient_targets_dict = dict(patient_true_labels)
    
    # 找到三方都有结果的患者（真实标签、模型预测、医生诊断）
    common_patients = []
    true_labels = []
    model_preds = []
    doctor_preds = []
    
    for patient_id in patient_predictions.keys():
        if patient_id in doctor_results and patient_id in patient_targets_dict:
            common_patients.append(patient_id)
            true_labels.append(patient_targets_dict[patient_id])
            model_pred = 1 if patient_predictions[patient_id] >= 0.5 else 0
            model_preds.append(model_pred)
            doctor_preds.append(doctor_results[patient_id])
    
    if len(common_patients) == 0:
        print(f"   ⚠️  没有找到三方都有结果的患者")
        return {}
    
    # 转换为numpy数组
    true_labels = np.array(true_labels)
    model_preds = np.array(model_preds)
    doctor_preds = np.array(doctor_preds)
    
    print(f"   三方共同患者数: {len(common_patients)} 个")
    print(f"   真实标签分布: 阳性 {np.sum(true_labels)} 个, 阴性 {len(true_labels) - np.sum(true_labels)} 个")
    
    # 计算模型准确性
    model_accuracy = accuracy_score(true_labels, model_preds)
    model_recall = recall_score(true_labels, model_preds)
    model_precision = precision_score(true_labels, model_preds)
    model_f1 = f1_score(true_labels, model_preds)
    
    # 计算医生准确性
    doctor_accuracy = accuracy_score(true_labels, doctor_preds)
    doctor_recall = recall_score(true_labels, doctor_preds)
    doctor_precision = precision_score(true_labels, doctor_preds)
    doctor_f1 = f1_score(true_labels, doctor_preds)
    
    print(f"   🤖 模型性能 (相对于真实标签):")
    print(f"      准确率: {model_accuracy:.4f}")
    print(f"      召回率: {model_recall:.4f}")
    print(f"      精确率: {model_precision:.4f}")
    print(f"      F1分数: {model_f1:.4f}")
    
    print(f"   👨‍⚕️ {doctor_level}性能 (相对于真实标签):")
    print(f"      准确率: {doctor_accuracy:.4f}")
    print(f"      召回率: {doctor_recall:.4f}")
    print(f"      精确率: {doctor_precision:.4f}")
    print(f"      F1分数: {doctor_f1:.4f}")
    
    # 比较分析
    print(f"   🏆 性能对比:")
    print(f"      准确率: {'模型' if model_accuracy > doctor_accuracy else doctor_level} 更高 ({max(model_accuracy, doctor_accuracy):.4f} vs {min(model_accuracy, doctor_accuracy):.4f})")
    print(f"      召回率: {'模型' if model_recall > doctor_recall else doctor_level} 更高 ({max(model_recall, doctor_recall):.4f} vs {min(model_recall, doctor_recall):.4f})")
    print(f"      精确率: {'模型' if model_precision > doctor_precision else doctor_level} 更高 ({max(model_precision, doctor_precision):.4f} vs {min(model_precision, doctor_precision):.4f})")
    print(f"      F1分数: {'模型' if model_f1 > doctor_f1 else doctor_level} 更高 ({max(model_f1, doctor_f1):.4f} vs {min(model_f1, doctor_f1):.4f})")
    
    # 详细的混淆矩阵比较
    print(f"   📊 详细分析:")
    
    # 模型的混淆矩阵
    model_cm = confusion_matrix(true_labels, model_preds)
    print(f"   模型混淆矩阵 (行:真实, 列:预测):")
    print(f"   [[{model_cm[0,0]:2d} {model_cm[0,1]:2d}]")
    print(f"    [{model_cm[1,0]:2d} {model_cm[1,1]:2d}]]")
    
    # 医生的混淆矩阵
    doctor_cm = confusion_matrix(true_labels, doctor_preds)
    print(f"   {doctor_level}混淆矩阵 (行:真实, 列:预测):")
    print(f"   [[{doctor_cm[0,0]:2d} {doctor_cm[0,1]:2d}]")
    print(f"    [{doctor_cm[1,0]:2d} {doctor_cm[1,1]:2d}]]")
    
    # 分析一致性和分歧
    model_correct = (model_preds == true_labels)
    doctor_correct = (doctor_preds == true_labels)
    
    both_correct = np.sum(model_correct & doctor_correct)
    both_wrong = np.sum((~model_correct) & (~doctor_correct))
    model_right_doctor_wrong = np.sum(model_correct & (~doctor_correct))
    doctor_right_model_wrong = np.sum((~model_correct) & doctor_correct)
    
    print(f"   🎯 一致性分析:")
    print(f"      都正确: {both_correct} 个 ({both_correct/len(common_patients)*100:.1f}%)")
    print(f"      都错误: {both_wrong} 个 ({both_wrong/len(common_patients)*100:.1f}%)")
    print(f"      模型对{doctor_level}错: {model_right_doctor_wrong} 个 ({model_right_doctor_wrong/len(common_patients)*100:.1f}%)")
    print(f"      {doctor_level}对模型错: {doctor_right_model_wrong} 个 ({doctor_right_model_wrong/len(common_patients)*100:.1f}%)")
    
    # 保存详细对比结果
    save_accuracy_comparison_results_internal(
        common_patients, true_labels, model_preds, doctor_preds, 
        patient_predictions, model_accuracy, doctor_accuracy, model_path, doctor_level
    )
    
    return {
        'model_accuracy': model_accuracy,
        'doctor_accuracy': doctor_accuracy,
        'model_f1': model_f1,
        'doctor_f1': doctor_f1,
        'num_common_patients': len(common_patients),
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'model_right_doctor_wrong': model_right_doctor_wrong,
        'doctor_right_model_wrong': doctor_right_model_wrong
    }

def compare_with_doctor_diagnosis_internal(patient_predictions, model_path):
    """与内部数据集的医生诊断结果进行对比"""
    from sklearn.metrics import cohen_kappa_score
    
    # 读取内部数据的医生诊断结果
    doctor_file_path = './data/内部数据集 -zwj-低年资普通骨科医师.xlsx'
    if not os.path.exists(doctor_file_path):
        print(f"\n⚠️  未找到内部数据医生诊断结果文件: {doctor_file_path}")
        return {
            'agreement_rate': 0.0,
            'kappa': 0.0,
            'num_common_patients': 0,
            'agreements': 0,
            'disagreements': 0
        }
    
    try:
        doctor_df = pd.read_excel(doctor_file_path, dtype={'病历号': str})
    except Exception as e:
        print(f"\n❌ 读取医生诊断结果文件时出错: {str(e)}")
        return {
            'agreement_rate': 0.0,
            'kappa': 0.0,
            'num_common_patients': 0,
            'agreements': 0,
            'disagreements': 0
        }
    
    # 处理医生诊断结果
    doctor_results = {}
    valid_diagnoses = 0
    
    for _, row in doctor_df.iterrows():
        try:
            patient_id_raw = str(row['病历号']).strip()
            diagnosis = row['结果']
            
            # 跳过空诊断
            if pd.isna(diagnosis) or diagnosis == '':
                continue
                
            # 标准化病历号（去除前导0）
            try:
                patient_id = str(int(patient_id_raw))
            except (ValueError, TypeError):
                continue
                
            # 转换医生诊断为数字
            if diagnosis == '是':
                doctor_results[patient_id] = 1  # 阳性
                valid_diagnoses += 1
            elif diagnosis == '否':
                doctor_results[patient_id] = 0  # 阴性
                valid_diagnoses += 1
        except Exception as e:
            continue
    
    print(f"\n🩺 内部数据医生诊断对比:")
    print(f"   医生诊断总数: {valid_diagnoses} 个有效诊断")
    print(f"   其中阳性: {sum(doctor_results.values())} 个")
    print(f"   其中阴性: {valid_diagnoses - sum(doctor_results.values())} 个")
    
    # 找到模型和医生都有结果的患者
    common_patients = []
    model_preds = []
    doctor_preds = []
    
    for patient_id in patient_predictions.keys():
        if patient_id in doctor_results:
            common_patients.append(patient_id)
            # 模型预测（使用0.5阈值）
            model_pred = 1 if patient_predictions[patient_id] >= 0.5 else 0
            model_preds.append(model_pred)
            doctor_preds.append(doctor_results[patient_id])
    
    if len(common_patients) == 0:
        print("   ⚠️  没有找到模型和医生都有结果的患者")
        return {
            'agreement_rate': 0.0,
            'kappa': 0.0,
            'num_common_patients': 0,
            'agreements': 0,
            'disagreements': 0
        }
    
    print(f"   共同患者数: {len(common_patients)} 个")
    
    # 计算一致性
    model_preds = np.array(model_preds)
    doctor_preds = np.array(doctor_preds)
    
    agreements = np.sum(model_preds == doctor_preds)
    disagreements = len(model_preds) - agreements
    agreement_rate = agreements / len(model_preds)
    
    # 计算Kappa系数
    kappa = cohen_kappa_score(doctor_preds, model_preds)
    
    # 详细分析
    print(f"   一致病例: {agreements} 个")
    print(f"   不一致病例: {disagreements} 个")
    print(f"   一致性: {agreement_rate:.4f}")
    print(f"   Kappa系数: {kappa:.4f}")
    
    # 分析不一致的情况
    model_pos_doctor_neg = np.sum((model_preds == 1) & (doctor_preds == 0))
    model_neg_doctor_pos = np.sum((model_preds == 0) & (doctor_preds == 1))
    
    print(f"   模型阳性-医生阴性: {model_pos_doctor_neg} 个")
    print(f"   模型阴性-医生阳性: {model_neg_doctor_pos} 个")
    
    # 详细的混淆矩阵对比
    print(f"\n   📊 模型vs医生混淆矩阵:")
    print(f"              医生诊断")
    print(f"              阴性  阳性")
    print(f"   模型阴性   {np.sum((model_preds == 0) & (doctor_preds == 0)):4d}  {model_neg_doctor_pos:4d}")
    print(f"   模型阳性   {model_pos_doctor_neg:4d}  {np.sum((model_preds == 1) & (doctor_preds == 1)):4d}")
    
    # 保存对比结果
    save_doctor_comparison_results_internal(
        common_patients, model_preds, doctor_preds, patient_predictions, 
        doctor_results, agreement_rate, kappa, model_path, "原始低年资医生"
    )
    
    return {
        'agreement_rate': agreement_rate,
        'kappa': kappa,
        'num_common_patients': len(common_patients),
        'agreements': agreements,
        'disagreements': disagreements,
        'model_pos_doctor_neg': model_pos_doctor_neg,
        'model_neg_doctor_pos': model_neg_doctor_pos
    }

def save_doctor_comparison_results_internal(common_patients, model_preds, doctor_preds, patient_predictions,
                                          doctor_results, agreement_rate, kappa, model_path, doctor_level):
    """保存与内部数据医生诊断对比的详细结果"""
    os.makedirs('train_result/internal_doctor_comparison', exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = os.path.basename(model_path).replace('.pth', '')
    
    # 创建详细对比表
    comparison_data = []
    for i, patient_id in enumerate(common_patients):
        model_prob = patient_predictions[patient_id]
        model_pred = model_preds[i]
        doctor_pred = doctor_preds[i]
        is_agreement = model_pred == doctor_pred
        
        comparison_data.append({
            'patient_id': patient_id,
            'model_probability': float(model_prob),
            'model_prediction': int(model_pred),
            'doctor_diagnosis': int(doctor_pred),
            'agreement': bool(is_agreement),
            'model_pred_text': '阳性' if model_pred == 1 else '阴性',
            'doctor_diag_text': '阳性' if doctor_pred == 1 else '阴性',
            'doctor_level': doctor_level
        })
    
    # 保存为Excel文件
    comparison_df = pd.DataFrame(comparison_data)
    excel_file = f'train_result/internal_doctor_comparison/{model_name}_internal_doctor_comparison_{timestamp}.xlsx'
    comparison_df.to_excel(excel_file, index=False)
    
    # 保存统计结果
    summary_results = {
        'comparison_info': {
            'timestamp': timestamp,
            'model_name': model_name,
            'model_path': model_path,
            'dataset': 'internal_test_set',
            'doctor_diagnosis_file': './data/内部数据_医生诊断结果.xlsx',
            'total_common_patients': len(common_patients),
            'doctor_level': doctor_level
        },
        'agreement_analysis': {
            'agreement_rate': float(agreement_rate),
            'kappa_score': float(kappa),
            'total_agreements': int(np.sum(model_preds == doctor_preds)),
            'total_disagreements': int(np.sum(model_preds != doctor_preds)),
            'model_pos_doctor_neg': int(np.sum((model_preds == 1) & (doctor_preds == 0))),
            'model_neg_doctor_pos': int(np.sum((model_preds == 0) & (doctor_preds == 1))),
            'both_positive': int(np.sum((model_preds == 1) & (doctor_preds == 1))),
            'both_negative': int(np.sum((model_preds == 0) & (doctor_preds == 0)))
        },
        'detailed_comparison_file': excel_file
    }
    
    # 保存JSON摘要
    json_file = f'train_result/internal_doctor_comparison/{model_name}_internal_comparison_summary_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    
    # 保存到CSV历史记录
    csv_file = 'train_result/internal_doctor_comparison/internal_doctor_comparison_history.csv'
    csv_exists = os.path.exists(csv_file)
    
    csv_row = {
        'timestamp': timestamp,
        'model_name': model_name,
        'common_patients': len(common_patients),
        'agreement_rate': agreement_rate,
        'kappa_score': kappa,
        'agreements': np.sum(model_preds == doctor_preds),
        'disagreements': np.sum(model_preds != doctor_preds),
        'excel_file': excel_file,
        'json_file': json_file,
        'doctor_level': doctor_level
    }
    
    df_row = pd.DataFrame([csv_row])
    if csv_exists:
        df_row.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df_row.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8')
    
    print(f"\n   💾 内部数据医生对比结果已保存:")
    print(f"      📊 详细对比表: {excel_file}")
    print(f"      📄 结果摘要: {json_file}")

def save_accuracy_comparison_results_internal(common_patients, true_labels, model_preds, doctor_preds,
                                            patient_predictions, model_accuracy, doctor_accuracy, model_path, doctor_level):
    """保存内部数据准确性对比的详细结果"""
    os.makedirs('train_result/internal_accuracy_comparison', exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = os.path.basename(model_path).replace('.pth', '')
    
    # 创建详细对比表
    comparison_data = []
    for i, patient_id in enumerate(common_patients):
        model_prob = patient_predictions[patient_id]
        model_pred = model_preds[i]
        doctor_pred = doctor_preds[i]
        true_label = true_labels[i]
        
        model_correct = model_pred == true_label
        doctor_correct = doctor_pred == true_label
        
        comparison_data.append({
            'patient_id': patient_id,
            'true_label': int(true_label),
            'true_label_text': '阳性' if true_label == 1 else '阴性',
            'model_probability': float(model_prob),
            'model_prediction': int(model_pred),
            'model_pred_text': '阳性' if model_pred == 1 else '阴性',
            'model_correct': bool(model_correct),
            'doctor_diagnosis': int(doctor_pred),
            'doctor_diag_text': '阳性' if doctor_pred == 1 else '阴性',
            'doctor_correct': bool(doctor_correct),
            'agreement': bool(model_pred == doctor_pred),
            'result_category': get_result_category_internal(model_correct, doctor_correct),
            'doctor_level': doctor_level
        })
    
    # 保存为Excel文件
    comparison_df = pd.DataFrame(comparison_data)
    excel_file = f'train_result/internal_accuracy_comparison/{model_name}_internal_accuracy_comparison_{timestamp}.xlsx'
    comparison_df.to_excel(excel_file, index=False)
    
    # 保存统计结果
    summary_results = {
        'comparison_info': {
            'timestamp': timestamp,
            'model_name': model_name,
            'model_path': model_path,
            'dataset': 'internal_test_set',
            'doctor_diagnosis_file': './data/内部数据_医生诊断结果.xlsx',
            'total_common_patients': len(common_patients),
            'true_positive_cases': int(np.sum(true_labels)),
            'true_negative_cases': int(len(true_labels) - np.sum(true_labels)),
            'doctor_level': doctor_level
        },
        'model_performance': {
            'accuracy': float(model_accuracy),
            'recall': float(recall_score(true_labels, model_preds)),
            'precision': float(precision_score(true_labels, model_preds)),
            'f1_score': float(f1_score(true_labels, model_preds))
        },
        'doctor_performance': {
            'accuracy': float(doctor_accuracy),
            'recall': float(recall_score(true_labels, doctor_preds)),
            'precision': float(precision_score(true_labels, doctor_preds)),
            'f1_score': float(f1_score(true_labels, doctor_preds))
        },
        'comparison_analysis': {
            'both_correct': int(np.sum((model_preds == true_labels) & (doctor_preds == true_labels))),
            'both_wrong': int(np.sum((model_preds != true_labels) & (doctor_preds != true_labels))),
            'model_right_doctor_wrong': int(np.sum((model_preds == true_labels) & (doctor_preds != true_labels))),
            'doctor_right_model_wrong': int(np.sum((model_preds != true_labels) & (doctor_preds == true_labels))),
            'both_correct_percentage': float(np.sum((model_preds == true_labels) & (doctor_preds == true_labels)) / len(common_patients) * 100),
            'both_wrong_percentage': float(np.sum((model_preds != true_labels) & (doctor_preds != true_labels)) / len(common_patients) * 100),
            'model_right_doctor_wrong_percentage': float(np.sum((model_preds == true_labels) & (doctor_preds != true_labels)) / len(common_patients) * 100),
            'doctor_right_model_wrong_percentage': float(np.sum((model_preds != true_labels) & (doctor_preds == true_labels)) / len(common_patients) * 100)
        }
    }
    
    # 保存JSON摘要
    json_file = f'train_result/internal_accuracy_comparison/{model_name}_internal_accuracy_summary_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n   💾 内部数据准确性对比结果已保存:")
    print(f"      📊 详细对比表: {excel_file}")
    print(f"      📄 结果摘要: {json_file}")

def get_result_category_internal(model_correct, doctor_correct):
    """根据模型和医生的正确性返回分类"""
    if model_correct and doctor_correct:
        return "都正确"
    elif not model_correct and not doctor_correct:
        return "都错误"
    elif model_correct and not doctor_correct:
        return "模型对医生错"
    else:
        return "医生对模型错"

# 主函数
def main():
    print("🚀 开始训练单图像模型...")
    print(f"📋 配置信息:")
    print(f"   - 使用文本特征: {use_text}")
    print(f"   - 批量大小: {batch_size}")
    print(f"   - 学习率: {learning_rate}")
    print(f"   - 随机种子: {seed}")
    print(f"   - 与医生诊断对比: {compare_with_doctor}")
    if compare_with_doctor:
        print(f"   - 医生诊断文件配置:")
        for doctor_level, file_path in doctor_files_config.items():
            file_exists = "✅" if os.path.exists(file_path) else "❌"
            print(f"     * {doctor_level}: {file_path} {file_exists}")
    
    print("\n开始加载数据...")
    # 根据use_text决定是否传入文本数据路径
    if use_text:
        all_image_paths, all_labels, all_patient_ids, unique_patient_ids, text_dict, gender_dict, age_dict = prepare_data(
            base_image_dir='./data/最终训练集/',
            col_name='查体_分词',
            positive_csv_path='./data/总阳_分词_processed.xlsx',
            negative_csv_path='./data/总阴_分词_processed.xlsx'
        )
    else:
        all_image_paths, all_labels, all_patient_ids, unique_patient_ids, text_dict, gender_dict, age_dict = prepare_data(
            base_image_dir='./data/最终训练集/'
        )
    
    # 根据病人ID划分数据集
    # 按病人ID进行划分
    total_patients = len(unique_patient_ids)
    train_size = int(total_patients * 0.6)
    val_size = int(total_patients * 0.2)
    
    # 随机打乱病人ID列表
    random.shuffle(unique_patient_ids)
    
    # 划分病人ID
    train_patient_ids_set = set(unique_patient_ids[:train_size])
    val_patient_ids_set = set(unique_patient_ids[train_size:train_size + val_size])
    test_patient_ids_set = set(unique_patient_ids[train_size + val_size:])
    
    # 根据病人ID划分数据集
    train_images = []
    train_labels = []
    train_patient_ids = []
    
    val_images = []
    val_labels = []
    val_patient_ids = []
    
    test_images = []
    test_labels = []
    test_patient_ids = []
    
    # 根据病人ID将图像分配到相应的数据集
    for img_path, label, patient_id in zip(all_image_paths, all_labels, all_patient_ids):
        if patient_id in train_patient_ids_set:
            train_images.append(img_path)
            train_labels.append(label)
            train_patient_ids.append(patient_id)
        elif patient_id in val_patient_ids_set:
            val_images.append(img_path)
            val_labels.append(label)
            val_patient_ids.append(patient_id)
        elif patient_id in test_patient_ids_set:
            test_images.append(img_path)
            test_labels.append(label)
            test_patient_ids.append(patient_id)
    
    # 打印病人ID分布
    print("\n各集合病人数量:")
    print(f"训练集病人数: {len(train_patient_ids_set)}")
    print(f"验证集病人数: {len(val_patient_ids_set)}")
    print(f"测试集病人数: {len(test_patient_ids_set)}")
    
    # 统计各数据集中阳性和阴性样本数量
    train_pos_count = sum(1 for label in train_labels if label == 1)
    train_neg_count = len(train_labels) - train_pos_count
    val_pos_count = sum(1 for label in val_labels if label == 1)
    val_neg_count = len(val_labels) - val_pos_count
    test_pos_count = sum(1 for label in test_labels if label == 1)
    test_neg_count = len(test_labels) - test_pos_count
    
    # 打印各数据集中阳性和阴性样本的数量
    print("\n各数据集阳性/阴性样本分布:")
    print(f"训练集 - 阳性: {train_pos_count}, 阴性: {train_neg_count}, 阳性比例: {train_pos_count/len(train_labels):.2%}")
    print(f"验证集 - 阳性: {val_pos_count}, 阴性: {val_neg_count}, 阳性比例: {val_pos_count/len(val_labels):.2%}")
    print(f"测试集 - 阳性: {test_pos_count}, 阴性: {test_neg_count}, 阳性比例: {test_pos_count/len(test_labels):.2%}")
    print("="*50)
    
    # 计算数据集的均值和标准差
    print("\n计算数据集的均值和标准差...")
    try:
        mean, std = compute_mean_std(train_images, max_images=1000)  # 使用训练集计算统计量
        print(f"计算得到的均值: {mean}")
        print(f"计算得到的标准差: {std}")
    except Exception as e:
        print(f"计算均值和标准差时出错: {str(e)}")
        # 如果计算失败，使用默认的ImageNet统计量
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        print(f"使用默认的ImageNet均值: {mean} 和标准差: {std}")
    print("="*50)
    
    # 使用计算得到的均值和标准差定义数据增强
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
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # 添加透视变换
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # 添加随机擦除
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),  # 使用计算得到的均值和标准差
    ])
    
    # 验证和测试集使用基础变换
    base_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),  # 使用计算得到的均值和标准差
    ])
    
    # 创建数据集
    train_dataset = SingleImageDataset(
        train_images, 
        train_labels, 
        train_patient_ids, 
        texts=text_dict,
        genders=gender_dict,
        ages=age_dict,
        transform=aug
    )
    
    val_dataset = SingleImageDataset(
        val_images, 
        val_labels, 
        val_patient_ids, 
        texts=text_dict,
        genders=gender_dict,
        ages=age_dict,
        transform=base_transform
    )
    
    test_dataset = SingleImageDataset(
        test_images, 
        test_labels, 
        test_patient_ids, 
        texts=text_dict,
        genders=gender_dict,
        ages=age_dict,
        transform=base_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 初始化模型
    print("\n初始化模型...")
    model = SingleImageNet(
        model_name='efficientnet_b0',
        dropout_rate=dropout_rate,
        use_text=use_text
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 损失函数和优化器
    loss_nn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器 - 使用余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 初始化指标跟踪
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    train_accs = []
    val_accs = []
    train_recalls = []
    val_recalls = []
    
    total_train_step = 0
    total_val_step = 0
    
    # 初始化评估指标
    train_auroc = AUROC(task="binary").to(device)
    val_auroc = AUROC(task="binary").to(device)
    train_acc = Accuracy(task="binary").to(device)
    val_acc = Accuracy(task="binary").to(device)
    train_recall = Recall(task="binary").to(device)
    val_recall = Recall(task="binary").to(device)
    
    # 早停参数
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    # 在模型文件名中包含重要参数
    best_model_path = f'model_save/single_img_model_bs{batch_size}_lr{learning_rate:.0e}_seed{seed}.pth'
    print(f"模型将保存为: {best_model_path}")
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            if use_text:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                gender = batch['gender'].to(device)
                age = batch['age'].to(device)
                outputs = model(images, input_ids, attention_mask, gender, age)
            else:
                outputs = model(images)
                
            loss = loss_nn(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            
            # 计算训练指标
            pred_probs = outputs.sigmoid().view(-1)  # 展平为一维
            labels_flat = labels.view(-1)  # 展平为一维
            
            train_auroc.update(pred_probs, labels_flat)
            train_acc.update(pred_probs, labels_flat)
            train_recall.update(pred_probs, labels_flat)
            
            total_train_step += 1
            epoch_train_loss += loss.item()
        
        # 计算训练指标
        train_auc = train_auroc.compute()
        train_accuracy = train_acc.compute()
        train_recall_value = train_recall.compute()
        
        # 记录训练指标
        train_losses.append(epoch_train_loss / len(train_loader))
        train_aucs.append(train_auc.item())
        train_accs.append(train_accuracy.item())
        train_recalls.append(train_recall_value.item())
        
        # 重置训练指标
        train_auroc.reset()
        train_acc.reset()
        train_recall.reset()
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                if use_text:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    gender = batch['gender'].to(device)
                    age = batch['age'].to(device)
                    outputs = model(images, input_ids, attention_mask, gender, age)
                else:
                    outputs = model(images)
                    
                loss = loss_nn(outputs, labels.unsqueeze(1).float())
                epoch_val_loss += loss.item() * images.size(0)  # 乘以batch size获得总loss
                
                # 计算验证指标
                pred_probs = outputs.sigmoid().view(-1)  # 展平为一维
                labels_flat = labels.view(-1)  # 展平为一维
                
                val_auroc.update(pred_probs, labels_flat)
                val_acc.update(pred_probs, labels_flat)
                val_recall.update(pred_probs, labels_flat)
                
                total_val_step += 1
        
        # 计算验证指标
        val_auc = val_auroc.compute()
        val_accuracy = val_acc.compute()
        val_recall_value = val_recall.compute()
        
        # 记录验证指标
        val_losses.append(epoch_val_loss / len(val_dataset))
        val_aucs.append(val_auc.item())
        val_accs.append(val_accuracy.item())
        val_recalls.append(val_recall_value.item())
        
        # 更新学习率
        scheduler.step()
        
        # 检查是否保存最佳模型
        if val_accuracy.item() > best_val_acc:
            best_val_acc = val_accuracy.item()
            patience_counter = 0
            
            # 保存模型
            try:
                os.makedirs('model_save', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_losses[-1],
                    'val_loss': val_losses[-1],
                    'val_auc': val_auc.item(),
                    'val_acc': val_accuracy.item(),
                    'val_recall': val_recall_value.item(),
                }, best_model_path)
                print(f"保存最佳模型，验证集ACC: {val_accuracy.item():.4f}")
            except Exception as e:
                print(f"保存模型时发生错误: {str(e)}")
        else:
            patience_counter += 1
            
        # 重置验证指标
        val_auroc.reset()
        val_acc.reset()
        val_recall.reset()
        
        # 打印当前轮次的指标
        print(f"Epoch: {epoch}")
        print(f"Train - Loss: {train_losses[-1]:.4f}, AUC: {train_aucs[-1]:.4f}, Acc: {train_accs[-1]:.4f}, Recall: {train_recalls[-1]:.4f}")
        print(f"Val   - Loss: {val_losses[-1]:.4f}, AUC: {val_aucs[-1]:.4f}, Acc: {val_accs[-1]:.4f}, Recall: {val_recalls[-1]:.4f}")
        print("-" * 60)
        
        # 早停检查
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 在测试集上评估模型
    evaluate_model(model, test_loader, device, best_model_path, batch_size, learning_rate, seed, train_losses, val_losses, train_aucs, val_aucs, train_accs, val_accs)

# 入口点
if __name__ == "__main__":
    main() 