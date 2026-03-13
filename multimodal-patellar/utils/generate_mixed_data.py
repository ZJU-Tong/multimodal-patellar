import pandas as pd
import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='生成混合数据集')
    parser.add_argument('--src_root', type=str, default='data', help='源数据根目录')
    parser.add_argument('--dest_root', type=str, default='data_mixed', help='目标数据根目录')
    parser.add_argument('--test_ratio', type=float, default=0.25, help='测试集比例')
    parser.add_argument('--test_num', type=int, default=None, help='测试集病人数量 (指定此参数时忽略 test_ratio)')
    parser.add_argument('--test_pos_num', type=int, default=None, help='外部验证集阳性病人数量 (指定此参数时需同时指定 test_neg_num，且忽略 test_ratio 和 test_num)')
    parser.add_argument('--test_neg_num', type=int, default=None, help='外部验证集阴性病人数量 (指定此参数时需同时指定 test_pos_num，且忽略 test_ratio 和 test_num)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Source Root: {args.src_root}")
    print(f"Dest Root: {args.dest_root}")
    
    if args.test_pos_num is not None and args.test_neg_num is not None:
        print(f"Test Pos Num: {args.test_pos_num}")
        print(f"Test Neg Num: {args.test_neg_num}")
        test_size = None # Not used in this mode
    elif args.test_num is not None:
        print(f"Test Num: {args.test_num}")
        test_size = args.test_num
    else:
        print(f"Test Ratio: {args.test_ratio}")
        test_size = args.test_ratio

    # 路径
    train_dir = os.path.join(args.src_root, '训练集')
    ext_dir = os.path.join(args.src_root, '外部验证集')
    
    # 检查源文件是否存在
    if not os.path.exists(os.path.join(train_dir, 'metadata.csv')):
        print(f"Error: {os.path.join(train_dir, 'metadata.csv')} does not exist.")
        return
    if not os.path.exists(os.path.join(ext_dir, 'metadata.csv')):
        print(f"Error: {os.path.join(ext_dir, 'metadata.csv')} does not exist.")
        return

    # 读取 metadata
    print("Reading metadata...")
    train_df = pd.read_csv(os.path.join(train_dir, 'metadata.csv'))
    ext_df = pd.read_csv(os.path.join(ext_dir, 'metadata.csv'))
    
    # 合并 DataFrame
    all_df = pd.concat([train_df, ext_df], ignore_index=True)
    print(f"Total samples: {len(all_df)}")
    
    # 按病人分组
    patient_groups = defaultdict(list)
    for idx, row in all_df.iterrows():
        pid = str(row['patient_id'])
        patient_groups[pid].append(row)
        
    patient_ids = list(patient_groups.keys())
    patient_labels = []
    for pid in patient_ids:
        # 取第一个样本的 label
        label = int(patient_groups[pid][0]['label'])
        patient_labels.append(label)
        
    print(f"Total patients: {len(patient_ids)}")
    
    # 划分
    train_pids = []
    test_pids = []
    
    if args.test_pos_num is not None and args.test_neg_num is not None:
        # 分别处理阳性和阴性
        pos_pids = [pid for pid, label in zip(patient_ids, patient_labels) if label == 1]
        neg_pids = [pid for pid, label in zip(patient_ids, patient_labels) if label == 0]
        
        print(f"Total Pos: {len(pos_pids)}, Total Neg: {len(neg_pids)}")
        
        if args.test_pos_num > len(pos_pids):
            raise ValueError(f"Requested {args.test_pos_num} positive patients, but only {len(pos_pids)} available.")
        if args.test_neg_num > len(neg_pids):
            raise ValueError(f"Requested {args.test_neg_num} negative patients, but only {len(neg_pids)} available.")
            
        # 随机抽取
        test_pos = random.sample(pos_pids, args.test_pos_num)
        test_neg = random.sample(neg_pids, args.test_neg_num)
        
        test_pids = test_pos + test_neg
        
        # 剩余作为训练集
        test_pids_set = set(test_pids)
        train_pids = [pid for pid in patient_ids if pid not in test_pids_set]
        
    else:
        # 检查类别数量
        unique_labels = set(patient_labels)
        stratify_labels = patient_labels if len(unique_labels) > 1 else None
        
        train_pids, test_pids = train_test_split(
            patient_ids, 
            test_size=test_size, 
            stratify=stratify_labels, 
            random_state=args.seed
        )
    
    train_pids_set = set(train_pids)
    test_pids_set = set(test_pids)
    
    print(f"Train patients: {len(train_pids)}")
    print(f"Test patients: {len(test_pids)}")
    
    # 准备目标目录
    dest_train_dir = os.path.join(args.dest_root, '训练集')
    dest_test_dir = os.path.join(args.dest_root, '外部验证集')
    
    if os.path.exists(args.dest_root):
        print(f"Removing existing {args.dest_root}...")
        shutil.rmtree(args.dest_root)
    
    os.makedirs(os.path.join(dest_train_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(dest_train_dir, 'negative'), exist_ok=True)
    os.makedirs(os.path.join(dest_test_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(dest_test_dir, 'negative'), exist_ok=True)
    
    # 处理数据
    new_train_rows = []
    new_test_rows = []
    
    # 统计
    processed_count = 0
    copy_count = 0
    
    print("Processing and copying files...")
    for pid in patient_ids:
        rows = patient_groups[pid]
        is_train = pid in train_pids_set
        dest_base = dest_train_dir if is_train else dest_test_dir
        split_name = '训练集' if is_train else '外部验证集'
        
        for row in rows:
            new_row = row.copy()
            new_row['split'] = split_name
            
            # 处理图像
            old_images_str = row['images']
            if pd.isna(old_images_str):
                continue
                
            old_paths = [p.strip() for p in old_images_str.split(';')]
            new_paths = []
            
            for old_rel_path in old_paths:
                # old_rel_path 如 "训练集/positive/xxx.png"
                # 真实源路径
                src_file = os.path.join(args.src_root, old_rel_path)
                
                if not os.path.exists(src_file):
                    # 尝试处理路径不匹配的情况 (例如 windows/linux 分隔符问题，或者相对路径问题)
                    # 这里的 old_rel_path 假设是相对于 src_root 的
                    print(f"Warning: File not found {src_file}")
                    continue
                    
                filename = os.path.basename(src_file)
                label_str = 'positive' if row['label'] == 1 else 'negative'
                
                # 构造新路径: split_name/label_str/filename
                # 例如 "训练集/positive/xxx.png"
                new_rel_path = os.path.join(split_name, label_str, filename)
                new_full_path = os.path.join(args.dest_root, new_rel_path)
                
                # 复制文件
                shutil.copy2(src_file, new_full_path)
                new_paths.append(new_rel_path)
                copy_count += 1
            
            new_row['images'] = ';'.join(new_paths)
            
            if is_train:
                new_train_rows.append(new_row)
            else:
                new_test_rows.append(new_row)
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count}/{len(patient_ids)} patients...")
            
    # 保存 metadata
    train_csv_path = os.path.join(dest_train_dir, 'metadata.csv')
    test_csv_path = os.path.join(dest_test_dir, 'metadata.csv')
    
    # 确保列顺序一致
    columns = list(all_df.columns)
    
    pd.DataFrame(new_train_rows, columns=columns).to_csv(train_csv_path, index=False)
    pd.DataFrame(new_test_rows, columns=columns).to_csv(test_csv_path, index=False)
    
    print(f"Finished! Copied {copy_count} images.")
    print(f"Generated data in {args.dest_root}")

if __name__ == '__main__':
    main()
