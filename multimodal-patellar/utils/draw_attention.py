import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import os
import sys
sys.path.append('../')
from model import CTTransformer_text

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_text_attention(tokens, attention_weights, save_path=None, layer_idx=-1, head_idx=0):
    """
    绘制文本注意力热力图
    
    参数:
        tokens: 分词后的文本列表
        attention_weights: 注意力权重张量 (来自BERT的attention输出)
        save_path: 保存图像的路径
        layer_idx: 要可视化的transformer层索引 (-1表示最后一层)
        head_idx: 要可视化的注意力头索引
    """
    # 获取注意力矩阵并确保维度匹配
    attention_matrix = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()
    
    # 只取实际词的数量
    num_tokens = len(tokens)
    attention_matrix = attention_matrix[:num_tokens, :num_tokens]
    
    # 对每行进行softmax归一化，确保每行和为1
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
    
    # 创建图像
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图
    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        square=True,
        annot=True,  # 显示具体数值
        fmt='.2f',   # 数值格式为2位小数
        cbar_kws={'label': 'Attention Weight'}  # 添加颜色条标签
    )
    
    # 设置标题和标签
    plt.title(f'Attention Weight Visualization')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    
    # 调整布局
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_attention_visualization(text, attention_weights, save_path, max_tokens=50):
    """
    创建带有颜色编码的文本注意力可视化
    """
    # 确保注意力权重是一维numpy数组
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # 确保是一维数组
    attention_weights = attention_weights.flatten()
    
    # 对注意力权重进行归一化到[0,1]区间
    min_val = attention_weights.min()
    max_val = attention_weights.max()
    normalized_attention = (attention_weights - min_val) / (max_val - min_val)
    
    # 创建背景图像
    img_width = 1200
    img_height = 200
    background = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(background)
    
    # 加载中文字体
    try:
        font = ImageFont.truetype("SimHei", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # 绘制文本和注意力高亮
    x, y = 10, 10
    for i, (token, attn) in enumerate(zip(text[:max_tokens], normalized_attention[:max_tokens])):
        # 计算颜色强度（红色）
        color = int(255 * (1 - float(attn)))  # 确保转换为float
        text_color = (255, color, color)
        
        # 绘制文本
        draw.text((x, y), token, font=font, fill=text_color)
        
        # 更新位置
        bbox = font.getbbox(token)
        text_width = bbox[2] - bbox[0]
        x += text_width + 5
        
        # 换行处理
        if x > img_width - 100:
            x = 10
            y += 40
    
    # 保存图像
    background.save(save_path)

def visualize_text_attention(model, text_input, save_dir='attention_viz', device='cuda', layers_to_visualize=None):
    """
    生成文本注意力的可视化，并计算每个token的整体重要性得分
    
    参数:
        model: 训练好的模型
        text_input: 已分词的文本
        save_dir: 保存可视化的目录
        device: 设备
        layers_to_visualize: 要可视化的层索引列表，None表示所有层
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 处理已分词的文本
    tokens = text_input.split('/')
    num_tokens = len(tokens)
    
    # 获取注意力权重
    model.eval()
    with torch.no_grad():
        attention_weights = model.get_text_attention(tokens)
    
    # 如果没有指定层，则可视化所有层
    if layers_to_visualize is None:
        layers_to_visualize = list(range(len(attention_weights)))
    
    # 为每一层创建子目录
    for layer_idx in layers_to_visualize:
        layer_dir = os.path.join(save_dir, f'layer_{layer_idx}')
        os.makedirs(layer_dir, exist_ok=True)
        
        # 获取当前层的注意力
        layer_attention = attention_weights[layer_idx]
        
        # 为每个注意力头生成可视化
        for head_idx in range(layer_attention.size(1)):
            save_path = os.path.join(layer_dir, f'attention_head{head_idx}.png')
            plot_text_attention(tokens, attention_weights, save_path, layer_idx, head_idx)
        
        # 计算当前层的token重要性得分
        calculate_token_importance(tokens, attention_weights, layer_idx, save_dir=layer_dir)
    
    # 计算最后一层的token重要性作为总体结果
    token_importance = calculate_token_importance(tokens, attention_weights, -1, save_dir=save_dir)
    
    return token_importance

def calculate_token_importance(tokens, attention_weights, layer_idx, save_dir):
    """计算指定层的token重要性得分"""
    num_tokens = len(tokens)
    layer_attention = attention_weights[layer_idx]
    
    # 提取所有注意力头的注意力矩阵
    all_head_attentions = []
    for head_idx in range(layer_attention.size(1)):
        head_attention = attention_weights[layer_idx][0, head_idx].detach().cpu()
        head_attention = head_attention[:num_tokens, :num_tokens]
        all_head_attentions.append(head_attention)
    
    # 计算平均注意力矩阵
    stacked_attentions = torch.stack(all_head_attentions)
    mean_attention = torch.mean(stacked_attentions, dim=0)
    
    # 计算两种重要性指标
    column_sum = torch.sum(mean_attention, dim=0)  # 被其他词关注的程度
    row_sum = torch.sum(mean_attention, dim=1)     # 关注其他词的程度
    
    # 综合两种重要性并大幅增加对比度
    # 使用不平等加权，偏重"被关注度"
    token_importance_combined = (column_sum * 2.0 + row_sum * 0.5) / 2.5
    
    # 应用指数函数来放大差异 - 平方放大差异
    token_importance_powered = token_importance_combined ** 2
    
    # 使用更高温度参数的softmax增强对比度
    token_importance_norm = torch.nn.functional.softmax(token_importance_powered * 20, dim=0)
    
    # 获取排序后的索引和数据
    sorted_indices = torch.argsort(token_importance_norm, descending=True)
    sorted_tokens = [tokens[idx] for idx in sorted_indices]
    sorted_importance = token_importance_norm[sorted_indices].numpy()
    
    # 使用更高对比度的颜色映射
    cmap = plt.cm.Reds
    colors = cmap(np.power(sorted_importance/max(sorted_importance), 0.5))  # 使用幂函数压缩颜色范围
    
    # 创建水平条形图
    plt.figure(figsize=(10, 6), dpi=300)
    bars = plt.barh(range(len(sorted_tokens)), sorted_importance, color=colors, 
                   alpha=0.85, height=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    for i, (v, bar) in enumerate(zip(sorted_importance, bars)):
        label_size = 9 + 3 * (v/max(sorted_importance))  # 根据重要性动态调整标签大小
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', 
                fontweight='bold', fontsize=label_size, color='black')
    
    # 设置轴标签和标题
    plt.yticks(range(len(sorted_tokens)), sorted_tokens, fontsize=10)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title(f'Token Importance', fontsize=14, fontweight='bold')
    
    # 美化图表
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.xlim(0, max(sorted_importance) * 1.15)
    
    # 添加平均值和中位数参考线
    mean_imp = sorted_importance.mean()
    median_imp = np.median(sorted_importance)
    plt.axvline(x=mean_imp, color='blue', linestyle='--', alpha=0.5, 
               label=f'Mean: {mean_imp:.3f}')
    plt.axvline(x=median_imp, color='green', linestyle=':', alpha=0.5,
               label=f'Median: {median_imp:.3f}')
    plt.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'token_importance_layer{layer_idx}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 同时生成垂直柱状图，也使用增强的对比度
    plt.figure(figsize=(12, 6), dpi=300)
    bars = plt.bar(range(len(sorted_tokens)), sorted_importance, color=colors, 
                  alpha=0.85, width=0.8, edgecolor='black', linewidth=0.5)
    
    # 在条形顶部添加值标签
    for i, v in enumerate(sorted_importance):
        label_size = 9 + 3 * (v/max(sorted_importance))
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', 
                fontsize=label_size, fontweight='bold')
    
    # 美化垂直图表
    plt.xticks(range(len(sorted_tokens)), sorted_tokens, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Tokens', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title(f'Token Importance', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'token_importance_vertical_layer{layer_idx}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成带颜色的文本可视化 - 使用增强对比度的权重
    save_path = os.path.join(save_dir, f'text_attention_layer{layer_idx}.png')
    create_attention_visualization(tokens, token_importance_norm, save_path)
    
    # 输出排序后的重要性得分
    print(f"\n层 {layer_idx} 的Token重要性得分:")
    for token, score in zip(sorted_tokens, sorted_importance):
        print(f"{token}: {score:.4f}")
    
    return token_importance_norm.numpy()

def load_model(model_path, device='cuda'):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型权重文件路径
        device: 运行设备
    返回:
        加载好的模型
    """
    # 初始化模型
    model = CTTransformer_text(use_text=True)
    model = model.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model_path = '../model_save/best_multi_model.pth'
    model = load_model(model_path, device)
    
    # 已分词的测试文本
    text = '右侧/髌骨/外推/恐惧/试验/阳性/膝关节/间隙/压痛'
    # text = '右/膝关节/浮髌/试验/阳性/髌骨/研磨/实验/阳性'
    
    # 生成可视化
    visualize_text_attention(model, text, save_dir='attention_visualizations', device=device, layers_to_visualize=[9, 10, 11])
    

