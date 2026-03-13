import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
import os
import glob
from tqdm import tqdm
sys.path.append('../')  # 添加项目根目录到Python路径
from model import VGG11,ResNet18,ResNet18_dropout,ImprovedResNet,EfficientNet,CTTransformer

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义图像预处理
aug = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_cam(model, image_tensor, target_layer='layer3'):
    """
    生成CAM热力图
    """
    model.eval()
    
    # 注册hook来获取特征图
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    # 检查模型类型并获取目标层
    if isinstance(model, VGG11):
        target = model.vgg.features[20]
        fc_weights = model.vgg.classifier[-1].weight.data
    elif isinstance(model, ResNet18) or isinstance(model, ResNet18_dropout) or isinstance(model, ImprovedResNet):
        target = getattr(model.resnet18, target_layer)
        fc_weights = model.resnet18.fc[-1].weight.data
    elif isinstance(model, EfficientNet):
        # 使用最后一个卷积层
        target = model.efficient_net.features[-1]  # 改回最后一层
        # 获取分类器权重并进行归一化
        fc_weights = torch.nn.functional.softmax(model.classifier[-1].weight.data, dim=1)
    elif isinstance(model, CTTransformer):
        # 使用EfficientNet的最后一个卷积层
        target = model.efficient_net.features[-4]
        # 获取分类器权重并进行归一化
        fc_weights = torch.nn.functional.softmax(model.classifier[-1].weight.data, dim=1)
    else:
        raise ValueError("Unsupported model type")
    
    handle = target.register_forward_hook(hook_fn)
    
    # 前向传播
    if isinstance(model, CTTransformer):
        # CTTransformer期望输入形状为 [batch_size, num_images, channels, height, width]
        # 我们这里只有一张图像，所以添加一个额外的维度
        image_tensor = image_tensor.unsqueeze(1)  # [1, 1, 3, 224, 224]
        output = model(image_tensor)
    else:
        output = model(image_tensor)
    
    pred = output.sigmoid()
    
    # 生成CAM
    feature = feature_maps[0].squeeze()  # [C, H, W]
    cam = torch.zeros((feature.shape[1], feature.shape[2]), device=device)
    
    # 使用归一化的权重
    weights = fc_weights.squeeze()
    weights = torch.abs(weights)  # 使用绝对值
    weights = weights / weights.sum()  # 归一化权重
    
    # 加权求和
    for i in range(min(len(weights), feature.shape[0])):
        cam += weights[i] * feature[i]
    
    # 移除hook
    handle.remove()
    
    # 处理CAM
    cam = cam.cpu().detach().numpy()
    cam = np.maximum(cam, 0)  # ReLU
    
    # 使用更大的核进行高斯模糊
    cam = cv2.GaussianBlur(cam, (11, 11), 0)  # 增加核大小
    cam = cv2.resize(cam, (224, 224))
    
    # 增强对比度
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化
    cam = np.power(cam, 2)  # 平方增强对比度
    
    return cam, pred.item()

def visualize_cam(image_path, model, save_path=None):
    """
    可视化CAM并保存结果
    """
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = aug(image).unsqueeze(0).to(device)
    
    # 获取CAM
    cam, pred = get_cam(model, input_tensor)
    
    # 将原始图像转换为numpy数组
    orig_img = np.array(image.resize((224, 224)))
    
    # 创建热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 调整透明度叠加
    alpha = 0.7  # 增加原始图像的权重
    superimposed = cv2.addWeighted(orig_img, alpha, heatmap, (1-alpha), 0)
    
    # 显示结果
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(orig_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cam, cmap='jet')
    plt.title('CAM')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(superimposed)
    plt.title(f'Overlay (Pred: {pred:.3f})')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def process_all_images(input_dir, output_dir, model):
    """
    处理指定目录下所有图片并生成CAM热力图
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出热力图目录
        model: 模型
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有子目录
    for root, dirs, files in os.walk(input_dir):
        # 对每个文件进行处理
        for file in tqdm(files, desc=f"处理目录: {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 构建完整的输入路径
                input_path = os.path.join(root, file)
                
                # 构建对应的输出路径 - 保持相同的目录结构
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                output_dir_path = os.path.dirname(output_path)
                
                # 确保输出目录存在
                os.makedirs(output_dir_path, exist_ok=True)
                
                try:
                    # 生成并保存热力图
                    visualize_cam(input_path, model, save_path=output_path)
                    print(f"已处理: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"处理 {input_path} 时出错: {str(e)}")

if __name__ == '__main__':
    # 加载模型
    model = CTTransformer().to(device)
    
    # 加载最佳模型
    checkpoint = torch.load('../model_save/best_CTTransformer.pth')
    # 移除权重键中的"0."前缀
    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('0.'):
            new_key = k[2:]  # 移除"0."前缀
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    
    # 批量处理所有图片
    # 选择数据集目录
    input_dir = '../data/最终训练集'  # 包含 positive 和 negative 子目录
    output_dir = './cam_results/batch_results'
    
    print(f"开始处理目录 {input_dir} 下的所有图片...")
    process_all_images(input_dir, output_dir, model)
    print("处理完成！")