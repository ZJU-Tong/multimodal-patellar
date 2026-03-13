import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule
import torch
from transformers import BertModel, BertTokenizer
import math
import torch.nn.functional as F
import jieba

class EfficientNet(LightningModule):
    def __init__(self, model_name='efficientnet_b0', dropout_rate=0.5):
        super(EfficientNet, self).__init__()
        
        # 加载预训练的EfficientNet模型
        self.efficient_net = getattr(models, model_name)(weights='DEFAULT')
        
        # 获取最后一层的输入特征数
        if hasattr(self.efficient_net, 'classifier'):
            in_features = self.efficient_net.classifier[-1].in_features
        else:
            in_features = self.efficient_net.fc.in_features
            
        # 冻结主干网络的前面层
        for param in list(self.efficient_net.parameters())[:-20]:
            param.requires_grad = False
            
        # 改进分类器结构
        self.classifier = nn.Sequential(    
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(64, 1)
        )
        
        # 替换原始分类器
        if hasattr(self.efficient_net, 'classifier'):
            self.efficient_net.classifier = nn.Identity()
        else:
            self.efficient_net.fc = nn.Identity()
            
    def forward(self, x):
        x = self.efficient_net(x)
        x = self.classifier(x)
        return x

class EfficientNet_BERT(LightningModule):
    def __init__(self, model_name='efficientnet_b0', dropout_rate=0.5):
        super(EfficientNet_BERT, self).__init__()
        
        # 图像特征提取器 (使用EfficientNet)
        self.efficient_net = getattr(models, model_name)(weights='DEFAULT')
        
        # 获取EfficientNet特征维度
        if hasattr(self.efficient_net, 'classifier'):
            in_features = self.efficient_net.classifier[-1].in_features
            self.efficient_net.classifier = nn.Identity()
        else:
            in_features = self.efficient_net.fc.in_features
            self.efficient_net.fc = nn.Identity()
        
        # 文本特征提取器 (使用BERT)
        self.text_encoder = BertModel.from_pretrained('/home/users/liutong/dl_learning/patellar/bert-base-chinese')
        
        # 特征融合和分类器
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 768, 512),  # EfficientNet特征 + 768(BERT特征)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, image, input_ids, attention_mask):
        # 提取图像特征
        image_features = self.efficient_net(image)
        
        # 提取文本特征
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        
        # 特征融合
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # 分类
        output = self.classifier(combined_features)
        return output

class ResNet18_BERT(LightningModule):
    def __init__(self, dropout_rate=0.5):
        super(ResNet18_BERT, self).__init__()
        
        # 图像特征提取器 (使用ResNet18)
        self.resnet18 = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.resnet18.fc = nn.Identity()  # 移除最后的全连接层
        
        # 文本特征提取器 (使用BERT)
        self.text_encoder = BertModel.from_pretrained('/home/users/liutong/dl_learning/patellar/bert-base-chinese')
        
        # 特征融合和分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),  # 512(ResNet特征) + 768(BERT特征)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1)
        )
        
    def forward(self, image, input_ids, attention_mask):
        # 提取图像特征
        image_features = self.resnet18(image)
        
        # 提取文本特征
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        
        # 特征融合
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # 分类
        output = self.classifier(combined_features)
        return output

class MobileNet(LightningModule):
    def __init__(self, dropout_rate=0.5, freeze_backbone=True):
        super(MobileNet, self).__init__()
        
        # 加载预训练的MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False
        
        # 获取最后一层的输入特征数
        last_channel = self.mobilenet.classifier[-1].in_features
        
        # 修改分类器部分，增加正则化
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(last_channel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        # 添加权重初始化
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.mobilenet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 添加训练时的随机噪声增强
        if self.training:
            x = x + torch.randn_like(x) * 0.1
        return self.mobilenet(x)
    
    def unfreeze_backbone(self, start_layer=14):
        """解冻backbone的后几层，允许微调"""
        # MobileNetV2有18个特征层，这里默认解冻最后4层
        for i, param in enumerate(self.mobilenet.features.parameters()):
            if i > start_layer * 3:  # 每层有3个参数（权重、偏置、BN）
                param.requires_grad = True
                
class CTTransformer_text(LightningModule):
    def __init__(self, model_name='efficientnet_b0', nhead=8, num_layers=2, dropout_rate=0.4, use_text=False):
        super(CTTransformer_text, self).__init__()
        
        # CNN特征提取器
        self.efficient_net = getattr(models, model_name)(weights='DEFAULT')
        
        # 获取特征维度
        if hasattr(self.efficient_net, 'classifier'):
            in_features = self.efficient_net.classifier[-1].in_features
            self.efficient_net.classifier = nn.Identity()
        else:
            in_features = self.efficient_net.fc.in_features
            self.efficient_net.fc = nn.Identity()
            
        # 冻结CNN主干网络
        for param in list(self.efficient_net.parameters())[:-20]:
            param.requires_grad = False
            
        # 特征降维
        self.image_proj = nn.Linear(in_features, 256)
        
        # 是否使用文本特征
        self.use_text = use_text
        if use_text:
            # 文本特征提取器 (使用BERT)
            self.text_encoder = BertModel.from_pretrained('/home/users/liutong/dl_learning/patellar/bert-base-chinese')
            self.text_proj = nn.Linear(768, 256)  # BERT输出维度是768
            
            # 冻结BERT参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # 性别和年龄特征的处理
        self.gender_embedding = nn.Embedding(3, 16)  # 假设性别有3种可能：男(1)、女(2)和未知(0)
        self.age_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # 为单个CT图像定义分类器
        self.image_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        # 如果使用文本特征，需要另一个分类器
        if use_text:
            # 文本和人口统计学特征分类器
            self.text_demo_classifier = nn.Sequential(
                nn.Linear(256 + 32, 128),  # 256(文本) + 32(性别和年龄)
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )
        
    def forward(self, images, input_ids=None, attention_mask=None, gender=None, age=None, return_attention=False):
        # 图像特征提取
        batch_size, num_images, c, h, w = images.shape
        images = images.view(batch_size * num_images, c, h, w)
        image_features = self.efficient_net(images)
        image_features = self.image_proj(image_features)
        image_features = image_features.view(batch_size, num_images, -1)  # (batch, num_images, 256)
        
        # 处理人口统计学特征
        if gender is not None and age is not None:
            # 处理性别特征
            gender_features = self.gender_embedding(gender)  # (batch, 16)
            
            # 处理年龄特征
            age = age.unsqueeze(1)  # 添加特征维度 (batch, 1)
            age_features = self.age_proj(age)  # (batch, 16)
            
            # 合并人口统计学特征
            demographic_features = torch.cat([gender_features, age_features], dim=1)  # (batch, 32)
        
        # 对每张CT图像分别进行预测
        image_outputs = []
        for i in range(num_images):
            # 获取当前图像的特征
            current_image_features = image_features[:, i, :]  # (batch, 256)
            # 对单张图像进行分类
            image_output = self.image_classifier(current_image_features)
            image_outputs.append(image_output)
        
        # 将所有图像的预测结果堆叠起来
        stacked_image_outputs = torch.stack(image_outputs, dim=1)  # (batch, num_images, 1)
        
        # 使用平均投票法得到最终结果
        image_predictions = stacked_image_outputs.mean(dim=1)  # (batch, 1)
        
        # 如果使用文本特征，需要结合文本和人口统计学特征进行预测
        if self.use_text and input_ids is not None and attention_mask is not None:
            # 获取文本特征
            if not return_attention:
                text_output = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).pooler_output
                text_features = self.text_proj(text_output)
            else:
                with torch.no_grad():
                    text_outputs = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                    text_features = self.text_proj(text_outputs.pooler_output)
                    attention_weights = text_outputs.attentions
            
            # 如果有人口统计学特征，结合文本和人口统计学特征
            if gender is not None and age is not None:
                text_demo_features = torch.cat([text_features, demographic_features], dim=1)
                text_demo_prediction = self.text_demo_classifier(text_demo_features)
                
                # 合并图像预测和文本预测 (简单平均)
                final_prediction = (image_predictions + text_demo_prediction) / 2
            else:
                # 没有人口统计学特征，只使用图像预测
                final_prediction = image_predictions
                
            if return_attention:
                return final_prediction, attention_weights
            return final_prediction
        else:
            # 不使用文本特征，直接返回图像预测
            return image_predictions

    def get_text_attention(self, text_tokens):
        """获取文本的注意力权重"""
        self.eval()
        with torch.no_grad():
            # 将文本转换为BERT的输入格式
            tokenizer = BertTokenizer.from_pretrained('/home/users/liutong/dl_learning/patellar/bert-base-chinese')
            inputs = tokenizer(text_tokens, is_split_into_words=True, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(next(self.parameters()).device)
            attention_mask = inputs['attention_mask'].to(next(self.parameters()).device)
            
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            return text_outputs.attentions

class CTTransformer(LightningModule):
    def __init__(self, model_name='efficientnet_b0', nhead=8, num_layers=2, dropout_rate=0.5):
        super(CTTransformer, self).__init__()
        
        # CNN特征提取器
        self.efficient_net = getattr(models, model_name)(weights='DEFAULT')
        
        # 获取特征维度
        if hasattr(self.efficient_net, 'classifier'):
            in_features = self.efficient_net.classifier[-1].in_features
            self.efficient_net.classifier = nn.Identity()
        else:
            in_features = self.efficient_net.fc.in_features
            self.efficient_net.fc = nn.Identity()
            
        # 冻结CNN主干网络
        for param in list(self.efficient_net.parameters())[:-20]:
            param.requires_grad = False
            
        # 特征降维，减少计算量
        self.feature_proj = nn.Linear(in_features, 256)
        
        # 单个CT图像的分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, return_attention=False):
        batch_size, num_images, c, h, w = x.shape
        x = x.view(batch_size * num_images, c, h, w)
        features = self.efficient_net(x)  # (batch*num_images, in_features)
        features = self.feature_proj(features)  # (batch*num_images, 256)
        features = features.view(batch_size, num_images, -1)  # (batch, num_images, 256)
        
        # 对每张CT图像分别进行预测
        image_outputs = []
        for i in range(num_images):
            # 获取当前图像的特征
            current_image_features = features[:, i, :]  # (batch, 256)
            # 对单张图像进行分类
            image_output = self.classifier(current_image_features)
            image_outputs.append(image_output)
            
        # 将所有图像的预测结果堆叠起来
        stacked_image_outputs = torch.stack(image_outputs, dim=1)  # (batch, num_images, 1)
        
        # 使用平均投票法得到最终结果
        image_predictions = stacked_image_outputs.mean(dim=1)  # (batch, 1)
        
        # 不再返回注意力权重，因为没有使用Transformer
        if return_attention:
            # 为了兼容性，返回一个空的注意力权重
            dummy_attention = torch.zeros((batch_size, num_images, num_images), device=x.device)
            return image_predictions, dummy_attention
            
        return image_predictions

class AlexNet(LightningModule):
    def __init__(self, dropout_rate=0.5, freeze_backbone=True):
        super(AlexNet, self).__init__()
        
        # 加载预训练的AlexNet
        self.alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.alexnet.features.parameters():
                param.requires_grad = False
        
        # 修改分类器部分，增加正则化
        self.alexnet.classifier = nn.Sequential(
            nn.Linear(9216, 1024),  # AlexNet的最后一个展平层特征数为9216
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, 1)
        )
        
        # 添加权重初始化
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.alexnet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 添加训练时的随机噪声增强
        if self.training:
            x = x + torch.randn_like(x) * 0.1
        return self.alexnet(x)

class VGG11_Improved(LightningModule):
    def __init__(self, dropout_rate=0.5, freeze_backbone=True):
        super(VGG11_Improved, self).__init__()
        
        # 加载预训练的VGG11
        self.vgg11 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.vgg11.features.parameters():
                param.requires_grad = False
        
        # 修改分类器部分，增加正则化
        self.vgg11.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, 1)
        )
        
        # 添加权重初始化
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.vgg11.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 添加训练时的随机噪声增强
        if self.training:
            x = x + torch.randn_like(x) * 0.1
        return self.vgg11(x)

class ResNet34_Improved(LightningModule):
    def __init__(self, dropout_rate=0.5, freeze_backbone=True):
        super(ResNet34_Improved, self).__init__()
        
        # 加载预训练的ResNet34
        self.resnet34 = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in list(self.resnet34.parameters())[:-20]:
                param.requires_grad = False
        
        # 修改分类器部分，增加正则化
        self.resnet34.fc = nn.Sequential(
            nn.Linear(self.resnet34.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, 1)
        )
        
        # 添加权重初始化
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.resnet34.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 添加训练时的随机噪声增强
        if self.training:
            x = x + torch.randn_like(x) * 0.1
        return self.resnet34(x)

class DenseNet_Improved(LightningModule):
    def __init__(self, dropout_rate=0.5, freeze_backbone=True, model_name='densenet121'):
        super(DenseNet_Improved, self).__init__()
        
        # 加载预训练的DenseNet模型
        self.densenet = getattr(models, model_name)(weights='DEFAULT')
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in list(self.densenet.parameters())[:-20]:
                param.requires_grad = False
        
        # 获取最后分类器的输入特征数
        in_features = self.densenet.classifier.in_features
        
        # 修改分类器部分，增加正则化
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, 1)
        )
        
        # 添加权重初始化
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.densenet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 添加训练时的随机噪声增强
        if self.training:
            x = x + torch.randn_like(x) * 0.1
        return self.densenet(x)

class MobileNetV2_Improved(LightningModule):
    def __init__(self, dropout_rate=0.5, freeze_backbone=True):
        super(MobileNetV2_Improved, self).__init__()
        
        # 加载预训练的MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False
        
        # 获取最后一层的输入特征数
        last_channel = self.mobilenet.classifier[-1].in_features
        
        # 修改分类器部分，增加正则化
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(last_channel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, 1)
        )
        
        # 添加权重初始化
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.mobilenet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 添加训练时的随机噪声增强
        if self.training:
            x = x + torch.randn_like(x) * 0.1
        return self.mobilenet(x)

class RegNet_Improved(LightningModule):
    def __init__(self, dropout_rate=0.5, freeze_backbone=True, model_name='regnet_y_400mf'):
        super(RegNet_Improved, self).__init__()
        
        # 加载预训练的RegNet模型
        self.regnet = getattr(models, model_name)(weights='DEFAULT')
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in list(self.regnet.parameters())[:-20]:
                param.requires_grad = False
        
        # 获取最后一层的输入特征数
        in_features = self.regnet.fc.in_features
        
        # 修改分类器部分，增加正则化
        self.regnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, 1)
        )
        
        # 添加权重初始化
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.regnet.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 添加训练时的随机噪声增强
        if self.training:
            x = x + torch.randn_like(x) * 0.1
        return self.regnet(x)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=768, hidden_dim=256, num_classes=1):
        super(TextClassifier, self).__init__()
        
        # 文本处理部分
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=4
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # 文本特征提取
        text_embeds = self.embedding(input_ids)
        
        # 注意力掩码处理
        padding_mask = (attention_mask == 0)
        text_features = self.transformer_encoder(text_embeds.transpose(0, 1), src_key_padding_mask=padding_mask)
        text_features = text_features.transpose(0, 1)
        
        # 获取序列的平均表示
        masked_features = text_features * attention_mask.unsqueeze(-1)
        sum_features = masked_features.sum(dim=1)
        seq_lengths = attention_mask.sum(dim=1, keepdim=True)
        seq_lengths = torch.clamp(seq_lengths, min=1)  # 防止除零
        mean_features = sum_features / seq_lengths
        
        # 分类
        output = self.classifier(mean_features)
        return output

class EfficientNet_text(LightningModule):
    def __init__(self, model_name='efficientnet_b0', dropout_rate=0.4, use_text=False):
        super(EfficientNet_text, self).__init__()
        
        # CNN特征提取器
        self.efficient_net = getattr(models, model_name)(weights='DEFAULT')
        
        # 获取特征维度
        if hasattr(self.efficient_net, 'classifier'):
            in_features = self.efficient_net.classifier[-1].in_features
            self.efficient_net.classifier = nn.Identity()
        else:
            in_features = self.efficient_net.fc.in_features
            self.efficient_net.fc = nn.Identity()
            
        # 冻结CNN主干网络
        for param in list(self.efficient_net.parameters())[:-20]:
            param.requires_grad = False
            
        # 特征降维
        self.image_proj = nn.Linear(in_features, 256)
        
        # 是否使用文本特征
        self.use_text = use_text
        if use_text:
            # 文本特征提取器 (使用BERT)
            self.text_encoder = BertModel.from_pretrained('/home/users/liutong/dl_learning/patellar/bert-base-chinese')
            self.text_proj = nn.Linear(768, 256)  # BERT输出维度是768
            
            # 冻结BERT参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # 性别和年龄特征的处理
        self.gender_embedding = nn.Embedding(3, 16)  # 假设性别有3种可能：男(1)、女(2)和未知(0)
        self.age_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # 为单个CT图像定义分类器
        self.image_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        # 如果使用文本特征，需要另一个分类器
        if use_text:
            # 文本和人口统计学特征分类器
            self.text_demo_classifier = nn.Sequential(
                nn.Linear(256 + 32, 128),  # 256(文本) + 32(性别和年龄)
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )
        
    def forward(self, images, input_ids=None, attention_mask=None, gender=None, age=None, return_attention=False):
        # 图像特征提取
        batch_size, num_images, c, h, w = images.shape
        images = images.view(batch_size * num_images, c, h, w)
        image_features = self.efficient_net(images)
        image_features = self.image_proj(image_features)
        image_features = image_features.view(batch_size, num_images, -1)  # (batch, num_images, 256)
        
        # 处理人口统计学特征
        if gender is not None and age is not None:
            # 处理性别特征
            gender_features = self.gender_embedding(gender)  # (batch, 16)
            
            # 处理年龄特征
            age = age.unsqueeze(1)  # 添加特征维度 (batch, 1)
            age_features = self.age_proj(age)  # (batch, 16)
            
            # 合并人口统计学特征
            demographic_features = torch.cat([gender_features, age_features], dim=1)  # (batch, 32)
        
        # 对每张CT图像分别进行预测
        image_outputs = []
        for i in range(num_images):
            # 获取当前图像的特征
            current_image_features = image_features[:, i, :]  # (batch, 256)
            # 对单张图像进行分类
            image_output = self.image_classifier(current_image_features)
            image_outputs.append(image_output)
        
        # 将所有图像的预测结果堆叠起来
        stacked_image_outputs = torch.stack(image_outputs, dim=1)  # (batch, num_images, 1)
        
        # 使用平均投票法得到最终结果
        image_predictions = stacked_image_outputs.mean(dim=1)  # (batch, 1)
        
        # 如果使用文本特征，需要结合文本和人口统计学特征进行预测
        if self.use_text and input_ids is not None and attention_mask is not None:
            # 获取文本特征
            if not return_attention:
                text_output = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).pooler_output
                text_features = self.text_proj(text_output)
            else:
                with torch.no_grad():
                    text_outputs = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                    text_features = self.text_proj(text_outputs.pooler_output)
                    attention_weights = text_outputs.attentions
            
            # 如果有人口统计学特征，结合文本和人口统计学特征
            if gender is not None and age is not None:
                text_demo_features = torch.cat([text_features, demographic_features], dim=1)
                text_demo_prediction = self.text_demo_classifier(text_demo_features)
                
                # 合并图像预测和文本预测 (简单平均)
                final_prediction = (image_predictions + text_demo_prediction) / 2
            else:
                # 没有人口统计学特征，只使用图像预测
                final_prediction = image_predictions
                
            if return_attention:
                return final_prediction, attention_weights
            return final_prediction
        else:
            # 不使用文本特征，直接返回图像预测
            return image_predictions

    def get_text_attention(self, text_tokens):
        """获取文本的注意力权重"""
        self.eval()
        with torch.no_grad():
            # 将文本转换为BERT的输入格式
            tokenizer = BertTokenizer.from_pretrained('/home/users/liutong/dl_learning/patellar/bert-base-chinese')
            inputs = tokenizer(text_tokens, is_split_into_words=True, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(next(self.parameters()).device)
            attention_mask = inputs['attention_mask'].to(next(self.parameters()).device)
            
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            return text_outputs.attentions

class SingleImageNet(LightningModule):
    def __init__(self, model_name='efficientnet_b0', dropout_rate=0.5, use_text=False):
        super(SingleImageNet, self).__init__()
        
        # CNN特征提取器
        self.efficient_net = getattr(models, model_name)(weights='DEFAULT')
        
        # 获取特征维度
        if hasattr(self.efficient_net, 'classifier'):
            in_features = self.efficient_net.classifier[-1].in_features
            self.efficient_net.classifier = nn.Identity()
        else:
            in_features = self.efficient_net.fc.in_features
            self.efficient_net.fc = nn.Identity()
            
        # 特征降维
        self.image_proj = nn.Linear(in_features, 256)
        
        # 是否使用文本特征
        self.use_text = use_text
        if use_text:
            # 文本特征提取器 (使用BERT)
            self.text_encoder = BertModel.from_pretrained('/home/users/liutong/dl_learning/patellar/bert-base-chinese')
            self.text_proj = nn.Linear(768, 256)  # BERT输出维度是768
            
            # 冻结BERT参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # 性别和年龄特征的处理
        self.gender_embedding = nn.Embedding(3, 16)  # 假设性别有3种可能：男(1)、女(2)和未知(0)
        self.age_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # 单图像分类器
        self.image_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        # 如果使用文本特征，需要另一个分类器
        if use_text:
            # 结合图像、文本和人口统计学特征的分类器
            self.combined_classifier = nn.Sequential(
                nn.Linear(256 + 256 + 32, 512),  # 256(图像) + 256(文本) + 32(性别和年龄)
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1)
            )
            
    def forward(self, image, input_ids=None, attention_mask=None, gender=None, age=None):
        """
        处理单个图像
        
        参数:
        - image: 形状为 [batch_size, channels, height, width] 的图像张量
        - input_ids: 形状为 [batch_size, seq_len] 的文本输入ID
        - attention_mask: 形状为 [batch_size, seq_len] 的注意力掩码
        - gender: 形状为 [batch_size] 的性别特征
        - age: 形状为 [batch_size] 的年龄特征
        """
        # 图像特征提取
        image_features = self.efficient_net(image)
        image_features = self.image_proj(image_features)  # [batch_size, 256]
        
        # 仅图像分类
        image_output = self.image_classifier(image_features)
        
        # 如果不使用文本，直接返回图像分类结果
        if not self.use_text or input_ids is None or attention_mask is None:
            return image_output
        
        # 提取文本特征
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        text_features = self.text_proj(text_output)  # [batch_size, 256]
        
        # 处理人口统计学特征（如果有）
        if gender is not None and age is not None:
            # 处理性别特征
            gender_features = self.gender_embedding(gender)  # [batch_size, 16]
            
            # 处理年龄特征
            age = age.unsqueeze(1)  # 添加特征维度 [batch_size, 1]
            age_features = self.age_proj(age)  # [batch_size, 16]
            
            # 合并人口统计学特征
            demographic_features = torch.cat([gender_features, age_features], dim=1)  # [batch_size, 32]
            
            # 合并所有特征
            combined_features = torch.cat([image_features, text_features, demographic_features], dim=1)
            
            # 结合所有特征进行分类
            combined_output = self.combined_classifier(combined_features)
            
            # 返回综合结果
            return combined_output
        
        # 如果没有人口统计学特征，只返回图像分类结果
        return image_output

class ImprovedSingleImageNet(LightningModule):
    def __init__(self, model_name='efficientnet_b0', dropout_rate=0.7, use_text=False):
        super(ImprovedSingleImageNet, self).__init__()
        
        # CNN特征提取器 - 使用更强的正则化
        self.efficient_net = getattr(models, model_name)(weights='DEFAULT')
        
        # 冻结前面的层以减少过拟合
        for param in list(self.efficient_net.parameters())[:-30]:
            param.requires_grad = False
            
        # 获取特征维度
        if hasattr(self.efficient_net, 'classifier'):
            in_features = self.efficient_net.classifier[-1].in_features
            self.efficient_net.classifier = nn.Identity()
        else:
            in_features = self.efficient_net.fc.in_features
            self.efficient_net.fc = nn.Identity()
            
        # 增强的特征投影层
        self.image_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # 是否使用文本特征
        self.use_text = use_text
        if use_text:
            # 文本特征提取器
            self.text_encoder = BertModel.from_pretrained('/home/users/liutong/dl_learning/patellar/bert-base-chinese')
            self.text_proj = nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2)
            )
            
            # 冻结BERT参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # 性别和年龄特征的处理 - 增强版
        self.gender_embedding = nn.Sequential(
            nn.Embedding(3, 32),
            nn.Dropout(dropout_rate/4)
        )
        self.age_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate/4)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout_rate/2,
            batch_first=True
        )
        
        # 改进的分类器 - 使用残差连接
        if use_text:
            classifier_input_dim = 256 + 256 + 64  # 图像 + 文本 + 人口统计学
        else:
            classifier_input_dim = 256  # 仅图像
            
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(128, 1)
        )
        
    def forward(self, image, input_ids=None, attention_mask=None, gender=None, age=None):
        # 添加训练时的噪声增强
        if self.training:
            image = image + torch.randn_like(image) * 0.02
            
        # 图像特征提取
        image_features = self.efficient_net(image)
        image_features = self.image_proj(image_features)  # [batch_size, 256]
        
        features_list = [image_features]
        
        # 如果使用文本特征
        if self.use_text and input_ids is not None and attention_mask is not None:
            # 提取文本特征
            text_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output
            text_features = self.text_proj(text_output)  # [batch_size, 256]
            features_list.append(text_features)
            
        # 处理人口统计学特征
        if gender is not None and age is not None:
            gender_features = self.gender_embedding(gender).squeeze(1)  # [batch_size, 32]
            age_features = self.age_proj(age.unsqueeze(1)).squeeze(1)   # [batch_size, 32]
            demographic_features = torch.cat([gender_features, age_features], dim=1)  # [batch_size, 64]
            features_list.append(demographic_features)
        
        # 合并所有特征
        if len(features_list) > 1:
            combined_features = torch.cat(features_list, dim=1)
        else:
            combined_features = features_list[0]
            
        # 分类
        output = self.classifier(combined_features)
        
        return output