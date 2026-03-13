import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
        
        # 解冻最后几层特征提取层
        for param in list(self.model.features.parameters())[-6:]:
            param.requires_grad = True
            
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ResNet34(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(DenseNet, self).__init__()
        self.model = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class RegNet(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(RegNet, self).__init__()
        self.model = models.regnet_y_400mf(weights='RegNet_Y_400MF_Weights.IMAGENET1K_V2')
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class VGG11(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(VGG11, self).__init__()
        self.model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 模型字典，方便调用
MODEL_DICT = {
    'alexnet': AlexNet,
    'resnet34': ResNet34,
    'densenet': DenseNet,
    'mobilenetv2': MobileNetV2,
    'regnet': RegNet,
    'vgg11': VGG11,
    'efficientnet_b0': EfficientNetB0
}
