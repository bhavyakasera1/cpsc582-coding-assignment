from torchvision.models import vgg16, VGG16_Weights, vgg19, VGG19_Weights, resnet18, ResNet18_Weights
import torch.nn as nn
import torch

class ClassificationModelVGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(ClassificationModelVGG16, self).__init__()
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        # Modify the last fully connected layer for classification
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)
    
class ClassificationModelVGG19(nn.Module):
    def __init__(self, num_classes=2):
        super(ClassificationModelVGG19, self).__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        # Modify the last fully connected layer for classification
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)
    
class ClassificationModelResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ClassificationModelResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the last fully connected layer for classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
class FeatureResNet18(nn.Module):
    def __init__(self):
        super(FeatureResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
 
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    
class FeatureVGG19(nn.Module):
    def __init__(self):
        super(FeatureVGG19, self).__init__()
        self.model = ClassificationModelVGG19()
        checkpoint = torch.load("classfication_model_vgg19_lr_0.0001_epochs_20.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        for i in range(37):
            x = self.model.vgg.features[i](x)
        x = self.model.vgg.avgpool(x)
        for i in range(4):
            x = self.model.vgg.classifier[i](x)
        x = torch.flatten(x, 1)
        return x