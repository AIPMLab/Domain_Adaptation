from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet-b2', pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self._feature_dim = self.model._fc.in_features

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

class ResNeStBackbone(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNeStBackbone, self).__init__()
        self.model = resnet_dict['resnet50'](pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2048)
        self._feature_dim = self.model.fc.in_features

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

    def output_num(self):
        return self._feature_dim
