import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, Conv2d=nn.Conv2d):
        super().__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        norm_layer = nn.BatchNorm2d if use_bn_layer else lambda c: nn.GroupNorm(2, c)
        self.bn1 = norm_layer(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        *,
        l2_norm=False,
        use_pretrained=False,
        use_bn_layer=False,
        last_feature_dim=512,
    ):
        super().__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        conv1_kernel_size = 7 if use_pretrained else 3

        Conv2d = self.get_conv()
        Linear = self.get_linear()

        self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size, stride=1, padding=1, bias=False)
        norm_layer = nn.BatchNorm2d if use_bn_layer else lambda c: nn.GroupNorm(2, c)
        self.bn1 = norm_layer(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
        self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

        if use_pretrained:
            resnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
            self.layer1.load_state_dict(resnet.layer1.state_dict(), strict=False)
            self.layer2.load_state_dict(resnet.layer2.state_dict(), strict=False)
            self.layer3.load_state_dict(resnet.layer3.state_dict(), strict=False)
            self.layer4.load_state_dict(resnet.layer4.state_dict(), strict=False)

        feature_dim = last_feature_dim * block.expansion
        if l2_norm:
            self.fc = Linear(feature_dim, num_classes, bias=False)
        else:
            self.fc = Linear(feature_dim, num_classes)

    def get_conv(self):
        return nn.Conv2d

    def get_linear(self):
        return nn.Linear

    def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, Conv2d=self.get_conv()))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        if self.l2_norm:
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
            logit = self.fc(out)
        else:
            logit = self.fc(out)

        if return_feature:
            return out, logit
        return logit


class ResNet18(ResNet):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
