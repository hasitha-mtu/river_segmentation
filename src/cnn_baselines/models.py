"""
Segmentation Models for River Water Detection
Implements: U-Net, U-Net++, ResUNet++, DeepLabV3+, DeepLabV3+ + CBAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ====================== CBAM Attention Module ======================
class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        residual = x
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out + residual  # Residual connection for stability


# ====================== U-Net ======================
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net Architecture"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ====================== U-Net++ ======================
class NestedDoubleConv(nn.Module):
    """Nested convolution block for U-Net++"""
    def __init__(self, in_channels, out_channels):
        super(NestedDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """U-Net++ (Nested U-Net) Architecture"""
    def __init__(self, n_channels=3, n_classes=1, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        
        filters = [32, 64, 128, 256, 512]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder
        self.conv0_0 = NestedDoubleConv(n_channels, filters[0])
        self.conv1_0 = NestedDoubleConv(filters[0], filters[1])
        self.conv2_0 = NestedDoubleConv(filters[1], filters[2])
        self.conv3_0 = NestedDoubleConv(filters[2], filters[3])
        self.conv4_0 = NestedDoubleConv(filters[3], filters[4])
        
        # Decoder - Level 1
        self.conv0_1 = NestedDoubleConv(filters[0] + filters[1], filters[0])
        self.conv1_1 = NestedDoubleConv(filters[1] + filters[2], filters[1])
        self.conv2_1 = NestedDoubleConv(filters[2] + filters[3], filters[2])
        self.conv3_1 = NestedDoubleConv(filters[3] + filters[4], filters[3])
        
        # Decoder - Level 2
        self.conv0_2 = NestedDoubleConv(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = NestedDoubleConv(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = NestedDoubleConv(filters[2]*2 + filters[3], filters[2])
        
        # Decoder - Level 3
        self.conv0_3 = NestedDoubleConv(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = NestedDoubleConv(filters[1]*3 + filters[2], filters[1])
        
        # Decoder - Level 4
        self.conv0_4 = NestedDoubleConv(filters[0]*4 + filters[1], filters[0])
        
        # Output layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        # Decoder - Level 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        # Decoder - Level 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        
        # Decoder - Level 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        
        # Decoder - Level 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


# ====================== ResUNet++ ======================
class ResidualBlock(nn.Module):
    """Residual Block with Squeeze-and-Excitation"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        # SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        out += residual
        out = F.relu(out, inplace=True)
        return out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()[2:]
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.conv5(x), size=size, mode='bilinear', align_corners=True)
        
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.conv_out(out)
        return out


class ResUNetPlusPlus(nn.Module):
    """ResUNet++ Architecture"""
    def __init__(self, n_channels=3, n_classes=1):
        super(ResUNetPlusPlus, self).__init__()
        
        filters = [64, 128, 256, 512]
        
        # Input conv
        self.input_conv = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.res1 = ResidualBlock(filters[0], filters[0])
        self.res2 = ResidualBlock(filters[0], filters[1], stride=2)
        self.res3 = ResidualBlock(filters[1], filters[2], stride=2)
        self.res4 = ResidualBlock(filters[2], filters[3], stride=2)
        
        # Bridge with ASPP
        self.aspp = ASPP(filters[3], filters[3])
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec4 = ResidualBlock(filters[3], filters[2])
        
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec3 = ResidualBlock(filters[2], filters[1])
        
        self.up2 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec2 = ResidualBlock(filters[1], filters[0])
        
        # Output
        self.output_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.input_conv(x)
        
        e1 = self.res1(x)
        e2 = self.res2(e1)
        e3 = self.res3(e2)
        e4 = self.res4(e3)
        
        # Bridge
        bridge = self.aspp(e4)
        
        # Decoder
        d4 = self.up4(bridge)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.output_conv(d2)
        return out


# ====================== DeepLabV3+ ======================
class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet50 backbone"""
    def __init__(self, n_channels=3, n_classes=1, pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet50(weights=None)
        
        # Modify first conv if input channels != 3
        if n_channels != 3:
            self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, 
                                   padding=3, bias=False)
            # Initialize with average of pretrained weights if pretrained
            if pretrained:
                with torch.no_grad():
                    self.conv1.weight[:, :min(3, n_channels), :, :] = \
                        resnet.conv1.weight[:, :min(3, n_channels), :, :]
        else:
            self.conv1 = resnet.conv1
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Encoder (ResNet layers)
        self.layer1 = resnet.layer1  # 256 channels, stride 4
        self.layer2 = resnet.layer2  # 512 channels, stride 8
        self.layer3 = resnet.layer3  # 1024 channels, stride 16
        self.layer4 = resnet.layer4  # 2048 channels, stride 32 (with dilation)
        
        # Modify layer4 to use dilation instead of stride
        self._modify_resnet_dilation(self.layer4, dilation=2)
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 256 + 48
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Conv2d(256, n_classes, 1)

    def _modify_resnet_dilation(self, layer, dilation):
        """Modify ResNet layer to use dilation"""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        low_level_feat = x  # Save for skip connection
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = F.interpolate(x, size=low_level_feat.size()[2:], 
                         mode='bilinear', align_corners=True)
        
        low_level_feat = self.decoder_conv1(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder_conv2(x)
        
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x


# ====================== DeepLabV3+ + CBAM + Boundary Loss ======================
class DeepLabV3PlusCBAM(nn.Module):
    """DeepLabV3+ with CBAM attention modules"""
    def __init__(self, n_channels=3, n_classes=1, pretrained=True):
        super(DeepLabV3PlusCBAM, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet50(weights=None)
        
        # Modify first conv if input channels != 3
        if n_channels != 3:
            self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, 
                                   padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    self.conv1.weight[:, :min(3, n_channels), :, :] = \
                        resnet.conv1.weight[:, :min(3, n_channels), :, :]
        else:
            self.conv1 = resnet.conv1
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Encoder with CBAM
        self.layer1 = resnet.layer1
        self.cbam1 = CBAM(256, reduction=16)
        
        self.layer2 = resnet.layer2
        self.cbam2 = CBAM(512, reduction=16)
        
        self.layer3 = resnet.layer3
        self.cbam3 = CBAM(1024, reduction=16)
        
        self.layer4 = resnet.layer4
        self.cbam4 = CBAM(2048, reduction=16)
        
        # Modify layer4 for dilation
        self._modify_resnet_dilation(self.layer4, dilation=2)
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Decoder with CBAM
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_cbam = CBAM(304, reduction=16)  # 256 + 48
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Conv2d(256, n_classes, 1)

    def _modify_resnet_dilation(self, layer, dilation):
        """Modify ResNet layer to use dilation"""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder with CBAM
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.cbam1(x)
        low_level_feat = x
        
        x = self.layer2(x)
        x = self.cbam2(x)
        
        x = self.layer3(x)
        x = self.cbam3(x)
        
        x = self.layer4(x)
        x = self.cbam4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder with CBAM
        x = F.interpolate(x, size=low_level_feat.size()[2:], 
                         mode='bilinear', align_corners=True)
        
        low_level_feat = self.decoder_conv1(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder_cbam(x)  # Apply CBAM attention
        x = self.decoder_conv2(x)
        
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x


# ====================== Model Factory ======================
def get_model(model_name, in_channels=3, num_classes=1, pretrained=True):
    """
    Factory function to get models by name
    
    Args:
        model_name: One of ['unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam']
        n_channels: Number of input channels
        n_classes: Number of output classes
        pretrained: Use pretrained weights (for DeepLabV3+ variants)
    """
    models = {
        'unet': UNet,
        'unetpp': UNetPlusPlus,
        'resunetpp': ResUNetPlusPlus,
        'deeplabv3plus': DeepLabV3Plus,
        'deeplabv3plus_cbam': DeepLabV3PlusCBAM
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    
    if 'deeplab' in model_name.lower():
        return model_class(n_channels=in_channels, n_classes=num_classes, pretrained=pretrained)
    else:
        return model_class(n_channels=in_channels, n_classes=num_classes)


if __name__ == "__main__":
    # Test all models
    x = torch.randn(2, 3, 512, 512)
    
    print("Testing models with input shape:", x.shape)
    print("-" * 60)
    
    for model_name in ['unet', 'unetpp', 'resunetpp', 'deeplabv3plus', 'deeplabv3plus_cbam']:
        print(f"\nTesting {model_name}...")
        model = get_model(model_name, n_channels=3, n_classes=1, pretrained=False)
        model.eval()
        
        with torch.no_grad():
            output = model(x)
            if isinstance(output, list):
                print(f"  Deep supervision outputs: {[o.shape for o in output]}")
            else:
                print(f"  Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
