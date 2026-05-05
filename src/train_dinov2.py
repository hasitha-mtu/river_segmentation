import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models



dino_backbones = {
    'vit_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'vit_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'vit_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    }
}



class conv_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(conv_head, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 64, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3,3), padding=(1,1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        x = torch.sigmoid(x)
        return x

class DINOv2Segmentation(nn.Module):
    def __init__(self, num_classes, encoder_variant, head = 'conv'):
        super(DINOv2Segmentation, self).__init__()
        self.heads = {
            'conv':conv_head
        }
        self.backbones = dino_backbones
        self.backbone = load('facebookresearch/dinov2', encoder_variant)
        self.backbone.eval()
        self.num_classes =  num_classes # add a class for background if needed
        self.embedding_size = self.backbones[encoder_variant]['embedding_size']
        self.patch_size = self.backbones[encoder_variant]['patch_size']
        self.head = self.heads[head](self.embedding_size,self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        with torch.no_grad():
            x = self.backbone.forward_features(x.cuda())
            x = x['x_norm_patchtokens']
            x = x.permute(0,2,1)
            x = x.reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1]))
        x = self.head(x)
        return x

def build_dinov2_segmentation(num_classes: int = 1, 
variant:  str = 'vit_b') -> DINOv2Segmentation:
    return DINOv2Segmentation(
        num_classes     = num_classes,
        encoder_variant = variant
    )

if __name__ == '__main__':
    pass