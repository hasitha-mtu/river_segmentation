import torch
import torch.nn as nn
from torch.hub import load

dino_backbones = {
    'vit_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14,
    },
    'vit_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14,
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
        # Return raw logits — BCEWithLogitsLoss (used by the combined loss)
        # applies sigmoid internally.  Applying sigmoid here would cause the
        # loss to compute sigmoid(sigmoid(x)), corrupting gradients.
        return x

class DINOv2Segmentation(nn.Module):
    def __init__(self, num_classes, encoder_variant, head = 'conv'):
        super(DINOv2Segmentation, self).__init__()
        self.heads = {
            'conv':conv_head
        }
        self.backbones = dino_backbones
        self.hub_name = self.backbones[encoder_variant]['name']
        self.backbone = load('facebookresearch/dinov2', self.hub_name)
        self.backbone.eval()
        self.num_classes =  num_classes # add a class for background if needed
        self.embedding_size = self.backbones[encoder_variant]['embedding_size']
        self.patch_size = self.backbones[encoder_variant]['patch_size']
        self.head = self.heads[head](self.embedding_size,self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        H_orig, W_orig = x.shape[2], x.shape[3]

        # ── Resize input to the nearest multiple of patch_size ────────────
        # DINOv2's patch_embed asserts H and W are exact multiples of
        # patch_size (14).  512 is not divisible by 14 (512 / 14 = 36.57),
        # so we pad up to the next valid size: ceil(512/14)*14 = 37*14 = 518.
        import math
        H_pad = math.ceil(H_orig / self.patch_size) * self.patch_size
        W_pad = math.ceil(W_orig / self.patch_size) * self.patch_size
        if H_pad != H_orig or W_pad != W_orig:
            x = torch.nn.functional.interpolate(
                x, size=(H_pad, W_pad), mode='bilinear', align_corners=False,
            )

        mask_dim = (H_pad // self.patch_size, W_pad // self.patch_size)

        with torch.no_grad():
            x = self.backbone.forward_features(x)
            x = x['x_norm_patchtokens']
            x = x.permute(0, 2, 1)
            x = x.reshape(batch_size, self.embedding_size,
                           int(mask_dim[0]), int(mask_dim[1]))

        x = self.head(x)

        # ── Upsample output back to original input resolution ─────────────
        # conv_head 4× upsamples the patch-grid feature map, producing a
        # spatial size that depends on the padded input (e.g. 148×148 for
        # 518 input).  Interpolate back to the original H×W so that the
        # output always matches the ground-truth mask dimensions.
        if x.shape[2] != H_orig or x.shape[3] != W_orig:
            x = torch.nn.functional.interpolate(
                x, size=(H_orig, W_orig), mode='bilinear', align_corners=False,
            )

        return x

def build_dinov2_segmentation(num_classes: int = 1, 
                              variant:  str = 'vit_b') -> DINOv2Segmentation:
        return DINOv2Segmentation(
            num_classes     = num_classes,
            encoder_variant = variant
        )

if __name__ == '__main__':
    pass