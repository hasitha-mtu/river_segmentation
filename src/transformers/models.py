"""
Transformer-based Segmentation Models for River Water Detection
RGB Input Only - SegFormer-B0, SegFormer-B2, Swin-UNet-Tiny
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== SegFormer Components ======================

class DWConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """MLP with Depthwise Convolution"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Efficient Self-Attention with Spatial Reduction"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping Patch Embedding"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                             stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """Mix Vision Transformer Encoder"""
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, 
                                             embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], 
                                             embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], 
                                             embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], 
                                             embed_dim=embed_dims[3])

        # Transformer blocks
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for _ in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.block2 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for _ in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.block3 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for _ in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.block4 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for _ in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class MLPDecoder(nn.Module):
    """All-MLP Decoder"""
    def __init__(self, in_channels=[64, 128, 256, 512], embedding_dim=256, num_classes=1):
        super(MLPDecoder, self).__init__()
        
        self.linear_c4 = nn.Sequential(
            nn.Conv2d(in_channels[3], embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c3 = nn.Sequential(
            nn.Conv2d(in_channels[2], embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c2 = nn.Sequential(
            nn.Conv2d(in_channels[1], embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c1 = nn.Sequential(
            nn.Conv2d(in_channels[0], embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c1.shape

        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1)

        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x


class SegFormer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation
    RGB Input Only
    """
    def __init__(self, variant='b0', num_classes=1):
        super(SegFormer, self).__init__()
        
        configs = {
            'b0': {
                'embed_dims': [32, 64, 160, 256],
                'num_heads': [1, 2, 5, 8],
                'mlp_ratios': [4, 4, 4, 4],
                'depths': [2, 2, 2, 2],
                'sr_ratios': [8, 4, 2, 1]
            },
            'b2': {
                'embed_dims': [64, 128, 320, 512],
                'num_heads': [1, 2, 5, 8],
                'mlp_ratios': [4, 4, 4, 4],
                'depths': [3, 4, 6, 3],
                'sr_ratios': [8, 4, 2, 1]
            }
        }
        
        config = configs[variant]
        
        self.encoder = MixVisionTransformer(
            in_chans=3,  # RGB only
            embed_dims=config['embed_dims'],
            num_heads=config['num_heads'],
            mlp_ratios=config['mlp_ratios'],
            depths=config['depths'],
            sr_ratios=config['sr_ratios']
        )
        
        self.decoder = MLPDecoder(
            in_channels=config['embed_dims'],
            embedding_dim=256,
            num_classes=num_classes
        )

    def forward(self, x):
        input_size = x.size()[2:]
        features = self.encoder(x)
        output = self.decoder(features)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        return output


# ====================== Swin Transformer Components ======================

class WindowAttention(nn.Module):
    """Window based multi-head self attention"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """Patch Merging Layer - merges 2x2 patches to reduce spatial resolution"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H*W}"
        assert H % 2 == 0 and W % 2 == 0, f"H ({H}) and W ({W}) must be even"

        x = x.view(B, H, W, C)

        # Merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C

        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C
        x = self.norm(x)
        x = self.reduction(x)  # B, H/2*W/2, 2*C

        return x


class PatchExpand(nn.Module):
    """Patch Expanding Layer"""
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x, H, W):
        x = self.expand(x)
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)
        x = x.view(B, H, W, 2, 2, C//4).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H*2, W*2, C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)

        return x


class SwinUNet(nn.Module):
    """
    Swin-UNet: Swin Transformer U-Net for Segmentation
    RGB Input Only - Tiny configuration
    """
    def __init__(self, num_classes=1, embed_dim=96, depths=[2, 2, 2, 2],
             num_heads=[3, 6, 12, 24], window_size=8):
        super(SwinUNet, self).__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding (RGB only)
        self.patch_embed = PatchEmbed(patch_size=4, in_chans=3, embed_dim=embed_dim)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=layer_dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.
                )
                for i in range(depths[i_layer])
            ])
            self.encoder_layers.append(layer)

        # Downsample (Patch Merging)
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            layer_dim = int(embed_dim * 2 ** i_layer)
            downsample = PatchMerging(dim=layer_dim, norm_layer=nn.LayerNorm)
            self.downsample_layers.append(downsample)

        # Decoder - dimensions match output of skip_projections
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            # After skip projection, dimension is embed_dim * 2^(num_layers-2-i_layer)
            layer_dim = int(embed_dim * 2 ** (self.num_layers - 2 - i_layer))
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=layer_dim,
                    num_heads=num_heads[self.num_layers - 2 - i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.
                )
                for i in range(depths[self.num_layers - 2 - i_layer])
            ])
            self.decoder_layers.append(layer)

        # Upsample
        self.upsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            layer_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            upsample = PatchExpand(dim=layer_dim, dim_scale=2, norm_layer=nn.LayerNorm)
            self.upsample_layers.append(upsample)

        # Skip projections
        self.skip_projections = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            layer_dim = int(embed_dim * 2 ** (self.num_layers - 2 - i_layer))
            proj = nn.Linear(layer_dim * 2, layer_dim)
            self.skip_projections.append(proj)

        # Final expansion and output
        self.final_expand = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 16, bias=False),
            nn.LayerNorm(embed_dim * 16)
        )
        self.output = nn.Conv2d(embed_dim * 16, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Patch embedding
        x, H, W = self.patch_embed(x)
        
        # Encoder
        skip_connections = []
        for i, (encoder_layer, downsample) in enumerate(zip(
            self.encoder_layers[:-1], self.downsample_layers)):
            
            for blk in encoder_layer:
                x = blk(x, H, W)
            
            skip_connections.append((x, H, W))
            x = downsample(x, H, W)  # PatchMerging needs H, W
            H, W = H // 2, W // 2

        # Bottleneck
        for blk in self.encoder_layers[-1]:
            x = blk(x, H, W)

        # Decoder
        for i, (decoder_layer, upsample, skip_proj) in enumerate(zip(
            self.decoder_layers, self.upsample_layers, self.skip_projections)):
            
            x = upsample(x, H, W)
            H, W = H * 2, W * 2
            
            skip_x, skip_H, skip_W = skip_connections[-(i+1)]
            x = torch.cat([x, skip_x], dim=-1)
            x = skip_proj(x)
            
            for blk in decoder_layer:
                x = blk(x, H, W)

        # Final expansion
        x = self.final_expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        x = self.output(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x


def get_model(model_name, variant='b0', num_classes=1):
    """
    Factory function to get models
    
    Args:
        model_name: 'segformer', or 'swin_unet'
        num_classes: Number of output classes (1 for binary segmentation)
    """
    if model_name == 'segformer':
        return SegFormer(variant=variant, num_classes=num_classes)
    elif model_name == 'swin_unet':
        return SwinUNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    print("Testing Transformer Models (RGB Only)")
    print("=" * 60)
    
    x = torch.randn(2, 3, 512, 512)  # RGB input
    
    models = ['segformer_b0', 'segformer_b2', 'swin_unet_tiny']
    
    for name in models:
        print(f"\nTesting {name}...")
        model = get_model(name)
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  Input:  {x.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Params: {params:,} ({params/1e6:.1f}M)")
    
    print("\n" + "=" * 60)
    print("All models tested successfully!")
