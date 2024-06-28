import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial

import timm.models.vision_transformer
from util.pos_embed import get_2d_sincos_pos_embed

# visualise SAR image and attention map
import numpy as np
import matplotlib.pyplot as plt


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # Added by Samar, need default pos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

    def forward_features(self, x):
        B, C, H, W = imgs.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, attn = blk(x, return_attention=True)
                
                imgs_np = []
                for i in range(imgs.shape[0]):
                    img_np = imgs[i].detach().cpu().numpy()
                    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
                    imgs_np.append(img_np)
                concat_imgs = np.hstack(imgs_np)
                SAR_imgs = Image.fromarray(np.uint8(concat_imgs))
                
                # return attention map if not training.
                attn = self.forward_encoder_test(imgs)
                attn = attn[:, :, 1:2, 1:]
    
                mask_weights = attn.mean(dim=1, keepdim=True)
                mask_weights = mask_weights.view(B, 1, H // patch_size, W // patch_size)
                mask_weights = F.interpolate(mask_weights, size=(H, W), mode='bilinear', align_corners=False)
                mask_weights = mask_weights.squeeze(1)     
    
                mask_weights_np = []
                for i in range(mask_weight.shape[0]):
                    mask_weight_np = mask_weights[i].detach().cpu().numpy()
                    mask_weight_np = (mask_weight_np - np.min(mask_weight_np)) / (np.max(mask_weight_np) - np.min(mask_weight_np))
                    mask_weights_np.append(mask_weight_np)
                concat_mask_weights = np.hstack(mask_weights_np)
                mask_weights_imgs = Image.fromarray(np.uint8(concat_mask_weights))

                fig, axes = plt.subplots(2, 1) 
                axes[0].imshow(SAR_imgs)
                axes[0].set_title('Difficult SAR images')
                axes[1].imshow(mask_weights_imgs)
                axes[1].set_title('Respective masking weight upsampled to image sizes!')
                plt.subplots_adjust(hspace=0.5)
                plt.show()
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
