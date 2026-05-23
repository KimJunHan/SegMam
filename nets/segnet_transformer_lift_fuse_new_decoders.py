import os
import sys

# import matplotlib
# matplotlib.use('tkagg')  # enables plt.show() in code
import matplotlib.pyplot as plt
import numpy as np
import torch


# When SEGMAM_DISABLE_PROFILING=1 the per-stage GPU profiling (torch.cuda.Event +
# synchronize) inside the forward path is bypassed. This is required for
# torch.compile(mode="reduce-overhead") / cuda graph capture, which is incompatible
# with arbitrary cuda.Event recording inside the traced graph.
# Evaluated lazily on every check so vis_eval.py can flip the env var inside main()
# before model.eval() and have it take effect for the next forward.
def _segmam_prof_enabled():
    return os.environ.get("SEGMAM_DISABLE_PROFILING", "0") != "1"


# Kept as a name in module scope only because the call-site rewrite uses it as a
# constructor argument. The actual decision is re-evaluated inside _SafeEvent.
_SEGMAM_PROF_ENABLED = True  # value is unused; kept to avoid touching every call site


class _SafeEvent:
    """Drop-in replacement for torch.cuda.Event(enable_timing=True) that becomes
    a no-op when profiling is disabled. Same .record() / .elapsed_time() API.
    The constructor argument is ignored — env var is re-checked at construction
    time so callers can flip SEGMAM_DISABLE_PROFILING any time before forward()."""
    __slots__ = ("real",)

    def __init__(self, _ignored=None):
        self.real = torch.cuda.Event(enable_timing=True) if _segmam_prof_enabled() else None

    def record(self):
        if self.real is not None:
            self.real.record()

    def elapsed_time(self, other):
        if self.real is not None and other.real is not None:
            return self.real.elapsed_time(other.real)
        return 0.0


def _maybe_sync():
    if _segmam_prof_enabled():
        torch.cuda.synchronize()
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import resnet18
import torchvision.models as models
#from pruning import ModelPruning


import utils.basic
import utils.geom
import utils.misc
import utils.vox
from nets.dino_v2_with_adapter.dino_v2_adapter.dinov2_adapter import (
    DinoAdapter,
)
from nets.ops.modules import MSDeformAttn, MSDeformAttn3D
from nets.voxelnet import VoxelNet
#from nets.vovnet import vovnet39,vovnet27_slim, vovnet57

import time
import pandas as pd
import os

# import cross & self mamba 
from CrossMamba.example import cross_mamba
from CrossMamba.example import self_mamba
from CrossMamba.example import FusingCrossMambaV2
from CrossMamba.example import SpatialCrossMambaFromCross
from torch.utils.checkpoint import checkpoint



sys.path.append("..")
EPS = 1e-4


def set_bn_momentum(model: nn.Module, momentum: float = 0.1) -> None:
    """
    Set the momentum for all instance normalization layers in the given model.

    Args:
        model (nn.Module): The PyTorch model containing the layers to be modified.
        momentum (float, optional): The momentum value to set for the instance normalization layers. Defaults to 0.1.

    Returns:
        None
    """
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum


#for resnet encoder
class UpsamplingConcat(nn.Module):
    """
    A module that performs upsampling and concatenation followed by convolution operations.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (float, optional): Upsampling scale factor. Defaults to 2.

    Attributes:
        upsample (torch.nn.Upsample): Upsampling layer.
        conv (torch.nn.Sequential): Sequential convolutional layers with instance normalization and ReLU.

    Methods:
        forward(x_to_upsample, x) -> torch.Tensor: Perform a forward pass of the module.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float = 2):
        """
        Initialize UpsamplingConcat module with the given parameters.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            scale_factor (float, optional): Upsampling scale factor. Defaults to 2.
        """
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the module.

        Args:
            x_to_upsample (torch.Tensor): Input tensor to be upsampled and concatenated.
            x (torch.Tensor): Input tensor to be concatenated.

        Returns:
            torch.Tensor: Output tensor after upsampling, concatenation, and convolution.
        """

        x_to_upsample = F.interpolate(x_to_upsample, size=x.shape[2:], mode='bilinear', align_corners=False)

        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)

        return self.conv(x_to_upsample)


class UpsamplingAdd(nn.Module):
    """
    UpsamplingAdd module that upsamples an input tensor and adds it to a skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (float, optional): Factor by which to upsample the input tensor. Default is 2.

    Attributes:
        upsample_layer (nn.Sequential): A sequential container with an upsampling layer,
            a convolutional layer, and an instance normalization layer.

    Methods:
        forward(x, x_skip): Forward pass that upsamples the input tensor and adds it to the skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float = 2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that upsamples the input tensor and adds it to the skip connection.

        Args:
            x (torch.Tensor): Input tensor to be upsampled.
            x_skip (torch.Tensor): Skip connection tensor to be added to the upsampled input.

        Returns:
            torch.Tensor: The result of the element-wise addition of the upsampled input
            and the skip connection.
        """
        x = self.upsample_layer(x)
        return x + x_skip


class VanillaSelfAttention(nn.Module):
    # adapted from https://github.com/zhiqi-li/BEVFormer
    def __init__(self, dim: int = 128, dropout: float = 0.1, vis_feats: bool = False):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        # Deform.DETR: n_heads=8 n_points=4
        self.deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=8, n_points=4)
        self.vis_feats = vis_feats

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Applies self-attention utilizing deformable attention from deformable DETR

        Args:
            query: (B, N, C) - input query
        """
        inp_residual = query.clone()
        B, N, C = query.shape

        Z, X = 200, 200
        # generate reference points in the BEV plane for spatial self-attention
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z - 0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X - 0.5, X, dtype=torch.float, device=query.device),
            indexing='ij'
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2)  # (B, N, 1, 2)

        if self.vis_feats:
            reference_points_reshape = reference_points.reshape(1, 200, 200, 1, 2)
            reference_points_reshape_np = reference_points_reshape.detach().cpu().numpy()
            img = plt.imshow(reference_points_reshape_np[0, :, :, 0, 0])
            plt.savefig("reference_points_z.png")

            img = plt.imshow(reference_points_reshape_np[0, :, :, 0, 1])
            plt.savefig("reference_points_x.png")

            img = plt.imshow(reference_points_reshape_np[0, :, :, 0, 0] + reference_points_reshape_np[0, :, :, 0, 1])
            plt.show()
            # UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
            # --> changed from 'agg' to 'tkagg'
            plt.savefig("reference_points.png")

        input_spatial_shapes = query.new_zeros([1, 2]).long()
        input_spatial_shapes[:] = 200
        input_level_start_index = query.new_zeros([1, ]).long()
        #with torch.amp.autocast('cuda',enabled=False):
        queries = self.deformable_attention(query, reference_points, query.clone(),
                                            input_spatial_shapes, input_level_start_index)

        return self.dropout(queries) + inp_residual


class SpatialCrossAttention(nn.Module):
    # adapted from https://github.com/zhiqi-li/BEVFormer

    def __init__(self, dim: int = 128, dropout: float = 0.1, num_levels: int = 1):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.num_levels = num_levels
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim,
                                                   num_heads=8,
                                                   num_levels=self.num_levels,
                                                   num_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_pos: torch.Tensor = None,
                reference_points_cam:  torch.Tensor = None, spatial_shapes: torch.Tensor = None,
                bev_mask: torch.Tensor = None) -> torch.Tensor:
        """
        # Attention-based lifting procedure

        Args:
            query: (B, N, C)
            key: (S, M, B, C)
            value:
            query_pos:
            reference_points_cam: (S, B, N, D, 2), in 0-1
            spatial_shapes:
            bev_mask: (S. B, N, D)
        """
        inp_residual = query
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        S, M, _, _ = key.shape

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros(
            [B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B, S, max_len, D, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]

        key = key.permute(2, 0, 1, 3).reshape(
            B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(
            B * S, M, C)

        if len(spatial_shapes) > 1:
            level_start_index = query.new_zeros([len(spatial_shapes), ]).long()
            level_start_index[1] = spatial_shapes[0, 0] * spatial_shapes[0, 1]
            level_start_index[2] = level_start_index[1] + (spatial_shapes[1, 0] * spatial_shapes[1, 1])
            level_start_index[3] = level_start_index[2] + (spatial_shapes[2, 0] * spatial_shapes[2, 1])
        else:
            level_start_index = query.new_zeros([1, ]).long()

        #with torch.amp.autocast('cuda',enabled=False):
        queries = self.deformable_attention(query=queries_rebatch.view(B * S, max_len, self.dim),
                                            key=key, value=value,
                                            reference_points=reference_points_rebatch.view(B * S, max_len, D, 2),
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(B, S, max_len, self.dim)

        for j in range(B):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]  # mean over overlapping regions
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual



class Mamba_SpatialCrossAttention(nn.Module):
    # adapted from https://github.com/zhiqi-li/BEVFormer

    def __init__(self, dim: int = 128, dropout: float = 0.1, num_levels: int = 1):
        super(Mamba_SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.num_levels = num_levels
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim,
                                                   num_heads=8,
                                                   num_levels=self.num_levels,
                                                   num_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_pos: torch.Tensor = None,
                reference_points_cam:  torch.Tensor = None, spatial_shapes: torch.Tensor = None,
                bev_mask: torch.Tensor = None) -> torch.Tensor:
        """
        # Attention-based lifting procedure

        Args:
            query: (B, N, C)
            key: (S, M, B, C)
            value:
            query_pos:
            reference_points_cam: (S, B, N, D, 2), in 0-1
            spatial_shapes:
            bev_mask: (S. B, N, D)
        """
        inp_residual = query
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        S, M, _, _ = key.shape

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros(
            [B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B, S, max_len, D, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]

        key = key.permute(2, 0, 1, 3).reshape(
            B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(
            B * S, M, C)

        if len(spatial_shapes) > 1:
            level_start_index = query.new_zeros([len(spatial_shapes), ]).long()
            level_start_index[1] = spatial_shapes[0, 0] * spatial_shapes[0, 1]
            level_start_index[2] = level_start_index[1] + (spatial_shapes[1, 0] * spatial_shapes[1, 1])
            level_start_index[3] = level_start_index[2] + (spatial_shapes[2, 0] * spatial_shapes[2, 1])
        else:
            level_start_index = query.new_zeros([1, ]).long()

        #with torch.amp.autocast('cuda',enabled=False):
        queries = self.deformable_attention(query=queries_rebatch.view(B * S, max_len, self.dim),
                                            key=key, value=value,
                                            reference_points=reference_points_rebatch.view(B * S, max_len, D, 2),
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(B, S, max_len, self.dim)

        for j in range(B):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]  # mean over overlapping regions
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual

class LiteSpatialCrossAttention(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super(LiteSpatialCrossAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                query_pos: torch.Tensor = None, bev_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            query: (B, N, C)
            key: (S, M, B, C)
            value: same as key
            query_pos: optional
            bev_mask: optional (S, B, N)

        Returns:
            (B, N, C)
        """
        inp_residual = query

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        S, M, _, _ = key.shape

        q = self.q_proj(query).view(B, N, self.heads, -1).transpose(1, 2)  # (B, H, N, C//H)
        k = self.k_proj(key).permute(2, 0, 1, 3).contiguous().view(B, S * M, self.heads, -1).transpose(1, 2)
        v = self.v_proj(value).permute(2, 0, 1, 3).contiguous().view(B, S * M, self.heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if bev_mask is not None:
            # bev_mask: (S, B, N)
            # Expand to match attn shape: (B, H, N, S*M)
            mask = bev_mask.permute(1, 0, 2).reshape(B, S, N).unsqueeze(1).expand(-1, self.heads, -1, -1).repeat(1, 1, 1, M)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)

        out = self.out_proj(out)

        return self.dropout(out) + inp_residual

        
class FusingCrossAttentionV2(nn.Module):
    """
    utilizes deformable attention to fuse camera- and pseudolidar BEV embeddings
    """

    def __init__(self, dim: int = 128, dropout: float = 0.1):
        super(FusingCrossAttentionV2, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        # Deform.DETR: n_heads=8 n_points=4
        self.fusing_deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=8, n_points=4)

    def forward(self, query: torch.Tensor, input_feats: torch.Tensor, query_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Utilizes deformable attention to fuse camera- and pseudolidar BEV embeddings
        Args:
            query: pseudolidar BEV embeddings
            input_feats: camera BEV embeddings
            query_pos: (optional) additional position embedding

        Returns:
            torch.Tensor: BEV feature embedding after one fusion block
        """
        query_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        Z, X = 200, 200
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z - 0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X - 0.5, X, dtype=torch.float, device=query.device),
            indexing='ij'
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2)  # (B, N, 1, 2)

        input_spatial_shapes = query.new_zeros([1, 2]).long()
        input_spatial_shapes[:] = 200
        input_level_start_index = query.new_zeros([1, ]).long()
        queries = self.fusing_deformable_attention(query, reference_points, input_feats,
                                                   input_spatial_shapes, input_level_start_index)

        return self.dropout(queries) + query_residual




       
class Mamba_FusingCrossAttentionV2(nn.Module):
    """
    utilizes deformable attention to fuse camera- and pseudolidar BEV embeddings
    """

    def __init__(self, dim: int = 128, dropout: float = 0.1):
        super(Mamba_FusingCrossAttentionV2, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        # Deform.DETR: n_heads=8 n_points=4
        #self.fusing_deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=8, n_points=4)
        self.fusing = FusingCrossMambaV2(
            dim=dim, dropout=0.1,
            n_levels=1,          # 사용 중인 멀티스케일 수
            n_points=32,          # Deformable 포인트 수
            d_state=32,          # Mamba state 차원
            weight_global=True,  # L*K 전체 softmax (권장)
            pool='attn',         # 'mean' or 'attn'
        )

    def forward(self, query: torch.Tensor, input_feats: torch.Tensor, query_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Utilizes deformable attention to fuse camera- and pseudolidar BEV embeddings
        Args:
            query: pseudolidar BEV embeddings
            input_feats: camera BEV embeddings
            query_pos: (optional) additional position embedding

        Returns:
            torch.Tensor: BEV feature embedding after one fusion block
        """
        query_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        Z, X = 200, 200
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z - 0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X - 0.5, X, dtype=torch.float, device=query.device),
            indexing='ij'
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2)  # (B, N, 1, 2)

        input_spatial_shapes = query.new_zeros([1, 2]).long()
        input_spatial_shapes[:] = 200
        input_level_start_index = query.new_zeros([1, ]).long()
        # queries = self.fusing_deformable_attention(query, reference_points, input_feats,
        #                                            input_spatial_shapes, input_level_start_index)

        queries = self.fusing(query, reference_points, input_feats, input_spatial_shapes, input_level_start_index, query_pos)


        return self.dropout(queries) + query_residual
    


#for SegnetTransformerLiftFuse
class FeatureEncoderDecoder(nn.Module):
    """
    After the fusion, this module forces the model to refine the embeddings by utilizing an encoder-decoder architecture
    that embodies a latent space with reduced spatial resolution in the BEV dims, supplemented by skip connections.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # to avoid torchvision deprecation warning for the parameter "pretrained=False"
        # backbone = resnet18(pretrained=False, zero_init_residual=True)
        backbone = resnet18(weights=None, zero_init_residual=True)
        # changes in_channels from 3(resnet-18) to 128
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=False)
        # kernel_size=7, stride=2, padding=3 -> kernel_size=5, stride=1, padding=2 # 128 -> 64; HW -> HW
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        # maxpool from original resnet-18 is omitted

        self.layer1 = backbone.layer1  # 64 -> 64;     HW -> HW/2 (unchanged)
        self.layer2 = backbone.layer2  # 64 -> 128;    HW/2 -> HW/4
        self.layer3 = backbone.layer3  # 128 -> 256;   HW/4 -> HW/8

        # layer 4 and final pooling + fc layer are omitted

        shared_out_channels = in_channels  # 640
        # definition of additive skip connections
        # - it first upsamples the maps by factor 2 in H and W
        # - then 1x1 convolution -> only reduce number of channels
        # - instance norm along channel dim
        # - in forward pass: add upsampled data to skipped data
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)  # HW/8 -> HW/4
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)  # HW/4 -> HW/2
        self.skip_conv1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False),
                                        nn.InstanceNorm2d(128))  # HW/2 -> HW

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shap: (B,128,200,200)
        # (H, W) -> (200,200)
        skip_x = {'1': x}  # first skip connection before first layer  (B,128,100,100)
        x = self.first_conv(x)  # (B,128,200,200) -> (B,64,200,200)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # (B,64,200,200) -> (B,64,100,100)
        skip_x['2'] = x  # skip connection before layer 2  (B,64,100,100)
        x = self.layer2(x)  # (B,64,100,100) -> (B,128,50,50)
        skip_x['3'] = x  # skip connection before layer 3  (B,128,50,50)
        x = self.layer3(x)  # output after last decoder layer (B,128,50,50) -> (B,256,25,25)
        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])  # upsamples x to match dims of layer2 output and adds them (+conv)
        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])  # upsamples x to match dims of layer1 output and adds them (+conv)
        # Third skip and add to (H, W)
        x = self.skip_conv1(x)  # 1x1 conv to get matching feature dim
        x = x + skip_x['1']

        return x  # (B,128,200,200)


class TaskSpecificDecoder(nn.Module):
    """
    Decoder that handles both tasks (object seg and semantic map seg.) either combined or separately
    """
    def __init__(self, in_channels: int, task: str, n_classes: int, use_feat_head: bool = False,
                 predict_future_flow=False, use_obj_layer_only_on_map=False):
        super(TaskSpecificDecoder, self).__init__()
        self.out_channels = 128
        self.n_classes = n_classes
        self.task = task
        self.predict_future_flow = predict_future_flow
        self.use_feat_head = use_feat_head
        self.use_obj_layer_only_on_map = use_obj_layer_only_on_map

        # structure like in a ResNet18 --> one convolution block and one identity block
        self.upsample_conv_layer = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
        )

        self.upsample_skip_layer = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # here: no upsampling since transformer operates on 200x200 already
            nn.Conv2d(in_channels, 128, kernel_size=1, padding=0, bias=False),
        )

        # first conv block
        self.first_conv_block = nn.Sequential(
            self.upsample_conv_layer,
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)   nn.BatchNorm2d(512)
            nn.ReLU(inplace=True),  # inplace=True
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)
        )
        self.skip_conv1_1 = self.upsample_skip_layer

        # second conv block
        self.second_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)   nn.BatchNorm2d(512)
            nn.ReLU(inplace=True),  # inplace=True
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)
        )
        self.skip_conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # third conv block
        self.third_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)   nn.BatchNorm2d(512)
            nn.ReLU(inplace=True),  # inplace=True
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)
        )
        self.skip_conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # definition of output head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.out_channels, self.n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, bev_flip_indices: torch.Tensor = None) -> dict:
        b, c, h, w = x.shape  # (B,128,200,200)

        # first conv block
        # (H, W) -> (100,100)  -> (200, 200)
        skip_1_1 = self.skip_conv1_1(x)  # (B,640,100,100) -> (B,512,200,200)
        x = self.first_conv_block(x)
        x = x + skip_1_1
        x = F.relu(x, inplace=True)

        # second conv block
        skip_2_1 = self.skip_conv2_1(x)  # (B,512,200,200) -> (B,256,200,200)
        x = self.second_conv_block(x)  # (B,512,200,200) -> (B,256,200,200)
        x = x + skip_2_1
        x = F.relu(x, inplace=True)

        # third conv block
        skip_3_1 = self.skip_conv3_1(x)  # (B,256,200,200) -> (B,128,200,200)
        x = self.third_conv_block(x)  # (B,256,200,200) -> (B,128,200,200)
        x = x + skip_3_1
        # (B, 128, 200, 200)

        # 'unflip' if flipped before
        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2])  # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        # apply task specific head
        # run model output through respective heads
        out_dict = {}
        segmentation_output = self.segmentation_head(x)  # (B,X,200,200)
        
        if self.task == 'object_decoder':
            out_dict = {
                'obj_segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),  # (B,1,200,200)
            }
        elif self.task == 'map_decoder':
            out_dict = {
                'bev_map_segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),  # (B,7,200,200)
                                # (B,7,200,200)
                #'bev_map_segmentation': segmentation_output[:, :-1],
            }
            #print(f"segnet519 out_dict{out_dict}")
            
        elif self.task == 'shared_decoder':
            out_dict = {
                # (B,7,200,200)
                'bev_map_segmentation': segmentation_output[:, :-1],
                # (B,1,200,200)
                'obj_segmentation': segmentation_output[:, -1:],
            }
            
        return out_dict


class Encoder_res101(nn.Module):
    """
    Adapted version of ResNet-101
    """
    def __init__(self, C: int, use_multi_scale_img_feats: bool):
        super().__init__()
        self.C = C  # C = 128 (latent_dim)
        self.use_multi_scale_img_feats = use_multi_scale_img_feats
        # to avoid torchvision deprecation warning for the parameter "pretrained"
        # resnet = torchvision.models.resnet101(pretrained=True)
        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        

        if self.use_multi_scale_img_feats:
            self.backbone = nn.Sequential(*list(resnet.children())[:-6])  # holds layer 1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3

            self.h4_2_channels = nn.Conv2d(256, self.C, kernel_size=1, padding=0)
            self.h8_2_channels = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
            self.h16_2_channels = nn.Conv2d(1024, self.C, kernel_size=1, padding=0)
        else:
            # get all layers except the last 4
            # -> we don't use the average pooling layer and all three blocks of layer 4 from the original ResNet
            self.backbone = nn.Sequential(*list(resnet.children())[:-4])

        # explicitly create a layer of the type block 3 from resnet-101
        # layer 3_x:
        #   conv1 1x1, 256
        #   conv2 3x3, 256
        #   conv3 1x1, 1024
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x: torch.Tensor) -> dict:
        feat_dict = {}
        if self.use_multi_scale_img_feats:
            x0 = self.backbone(x)  # (B*S, 3, H, W) --> (B*S, 64, H/4, W/4)
            x1 = self.layer1(x0)  # (B*S, 64, H/4, W/4) --> (B*S, 256, H/4, W/4)
            x2 = self.layer2(x1)  # (B*S, 256, H/4, W/4) --> (B*S, 512, H/8, W/8)
            x3 = self.layer3(x2)  # (B*S, 512, H/8, W/8) --> (B*S, 1024, H/16, W/16)

            # feature extraction with same channel depth:
            x1_ = self.h4_2_channels(x1)
            x2_ = self.h8_2_channels(x2)
            x3_ = self.h16_2_channels(x3)

            feat_dict = {
                # "feats_2": x0,
                "feats_4": x1_,
                "feats_8": x2_,
                "feats_16": x3_,
            }
        else:
            x2 = self.backbone(x)  # passes input in net and runs it through all layers -> x1=output of the model
            x3 = self.layer3(x2)  # define x2 to be the output of layer 3

        x = self.upsampling_layer(x3, x2)  # input: x_to_upsample=x2 (1024 channels), x=x1 (512 channels)
        # in_channels=1536 (output of layer3 = 1024 + x1 = 512), out_channels=512
        x = self.depth_layer(x)  # 1x1 convolution from 512 channels to 128 (no padding)
        feat_dict["output"] = x

        return feat_dict  # x


class Encoder_res50(nn.Module):
    """
    ResNet-50 인코더 (이미지넷 X)
    - backbone_pretrained_path: 필수. DeepLabv3(+)-ResNet Cityscapes/HF .bin 또는 .pth
      파일에서 'backbone'만 추출해 self.resnet에 주입. 없으면 즉시 에러.
    - strict: True면 백본 키 하나라도 빠지면 에러; False면 겹치는 키만 로드(기본).
    """
    def __init__(self, C: int, use_multi_scale_img_feats: bool,strict: bool = False):
        super().__init__()
        backbone_pretrained_path = "/SegMam/deeplabv3-resnet50-cityscapes/pytorch_model.bin"
        if not backbone_pretrained_path :
            raise FileNotFoundError(f"[Encoder_res50] ckpt needed {backbone_pretrained_path}")

        self.C = C
        self.use_multi_scale_img_feats = use_multi_scale_img_feats
        
        # ImageNet 사용 안 함
        self.resnet = torchvision.models.resnet50(weights=None)  # or pretrained=False (구버전)

        # 반드시 외부 ckpt에서 백본 로드 (실패 시 예외)
        self._load_backbone_from(backbone_pretrained_path, strict=strict)

        if self.use_multi_scale_img_feats:
            self.backbone = nn.Sequential(*list(self.resnet.children())[:-6])  # conv1..maxpool
            self.layer1   = self.resnet.layer1
            self.layer2   = self.resnet.layer2
            self.layer3   = self.resnet.layer3

            self.h4_2_channels  = nn.Conv2d(256,  self.C, 1, padding=0)
            self.h8_2_channels  = nn.Conv2d(512,  self.C, 1, padding=0)
            self.h16_2_channels = nn.Conv2d(1024, self.C, 1, padding=0)
        else:
            self.backbone = nn.Sequential(*list(self.resnet.children())[:-4])  # conv1..layer2
            self.layer3   = self.resnet.layer3

        self.depth_layer      = nn.Conv2d(512, self.C, 1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def _load_backbone_from(self, ckpt_path: str, strict: bool = False):
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:  # weights_only 인자를 모르는 구버전
            sd = torch.load(ckpt_path, map_location="cpu")

        # 포맷 통일
        if isinstance(sd, dict) and "state_dict" in sd:  # HF/torchvision 스타일
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model_state" in sd: # VainF 스타일
            sd = sd["model_state"]
        if not isinstance(sd, dict):
            raise RuntimeError(f"[Encoder_res50] 알 수 없는 ckpt 포맷: {type(sd)}")

        def strip(k: str) -> str:
            for p in ("module.", "model."):
                if k.startswith(p): return k[len(p):]
            return k
        sd = {strip(k): v for k, v in sd.items() if isinstance(v, torch.Tensor)}

        # backbone/body 키만 추출 → torchvision resnet 키로 정규화
        def take(k: str):
            for p in ("backbone.body.", "backbone.", "resnet."):
                if k.startswith(p): return k[len(p):]
            return None

        bb = {}
        for k, v in sd.items():
            kk = take(k)
            if kk is None: 
                continue
            kk = kk.replace("body.", "")  # body.* 한 번 더 싸인 경우 제거
            bb[kk] = v

        if not bb:
            raise RuntimeError("[Encoder_res50] ckpt에서 backbone 키를 찾지 못했습니다. "
                               "기대한 접두사: backbone.body.*, backbone.*, resnet.*")

        tgt = self.resnet.state_dict()
        mapped = {k: v for k, v in bb.items() if (k in tgt and v.shape == tgt[k].shape)}

        # 매핑된 게 너무 없으면 의미 없는 ckpt로 판단
        if len(mapped) == 0:
            raise RuntimeError("[Encoder_res50] 일치하는 백본 파라미터가 없습니다. "
                               "ResNet-50 기반 ckpt인지 확인하세요.")

        missing, unexpected = self.resnet.load_state_dict(mapped, strict=strict)

        # strict=True면 누락/예상외 키 존재 시 load_state_dict가 이미 예외 발생
        if not strict:
            # strict=False에서도, 매핑률이 너무 낮으면 경고 대신 에러로 처리하고 싶다면 임계치 사용:
            total_needed = sum(p.numel() for n, p in tgt.items())
            total_mapped = sum(p.numel() for n, p in mapped.items())
            ratio = total_mapped / max(1, total_needed)
            if ratio < 0.2:  # 필요시 조정
                raise RuntimeError(f"[Encoder_res50] 매핑율이 너무 낮습니다 ({ratio:.1%}). "
                                   "ckpt가 ResNet-50 백본이 맞는지 확인하세요.")

        print(f"[Cityscapes->ResNet50] loaded={len(mapped)} tensors from {ckpt_path} (strict={strict})")

    def forward(self, x: torch.Tensor) -> dict:
        feat = {}
        if self.use_multi_scale_img_feats:
            x0 = self.backbone(x)   # /4
            x1 = self.layer1(x0)    # 256, /4
            x2 = self.layer2(x1)    # 512, /8
            x3 = self.layer3(x2)    # 1024, /16
            feat["feats_4"]  = self.h4_2_channels(x1)
            feat["feats_8"]  = self.h8_2_channels(x2)
            feat["feats_16"] = self.h16_2_channels(x3)
        else:
            x2 = self.backbone(x)
            x3 = self.layer3(x2)
        y = self.upsampling_layer(x3, x2)  # (1024 ↑ ⊕ 512) → 1536 → 512
        y = self.depth_layer(y)            # 512 → C
        feat["output"] = y
        return feat
    
    
class Encoder_res50_default(nn.Module):
    """
        Adapted version of ResNet-50
    """
    def __init__(self, C: int, use_multi_scale_img_feats: bool):
        super().__init__()
        self.C = C
        self.use_multi_scale_img_feats = use_multi_scale_img_feats
        resnet = torchvision.models.resnet50(pretrained=True)

        if self.use_multi_scale_img_feats:
            self.backbone = nn.Sequential(*list(resnet.children())[:-6])  # holds layer 1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3

            self.h4_2_channels = nn.Conv2d(256, self.C, kernel_size=1, padding=0)
            self.h8_2_channels = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
            self.h16_2_channels = nn.Conv2d(1024, self.C, kernel_size=1, padding=0)

        else:
            self.backbone = nn.Sequential(*list(resnet.children())[:-4])  # holds layer 1 and 2
            self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x: torch.Tensor) -> dict:
        feat_dict = {}
        if self.use_multi_scale_img_feats:
            x0 = self.backbone(x)  # (B*S, 3, H, W) --> (B*S, 64, H/4, W/4)
            x1 = self.layer1(x0)  # (B*S, 64, H/4, W/4) --> (B*S, 256, H/4, W/4)
            x2 = self.layer2(x1)  # (B*S, 256, H/4, W/4) --> (B*S, 512, H/8, W/8)
            x3 = self.layer3(x2)  # (B*S, 512, H/8, W/8) --> (B*S, 1024, H/16, W/16)

            # feature extraction with same channel depth:
            x1_ = self.h4_2_channels(x1)
            x2_ = self.h8_2_channels(x2)
            x3_ = self.h16_2_channels(x3)

            feat_dict = {
                # "feats_2": x0,
                "feats_4": x1_,
                "feats_8": x2_,
                "feats_16": x3_,
            }
        else:
            x2 = self.backbone(x)  # res:
            x3 = self.layer3(x2)  #

        x = self.upsampling_layer(x3, x2)
        x = self.depth_layer(x)
        feat_dict["output"] = x

        return feat_dict  # x
    
    
    
# class Encoder_ghost(nn.Module):
#     """
#         Adapted version of Encoder_ghost
#     """
#     def __init__(self, C: int, use_multi_scale_img_feats: bool):
        
#         super().__init__()
#         self.C = C
#         self.use_multi_scale_img_feats = use_multi_scale_img_feats
#         ghost = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
#         model_sturcture = ghost.eval()
#         #print(f"model_sturcture {model_sturctur}")
                
        
#         if self.use_multi_scale_img_feats:
            
#             self.ghostnet = nn.Sequential(*list(ghost.children())[:-6])  # holds layer 1
            
#             '''reference  resent layer1, layer2, layer3 '''
            
#             self.layer1 = nn.Sequential(
                
#                 #basic block 0
#                 nn.Conv2d(16,256, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256,256, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
#                 #basic block 1
#                 nn.Conv2d(256,256, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256,256, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             )
            
#             self.layer2 = nn.Sequential(
                
#                 #basic block 0
#                 nn.Conv2d(256,512, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(512,512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                 #down sampling
#                 nn.Conv2d(512,512,kernel_size=(1,1), stride=(2,2), bias=False),
#                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                 #basic block 1
#                 nn.Conv2d(512,512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(512,512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             )   
            
#             self.layer3 = nn.Sequential(
                
#                 #basic block 0
#                 nn.Conv2d(512,1024, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(1024,1024, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
#                 #down sampling 
#                 nn.Conv2d(1024,1024,kernel_size=(1,1), stride=(2,2), bias=False),
#                 nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
#                 #basic block 1
#                 nn.Conv2d(1024,1024, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(1024,1024, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
#                 nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             )   
            
            
#             self.h4_2_channels = nn.Conv2d(256, self.C, kernel_size=1, padding=0)
#             self.h8_2_channels = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
#             self.h16_2_channels = nn.Conv2d(1024, self.C, kernel_size=1, padding=0)
        
#         else:
#             self.ghostnet = nn.Sequential(*list(ghost.children())[:-4])  # holds layer 1 and 2
#             self.layer3 = ghost.classifier

#         self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
    
#         self.upsampling_layer = UpsamplingConcat(1536, 512)

#     def forward(self, x: torch.Tensor) -> dict:
#         feat_dict = {}
#         if self.use_multi_scale_img_feats:
#             x0 = self.ghostnet(x)  # (B*S, 3, H, W) --> (B*S, 64, H/4, W/4)
#             x1 = self.layer1(x0)  # (B*S, 64, H/4, W/4) --> (B*S, 256, H/4, W/4)
#             x2 = self.layer2(x1)  # (B*S, 256, H/4, W/4) --> (B*S, 512, H/8, W/8)
#             x3 = self.layer3(x2)  # (B*S, 512, H/8, W/8) --> (B*S, 1024, H/16, W/16)

#             # feature extraction with same channel depth:
#             x1_ = self.h4_2_channels(x1)
#             x2_ = self.h8_2_channels(x2)
#             x3_ = self.h16_2_channels(x3)

#             feat_dict = {
#                 # "feats_2": x0,
#                 "feats_4": x1_,
#                 "feats_8": x2_,
#                 "feats_16": x3_,
#             }
#         else:
#             x2 = self.backbone(x)  # res:
#             x3 = self.layer3(x2)  #

#         x = self.upsampling_layer(x3, x2)
#         x = self.depth_layer(x)
#         feat_dict["output"] = x

#         return feat_dict  # x


# import torch.nn.utils.prune as prune
# import torch 
# import torch.nn as nn

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Depthwise separable convolution 적용
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(x1, size=x2.shape[-2:], mode='nearest')
        x = torch.cat([x1, x2], dim=1)
        x = self.depthwise(x)
        return self.pointwise(x)


# class Encoder_ghost(nn.Module):
#     def __init__(self, C: int, use_multi_scale_img_feats: bool, prune_ratio: float = 0.3):
#         super().__init__()
#         self.C = C
#         self.use_multi_scale_img_feats = use_multi_scale_img_feats
#         self.prune_ratio = prune_ratio

#         # GhostNet 로드 (채널 수 줄이기)
#         ghost = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)

#         if self.use_multi_scale_img_feats:
#             self.ghostnet = nn.Sequential(*list(ghost.children())[:-6])

#             # Depthwise Separable Convolution 적용
#             self.layer1 = nn.Sequential(
#                 nn.Conv2d(16, 128, 3, 1, 1, bias=False),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True),
#             )
#             self.layer2 = nn.Sequential(
#                 nn.Conv2d(128, 256, 3, 2, 1, bias=False),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(inplace=True),
#             )
#             self.layer3 = nn.Sequential(
#                 nn.Conv2d(256, 512, 3, 2, 1, bias=False),
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(inplace=True),
#             )

#             self.h4_2_channels = nn.Conv2d(128, self.C, 1)
#             self.h8_2_channels = nn.Conv2d(256, self.C, 1)
#             self.h16_2_channels = nn.Conv2d(512, self.C, 1)
#         else:
#             self.ghostnet = nn.Sequential(*list(ghost.children())[:-4])
#             self.layer3 = ghost.classifier

#         # Depthwise Separable Convolution 적용
#         self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, bias=False)
#         self.upsampling_layer = UpsamplingConcat(768, 512)

#         self._apply_structured_pruning()

#     def _apply_structured_pruning(self):
#         def prune_module(module):
#             for name, submodule in module.named_modules():
#                 if isinstance(submodule, nn.Conv2d):
#                     prune.ln_structured(submodule, name="weight", amount=self.prune_ratio, n=2, dim=0)
#                     prune.remove(submodule, "weight")  

#         prune_module(self.ghostnet)
#         prune_module(self.layer1)
#         prune_module(self.layer2)
#         prune_module(self.layer3)
#         prune_module(self.depth_layer)

#         if self.use_multi_scale_img_feats:
#             prune_module(self.h4_2_channels)
#             prune_module(self.h8_2_channels)
#             prune_module(self.h16_2_channels)

    
#     def forward(self, x: torch.Tensor) -> dict:
#         feat_dict = {}
#         if self.use_multi_scale_img_feats:
#             x0 = self.ghostnet(x)
#             x1 = self.layer1(x0)
#             x2 = self.layer2(x1)
#             x3 = self.layer3(x2)

#             # 순차 실행 (병렬 아님)
#             x1_ = self.h4_2_channels(x1)
#             x2_ = self.h8_2_channels(x2)
#             x3_ = self.h16_2_channels(x3)

#             feat_dict = {
#                 "feats_4": x1_,
#                 "feats_8": x2_,
#                 "feats_16": x3_,
#             }
#         else:
#             x2 = self.ghostnet(x)
#             x3 = self.layer3(x2)

#         x = self.upsampling_layer(x3, x2)
#         x = self.depth_layer(x)
#         feat_dict["output"] = x
#         return feat_dict


class SegnetTransformerLiftFuse(nn.Module):
    def __init__(self, Z_cam: int, Y_cam: int, X_cam: int, Z_rad: int, Y_rad: int, X_rad: int,
                 vox_util: torch.Tensor = None,
                 use_pseudolidar: bool = False,
                 use_metapseudolidar: bool = False,
                 use_shallow_metadata: bool = False,
                 use_pseudolidar_encoder: bool = False,
                 do_rgbcompress: bool = False,
                 rand_flip: bool = False,
                 latent_dim: int = 128,
                 encoder_type: str = "vit_s",
                 pseudolidar_encoder_type: str = "voxel_net",
                 train_task: str = "both",
                 init_query_with_image_feats: bool = False,
                 use_obj_layer_only_on_map: bool = False,
                 do_feat_enc_dec: bool = False,
                 use_multi_scale_img_feats: bool = False,
                 num_layers: int = 4,
                 compress_adapter_output: bool = True,
                 use_pseudolidar_as_k_v: bool = False,
                 combine_feat_init_w_learned_q: bool = False,
                 use_rpn_pseudolidar: bool = False,
                 use_pseudolidar_occupancy_map: bool = False,
                 freeze_dino: bool = True,
                 learnable_fuse_query: bool = False,
                 is_master: bool = False):
        super(SegnetTransformerLiftFuse, self).__init__()
        assert (encoder_type in ["res101", "res50", "dino_v2", "vit_s","mobile","ghost"])
        assert (pseudolidar_encoder_type in ["voxel_net", None])
        assert (train_task in ["object", "map", "both"])

        self.Z_cam, self.Y_cam, self.X_cam = Z_cam, Y_cam, X_cam  # Z=200, Y=8, X=200
        self.Z_rad, self.Y_rad, self.X_rad = Z_rad, Y_rad, X_rad  # Z=200, Y=8, X=200
        self.use_pseudolidar = use_pseudolidar
        self.use_metapseudolidar = use_metapseudolidar
        self.use_shallow_metadata = use_shallow_metadata
        self.use_pseudolidar_encoder = use_pseudolidar_encoder
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.pseudolidar_encoder_type = pseudolidar_encoder_type
        self.train_task = train_task
        self.init_query_with_image_feats = init_query_with_image_feats  # use simple lifting for BEV query init.
        self.use_obj_layer_only_on_map = use_obj_layer_only_on_map
        self.do_feat_enc_dec = do_feat_enc_dec
        self.use_multi_scale_img_feats = use_multi_scale_img_feats
        self.num_layers = num_layers
        self.compress_adapter_output = compress_adapter_output
        self.use_pseudolidar_as_k_v = use_pseudolidar_as_k_v
        self.combine_feat_init_w_learned_q = combine_feat_init_w_learned_q

        self.use_pseudolidar_only_init = False
        self.use_rpn_pseudolidar = use_rpn_pseudolidar
        self.use_pseudolidar_occupancy_map = use_pseudolidar_occupancy_map
        self.freeze_dino = freeze_dino
        self.learnable_fuse_query = learnable_fuse_query
        self.is_master = is_master

        if self.is_master:
            print("latent_dim: ", latent_dim)

        # mean and std for every color channel -> how did they obtain the values?
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float().cuda()

        # Image Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim, use_multi_scale_img_feats=use_multi_scale_img_feats)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim, use_multi_scale_img_feats=use_multi_scale_img_feats)
        # elif encoder_type == "ghost":
        #     self.encoder = Encoder_ghost(feat2d_dim, use_multi_scale_img_feats=use_multi_scale_img_feats)
        elif encoder_type == "dino_v2" or encoder_type == "vit_s":
            if encoder_type == "vit_s":
                self.encoder = DinoAdapter(add_vit_feature=False, pretrain_size=518, pretrained_vit=True,
                                           freeze_dino=freeze_dino, embed_dim=384, num_heads=6)  # VIT-S14
            else:  # VIT-B14
                self.encoder = DinoAdapter(add_vit_feature=False, pretrain_size=518, pretrained_vit=True,
                                           freeze_dino=freeze_dino)
            if self.compress_adapter_output:  # dino embed dim: 768 --> desired: 128
                self.img_feats_compr_4 = nn.Sequential(
                    nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=latent_dim,
                              kernel_size=1, stride=1, bias=True),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
                self.img_feats_compr_8 = nn.Sequential(
                    nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=latent_dim,
                              kernel_size=1, stride=1, bias=True),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
                self.img_feats_compr_16 = nn.Sequential(
                    nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=latent_dim,
                              kernel_size=1, stride=1, bias=True),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
                self.img_feats_compr_32 = nn.Sequential(
                    nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=latent_dim,
                              kernel_size=1, stride=1, bias=True),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )

        # Pseudolidar Encoder
        if self.use_pseudolidar_encoder and self.use_pseudolidar:
            if self.pseudolidar_encoder_type == "voxel_net":
                # if reduced_zx==True -> 100x100 instead of 200x200
                # if use_col=True: added RPN after CML -->  in our case RPN lead to worse performance
                self.pseudolidar_encoder = VoxelNet(use_col=self.use_rpn_pseudolidar, reduced_zx=False,
                                              output_dim=latent_dim,
                                              use_pseudolidar_occupancy_map=self.use_pseudolidar_occupancy_map)
            else:
                print("Pseudolidar encoder not found ")
        elif not self.use_pseudolidar_encoder and self.use_pseudolidar and self.is_master:
            print("#############    NO PSEUDOLIDAR ENCODING    ##############")
        else:
            if self.is_master:
                print("#############    CAM ONLY    ##############")

        # image BEV 3D feature volume compressor:
        if self.init_query_with_image_feats:
            self.imgs_bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim * Y_cam, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )

            if self.use_pseudolidar:
                #self.image_based_query_attention = FusingCrossAttentionV2(dim=latent_dim)
                self.image_based_query_attention = Mamba_FusingCrossAttentionV2(dim=latent_dim) 

        # init queries
        self.bev_queries = nn.Parameter(0.1 * torch.randn(latent_dim, Z_cam, X_cam).float(), requires_grad=True)
        # positional encoding
        self.bev_queries_enc_pos = nn.Parameter(0.1 * torch.randn(latent_dim, Z_cam, X_cam).float(), requires_grad=True)

        if self.use_multi_scale_img_feats:
            num_levels = 4
        else:
            num_levels = 1

        self.self_attn_layers_encoder = nn.ModuleList([
            VanillaSelfAttention(dim=latent_dim) for _ in range(num_layers)
        ])  # deformable self attention

        
        # Lightweight self_mamba: d_state and expand are configurable via env vars.
        #   SEGMAM_MAMBA_D_STATE  (default 8 ; "light" uses 4)
        #   SEGMAM_MAMBA_EXPAND   (default 2 ; "light" uses 1)
        # The selective_scan cost ~ d_inner * d_state where d_inner = latent_dim * expand;
        # the causal conv / in-out proj cost ~ d_inner. So expand 2->1 ~halves the block,
        # d_state 8->4 ~halves the scan, both ~quarters the scan.
        # SEGMAM_MAMBA_LIGHT=1 is a convenience alias = d_state 4, expand 1 (used if the
        # granular vars are not set). Any non-(8,2) config needs a matching sliced checkpoint.
        _light_alias = os.environ.get("SEGMAM_MAMBA_LIGHT", "0") == "1"
        _mamba_d_state = int(os.environ.get("SEGMAM_MAMBA_D_STATE", "4" if _light_alias else "8"))
        _mamba_expand  = int(os.environ.get("SEGMAM_MAMBA_EXPAND",  "1" if _light_alias else "2"))
        if (_mamba_d_state, _mamba_expand) != (8, 2):
            self.self_mamba_attn_layers_fuser = nn.ModuleList([
                self_mamba(dim=latent_dim, d_state=_mamba_d_state, expand=_mamba_expand) for _ in range(num_layers)
            ])
        else:
            self.self_mamba_attn_layers_fuser = nn.ModuleList([
                self_mamba(dim=latent_dim) for _ in range(num_layers)
            ])  # sefl mamba

        #print(f" latne dim {latent_dim}")
        #print(f" num_layers  {num_layers}")

              
        self.spaptialcrossmamba = nn.ModuleList([
            SpatialCrossMambaFromCross(dim=latent_dim) for _ in range(num_layers)
        ]) # spatial crossmamba

        self.norm1_layers_encoder = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        self.cross_attn_layers_encoder = nn.ModuleList([
            SpatialCrossAttention(dim=latent_dim, num_levels=num_levels) for _ in range(num_layers)
        ])
        self.norm2_layers_encoder = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        
        self.norm4_layers_encoder = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])

        ffn_dim = 1028

        self.ffn_layers_encoder = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, latent_dim)) for _ in
            range(num_layers)
        ])
        self.norm3_layers_encoder = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])

        # TransFusion
        if self.use_pseudolidar:
            self.bev_fuse_queries = nn.Parameter(0.1 * torch.randn(latent_dim, Z_cam, X_cam).float(),
                                                 requires_grad=True)
            self.bev_queries_fuse_pos = nn.Parameter(0.1 * torch.randn(latent_dim, Z_cam, X_cam).float(),
                                                     requires_grad=True)  # C, Z, X
            self.self_attn_layers_fuser = nn.ModuleList([
                VanillaSelfAttention(dim=latent_dim) for _ in range(num_layers)
            ])  # deformable self attention


            # selfmamba
            # self.self_mamba_attn_layers_fuser = nn.ModuleList([
            #     self_mamba(dim=latent_dim) for _ in range(num_layers)
            # ])  # deformable self attention

            # # crossmamba
            self.cross_mamba_attn_layers_fuser = nn.ModuleList([
                cross_mamba(dim=latent_dim) for _ in range(num_layers)
            ])  


            self.Mamba_cross_mamba_attn_layers_fuser = nn.ModuleList([
                Mamba_FusingCrossAttentionV2(dim=latent_dim) for _ in range(num_layers)
            ])  


            self.norm1_layers_fuser = nn.ModuleList([
                nn.LayerNorm(latent_dim) for _ in range(num_layers)
            ])
            

            self.fusing_attn_layers_fuser = nn.ModuleList([
                FusingCrossAttentionV2(dim=latent_dim) for _ in range(num_layers)
            ])

            self.norm2_layers_fuser = nn.ModuleList([
                nn.LayerNorm(latent_dim) for _ in range(num_layers)
            ])
            
            self.norm4_layers_fuser = nn.ModuleList([
                nn.LayerNorm(latent_dim) for _ in range(num_layers)
            ])

            ffn_dim = 1028

            self.ffn_layers_fuser = nn.ModuleList([
                nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, latent_dim)) for _ in
                range(num_layers)
            ])
            self.norm3_layers_fuser = nn.ModuleList([
                nn.LayerNorm(latent_dim) for _ in range(num_layers)
            ])

        if self.is_master:
            print("Transformer initialized")

        # Apply Feature Encoder - Decoder on fused features
        if do_feat_enc_dec:
            self.feat_enc_dec = FeatureEncoderDecoder(in_channels=self.latent_dim)
            

        # Decoder
        """
            class 0:    'drivable_area' --- color in rbg: (1.00, 0.50, 0.31)\n
            class 1:    'carpark_area'  --- color '#FFD700' in rbg: (255./255., 215./255., 0./255)\n
            class 2:    'ped_crossing'  --- color '#069AF3' in rbg: (6./255., 154/255., 243/255.) \n
            class 3:    'walkway'       --- color '#FF00FF' in rbg: (255./255., 0./255., 255./255.) \n
            class 4:    'stop_line'     --- color '#FF0000' in rbg: (255./255., 0./255., 0./255.) \n
            class 5:    'road_divider'  --- color in rbg: (0.0, 0.0, 1.0)\n
            class 6:    'lane_divider'  --- color in rbg: (159./255., 0.0, 1.0)\n

            optional:
            class 7: Objects --> Vehicles as additional class for Object segmentation
            other -> considered background
        """
        if self.train_task == "object":
            self.object_decoder = TaskSpecificDecoder(in_channels=self.latent_dim,
                                                      task='object_decoder',
                                                      n_classes=1,
                                                      use_feat_head=False,
                                                      predict_future_flow=False)
            self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        elif self.train_task == "map":
            self.map_decoder = TaskSpecificDecoder(in_channels=self.latent_dim,
                                                   task='map_decoder',
                                                   n_classes=7,
                                                   use_feat_head=True,
                                                   predict_future_flow=False)
            
            #print(f'segnet 1073 self.map_decoder {self.map_decoder }')
            self.fc_map_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        elif self.train_task == "both":
            self.shared_decoder = TaskSpecificDecoder(in_channels=self.latent_dim,
                                                      task='shared_decoder',
                                                      n_classes=8,
                                                      use_feat_head=False,
                                                      predict_future_flow=False,
                                                      use_obj_layer_only_on_map=use_obj_layer_only_on_map)
            self.fc_map_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            print("invalid task")

        if self.is_master:
            print("Decoder initialized")

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z_cam, Y_cam, X_cam, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z_cam, Y_cam, X_cam,
                                             assert_cube=False)  # transforms mem coordinates into ref coordinates
        else:
            self.xyz_camA = None

    def forward(self, rgb_camXs: torch.Tensor, pix_T_cams: torch.Tensor, cam0_T_camXs: torch.Tensor,
                vox_util: torch.Tensor, rad_occ_mem0: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            B = batch size, S = number of cameras, C = 3, H = img height, W = img width
            rgb_camXs: (B,S,C,H,W)
            pix_T_cams: (B,S,4,4)
            cam0_T_camXs: (B,S,4,4)
            vox_util: vox util object
            rad_occ_mem0:
                - None when use_pseudolidar = False
                - (B, 1, Z, Y, X) when use_pseudolidar = True, use_metapseudolidar = False
                - (B, 16, Z, Y, X) when use_pseudolidar = True, use_metapseudolidar = True
        Returns:
            torch.Tensor: predicted segmentation
        """
        
        ''' for Preprocessing start '''
        timings = {}

        processing_starter, processing_ender = _SafeEvent(_SEGMAM_PROF_ENABLED), _SafeEvent(_SEGMAM_PROF_ENABLED)
        processing_starter.record()
        
        B, S, C, H, W = rgb_camXs.shape
        assert (C == 3)

        def __p(x: torch.Tensor) -> torch.Tensor:
            # Wrapper function: e.g. unites B,S dim to B*S -> pack_seqdim: reshaping: (B,S,C,H,W) -> ([B*S],C,H,W)
            return utils.basic.pack_seqdim(x, B)

        def __u(x: torch.Tensor) -> torch.Tensor:
            # Wrapper function: e.g. splits B*S dim into B,S -> unpack_seqdim: reshaping: ([B*S],C,H,W) -> (B,S,C,H,W)
            return utils.basic.unpack_seqdim(x, B)

        # reshape tensors: __p -> "pack" input
        rgb_camXs_ = __p(rgb_camXs)  # (B,S,C,H,W)   ->  ([B*S],C,H,W)
        pix_T_cams_ = __p(pix_T_cams)  # (B,S,4,4)     ->  ([B*S],4,4)
        cam0_T_camXs_ = __p(cam0_T_camXs)  # (B,S,4,4)     ->  ([B*S],4,4)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)  # inverse of transformation matrix

        # rgb encoder
        device = rgb_camXs_.device
        # input normalization: add 0.5 and subtract the color-channel-specific mean
        # divide through the color-specific std
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)

        B0, _, _, _ = rgb_camXs_.shape

        dinovoxel = None

        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            # decide which images in one batch should be flipped
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            # -1: flip on last dim -> W -> flip vertically
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])

        ''' for Preprocessing end '''
        #end_preprocessing_time = time.time()
        processing_ender.record()
        _maybe_sync()
        Preprocessing = processing_starter.elapsed_time(processing_ender)
        #print(f"Preprocessing {Preprocessing}")
        timings['preprocessing time(ms)'] = Preprocessing
        
        # put randomly flipped input data into encoder
        # image features as output of modified encoder -> 128 x H/8 x W/8
        
        ''' for start_image_encoder_time start '''
        image_encoder_starter, image_encoder_processing_ender = _SafeEvent(_SEGMAM_PROF_ENABLED), _SafeEvent(_SEGMAM_PROF_ENABLED)
        image_encoder_starter.record()



        feats_4 = 0
        feats_8 = 0
        feats_16 = 0
        feats_32 = 0
        if self.use_multi_scale_img_feats:
            if self.encoder_type == 'dino_v2' or self.encoder_type == 'vit_s':
                # we need to get the feature space down to 128
                img_encoder_feats, dino_out = self.encoder(rgb_camXs_)
                if self.compress_adapter_output:
                    feats_4 = self.img_feats_compr_4(img_encoder_feats[0])
                    feats_8 = self.img_feats_compr_8(img_encoder_feats[1])
                    feats_16 = self.img_feats_compr_16(img_encoder_feats[2])
                    feats_32 = self.img_feats_compr_32(img_encoder_feats[3])
                    feat_camXs_ = feats_8

                else:
                    feats_4 = img_encoder_feats[0]
                    feats_8 = img_encoder_feats[1]
                    feats_16 = img_encoder_feats[2]
                    feats_32 = img_encoder_feats[3]
                    feat_camXs_ = feats_8

            else:
                img_encoder_feats = self.encoder(rgb_camXs_)
                feats_4 = img_encoder_feats["feats_4"]
                feats_8 = img_encoder_feats["feats_8"]
                feats_16 = img_encoder_feats["feats_16"]
                feat_camXs_ = img_encoder_feats["output"]

        else:
            feat_camXs_ = self.encoder(rgb_camXs_)["output"]

        if self.use_multi_scale_img_feats:
            # "unflip" the image feature maps based on the same random order of the image flipping
            if self.rand_flip:
                feats_4[self.rgb_flip_index] = torch.flip(feats_4[self.rgb_flip_index], [-1])
                feats_8[self.rgb_flip_index] = torch.flip(feats_8[self.rgb_flip_index], [-1])
                feats_16[self.rgb_flip_index] = torch.flip(feats_16[self.rgb_flip_index], [-1])
                feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
                if self.encoder_type == 'dino_v2' or self.encoder_type == 'vit_s':
                    feats_32[self.rgb_flip_index] = torch.flip(feats_32[self.rgb_flip_index], [-1])
            # unpack
            feats_4 = __u(feats_4)  # (B, S, C, H/4, W/4)
            feats_8 = __u(feats_8)  # (B, S, C, H/8, W/8)
            feats_16 = __u(feats_16)  # (B, S, C, H/16, W/16)
            feat_camXs = __u(feat_camXs_)  # (B, S, C, H/8, W/8)
            if self.encoder_type == 'dino_v2' or self.encoder_type == 'vit_s':
                feats_32 = __u(feats_32)
        else:
            # "unflip" the image feature maps based on the same random order of the image flipping
            if self.rand_flip:
                feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
            feat_camXs = __u(feat_camXs_)  # (B, S, C, Hf, Wf)

        _, C, Hf, Wf = feat_camXs_.shape  # C=128, Hf=H/8, Wf=W/8

        image_encoder_processing_ender.record()
        _maybe_sync()
        Image_Encoder= image_encoder_starter.elapsed_time(image_encoder_processing_ender)
        timings['Image Encoder(ms)'] = Image_Encoder
        
        '''Image Encoder end '''

        sy = Hf / float(H)  # sy = 1/8
        sx = Wf / float(W)  # sx = 1/8
        
        Z_cam, Y_cam, X_cam = self.Z_cam, self.Y_cam, self.X_cam  # 100, 4, 100
        
        '''start comput img time test'''
        #compute the image locations (no flipping for now)
        comput_img_starter, comput_img_ender = _SafeEvent(_SEGMAM_PROF_ENABLED), _SafeEvent(_SEGMAM_PROF_ENABLED)
        comput_img_starter.record()

        # compute the image locations (no flipping for now)
        xyz_mem_ = utils.basic.gridcloud3d(B0, Z_cam, Y_cam, X_cam, norm=False, device=rgb_camXs.device)  # B0, Z*Y*X,3
        xyz_cam0_ = vox_util.Mem2Ref(xyz_mem_, Z_cam, Y_cam, X_cam, assert_cube=False)
        xyz_camXs_ = utils.geom.apply_4x4(camXs_T_cam0_, xyz_cam0_)
        xy_camXs_ = utils.geom.camera2pixels(xyz_camXs_, pix_T_cams_)  # B0, N, 2
        xy_camXs = __u(xy_camXs_)  # B, S, N, 2, where N=Z*Y*X
        reference_points_cam = xy_camXs_.reshape(B, S, Z_cam, Y_cam, X_cam, 2).permute(1, 0, 2, 4, 3, 5).\
            reshape(S, B, Z_cam * X_cam, Y_cam, 2)
        reference_points_cam[..., 0:1] = reference_points_cam[..., 0:1] / float(
            W)  # scale x  -> should be [ -1,1] for grid sampling
        reference_points_cam[..., 1:2] = reference_points_cam[..., 1:2] / float(H)  # scale y
        bev_mask = ((reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0)).squeeze(-1)

        
        comput_img_ender.record()
        _maybe_sync()
        comput_img = comput_img_starter.elapsed_time(comput_img_ender)
        timings['comput_img time(ms)'] = comput_img

        '''end_compute_img_time'''
        

        '''start_pseudolidar_encoder_time'''
        pseudolidar_encoder_starter, pseudolidar_encoder_processing_ender = _SafeEvent(_SEGMAM_PROF_ENABLED), _SafeEvent(_SEGMAM_PROF_ENABLED)
        pseudolidar_encoder_starter.record()
        
        # first get pseudolidar feats from rad encoder
        rad_bev_ = 0
        if self.use_pseudolidar:
            assert (rad_occ_mem0 is not None)
            Z_rad, Y_rad, X_rad = self.Z_rad, self.Y_rad, self.X_rad

            # add pseudolidar encoding branch
            if self.use_pseudolidar_encoder:
                if self.pseudolidar_encoder_type == 'voxel_net':
                    rad_bev_ = self.pseudolidar_encoder(voxel_features=rad_occ_mem0[0],
                                                  voxel_coords=rad_occ_mem0[1],
                                                  number_of_occupied_voxels=rad_occ_mem0[2],
                                                  dinovoxel=dinovoxel)
                elif self.use_shallow_metadata:
                    rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 4 * Y_rad, Z_rad, X_rad)
                    rad_bev_ = self.pseudolidar_encoder(rad_bev_)
                else:
                    rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16 * Y_rad, Z_rad, X_rad)
                    rad_bev_ = self.pseudolidar_encoder(rad_bev_)

            elif self.use_shallow_metadata and not self.use_pseudolidar_encoder:
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 4 * Y_rad, Z_rad, X_rad)  # B,32,200,200
                # for the transformer, we need matching feature dims! --> apply zero-padding here!
                zero_padding = torch.zeros((B, self.latent_dim - (4 * Y_rad), Z_rad, X_rad)).to(device)
                rad_bev_ = torch.cat((rad_bev_, zero_padding), dim=1).to(device)  # C=128

            else:
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16 * Y_rad, Z_rad, X_rad)  # C=128

        pseudolidar_encoder_processing_ender.record()
        _maybe_sync()
        pseudolidar_encoder = pseudolidar_encoder_starter.elapsed_time(pseudolidar_encoder_processing_ender)
        #print(f"pseudolidar_encoder {pseudolidar_encoder}")
        timings['pseudolidar_encoder time(ms)'] = pseudolidar_encoder
        
        '''end_pseudolidar_encoder_time'''

        

        
        # #### Transformer STAGE ####
        '''start lifting time'''
        lifting_starter, lifting_ender = _SafeEvent(_SEGMAM_PROF_ENABLED), _SafeEvent(_SEGMAM_PROF_ENABLED)
        lifting_starter.record()
        
        if self.init_query_with_image_feats:
            # new image feat init
            featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)

            if self.xyz_camA is not None:
                # 3d mem in view of reference cam??? (in meters)
                xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B * S, 1, 1)
            else:
                xyz_camA = None

            # unproject_image_to_mem: transforms image features from all cams into shared 3D feature space
            feat_mems_ = vox_util.unproject_image_to_mem(
                feat_camXs_,
                utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
                camXs_T_cam0_, Z_cam, Y_cam, X_cam,
                xyz_camA=xyz_camA)

            # unpack features from ([B*S], C, Z, Y, X) -> (B, S, C, Z, Y, X)
            feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X

            # mask is 1 if abs value is != 0 else zero
            mask_mems = (torch.abs(feat_mems) > 0).float()
            # S = 0 since in 3D feature space we don't need the number of cams dim -> reduce this dim
            feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X
            # reshape to get right feat dim
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y_cam, Z_cam, X_cam)

            # do uncompression to get query shape
            # goal: learns the relevant features that correspond to pseudolidar readings
            feat_bev_q_dim = self.imgs_bev_compressor(feat_bev_)
            img_feat_bev_q_dim = feat_bev_q_dim.permute(0, 2, 3, 1).reshape(B, -1, self.latent_dim)

            if self.use_pseudolidar:
                rad_bev_q_dim = rad_bev_.permute(0, 2, 3, 1).reshape(B, -1, self.latent_dim)
                # use pseudolidar-initialized query on simple 3D image space
                bev_queries = self.image_based_query_attention(rad_bev_q_dim,
                                                               img_feat_bev_q_dim)  # problem: shallow metadata -> feature dim is wrong...
            else:
                bev_queries = img_feat_bev_q_dim

        elif self.use_pseudolidar_only_init:
            # new: use pseudolidar features as bev query
            bev_queries = rad_bev_.permute(0, 2, 3, 1).reshape(B, -1, self.latent_dim)
        else:
            bev_queries = self.bev_queries.clone().unsqueeze(0).repeat(B, 1, 1, 1).\
                reshape(B, self.latent_dim, -1).permute(0, 2, 1)

        bev_queries_enc_pos = self.bev_queries_enc_pos.clone().unsqueeze(0).repeat(B, 1, 1, 1).\
            reshape(B, self.latent_dim, -1).permute(0, 2, 1)  # B, Z*X, C
        bev_queries_learned = self.bev_queries.clone().unsqueeze(0).repeat(B, 1, 1, 1).\
            reshape(B, self.latent_dim, -1).permute(0, 2, 1)  # B, Z*X, C

        bev_queries_fuse_pos = torch.zeros_like(bev_queries)
        bev_fuse_queries_learned = torch.zeros_like(bev_queries)
        if self.use_pseudolidar:
            bev_queries_fuse_pos = self.bev_queries_fuse_pos.clone().unsqueeze(0).repeat(B, 1, 1, 1) \
                .reshape(B, self.latent_dim, -1).permute(0, 2, 1)  # B, Z*X, C
            bev_fuse_queries_learned = self.bev_fuse_queries.clone().unsqueeze(0).repeat(B, 1, 1, 1).\
                reshape(B, self.latent_dim, -1).permute(0, 2, 1)  # B, Z*X, C

        if self.use_multi_scale_img_feats:
            # collect bev keys and store spatial shapes
            spatial_shapes = bev_queries.new_zeros([4, 2]).long()
            bev_keys = []
            # feats_4:
            _, _, c_4, h4, w4 = feats_4.shape
            _, _, c_8, h8, w8 = feats_8.shape
            _, _, c_16, h16, w16 = feats_16.shape

            spatial_shapes[0, 0] = h4
            spatial_shapes[0, 1] = w4
            spatial_shapes[1, 0] = h8
            spatial_shapes[1, 1] = w8
            spatial_shapes[2, 0] = h16
            spatial_shapes[2, 1] = w16

            bev_keys.append(feats_4.reshape(B, S, c_4, h4 * w4).permute(1, 3, 0, 2))  # S, M, B, C
            bev_keys.append(feats_8.reshape(B, S, c_8, h8 * w8).permute(1, 3, 0, 2))  # S, M, B, C
            bev_keys.append(feats_16.reshape(B, S, c_16, h16 * w16).permute(1, 3, 0, 2))  # S, M, B, C

            if self.encoder_type == 'dino_v2' or self.encoder_type == 'vit_s':
                _, _, c_32, h32, w32 = feats_32.shape
                spatial_shapes[3, 0] = h32
                spatial_shapes[3, 1] = w32
                bev_keys.append(feats_32.reshape(B, S, c_32, h32 * w32).permute(1, 3, 0, 2))  # S, M, B, C
            else:
                _, _, c_f, hf, wf = feat_camXs.shape
                spatial_shapes[3, 0] = hf
                spatial_shapes[3, 1] = wf
                bev_keys.append(feat_camXs.reshape(B, S, c_f, hf * wf).permute(1, 3, 0, 2))  # S, M, B, C

            # make contiguous in memory and store as tensor
            bev_keys = torch.cat(bev_keys, dim=1).contiguous()
        else:
            bev_keys = feat_camXs.reshape(B, S, C, Hf * Wf).permute(1, 3, 0, 2)  # S, M, B, C
            spatial_shapes = bev_queries.new_zeros([1, 2]).long()
            spatial_shapes[0, 0] = Hf
            spatial_shapes[0, 1] = Wf

        # positional encoding
        if self.combine_feat_init_w_learned_q:
            bev_queries = bev_queries + bev_queries_learned + bev_queries_enc_pos
        else:
            bev_queries = bev_queries + bev_queries_enc_pos

        # encoder
        bev_queries_cam_out = torch.zeros_like(bev_queries)

        '''default'''
        # for i in range(self.num_layers):
        #     # self attention within the features (B, N, C)
        #     bev_queries = self.self_attn_layers_encoder[i](bev_queries)

        #     # normalize (B, N, C)
        #     bev_queries = self.norm1_layers_encoder[i](bev_queries)

        #     # cross attention into the images
        #     bev_queries = self.cross_attn_layers_encoder[i](bev_queries, bev_keys, bev_keys,
        #                                                     query_pos=None,
        #                                                     reference_points_cam=reference_points_cam,
        #                                                     spatial_shapes=spatial_shapes,
        #                                                     bev_mask=bev_mask)

        #     # normalize (B, N, C)
        #     bev_queries = self.norm2_layers_encoder[i](bev_queries)

        #     # feedforward layer (B, N, C)
        #     bev_queries = bev_queries + self.ffn_layers_encoder[i](bev_queries)

        #     # normalize (B, N, C)
        #     bev_queries_cam_out = self.norm3_layers_encoder[i](bev_queries)

        #     bev_queries = bev_queries_cam_out

        '''mamba lift'''
        # VIS_EVAL_LIFT_BREAKDOWN=1 (one-shot, set after warmup) prints per-sub-module
        # timing for one layer of the lift loop. Strictly diagnostic.
        _lift_break = os.environ.get("VIS_EVAL_LIFT_BREAKDOWN", "0") == "1"
        if _lift_break:
            torch.cuda.synchronize()
            _lift_t = [("start", torch.cuda.Event(enable_timing=True))]
            _lift_t[-1][1].record()
        for i in range(self.num_layers):

            # self attention within the features (B, N, C)
            #bev_queries = self.self_attn_layers_encoder[i](bev_queries)

            #bev_queries = self.self_mamba_attn_layers_fuser[i](bev_queries)

            # In inference_mode / no_grad, torch.utils.checkpoint adds no real recompute
            # cost but still hits autograd bookkeeping. Skip the wrapper outside of
            # gradient-tracking contexts to remove that per-layer overhead.
            if torch.is_grad_enabled():
                bev_queries = checkpoint(self.self_mamba_attn_layers_fuser[i], bev_queries, use_reentrant=False)
            else:
                bev_queries = self.self_mamba_attn_layers_fuser[i](bev_queries)
            if _lift_break and i == 0:
                _lift_t.append(("mamba", torch.cuda.Event(enable_timing=True))); _lift_t[-1][1].record()

            bev_queries = self.self_attn_layers_encoder[i](bev_queries)
            if _lift_break and i == 0:
                _lift_t.append(("self_attn", torch.cuda.Event(enable_timing=True))); _lift_t[-1][1].record()

                        


            # normalize (B, N, C)
            bev_queries = self.norm1_layers_encoder[i](bev_queries)

            #cross attention into the images
            # bev_queries = self.cross_attn_layers_encoder[i](bev_queries, bev_keys, bev_keys,
            #                                                 query_pos=None,
            #                                                 reference_points_cam=reference_points_cam,
            #                                                 spatial_shapes=spatial_shapes,
            #                                                 bev_mask=bev_mask)

            # bev_queries = checkpoint(self.spaptialcrossmamba[i](bev_queries, bev_keys, bev_keys,
            #                                                 query_pos=None,
            #                                                 reference_points_cam=reference_points_cam,
            #                                                 spatial_shapes=spatial_shapes,
            #                                                 bev_mask=bev_mask))

            # bev_queries = checkpoint(
            #     lambda q, k, v, rp, ss, bm: self.spaptialcrossmamba[i](
            #         q, k, v,
            #         query_pos=None,
            #         reference_points_cam=rp,
            #         spatial_shapes=ss,
            #         bev_mask=bm
            #     ),
            #     bev_queries,            # q
            #     bev_keys,               # k
            #     bev_keys,               # v
            #     reference_points_cam,   # rp
            #     spatial_shapes,         # ss
            #     bev_mask,               # bm
            #     use_reentrant=False
            # )
            
            
            bev_queries = self.cross_attn_layers_encoder[i](bev_queries, bev_keys, bev_keys,
                                                            query_pos=None,
                                                            reference_points_cam=reference_points_cam,
                                                            spatial_shapes=spatial_shapes,
                                                            bev_mask=bev_mask)
            if _lift_break and i == 0:
                _lift_t.append(("cross_attn", torch.cuda.Event(enable_timing=True))); _lift_t[-1][1].record()

            # normalize (B, N, C)
            bev_queries = self.norm2_layers_encoder[i](bev_queries)
            
            # bev_queries = self.spaptialcrossmamba[i](bev_queries, bev_keys, bev_keys,
            #                                                 query_pos=None,
            #                                                 reference_points_cam=reference_points_cam,
            #                                                 spatial_shapes=spatial_shapes,
            #                                                 bev_mask=bev_mask)

                    

            
            # normalize (B, N, C)
            #bev_queries = self.norm3_layers_encoder[i](bev_queries)

            # feedforward layer (B, N, C)
            bev_queries = bev_queries + self.ffn_layers_encoder[i](bev_queries)
            if _lift_break and i == 0:
                _lift_t.append(("ffn", torch.cuda.Event(enable_timing=True))); _lift_t[-1][1].record()

            # normalize (B, N, C)
            bev_queries_cam_out = self.norm3_layers_encoder[i](bev_queries)

            bev_queries = bev_queries_cam_out
            if _lift_break and i == 0:
                _lift_t.append(("layer_end", torch.cuda.Event(enable_timing=True))); _lift_t[-1][1].record()
                torch.cuda.synchronize()
                print("[lift-breakdown layer0] " + ", ".join(
                    f"{name}={prev_ev.elapsed_time(ev):.2f}ms"
                    for (prev_name, prev_ev), (name, ev) in zip(_lift_t[:-1], _lift_t[1:])))


        lifting_ender.record()
        _maybe_sync()
        lifting = lifting_starter.elapsed_time(lifting_ender)
        #print(f"lifting {lifting}")
        timings['lifting time(ms)'] = lifting

        '''end_lifting_time'''


        # fuser
        '''start_fuser_time'''
        fuser_starter, fuser_ender = _SafeEvent(_SEGMAM_PROF_ENABLED), _SafeEvent(_SEGMAM_PROF_ENABLED)
        fuser_starter.record()


        bev_queries_fuser_out = torch.zeros_like(bev_queries)
        if self.use_pseudolidar:
            # convert pseudolidar features into matching query
            rad_bev_query = rad_bev_.permute(0, 2, 3, 1).reshape(B, -1, self.latent_dim)

            # positional encoding
            if self.use_pseudolidar_as_k_v:
                if self.combine_feat_init_w_learned_q:
                    fuse_bev_queries = bev_queries_cam_out + bev_fuse_queries_learned + bev_queries_fuse_pos
                else:
                    fuse_bev_queries = bev_queries_cam_out + bev_queries_fuse_pos
            else:  # use pseudolidar as query
                if self.combine_feat_init_w_learned_q or self.learnable_fuse_query:
                    fuse_bev_queries = rad_bev_query + bev_fuse_queries_learned + bev_queries_fuse_pos
                else:
                    fuse_bev_queries = rad_bev_query + bev_queries_fuse_pos
            # rad_bev_query = rad_bev_query + bev_queries_fuse_pos

            '''default fuser start'''
            # for i in range(self.num_layers):
            #     # self attention within the features (B, N, C)
            #     bev_queries = self.self_attn_layers_fuser[i](fuse_bev_queries)

            #     # normalize (B, N, C)
            #     bev_queries = self.norm1_layers_fuser[i](bev_queries)
            #     # fusing layer
            #     if self.use_pseudolidar_as_k_v:
            #         bev_queries = self.fusing_attn_layers_fuser[i](bev_queries, rad_bev_query)  # rad_bev_query
            #     else:
            #         bev_queries = self.fusing_attn_layers_fuser[i](bev_queries, bev_queries_cam_out)

            #     # normalize (B, N, C)
            #     bev_queries = self.norm2_layers_fuser[i](bev_queries)
            #     # feedforward layer (B, N, C)
            #     bev_queries = bev_queries + self.ffn_layers_fuser[i](bev_queries)

            #     # normalize (B, N, C)
            #     bev_queries_fuser_out = self.norm3_layers_fuser[i](bev_queries)

            #     # rad_bev_query = bev_queries_fuser_out
            #     fuse_bev_queries = bev_queries_fuser_out

            '''default fuser end'''


            '''default fuser start'''
            for i in range(self.num_layers):
                # self attention within the features (B, N, C)
                
                #bev_queries = self.self_mamba_attn_layers_fuser[i](fuse_bev_queries)
                #bev_queries = checkpoint(self.self_mamba_attn_layers_fuser[i], fuse_bev_queries, use_reentrant=False)
                
                bev_queries = self.cross_mamba_attn_layers_fuser[i](bev_queries, rad_bev_query)
                
                bev_queries = self.self_attn_layers_fuser[i](fuse_bev_queries)
                #bev_queries = self.self_mamba_attn_layers_fuser[i](bev_queries)
                
                # normalize (B, N, C)
                bev_queries = self.norm1_layers_fuser[i](bev_queries)

                
                if self.use_pseudolidar_as_k_v:
                    bev_queries = self.fusing_attn_layers_fuser[i](bev_queries, rad_bev_query)  # rad_bev_query
                else:
                    bev_queries = self.fusing_attn_layers_fuser[i](bev_queries, bev_queries_cam_out)



                bev_queries = self.norm2_layers_fuser[i](bev_queries)
                
                #bev_queries = self.norm3_layers_fuser[i](bev_queries)
                
                # feedforward layer (B, N, C)
                bev_queries = bev_queries + self.ffn_layers_fuser[i](bev_queries)

                # normalize (B, N, C)
                bev_queries_fuser_out = self.norm3_layers_fuser[i](bev_queries)

                # rad_bev_query = bev_queries_fuser_out
                fuse_bev_queries = bev_queries_fuser_out

            '''default fuser end'''



        if self.use_pseudolidar:
            feat_bev_ = bev_queries_fuser_out.permute(0, 2, 1).reshape(B, self.latent_dim, self.Z_cam, self.X_cam)
        else:
            feat_bev_ = bev_queries_cam_out.permute(0, 2, 1).reshape(B, self.latent_dim, self.Z_cam, self.X_cam)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_bev_[self.bev_flip1_index] = torch.flip(feat_bev_[self.bev_flip1_index], [-1])
            feat_bev_[self.bev_flip2_index] = torch.flip(feat_bev_[self.bev_flip2_index], [-2])

            if rad_occ_mem0 is not None and not (self.pseudolidar_encoder_type == "voxel_net"):
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])
        

        fuser_ender.record()
        _maybe_sync()
        fuser = fuser_starter.elapsed_time(fuser_ender)
        #print(f"fuser {fuser}")
        timings['fuser time(ms)'] = fuser
        '''end_fuser_time'''

        '''start bev Enc-Dec'''
        # bev Enc-Dec
        enc_dec_starter, enc_dec_ender = _SafeEvent(_SEGMAM_PROF_ENABLED), _SafeEvent(_SEGMAM_PROF_ENABLED)
        enc_dec_starter.record()

        if self.do_feat_enc_dec:
            feat_bev = self.feat_enc_dec(feat_bev_)
        else:
            feat_bev = feat_bev_

        enc_dec_ender.record()
        _maybe_sync()
        enc_dec = enc_dec_starter.elapsed_time(enc_dec_ender)
        #print(f"enc_dec {enc_dec}")
        timings['enc_dec time(ms)'] = enc_dec

        '''end_bev_encoder_time'''


        '''strat bev decoder '''
        # bev decoder
        decoder_starter, decoder_ender = _SafeEvent(_SEGMAM_PROF_ENABLED), _SafeEvent(_SEGMAM_PROF_ENABLED)
        decoder_starter.record()

        seg_e = {}

        if self.train_task == "object":
            out_dict_objects = self.object_decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index)
                                                   if self.rand_flip else None)
            # object estimation data
            obj_seg_e = out_dict_objects['obj_segmentation']
            seg_e = obj_seg_e
            
        if self.train_task == "map":
            
            '''default map seg'''
            out_dict_map = self.map_decoder(feat_bev,
                                            (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
            # map estimation data
            bev_map_seg_e = out_dict_map['bev_map_segmentation']
            seg_e = bev_map_seg_e
                        

        if self.train_task == "both":
            out_dict_shared = self.shared_decoder(feat_bev, (
                self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
            # map estimation data
            bev_map_seg_e = out_dict_shared['bev_map_segmentation']
            obj_seg_e = out_dict_shared['obj_segmentation']
            #print(f"obev_map_seg_e 1482  {bev_map_seg_e}")
            seg_e = torch.cat([bev_map_seg_e, obj_seg_e], dim=1)  # [b, 8, 200, 200]
        
        '''end_bev_decoder_time'''
        decoder_ender.record()
        _maybe_sync()
        decoder = decoder_starter.elapsed_time(decoder_ender)
        #print(f"decoder {decoder}")
        timings['decoder time(ms)'] = decoder

        total =  processing_starter.elapsed_time(decoder_ender)
        #print(f"total {total}")
        timings['total Time(ms)'] = total
        '''end_bev_decoder_time'''
    

        return seg_e,timings #for inference time return # for visualization 
        #return seg_e # for training
        