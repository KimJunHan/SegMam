# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import MultiScaleDeformableAttention as MSDA
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        # The CUDA kernel is fp32-only ("ms_deform_attn_forward_cuda" not implemented for 'Half').
        # Under autocast (fp16/bf16) we must run this op in fp32, then cast back.
        ctx.im2col_step = im2col_step
        ctx._orig_dtype = value.dtype
        if value.dtype != torch.float32:
            value_f = value.float()
            sampling_locations_f = sampling_locations.float()
            attention_weights_f = attention_weights.float()
        else:
            value_f, sampling_locations_f, attention_weights_f = value, sampling_locations, attention_weights
        output = MSDA.ms_deform_attn_forward(
            value_f, value_spatial_shapes, value_level_start_index,
            sampling_locations_f, attention_weights_f, ctx.im2col_step)
        ctx.save_for_backward(value_f, value_spatial_shapes, value_level_start_index,
                              sampling_locations_f, attention_weights_f)
        if output.dtype != ctx._orig_dtype:
            output = output.to(ctx._orig_dtype)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        if grad_output.dtype != torch.float32:
            grad_output = grad_output.float()
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        orig_dtype = getattr(ctx, "_orig_dtype", torch.float32)
        if orig_dtype != torch.float32:
            grad_value = grad_value.to(orig_dtype)
            grad_sampling_loc = grad_sampling_loc.to(orig_dtype)
            grad_attn_weight = grad_attn_weight.to(orig_dtype)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
    

# class MSDeformAttnFunction(Function):
#     @staticmethod
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
#         ctx.im2col_step = im2col_step
#         ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
#         output = ms_deform_attn_core_pytorch(
#             value, value_spatial_shapes, sampling_locations, attention_weights
#         )
#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors

#         # 모든 입력값에 대해 requires_grad=True로 복사
#         value = value.detach().requires_grad_(True)
#         value_spatial_shapes = value_spatial_shapes.detach() if torch.is_tensor(value_spatial_shapes) else value_spatial_shapes
#         sampling_locations = sampling_locations.detach().requires_grad_(True)
#         attention_weights = attention_weights.detach().requires_grad_(True)

#         # forward를 다시 호출하여 output을 얻고, grad_output으로 backward
#         output = ms_deform_attn_core_pytorch(
#             value, value_spatial_shapes, sampling_locations, attention_weights
#         )
#         grads = torch.autograd.grad(
#             outputs=output,
#             inputs=(value, sampling_locations, attention_weights),
#             grad_outputs=grad_output,
#             allow_unused=True,
#             retain_graph=True
#         )

#         grad_value = grads[0]
#         grad_sampling_locations = grads[1]
#         grad_attention_weights = grads[2]

#         # value_level_start_index, im2col_step은 미분 불필요
#         return grad_value, None, None, grad_sampling_locations, grad_attention_weights, None

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)  (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()

# def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
#     # for debug and test only,
#     # need to use cuda version instead
#     N_, S_, M_, D_ = value.shape
#     _, Lq_, M_, L_, P_, _ = sampling_locations.shape

#     # value_spatial_shapes가 Tensor가 아닐 경우 Tensor로 변환
#     if not torch.is_tensor(value_spatial_shapes):
#         value_spatial_shapes = torch.as_tensor(value_spatial_shapes, device=value.device)
#     Hs = value_spatial_shapes[:, 0].tolist()
#     Ws = value_spatial_shapes[:, 1].tolist()
#     splits = [int(H * W) for H, W in zip(Hs, Ws)]
#     value_list = value.split(splits, dim=1)

#     sampling_grids = 2 * sampling_locations - 1
#     sampling_value_list = []
#     for lid_, (H_, W_) in enumerate(zip(Hs, Ws)):
#         # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
#         value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
#         # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
#         sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
#         # N_*M_, D_, Lq_, P_
#         sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
#                                           mode='bilinear', padding_mode='zeros', align_corners=False)
#         sampling_value_list.append(sampling_value_l_)
#     # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)  (N_*M_, 1, Lq_, L_*P_)
#     attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
#     output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
#     return output.transpose(1, 2).contiguous()

