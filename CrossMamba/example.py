from .mamba_block import CrossBlock as Cross_Block
from .cross_mamba_simple import Mamba as Cross_Mamba
from .mamba_block import Block as Block
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from functools import partial
from torch import nn
import os
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


def _level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    device = spatial_shapes.device
    sizes = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(torch.long)
    return torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                      sizes.cumsum(0)[:-1]], dim=0)

# class SpatialCrossMambaFromCross(nn.Module):
#     def __init__(self, dim: int = 128, num_levels: int = 4, d_state: int = 4,
#                  dropout: float = 0.1, q_chunk: int = 512,
#                  inner_dim: int | None = None,      # ← 채널 축 축소 차원 r
#                  seq_down: int = 4):                # ← 시퀀스 다운샘플 비율
#         super().__init__()
#         self.dim = dim
#         self.num_levels = num_levels
#         self.dropout = nn.Dropout(dropout)

#         # cross_mamba 서명 호환 + alias
#         try:
#             self.cross = cross_mamba(d_model=(inner_dim or max(16, dim // 2)), d_state=d_state)
#         except TypeError:
#             self.cross = cross_mamba(dim=(inner_dim or max(16, dim // 2)), d_state=d_state)
#         self.cross_mamba = self.cross

#         # 하이퍼
#         self.q_chunk = q_chunk
#         self.seq_down = max(1, seq_down)
#         self.r = inner_dim or max(16, dim // 2)     # 기본 r = dim//2 (최소 16)

#         # 레벨 게이트
#         self.level_gate = nn.Parameter(torch.zeros(self.num_levels))

#         # 채널 축 축소/확장
#         self.in_proj = nn.Linear(dim, self.r)       # dim -> r
#         self.out_proj = nn.Linear(self.r, dim)      # r -> dim
#         self.output_proj = nn.Linear(dim, dim)      # 최종 residual proj

#     @torch.no_grad()
#     def _build_indexes_any(self, bev_mask: torch.Tensor):
#         valid_any = bev_mask.any(dim=0).any(dim=-1)  # (B,N)
#         indexes = [valid_any[j].nonzero(as_tuple=False).squeeze(-1) for j in range(valid_any.size(0))]
#         max_len = max((len(idx) for idx in indexes), default=0)
#         return indexes, max_len

#     def _reshape_value_per_level(self, value_BS_M_C, spatial_shapes, B, S):
#         BS, M, C = value_BS_M_C.shape
#         L = 1  # 현재 한 레벨만 사용 중
#         lsi = _level_start_index(spatial_shapes)
#         feats_per_level = []
#         for l in range(L):
#             H, W = spatial_shapes[l].tolist()
#             start = lsi[l].item()
#             length = H * W
#             slice_l = value_BS_M_C[:, start:start+length, :]                 # (B*S, H*W, C)
#             feat_l = slice_l.transpose(1, 2).contiguous().view(B*S, C, H, W) # (B*S, C, H, W)
#             feat_l = feat_l.view(B, S, C, H, W)
#             feats_per_level.append(feat_l)
#         return feats_per_level  # len L, each (B,S,C,H,W)

#     def forward(self, query, key, value, query_pos=None,
#                 reference_points_cam=None, spatial_shapes=None, bev_mask=None):

#         B, N, C = query.shape
#         S, M, Bk, Ck = key.shape
#         assert Bk == B and Ck == C
#         assert reference_points_cam.dim() == 5
#         assert bev_mask.shape[:4] == reference_points_cam.shape[:4]

#         inp_residual = query
#         if query_pos is not None:
#             query = query + query_pos

#         device = query.device
#         spatial_shapes = spatial_shapes.to(device=device, dtype=torch.long)
#         reference_points_cam = reference_points_cam.to(device=device, dtype=torch.float32)

#         # 유효 BEV 인덱스
#         idx_list, max_len = self._build_indexes_any(bev_mask)
#         if max_len == 0:
#             return self.dropout(self.output_proj(query)) + inp_residual

#         D = reference_points_cam.size(3)
#         queries_sel = query.new_zeros((B, max_len, C))
#         refs_sel = reference_points_cam.new_zeros((B, S, max_len, D, 2))
#         mask_sel = bev_mask.new_zeros((B, S, max_len, D))

#         for j in range(B):
#             idx = idx_list[j]
#             Lq = idx.numel()
#             if Lq > 0:
#                 queries_sel[j, :Lq] = query[j, idx]
#                 refs_sel[j] = reference_points_cam[:, j, :, :, :][:, idx]
#                 mask_sel[j] = bev_mask[:, j, :, :][:, idx]

#         # 멀티스케일 복원 및 샘플링
#         value_BS_M_C = value.permute(2, 0, 1, 3).contiguous().view(B * S, M, C)
#         feats_per_level = self._reshape_value_per_level(value_BS_M_C, spatial_shapes, B, S)
#         L = len(feats_per_level)  # 현재 1

#         refs = refs_sel.permute(0, 1, 2, 3, 4).contiguous()      # (B,S,max_len,D,2)
#         grid = refs.view(B*S, max_len*D, 2) * 2.0 - 1.0
#         grid = grid.view(B*S, max_len*D, 1, 2)

#         tokens_lvl = []
#         for l, feat in enumerate(feats_per_level):
#             BS_feat = feat.view(B*S, C, feat.size(-2), feat.size(-1))
#             sampled = F.grid_sample(BS_feat, grid, mode='bilinear',
#                                     padding_mode='zeros', align_corners=False)    # (B*S,C,max_len*D,1)
#             sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()            # (B*S,max_len*D,C)
#             sampled = sampled.view(B, S, max_len, D, C)
#             tokens_lvl.append(sampled * (1.0 + self.level_gate[l]))

#         tokens = torch.stack(tokens_lvl, dim=5)                                    # (B,S,max_len,D,C,L)
#         tokens = tokens.permute(0, 2, 1, 3, 5, 4).contiguous()                     # (B,max_len,S,D,L,C)
#         B_, Lq_, S_, D_, L_, C_ = tokens.shape
#         seq_len = S_ * D_ * L_
#         tokens = tokens.view(B_, Lq_, seq_len, C_)                                 # (B,max_len,seq_len,C)

#         # 마스크도 같은 형태로
#         mask_seq = mask_sel.permute(0, 2, 1, 3).contiguous()                       # (B,max_len,S,D)
#         mask_seq = mask_seq.unsqueeze(-1).expand(B, max_len, S, D, L)              # (B,max_len,S,D,L)
#         mask_seq = mask_seq.reshape(B, max_len, seq_len).to(tokens.dtype)          # (B,max_len,seq_len)

#         # ====== (1) 채널 축 축소: dim -> r ======
#         tokens_r = self.in_proj(tokens)            # (B,max_len,seq_len,r)
#         q_r = self.in_proj(queries_sel)            # (B,max_len,r)

#         # ====== (2) 시퀀스 다운샘플: seq_len -> ceil(seq_len/seq_down) ======
#         BL = B * max_len
#         r = self.r
#         tokens_r_bl = tokens_r.view(BL, seq_len, r).transpose(1, 2).contiguous()   # (BL,r,seq_len)
#         mask_bl = mask_seq.view(BL, 1, seq_len).contiguous()                       # (BL,1,seq_len)

#         if self.seq_down > 1:
#             tokens_r_bl = F.avg_pool1d(tokens_r_bl, kernel_size=self.seq_down,
#                                        stride=self.seq_down, ceil_mode=True)      # (BL,r,L')
#             mask_bl = F.avg_pool1d(mask_bl, kernel_size=self.seq_down,
#                                     stride=self.seq_down, ceil_mode=True)         # (BL,1,L')
#         Lp = tokens_r_bl.shape[-1]                                                 # 다운샘플된 길이
#         tokens_r_seq = tokens_r_bl.transpose(1, 2).contiguous()                    # (BL,L',r)
#         mask_p = (mask_bl > 0).to(tokens_r_seq.dtype).transpose(1, 2).contiguous() # (BL,L',1)

#         # q를 L' 길이에 expand (expand는 뷰라서 비용 적음)
#         q_r_bl = q_r.view(BL, r)                                                   # (BL,r)
#         q_seq = q_r_bl.unsqueeze(1).expand(BL, Lp, r).contiguous()                 # (BL,L',r)
#         v_seq = tokens_r_seq                                                       # (BL,L',r)

#         # ====== (3) cross_mamba 실행 (쿼리 청킹) ======
#         fused_list = []
#         for s in range(0, BL, self.q_chunk):
#             e = min(s + self.q_chunk, BL)
#             fused_list.append(self.cross(q_seq[s:e], v_seq[s:e]))                  # (···,L',r)
#         fused_seq = torch.cat(fused_list, dim=0)                                   # (BL,L',r)

#         # ====== (4) 마스크 평균 풀링 → (BL,r) ======
#         fused_seq = fused_seq * mask_p                                             # (BL,L',r)
#         denom = mask_p.sum(dim=1).clamp_min(1.0)                                   # (BL,1)
#         pooled_r = fused_seq.sum(dim=1) / denom                                    # (BL,r)

#         # ====== (5) r -> dim 복원 및 슬롯 반영 ======
#         updates = self.out_proj(pooled_r).view(B, max_len, C)                      # (B,max_len,C)

#         slots = query.new_zeros((B, N, C))
#         for j in range(B):
#             idx = idx_list[j]
#             Lq = idx.numel()
#             if Lq > 0:
#                 slots[j, idx] = updates[j, :Lq]

#         out = self.output_proj(slots)
#         return self.dropout(out) + inp_residual

########### Spatial Cross Mamba END ###########

class SpatialCrossMambaFromCross(nn.Module):
    def __init__(self, dim: int = 128, num_levels: int = 8, d_state: int = 8,
                 dropout: float = 0.1, q_chunk: int = 128,
                 inner_dim: int | None = None,      # ← 채널 축 축소 차원 r
                 seq_down: int = 4):                # ← 시퀀스 다운샘플 비율
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.dropout = nn.Dropout(dropout)

        # cross_mamba 서명 호환 + alias
        try:
            self.cross = cross_mamba(d_model=(inner_dim or max(16, dim // 2)), d_state=d_state)
        except TypeError:
            self.cross = cross_mamba(dim=(inner_dim or max(16, dim // 2)), d_state=d_state)
        self.cross_mamba = self.cross

        # 하이퍼
        self.q_chunk = q_chunk
        self.seq_down = max(1, seq_down)
        self.r = inner_dim or max(16, dim // 2)     # 기본 r = dim//2 (최소 16)

        # 레벨 게이트
        self.level_gate = nn.Parameter(torch.zeros(self.num_levels))

        # 채널 축 축소/확장
        self.in_proj = nn.Linear(dim, self.r)       # dim -> r
        self.out_proj = nn.Linear(self.r, dim)      # r -> dim
        self.output_proj = nn.Linear(dim, dim)      # 최종 residual proj

    @torch.no_grad()
    def _build_indexes_any(self, bev_mask: torch.Tensor):
        valid_any = bev_mask.any(dim=0).any(dim=-1)  # (B,N)
        indexes = [valid_any[j].nonzero(as_tuple=False).squeeze(-1) for j in range(valid_any.size(0))]
        max_len = max((len(idx) for idx in indexes), default=0)
        return indexes, max_len

    def _reshape_value_per_level(self, value_BS_M_C, spatial_shapes, B, S):
        BS, M, C = value_BS_M_C.shape
        L = 1  # 현재 한 레벨만 사용 중
        lsi = _level_start_index(spatial_shapes)
        feats_per_level = []
        for l in range(L):
            H, W = spatial_shapes[l].tolist()
            start = lsi[l].item()
            length = H * W
            slice_l = value_BS_M_C[:, start:start+length, :]                 # (B*S, H*W, C)
            feat_l = slice_l.transpose(1, 2).contiguous().view(B*S, C, H, W) # (B*S, C, H, W)
            feat_l = feat_l.view(B, S, C, H, W)
            feats_per_level.append(feat_l)
        return feats_per_level  # len L, each (B,S,C,H,W)

    def forward(self, query, key, value, query_pos=None,
                reference_points_cam=None, spatial_shapes=None, bev_mask=None):

        B, N, C = query.shape
        S, M, Bk, Ck = key.shape
        assert Bk == B and Ck == C
        assert reference_points_cam.dim() == 5
        assert bev_mask.shape[:4] == reference_points_cam.shape[:4]

        inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        device = query.device
        compute_dtype = query.dtype  # ★ dtype 기준은 query로 통일
        spatial_shapes = spatial_shapes.to(device=device, dtype=torch.long)
        reference_points_cam = reference_points_cam.to(device=device, dtype=torch.float32)

        # 유효 BEV 인덱스
        idx_list, max_len = self._build_indexes_any(bev_mask)
        if max_len == 0:
            out = self.output_proj(query.to(compute_dtype))
            return self.dropout(out) + inp_residual

        D = reference_points_cam.size(3)
        queries_sel = query.new_zeros((B, max_len, C))                 # (B,max_len,C)  dtype=query.dtype
        refs_sel = reference_points_cam.new_zeros((B, S, max_len, D, 2))
        mask_sel = bev_mask.new_zeros((B, S, max_len, D))

        for j in range(B):
            idx = idx_list[j]
            Lq = idx.numel()
            if Lq > 0:
                queries_sel[j, :Lq] = query[j, idx]
                # refs_sel[j] = reference_points_cam[:, j, :, :, :][:, idx]
                # mask_sel[j] = bev_mask[:, j, :, :][:, idx]
                refs_sel[j, :, :Lq] = reference_points_cam[:, j, :, :, :][:, idx]  # [S, Lq, D, 2]
                mask_sel[j, :, :Lq] = bev_mask[:, j, :, :][:, idx]

        # 멀티스케일 복원 및 샘플링
        value_BS_M_C = value.permute(2, 0, 1, 3).contiguous().view(B * S, M, C)   # (B*S,M,C)
        feats_per_level = self._reshape_value_per_level(value_BS_M_C, spatial_shapes, B, S)
        L = len(feats_per_level)  # 현재 1

        refs = refs_sel.permute(0, 1, 2, 3, 4).contiguous()      # (B,S,max_len,D,2)
        grid = refs.view(B*S, max_len*D, 2) * 2.0 - 1.0          # (-1,1) 정규화
        grid = grid.view(B*S, max_len*D, 1, 2)

        tokens_lvl = []
        for l, feat in enumerate(feats_per_level):
            BS_feat = feat.view(B*S, C, feat.size(-2), feat.size(-1))             # (B*S,C,H,W)

            # ★ dtype fix: grid_sample은 input/ grid dtype 일치가 안전
            grid_ = grid.to(BS_feat.dtype)

            sampled = F.grid_sample(BS_feat, grid_, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)    # (B*S,C,max_len*D,1)
            sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()            # (B*S,max_len*D,C)
            sampled = sampled.view(B, S, max_len, D, C)

            # ★ dtype fix: level_gate를 토큰 dtype으로
            gate = (1.0 + self.level_gate[l]).to(sampled.dtype)
            tokens_lvl.append(sampled * gate)

        tokens = torch.stack(tokens_lvl, dim=5)                                    # (B,S,max_len,D,C,L)
        tokens = tokens.permute(0, 2, 1, 3, 5, 4).contiguous()                     # (B,max_len,S,D,L,C)
        B_, Lq_, S_, D_, L_, C_ = tokens.shape
        seq_len = S_ * D_ * L_
        tokens = tokens.view(B_, Lq_, seq_len, C_)                                 # (B,max_len,seq_len,C)

        # 마스크도 같은 형태로
        mask_seq = mask_sel.permute(0, 2, 1, 3).contiguous()                       # (B,max_len,S,D)
        mask_seq = mask_seq.unsqueeze(-1).expand(B, max_len, S, D, L)              # (B,max_len,S,D,L)
        mask_seq = mask_seq.reshape(B, max_len, seq_len).to(tokens.dtype)          # (B,max_len,seq_len)

        # ====== (1) 채널 축 축소: dim -> r ======
        tokens_r = self.in_proj(tokens)            # (B,max_len,seq_len,r)
        q_r = self.in_proj(queries_sel)            # (B,max_len,r)

        # ====== (2) 시퀀스 다운샘플: seq_len -> ceil(seq_len/seq_down) ======
        BL = B * max_len
        r = self.r
        tokens_r_bl = tokens_r.view(BL, seq_len, r).transpose(1, 2).contiguous()   # (BL,r,seq_len)
        mask_bl = mask_seq.view(BL, 1, seq_len).contiguous()                       # (BL,1,seq_len)

        if self.seq_down > 1:
            tokens_r_bl = F.avg_pool1d(tokens_r_bl, kernel_size=self.seq_down,
                                       stride=self.seq_down, ceil_mode=True)      # (BL,r,L')
            mask_bl = F.avg_pool1d(mask_bl, kernel_size=self.seq_down,
                                    stride=self.seq_down, ceil_mode=True)         # (BL,1,L')
        Lp = tokens_r_bl.shape[-1]                                                 # 다운샘플된 길이
        tokens_r_seq = tokens_r_bl.transpose(1, 2).contiguous()                    # (BL,L',r)
        mask_p = (mask_bl > 0).to(tokens_r_seq.dtype).transpose(1, 2).contiguous() # (BL,L',1)

        # q를 L' 길이에 expand (expand는 뷰라서 비용 적음)
        q_r_bl = q_r.view(BL, r)                                                   # (BL,r)
        q_seq = q_r_bl.unsqueeze(1).expand(BL, Lp, r).contiguous()                 # (BL,L',r)
        v_seq = tokens_r_seq                                                       # (BL,L',r)

        # ====== (3) cross_mamba 실행 (쿼리 청킹) ======
        fused_list = []
        for s in range(0, BL, self.q_chunk):
            e = min(s + self.q_chunk, BL)
            fused_list.append(self.cross(q_seq[s:e], v_seq[s:e]))                  # (···,L',r)
        fused_seq = torch.cat(fused_list, dim=0)                                   # (BL,L',r)

        # ====== (4) 마스크 평균 풀링 → (BL,r) ======
        fused_seq = fused_seq * mask_p                                             # (BL,L',r)
        denom = mask_p.sum(dim=1).clamp_min(1.0)                                   # (BL,1)
        pooled_r = fused_seq.sum(dim=1) / denom                                    # (BL,r)

        # ====== (5) r -> dim 복원 및 슬롯 반영 ======
        updates = self.out_proj(pooled_r).view(B, max_len, C)                      # (B,max_len,C)

        slots = query.new_zeros((B, N, C))                                         # dtype=query.dtype
        # ★ dtype fix: 인덱스 대입 전 dtype 일치
        updates = updates.to(slots.dtype)

        for j in range(B):
            idx = idx_list[j]
            Lq = idx.numel()
            if Lq > 0:
                slots[j, idx] = updates[j, :Lq]

        out = self.output_proj(slots)
        return self.dropout(out) + inp_residual


#
class self_mamba(nn.Module):
    """
    Multi-Scale Deformable Self-Mamba
      x: (B, N, C)  where N = Z*X
    """
    def __init__(self, dim=128, d_state=8, Z=200, X=200, n_levels=4, n_points=4,
                 weight_global=True, pool='mean', expand=2):
        super(self_mamba, self).__init__()
        self.C = dim
        self.Z = Z
        self.X = X
        self.L = n_levels
        self.K = n_points
        self.weight_global = weight_global  # True: L*K 전체 softmax
        self.pool = pool                    # 'mean' | 'sum' | 'attn'

        # 쿼리 기반 deform 파라미터
        self.offset_proj = nn.Linear(dim, n_levels * n_points * 2)
        self.weight_proj = nn.Linear(dim, n_levels * n_points)

        # 레벨별 value 프로젝션
        self.value_proj = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(n_levels)])

        # 시퀀스 집계기: (양방향) Mamba Block 재사용.
        # d_state / expand 는 경량화 대상: d_inner = dim * expand 가 selective_scan 비용을
        # 결정하고 (∝ d_inner × d_state), d_conv=4 인 causal conv 도 ∝ d_inner.
        # 기본값(d_state=8, expand=2)은 원래 학습 config; 가벼운 변형은 d_state=4, expand=1.
        mk = dict(mixer_cls=partial(Mamba, d_state=d_state, d_conv=4, expand=expand),
                  norm_cls=partial(RMSNorm, eps=1e-5), fused_add_norm=False)
        self.mamba_fw = Block(dim, **mk)
        self.mamba_bw = Block(dim, **mk)

        # 출력/게이팅
        self.out_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

        # 안정 초기화
        nn.init.zeros_(self.offset_proj.weight); nn.init.zeros_(self.offset_proj.bias)
        nn.init.zeros_(self.weight_proj.bias)

        # 시퀀스 위치 임베딩(레벨/포인트)
        self.level_embed = nn.Parameter(torch.zeros(n_levels, dim))
        self.point_embed = nn.Parameter(torch.zeros(n_points, dim))
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        nn.init.trunc_normal_(self.point_embed, std=0.02)

        if self.pool == 'attn':
            self.pool_attn = nn.Linear(dim, 1)

    @staticmethod
    def _norm_xy_to_grid(xy, H, W):
        # [0,1] → [-1,1] (x,y)
        gx = xy[..., 0] * 2.0 - 1.0
        gy = xy[..., 1] * 2.0 - 1.0
        return torch.stack([gx, gy], dim=-1)

    def _build_pyramid(self, y4d):
        """
        y4d: (B,C,Z,X)
        returns list of (B,C,H_l,W_l) with L levels
        """
        l0 = y4d
        feats = [l0]
        while len(feats) < self.L:
            feats.append(F.avg_pool2d(feats[-1], kernel_size=2, stride=2))
            # 필요시 max_pool로 교체 테스트 가능
        return feats[:self.L]

    def forward(self, x):
        """
        x: (B, N, C) with N = Z*X
        """
        # --- (1) 입력 기본 체크 & shape 변수 정의 ---
        assert x.dim() == 3, f"Expected (B,N,C), got {tuple(x.shape)}"
        B, N, C = x.shape
        assert C == self.C, f"C mismatch: got {C} vs {self.C}"

        # --- (2) Z,X 일관성 체크/자동 유추 ---
        if self.Z * self.X != N:
            # 정사각형 근사 자동 유추 (필요시 명시적으로 Z,X 고정 권장)
            z = int(round(math.sqrt(N)))
            if z * z != N:
                raise ValueError(
                    f"[self_mamba] N={N} != Z*X ({self.Z}*{self.X}={self.Z*self.X}). "
                    f"Set (Z,X) explicitly or use N being a perfect square."
                )
            self.Z = self.X = z  # auto

        # --- (3) (B,N,C) → (B,C,Z,X) ---
        y4d = x.view(B, self.Z, self.X, C).permute(0, 3, 1, 2).contiguous()  # (B,C,Z,X)
        feats = self._build_pyramid(y4d)  # L levels

        # --- (4) 기준 좌표 (정규화) ---
        gy, gx = torch.meshgrid(
            torch.linspace(0.5, self.Z - 0.5, self.Z, device=x.device),
            torch.linspace(0.5, self.X - 0.5, self.X, device=x.device),
            indexing='ij'
        )
        ref = torch.stack([(gx.reshape(-1)[None] / self.X),
                           (gy.reshape(-1)[None] / self.Z)], dim=-1)  # (1,N,2)
        ref = ref.expand(B, N, 2)

        # --- (5) 오프셋/가중치 ---
        offsets = self.offset_proj(x).view(B, N, self.L, self.K, 2)
        offsets = torch.tanh(offsets) * 0.5  # [-0.5,0.5] 안정화

        if self.weight_global:
            w = torch.softmax(self.weight_proj(x).view(B, N, self.L * self.K), dim=-1)
            weights = w.view(B, N, self.L, self.K)
        else:
            weights = torch.softmax(self.weight_proj(x).view(B, N, self.L, self.K), dim=-1)

        level_emb = self.level_embed.view(1, 1, self.L, 1, C)
        point_emb = self.point_embed.view(1, 1, 1, self.K, C)

        # --- (6) 각 레벨에서 샘플링 ---
        sampled_list = []
        for l, feat in enumerate(feats):
            _, _, H, W = feat.shape
            v = self.value_proj[l](feat)                          # (B,C,H,W)
            base = self._norm_xy_to_grid(ref, H, W)               # (B,N,2)
            grid = base.unsqueeze(2) + offsets[:, :, l, :, :]     # (B,N,K,2)

            # (B,C,H,W) x (B,N,K,2) -> (B,C,N,K) -> (B,N,K,C)
            val = F.grid_sample(v, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            val = val.permute(0, 2, 3, 1)                          # (B,N,K,C)
            val = val * weights[:, :, l, :].unsqueeze(-1)          # 가중치 적용
            val = val + (level_emb[:, :, l:l+1, :, :] + point_emb) # 임베딩 추가
            sampled_list.append(val)

        # (B,N,S,C), S = L*K
        sampled = torch.cat(sampled_list, dim=2)
        S = self.L * self.K

        # --- (7) Mamba 집계 ---
        seq = sampled.reshape(B * N, S, C).contiguous()  # (B*N,S,C)

        # SELF_MAMBA_UNIDIRECTIONAL=1: forward direction only (halves runtime but
        # the model is trained bidirectional → noticeable IoU drop; not recommended).
        if os.environ.get("SELF_MAMBA_UNIDIRECTIONAL", "0") == "1":
            out_f, res_f = self.mamba_fw(seq, None, inference_params=None)
            if res_f is not None:
                out_f = out_f + res_f
            seq_out = out_f
        # SELF_MAMBA_PARALLEL_STREAMS=1: forward & backward passes are independent
        # (separate weights, independent inputs) — launch them on two CUDA streams
        # so the GPU can overlap them. Mathematically identical to the serial path.
        elif (os.environ.get("SELF_MAMBA_PARALLEL_STREAMS", "0") == "1"
              and seq.is_cuda and not torch.is_grad_enabled()):
            if getattr(self, "_mamba_bw_stream", None) is None:
                self._mamba_bw_stream = torch.cuda.Stream()
            seq_rev = torch.flip(seq, [1])
            bw_stream = self._mamba_bw_stream
            bw_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(bw_stream):
                out_b, res_b = self.mamba_bw(seq_rev, None, inference_params=None)
                if res_b is not None:
                    out_b = out_b + res_b
                out_b = torch.flip(out_b, [1])
            # forward on the default stream, concurrently
            out_f, res_f = self.mamba_fw(seq, None, inference_params=None)
            if res_f is not None:
                out_f = out_f + res_f
            torch.cuda.current_stream().wait_stream(bw_stream)
            seq_out = out_f + out_b  # (B*N,S,C)
        else:
            # forward
            out_f, res_f = self.mamba_fw(seq, None, inference_params=None)
            if res_f is not None:
                out_f = out_f + res_f
            # backward
            seq_rev = torch.flip(seq, [1])
            out_b, res_b = self.mamba_bw(seq_rev, None, inference_params=None)
            if res_b is not None:
                out_b = out_b + res_b
            out_b = torch.flip(out_b, [1])

            seq_out = out_f + out_b  # (B*N,S,C)

        # --- (8) 풀링 & 게이팅 ---
        if self.pool == 'mean':
            pooled = seq_out.mean(dim=1)
        elif self.pool == 'sum':
            pooled = seq_out.sum(dim=1)
        else:  # 'attn'
            alpha = torch.softmax(self.pool_attn(seq_out), dim=1)  # (B*N,S,1)
            pooled = (alpha * seq_out).sum(dim=1)

        # Calibration hook: if self._cal_capture is a list, stash the mean-pooled SSM output
        # `pooled` (B*N, C) and the input residual `x` (B, N, C). Used by the closed-form
        # out_proj calibration that makes a lightweight self_mamba reproduce the heavy one's
        # `fused = out_proj(pooled)` output (no gradient training). Off unless set externally.
        if getattr(self, "_cal_capture", None) is not None:
            self._cal_capture.append((pooled.detach().float().cpu(), x.detach().float().cpu()))

        fused = self.out_proj(pooled).view(B, N, C)
        gate = torch.sigmoid(self.gate_proj(x))
        return x + gate * fused



############cross mamba############
class cross_mamba(nn.Module):
    def __init__(self, dim=128, d_state=8):
        super().__init__()
        self.mamba = Cross_Block(
            dim,
            mixer_cls=partial(Cross_Mamba, d_state=d_state, d_conv=4, expand=2),
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=False,
        )
        self.mamba_bw = Cross_Block(
            dim,
            mixer_cls=partial(Cross_Mamba, d_state=d_state, d_conv=4, expand=2),
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=False,
        )

    def forward(self, _q, _v):  # _q,_v: (B, T, C)
        for_residual = None
        forward_f, for_residual = self.mamba(_q, _v, for_residual, inference_params=None)
        forward_f = forward_f + for_residual

        back_residual = None
        backward_q = torch.flip(_q, [1])
        backward_v = torch.flip(_v, [1])
        backward_f, back_residual = self.mamba_bw(backward_q, backward_v, back_residual, inference_params=None)
        backward_f = (backward_f + back_residual) if back_residual is not None else backward_f
        backward_f = torch.flip(backward_f, [1])

        return forward_f + backward_f  # (B, T, C)


# -----------------------------
# 2) 멀티스케일 Deformable 샘플러 + Cross-Mamba 집계 코어
#    (MSDeformAttn 호출 시그니처를 어댑트)
# -----------------------------
class MSDeformCrossMambaAdapter(nn.Module):
    """
    MSDeformAttn과 동일한 호출:
    forward(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index)

    - query:               (B, Nq, C)
    - reference_points:    (B, Nq, L, 2) 또는 (B, Nq, 1, 2), [0,1] 정규화된 (x,y)
    - input_flatten:       (B, sum(H_l*W_l), C)
    - input_spatial_shapes:(L, 2)  (H_l, W_l)
    - input_level_start_index:(L,)
    return:                (B, Nq, C)
    """
    def __init__(self, dim=128, n_levels=8, n_points=8, d_state=8, weight_global=True, pool='mean'):
        super().__init__()
        self.C = dim
        self.L = n_levels
        self.K = n_points
        self.weight_global = weight_global  # True면 L*K 전체에 대해 softmax, False면 레벨별 softmax(K)
        self.pool = pool                    # 'mean' | 'sum' | 'attn'

        # 쿼리→오프셋/가중치 예측
        self.offset_proj = nn.Linear(dim, n_levels * n_points * 2)
        self.weight_proj = nn.Linear(dim, (n_levels * n_points) if weight_global else (n_levels * n_points))

        # 레벨별 밸류 투영 (C->C)
        self.value_proj = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(n_levels)])

        # Cross-Mamba 집계기 (네 구현을 재사용)
        self.cross_mamba = cross_mamba(dim=dim, d_state=d_state)

        # 출력 정리 & 게이팅
        self.out_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

        # 안정 초기화
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        nn.init.zeros_(self.weight_proj.bias)

        # (선택) 시퀀스 포지셔널 임베딩: 레벨, 포인트 인덱스
        self.level_embed = nn.Parameter(torch.zeros(n_levels, dim))
        self.point_embed = nn.Parameter(torch.zeros(n_points, dim))
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        nn.init.trunc_normal_(self.point_embed, std=0.02)

        # (선택) 풀링용 어텐션
        if self.pool == 'attn':
            self.pool_attn = nn.Linear(dim, 1)

    @staticmethod
    def _norm_xy_to_grid(xy, H, W):
        # [0,1] → [-1,1], grid_sample 순서 (x,y)
        gx = xy[..., 0] * 2 - 1
        gy = xy[..., 1] * 2 - 1
        return torch.stack([gx, gy], dim=-1)

    def _unflatten_feats(self, input_flatten, input_spatial_shapes, input_level_start_index):
        """(B, sum(HW), C) -> list of (B, C, H, W)"""
        B, _, C = input_flatten.shape
        feats = []
        for l in range(self.L):
            H, W = input_spatial_shapes[l].tolist()
            s = input_level_start_index[l].item()
            e = s + H * W
            mem = input_flatten[:, s:e, :]                    # (B, H*W, C)
            mem = mem.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            feats.append(mem)                                 # (B, C, H, W)
        return feats

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        """
        query (B,Nq,C), reference_points (B,Nq,1|L,2)
        """
        B, Nq, C = query.shape
        assert C == self.C
        L = input_spatial_shapes.shape[0]
        assert L == self.L, f"n_levels mismatch: cfg={self.L}, input={L}"

        feats = self._unflatten_feats(input_flatten, input_spatial_shapes, input_level_start_index)

        # ref shape 정리
        if reference_points.size(2) == 1:
            ref = reference_points.expand(B, Nq, L, 2)       # (B,Nq,L,2)
        elif reference_points.size(2) == L:
            ref = reference_points
        else:
            raise ValueError(f"reference_points shape {reference_points.shape} incompatible.")

        # 오프셋/가중치
        offsets = self.offset_proj(query).view(B, Nq, L, self.K, 2)
        offsets = torch.tanh(offsets) * 0.5                  # 안정화: [-0.5,0.5]

        if self.weight_global:
            # (B,Nq,L*K) → softmax over L*K
            weights = self.weight_proj(query).view(B, Nq, L * self.K)
            weights = torch.softmax(weights, dim=-1).view(B, Nq, L, self.K)
        else:
            # 레벨별 softmax(K)
            weights = self.weight_proj(query).view(B, Nq, L, self.K)
            weights = torch.softmax(weights, dim=-1)

        sampled_list = []
        # 레벨·포인트 임베딩 준비
        level_emb = self.level_embed.view(1, 1, L, 1, C)     # (1,1,L,1,C)
        point_emb = self.point_embed.view(1, 1, 1, self.K, C)# (1,1,1,K,C)

        for l, feat in enumerate(feats):
            _, _, H, W = feat.shape
            v = self.value_proj[l](feat)                     # (B,C,H,W)

            base = self._norm_xy_to_grid(ref[:, :, l, :], H, W)     # (B,Nq,2)
            grid = base.unsqueeze(2) + offsets[:, :, l, :, :]       # (B,Nq,K,2)

            # F.grid_sample expects grid (B, Nq, K, 2), returns (B,C,Nq,K)
            val = F.grid_sample(v, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            val = val.permute(0, 2, 3, 1)                    # (B,Nq,K,C)

            wk = weights[:, :, l, :].unsqueeze(-1)          # (B,Nq,K,1)
            val = val * wk                                   # 가중 샘플

            # 샘플별 임베딩 추가(레벨/포인트)
            val = val + (level_emb[:, :, l:l+1, :, :] + point_emb)

            sampled_list.append(val)                         # (B,Nq,K,C)

        # (B,Nq,L*K,C)
        sampled = torch.cat(sampled_list, dim=2)
        S = L * self.K

        # -------- Cross-Mamba 집계 --------
        # 쿼리를 샘플 길이 S로 broadcast해 (B,Nq,S,C)
        q_seq = query.unsqueeze(2).expand(B, Nq, S, C)
        v_seq = sampled

        # (B·Nq, S, C)
        q_seq = q_seq.reshape(B * Nq, S, C).contiguous()
        v_seq = v_seq.reshape(B * Nq, S, C).contiguous()

        fused_seq = self.cross_mamba(q_seq, v_seq)           # (B·Nq, S, C)

        # 풀링 (S축)
        if self.pool == 'mean':
            pooled = fused_seq.mean(dim=1)                   # (B·Nq, C)
        elif self.pool == 'sum':
            pooled = fused_seq.sum(dim=1)
        elif self.pool == 'attn':
            alpha = torch.softmax(self.pool_attn(fused_seq), dim=1)  # (B·Nq,S,1)
            pooled = (alpha * fused_seq).sum(dim=1)
        else:
            raise ValueError(self.pool)

        fused = self.out_proj(pooled).view(B, Nq, C)

        gate = torch.sigmoid(self.gate_proj(query))
        return query + gate * fused                          # (B,Nq,C)


# -----------------------------
# 3) FusingCrossAttentionV2 교체 버전
#    (기존 MSDeformAttn 호출부와 호환)
# -----------------------------
class FusingCrossMambaV2(nn.Module):
    """
    Multi-Scale Deformable Cross-Mamba 기반 레이더↔카메라 BEV 융합 블록
    """
    def __init__(self, dim=128, dropout=0.1, n_levels=8, n_points=8, d_state=8,
                 weight_global=True, pool='mean'):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.fuser = MSDeformCrossMambaAdapter(
            dim=dim, n_levels=n_levels, n_points=n_points, d_state=d_state,
            weight_global=weight_global, pool=pool
        )

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, query_pos=None):
        """
        query:               (B, Nq, C)  (e.g., pseudolidar BEV)
        reference_points:    (B, Nq, 1|L, 2) in [0,1]
        input_flatten:       (B, sum(H_l*W_l), C)  (e.g., camera BEV multi-scale memory)
        input_spatial_shapes:(L,2)
        input_level_start_index:(L,)
        query_pos:           (B, Nq, C) optional
        """
        residual = query
        if query_pos is not None:
            query = query + query_pos

        out = self.fuser(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index)
        return self.dropout(out) + residual