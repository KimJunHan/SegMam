# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import math
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from ..functions import MSDeformAttnFunction
from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level

        d_model: feature 채널 수 (hidden dimension, 예: 256)  >   128
        n_levels: multi-scale feature level 수 (ex: P2, P3, P4, P5 → 4개)    4 8 16 32 > 1
        n_heads: multi-head attention head 수 > 8
        n_points: head당 샘플링 포인트 수 > 4 

        """
        super().__init__()
        if d_model % n_heads != 0:

            '''
            head별 feature 분배를 위해 d_model 256 은 n_heads 8 로 나눠 떨어져야 한다 16 128 / 8 n_level 1이다.
            '''
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))

        _d_per_head = d_model // n_heads
        #print(f"_d_per_head {_d_per_head}")  16이다.
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head): # _d_per_head는 2의 제곱승이여야 gpu에서 연산하기 편하다. 
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a "
                          "power of 2 which is more efficient in our CUDA implementation.")

        self.im2col_step = 64 # "im2col" (Image to Column
        # Convolution 연산을 빠르게 하기 위해, 이미지를 잘라서 행렬 형태로 만듦
        # 원래 Convolution은 커널을 이미지 위에 슬라이딩하면서 연산함
        # im2col로 미리 펼치면,
        # Matrix Multiplication (GEMM, matmul) 로 바꿔서 훨씬 빠르게 연산할 수 있다.
        # Convolution = Matrix 곱셈(matmul) 로 최적화하기 위한 중간 변환 기법이야.
        # im2col step = 64개 query씩 나눠서 deformable attention 수행. 쿼리 2만개
        #메모리 최적화를 위한 설정.


        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        #sampling_offsets: query를 입력 받아 "어디를 샘플링할지" 2D offset 예측 쿼리 주변을 예측하게된다.
        # d_model 256이고 현재 8 x 4 x 4 x 2니깐 256이다. input output 채널이 같다.
        # nn.Linear가 W와 b(bias)를 모두 학습가능한 파라미터로 최적화됨
        # _d_model 128, 8 * 1 * 4 =32 * 2 =64 > x만큼 얼마 y만큼 얼마떨어질까?
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)  # 2는 각 포인트는 (x, y) 2차원 오프셋을 가짐
        '''
        각 Head는 각 Level마다 n_points 개의 샘플링 포인트를 예측하고 각 포인트는 (x, y) 방향으로 2개의 값(오프셋)을 가져야 한다
        따라서 총 필요한 예측 값 개수는 n_heads × n_levels × n_points × 2

        예를 들어:
        n_heads = 8
        n_levels = 1
        n_points = 4
        이면:
        8 heads
        각 head마다 4개 level
        각 level당 1개 point 4개
        각 point는 (x, y) 오프셋

        즉, query 하나당 64 값을 예측해야 한다.
        이 64개는 샘플링 오프셋을 모두 나타낸다.

        입력: query (shape: (N, Length_query, d_model)) 
        출력: 각 query당
        n_heads x n_levels x n_points x 2개의 offset
        즉, 어디로 sample할지 모든 오프셋 예측

        Deformable Attention은 reference point 기준에서 조금 이동한 위치를 샘플링한다.
        이때 이동은 2D 공간(x축, y축)으로 이뤄진다. 그래서 각 샘플링 포인트마다 얼마나 x로 이동할지 얼마나 y로 이동할지
        두 개의 값이 필요하다. "Reference point + offset" → 실제 sampling 위치
        '''

        #attention_weights: 각 샘플링 포인트에 대해 attention 가중치 예측
        #nn.Linear 클래스는 두 개의 행렬 가중치(weight)와 편향(bias)을 학습하며, 입력 텐서를 선형 변환하여 출력 텐서를 생성합니다. 선형 변환은 입력 텐서와 가중치 행렬의 행렬 곱을 계산하고, 편향을 더하는 연산으로 이루어집니다.
        #nn.linear 벡터를 벡터로 변환하는 완전 연결(fully connected) 레이어"
        #nn.Linear는 "전체를 한 번에", Conv는 "부분을 스캔해서" 학습한다.
        # 128 8 1 4 = 32
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        '''
        (head, level, point) 조합마다 1개의 weight가 필요하기 때문에
        전체 n_heads × n_levels × n_points개의 attention weight를 예측하는 거야.
        "각 head, level, point 별로 feature를 얼마나 중요하게 반영할지 결정하기 위해 weight를 따로 예측한다."

        Bilinear interpolation?
        위치가 정확히 grid 위가 아니면, 주변 4개 pixel을 이용해서 값을 부드럽게 보간

        '''

        #value_proj: input feature를 internal dimension으로 변환
        # "value_proj는 이미지에서 input feature를 head별로 나눌 수 있게 가공(preprocess)하는 Linear layer다."
        # Deformable Attention에 Input feature (input_flatten)를 그냥 쓰지 않고, value_proj를 통해 한 번 변환해서 사용한다.

        '''
        head별 분할 준비	나중에 multi-head로 나누기 전에 feature를 조정해주기 위함
        feature enhancement	raw feature를 바로 쓰지 않고, 좀 더 학습된 표현(feature space)으로 바꾸려는 것
        parameter 공유	여러 레벨의 feature를 공유된 projection 공간으로 맞춰서 통합하려는 목적
        '''

        self.value_proj = nn.Linear(d_model, d_model)
        #output_proj: 최종 attention output을 다시 원래 dimension으로 맞춤
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):

        #sampling_offsets Linear Layer의 weight를 0으로 초기화한다.
        #초기에는 학습하기 전 weight가 전부 0이다
        #sampling offset은 주로 bias를 이용해서 초기 grid를 설정하고, weight는 학습하면서 업데이트함
        constant_(self.sampling_offsets.weight.data, 0.)

        #0부터 self.n_heads - 1까지 정수 배열을 만든 후,
        #각 head별로 원(circle) 위에서의 각도(theta) 를 계산한다.
        #각 head에 대해 다른 방향(0도, 45도, 90도...)을 설정하려는 것.
        # torch.arange는 start부터 end까지 step 간격으로 1D tensor를 생성하는 함수
        # 데이터 타입 지정 가능 (float32, int64 등)
        '''
        2π는 원(circle) 한 바퀴를 의미해 (360도 = 2π 라디안)    
        self.n_heads로 나누면
        각 head별로 균등하게 배치할 때 필요한 각도 간격이 나온다.
        
        2pi/8 = pi/4 almost 0.785 radians 즉 head하나당 약 45도 간격이다.
        thetas = (0., 1., 2., ..., 7.) *(2π / 8)

                tensor([
            0 * (π/4),
            1 * (π/4),
            2 * (π/4),
            3 * (π/4),
            4 * (π/4),
            5 * (π/4),
            6 * (π/4),
            7 * (π/4)
        ])

        0, 45도, 90도, 135도, ..., 315도에 해당하는 head별 방향각(theta) 값들이 생김
        "Multi-Head Attention의 각 head가 다른 방향(θ)을 바라보게 초기화"
        head마다 sampling 방향을 다르게 하기 위해
        원(circle) 위에 균등하게 나눠진 방향을 만든다.
        "torch.arange는 0부터 head 수 - 1까지 tensor를 만들고, 이걸 이용해 0부터 2π까지 균등하게 나눈 head별 방향각(θ)을 계산하는 것이다."
        '''
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)

        #theta 각도에 대해 cos, sin 값을 계산해서 쌍으로 묶으면 cos쎄타 sin쎄타 형태의 벡터를 만들어 단위 원위의 포인트를만들 수 있다.
        #"head별 θ(방향각)에 대해 (cosθ, sinθ) 좌표를 만들어서, 2D 원(circle) 위에 점을 생성하는 과정"
        #두 tensor를 마지막 차원(-1) 에 대해 쌓아버리는 거야.
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        #(cos, sin) 벡터를 정규화(normalization)한다.벡터 길이를 1로 맞추려는 과정. (엄밀히 말하면, 값이 1이지만 혹시나 수치 오류를 방지하는 목적도 있다.)
        # shape을 (n_heads, 1, 1, 2)로 바꾼 다음, 이걸 n_levels, n_points 수만큼 반복(repeat)한다.\
        # 각 head마다 각 level마다 각 point마다 (cos, sin) 방향 벡터를 복제해서 갖게 된다
        # 초기부터 head마다 방향을 다르게 줘서 다양한 방향에서 feature를 샘플링하도록 유도하는 거야.
        # head별 (cosθ, sinθ) 방향 벡터를 정규화(normalization)하고, level과 point마다 복제(repeat)하는 과정이다.

        '''
        grid_init: 각 head에 대한 (cosθ, sinθ) 2D 벡터였지. 
        여기서 .abs().max(-1, keepdim=True)[0]는:
        각 벡터의 (cos, sin) 요소 중 최댓값의 절댓값을 구한다.
        (cosθ, sinθ) 둘 중 큰 값을 기준으로 나눈다.
        cos, sin 값은 [-1, 1] 범위라서

        (cos, sin) 둘 다 -1이나 1 근처 값이 나올 수 있어.
        이걸 길이(norm) 를 1로 맞추려는 거야.
        벡터 길이 조정(normalization) 과정이야.
        (※ 완벽한 L2 normalization은 아니고, max값 기준 스케일 조정)

        grid_init shape을 (n_heads, 1, 1, 2)로 변형.
        head별로 고정된 방향을 level과 point에 대해 복제할 준비를 하는 것.

        repeat을 이용해
        level 수(self.n_levels)
        point 수(self.n_points) 방향으로 복제한다.
        ead별로 정해진 방향을
        모든 level과 모든 sampling point에 대해 복제해서 쓴다.
        (나중에 point마다 거리만 살짝 조정하고, 방향은 그대로 가져간다.)

        '''

        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).\
            repeat(1, self.n_levels, self.n_points, 1)
        
        # 각 point마다 방향 벡터의 길이(scale) 를 다르게 한다.
        #첫 번째 point는 1배, 두 번째 point는 2배, 세 번째 point는 3배
        #원 중심으로부터 멀리 퍼져가게 하려고 하는 거야.
        #즉, 샘플링 포인트들이 원 주변에 점점 멀어지도록 설정하는 것.
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        #gradient를 계산하지 않고 (no_grad),

        #위에서 만든 grid_init 값을 sampling_offsets.bias로 설정한다.
        #학습 초기에는 이 bias를 이용해 "원 모양으로 퍼진 샘플링 패턴"을 만든다.
        #학습이 시작되면 이 bias도 업데이트될 수 있다.
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            #  학습 대상이 아니다.
            # Backpropagation(역전파) 때 여기를 건너뛰고 계산한다.
            # sampling_offsets.bias를 직접 초기화하고 싶은 거지,
            # 이 초기화 자체가 학습돼야 하는 건 아니니까.
            # 초기화할 때는 gradient 필요 없으니 계산을 끈다.
            # "모델의 학습 가능한(weight update 되는) 파라미터로 등록하는 special tensor" 다.
            # Optimizer가 이걸 학습 대상으로 삼는다.
            #grid_init shape이 (n_heads, n_levels, n_points, 2)였지. .view(-1) 하면 다 1D로 펼쳐버린다.
            # 즉, bias로 넣을 때 1D 벡터 형태로 넣어야 하니까 펼치는 거야.
            # 초기 grid_init 값으로 직접 세팅해서
            #학습 가능한 파라미터로 등록하는데,
            #초기 세팅 과정에서는 gradient를 계산하지 않게 한다."
            #만약 bias를 0으로만 초기화하면 어떤 문제가 발생하는지 (학습 불안정, 수렴 느림)


        #attention_weights Linear Layer의 weight를 0으로 초기화한다.
        #초기 attention weight는 random이 아니라 0 weight를 갖는다.
        #초기엔 거의 균등하게 attention을 하게 되도록 유도.

        constant_(self.attention_weights.weight.data, 0.)
        #attention_weights의 bias도 0으로 초기화. (초기 bias가 없다는 의미)
        constant_(self.attention_weights.bias.data, 0.)

        #value_proj의 weight를 Xavier Uniform 초기화 한다.
        #입력/출력 차원을 고려해 weight를 균일하게 샘플링하는 초기화 방법.
        #주로 딥러닝의 깊은 층에서도 gradient가 너무 커지거나 작아지는 걸 방지하는 역할

        xavier_uniform_(self.value_proj.weight.data)

        #value_proj의 bias를 0으로 초기화한다.
        constant_(self.value_proj.bias.data, 0.)

        #output_proj의 weight를 Xavier Uniform 초기화 한다.
        #(값들이 골고루 분포되도록 초기화)
        xavier_uniform_(self.output_proj.weight.data)
        #output_proj의 bias를 0으로 초기화한다.
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)

        query: (B, Length_query, C) ➔ Query feature들
        reference_points: (B, Length_query, n_levels, 2 or 4) ➔ Sampling할 기준 좌표들
        input_flatten: (B, Σ(H_l×W_l), C) ➔ Flatten된 multi-level feature map
        input_spatial_shapes: (n_levels, 2) ➔ 각 level의 height, width
        input_level_start_index: (n_levels,) ➔ Flattened feature에서 level별 시작 인덱스
        input_padding_mask: (Optional) ➔ Padding mask
        """

        '''
        batch size: N   
        query 개수: Len_q
        input_flatten 전체 feature 개수: Len_in (1,20000, 128)
        '''
        
        N, Len_q, _ = query.shape
        # N= 1 배치사이즈는 1이다.
        #print(f"Len_q Len_qLen_qLen_q {Len_q}")  # Len_q 20000

        N, Len_in, _ = input_flatten.shape  # query.clone으로 들어온다.
        #print(f"Len_in Len_in {Len_in}")  #  Len_in 20000

        #print(f"Len_inLen_inLen_inLen_in {Len_in}")
        #print(f"i(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() {(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()}")

        '''
        Σ(H * W)가 flatten된 input feature 수(Len_in)와 같아야 한다.
        즉, spatial shapes로부터 예상한 픽셀 개수와 실제 input_flatten 길이가 같아야 한다. 틀리면 에러난다.

        여기선 (1,2) 1행 2열 0으로 초기화
        input_spatial_shapes[:, 0]tensor([100], device='cuda:0')
        input_spatial_shapes[:, 1]tensor([200], device='cuda:0')

        '''

        #print(f"input_spatial_shapes[:, 0]{input_spatial_shapes[:, 0]}")
        #print(f"input_spatial_shapes[:, 1]{input_spatial_shapes[:, 1]}")

        #print(f"(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()) {(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()}")# (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()) 20000
        #print(f"(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1])) {(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1])}")# (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1])) tensor([20000], device='cuda:0')
        # sum 안해주면 tensor로 나옴

        #print(f"Len_inLen_inLen_in{Len_in}") Len_inLen_inLen_in 20000


        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        
        #print(f"input_flatten input_flatten input_flatten.shape {input_flatten.shape}") (1,20000,128)
        #print(f"query shape {query.shape}") #(1,20000,128)

        '''
        input_flatten feature에 대해 nn.Linear(d_model, d_model) 수행.
        feature를 새롭게 정렬해서 Multi-Head Attention에 맞게 만든다.
        (여기서 value를 만들어내는 과정.)
        '''

        value = self.value_proj(input_flatten) # linear d_model d_model로 넣음


        '''
        만약 padding mask가 있다면, 
        padding 위치의 value를 전부 0으로 만들어서 attention이 거기로 가지 않게 한다.

        "padding 영역에 해당하는 feature 값을 0으로 만들어버리는 코드"
        Shape: (N, Len_in, n_heads, d_model//n_heads)
        Multi-level feature들을 flatten하고 head별로 나눈 것.
        이 value들로부터 샘플링을 하게 되는데, padding 부분도 포함되어 있을 수 있다.

        input_padding_mask
        Shape: (N, Len_in)

        각 위치가 True/False로 되어 있어.
        True	이 위치는 패딩(padding)이야 (의미 없음)
        False	실제 유효한 데이터야 (의미 있음)

        mask가 True인 곳에 지정한 value를 채운다.
        mask가 True인 곳 ➔ 패딩된 부분
        이 곳에 0.0을 채워넣는다.
        padding 부분 feature를 0으로 만든다.
        #... Ellipsis 나머지 차원은 건드리지 말고 그대로 유지해라

        #... 는 앞의 모든 차원을 그대로 두고" 라는 의미
        x = torch.randn(2, 3, 4)
        # x.shape = (2, 3, 4)
        x[..., 0]

        None는 새로운 차원 추가 (= numpy의 np.newaxis랑 똑같음

        x = torch.randn(2, 3)

        # x.shape = (2, 3)

        x[:, None, :]
        # shape = (2, 1, 3)
        가운데에 1 크기의 새로운 차원이 생긴다.
        맨 마지막에 1 크기의 차원을 하나 추가하되, 앞 차원은 건드리지 않는다.

        value 텐서의 shape이 (N, Len_in, n_heads, d_model//n_heads)라서
        broadcasting 할 때 shape을 맞추려고!

        (N, Len_in, 1) vs (N, Len_in, n_heads, d_model//n_heads)

        이렇게 하면 (1) 위치에서 broadcasting이 되어서 n_heads 개로 맞춰진다.

        "서로 다른 shape을 가진 tensor끼리 연산할 때, 자동으로 shape을 맞춰주는 규칙"

        에서부터 차원을 비교
        크기가 같거나, 하나가 1이면 OK
        필요하면 1인 차원을 복제
        같은 shape이 된 다음 element-wise 연산 수행



        '''

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))


        '''
        Multi-head Attention을 위해
        feature를 head 수만큼 분리한다.
        (B, Length_in, n_heads, d_model_per_head)        
        '''

        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        '''
        query를 input으로 해서
        각 head, 각 level, 각 point에 대해
        (x, y) offset을 예측한다.

        샘플 오프셋은 쿼리로 만든다. 벨류 키가 아니다.
        '''

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        #view는 samplig_offsets의 query인풋으로 넣은 아웃풋을 출력 텐서를 다차원 형태로 reshape 해줌
        #print(f"sampling_offsets {sampling_offsets.shape}") # sampling_offsets torch.Size([1, 20000, 8, 1, 4, 2])

        ''''

        query를 input으로 해서
        head * level * point 개수만큼 weight를 만든다.
        softmax를 통해 각 point마다 weight를 normalize한다 (합=1).
        
        sampling point로 얻은 feature들 > head별, level별, point별
        attention wegith 각각 가중치 w가 i, j, k로 정규화 됨
        최종 weighted sum 수식은 output w의 i, j, k 와 feature의 i, j, k의 sum으로 결정됨
        (1, 20000, 8, 4) → 각 query당 head별로 4개의 포인트를 볼때 중요도가 필요한데 그걸 attentio_weight로 쓰겠다.
        '''

        #(batch_size, num_queries, num_heads, num_levels * num_points)
        # Deformable Attention의 핵심인 가중치 정규화와 구조 재구성을 담당하는 코드야.

        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        #print(f"attention_weights shape {attention_weights.shape}") #attention_weights shape torch.Size([1, 20000, 8, 4])
        # 각 query, 각 head가 보고 있는 sampling point들의 가중치 총합이 1이 되도록 만듦.
        # 마지막 차원에 대해 softmax를 적용한다는 뜻. 헤드의 포인트 4개에 소프트 맥스를 적용해서 가중치를 확률값으로 출력하게됨.
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)


        #print(f"attention_weights.shape{attention_weights.shape}") #attention_weights.shapetorch.Size([1, 20000, 8, 1, 4])

        '''
        print(f"attention_weights{attention_weights}")
        20000개의 쿼리마다 8개의 헤드가 있고 그 헤드마다 4개의 포인트가 있고 그포인트의  weight 가중치값이 계산돼서 나온다.
        self.n_point마다 가중치가 나오게됨 

         [[[0.1220, 0.1161, 0.1845, 0.5774]],

          [[0.0848, 0.0724, 0.0722, 0.7707]],

          [[0.1326, 0.1013, 0.1787, 0.5874]],

          ...,

          [[0.0956, 0.0702, 0.1578, 0.6764]],

          [[0.1446, 0.1087, 0.2163, 0.5304]],

          [[0.0260, 0.0145, 0.0243, 0.9352]]]]], device='cuda:0')
        
        
        '''
        
        
        # N, Len_q, n_heads, n_levels, n_points, 2


        if reference_points.shape[-1] == 2:
            #torch.stack()은 동일한 크기의 텐서들을 새로운 차원에 따라 쌓아 하나의 텐서로 만듭니다. dim 파라미터는 이 새로운 차원이 삽입될 위치를 지정합니다.
            # 가장 마지막 차원을 추가해서 쌓겠다.
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)

            #print(f" input_spatial_shapes[..., 1] {input_spatial_shapes[..., 1]}") # input_spatial_shapes[..., 1] tensor([200], device='cuda:0')
            #print(f" input_spatial_shapes[..., 0] {input_spatial_shapes[..., 0]}") #  input_spatial_shapes[..., 0] tensor([100], device='cuda:0')
            #print(f"offset_normalizer {offset_normalizer}") # offset_normalizer tensor([[200, 100]], device='cuda:0')
            #print(f"offset_normalizer.shape {offset_normalizer.shape}") # offset_normalizer.shape torch.Size([1, 2])

            



            '''
            sampling_offsets는
            reference_points를 중심으로 "얼마나 이동해서 샘플링할지"를 알려주는 값이야.

            reference_points:
            "기준 위치"를 (0~1 사이) normalized 좌표로 잡아.
            sampling_offsets:
            이 기준 위치에서 (x, y) 방향으로 얼마나 이동할지 offset을 예측한다.
            offset을 feature map 크기로 나눠서 정규화하고,  
            reference_points에 더한다.
            최종적으로 "샘플링할 위치들"이 나온다.
            MSDeformAttnFunction 안에서는:
            이 sampling_locations 위치를 bilinear interpolation으로 실제 feature map에서 값을 가져온다.
            가져온 feature들에 attention_weights를 곱해서 가중합(sum)한다.

            처음 모델을 학습 시작할 때, 
            sampling_offsets를 0으로 초기화하면
            모든 sampling location이 reference point 중심에만 몰려버려.
            이러면 학습이 제대로 안 되고 gradient도 제대로 안 나온다.
            처음부터 sampling point들을 원형(circle)으로 분산 배치한다.
            head마다 방향(θ)을 다르게 줘서
            각 head가 서로 다른 방향으로 퍼져서 샘플링하게 한다.
            그리고 포인트 간 거리를 점점 늘려가면서 (1, 2, 3배 등) 원 주변에 퍼뜨린다

            '''

            '''
            reference_points: 기본 위치 (0~1 normalized)    
            sampling_offsets: 예측된 offset (raw pixel scale)
            offset_normalizer: feature map 크기 (W, H)로 normalize
            sampling location 계산 (reference + offset)
            sampling_locations=reference_points+ (sampling_offsets / feature_size) 기준점 주변으로 미세하게 이동한 실제 sampling 위치 계산된다.
            '''
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4: # 만약 reference_point가 x,y,w,h의 box형태라면  shape[-1] reference point의 마지막차원
            # 박스 크기에 따라 샘플링 위치를 다르게 조정하는경우
            # 여기선 보통 x,y 2d point를 쓰니 위 if문만 사용한다.
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
                #reference point는 반드시 2개(x,y)나 4개(x,y,w,h) 좌표로 구성돼야 함.
        # debug only:
        # print("DEFORM ATTN in DEBUG MODE!!! ")
        # output_debug = ms_deform_attn_core_pytorch(value=value, value_spatial_shapes=input_spatial_shapes,
        #                                            sampling_locations=sampling_locations,
        #                                            attention_weights=attention_weights)

        '''
        sampling_locations 위치에서 
        value feature를 bilinear interpolation으로 가져오고
        attention_weights를 곱해서
        가중합(weighted sum) 하고
        최종적으로 query마다 새로운 feature를 만든다.
        '''

        # CUDA kernel "ms_deform_attn_forward_cuda" is fp32-only.
        # Under autocast (fp16/bf16) we must cast inputs to fp32, run the op,
        # then cast back to the autocast dtype so subsequent ops keep their precision.
        _msda_in_dtype = value.dtype
        if _msda_in_dtype != torch.float32:
            value_f32 = value.float()
            sampling_locations_f32 = sampling_locations.float()
            attention_weights_f32 = attention_weights.float()
            with torch.cuda.amp.autocast(enabled=False):
                output = MSDeformAttnFunction.apply(
                    value_f32, input_spatial_shapes, input_level_start_index,
                    sampling_locations_f32, attention_weights_f32, self.im2col_step)
            output = output.to(_msda_in_dtype)
        else:
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        
        '''
        Multi-head attention 결과를 하나의 feature로 합치기 위한 마지막 선형 변환 (nn.Linear)   
        output은 각 head들의 결과가 이어져 있는 상태야.
        self.output_proj를 통해
        다시 원래 dimension(d_model)으로 맞춘다.
        정보를 압축/변형해서 다음 레이어로 넘길 준비 완료.

        sampling_locations에서 value feature를 샘플링해서 attention weight로 가중합하고, 
        마지막에 output_proj로 head별 feature를 합쳐 최종 output feature를 만든다.

        "interpolation"은 보간법, 
        내삽이라고 번역되며, 
        주어진 값들 사이의 값을 추정하거나 계산하는 것을 의미합니다. 
        즉, 특정 지점의 값을 알고 있을 때, 그 사이의 값을 예측하거나 채워 넣는 과정을 가리킵니다.

        샘플링 위치(sampling_locations)가 정확히 픽셀 중심이 아닐 때,
        그 주변 4개의 픽셀로부터 값을 보간(interpolate) 해서 feature 값을 계산함.

        이 sum을 통해 각 query마다 최종 feature가 만들어진다.
        각 head별 결과는 나중에 concat되고, output_proj를 통해 통합됨.

        '''

        # Multi-Head Attention 후처리 역할을 함.   
        # 각 head의 결과가 그냥 이어져있는 채로 남아버림.
        # output_proj는 (head × d_head) → d_model 로 통합하는 역할을 한다.
        # 없으면 각 head의 정보를 결합할 수 없다.
        #output_proj는 학습 가능한 가중치로 head간 조합을 학습함.
        #이걸 빼면 그냥 단순 concat 결과를 쓰는 셈.
        #→ 정보 융합 부족 → 성능 저하

        output = self.output_proj(output)

        '''
        Bilinear Interpolation	(x, y) 실수 좌표 기준으로 4픽셀 보간해서 feature 추출
        Attention Weight & Weighted Sum	softmax weight로 각 샘플링된 feature를 가중합
        Output Projection	여러 head에서 나온 feature를 통합하고 후처리하는 Linear layer (없으면 학습 성능 저하)
        '''


        return output


# Multi-Scale Deformable Attention을 3D 상황에 맞게 확장한 버전.
# Deformable DETR의 Multi-Scale Attention을 3D BEV에 맞게 확장한 클래스
class MSDeformAttn3D(nn.Module): 

    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.

            embed_dims: 전체 feature dimension (ex: 256)

            num_heads: multi-head attention head 수

            num_levels: multi-scale feature level 수

            num_points: sampling할 point 수

            im2col_step: 내부 연산 batch chunk size

            dropout, batch_first, norm_cfg, init_cfg: 부가 설정

    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__()
        #head 수로 나눠 떨어지지 않으면 에러 (ex. 256 / 8 = 32 OK)

        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '  # head별로 dimension이 나누어 떨어져야 한다. (ex: 256 / 8 = 32)
                             f'but got {embed_dims} and {num_heads}')
        

        dim_per_head = embed_dims // num_heads # head 하나당 dimension 크기

        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        #head별 dimension이 2의 거듭제곱이면 CUDA 최적화가 잘 됨.

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0


        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        '''
        sampling_offsets: 각 query가 어디를 샘플링할지 offset 예측
        attention_weights: 각 샘플 포인트에 대해 가중치 예측
        value_proj: 입력 feature를 head별로 나누기 전에 변환
        '''

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear( # CUDA에서 효율적인 연산을 위해 power of 2 체크
            embed_dims, num_heads * num_levels * num_points * 2)
        
        self.attention_weights = nn.Linear(embed_dims, # 각 query마다 (head × level × point) 개수의 (x,y) offset 예
                                           num_heads * num_levels * num_points)
        #각 sampling 위치에 대한 attention 가중치 예측
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights() # weight와 bias를 특별히 초기화해줌

    def init_weights(self):
        """Default initialization for Parameters of Module."""

        #sampling offset 가중치는 0으로 초기화.
        self.sampling_offsets.weight.data.fill_(0.0)
        self.sampling_offsets.bias.data.fill_(0.0)

        #head별 방향성을 원(circle) 위에 고르게 배치(cosθ, sinθ) 형태로 초기 방향 설정
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        # head마다 고유한 각도 θ 생성
        # (cosθ, sinθ)로 방향 벡터 만들기 → head별 원형 분포 생성
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1  # 원 밖으로 점점 멀어지도록 크기 스케일 조정 (1배, 2배, 3배...)

        # wegith 초기화
        self.sampling_offsets.bias.data = grid_init.view(-1)
        self.attention_weights.weight.data.fill_(0.0)
        self.attention_weights.bias.data.fill_(0.0)
        torch.nn.init.xavier_uniform_(self.value_proj.weight) # bias에 방향 초기값 저장 Xavier 초기화 → 분산을 고려한 weight 초기화
        self.value_proj.bias.data.fill_(0.0)
        self._is_init = True
        #attention weight도 0 초기화 value_proj는 Xavier 초기화 (초기 분포 넓게)



    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].

             Query 기반 Deformable Attention 수행   
             Key, Value 입력 없으면 Query 자체를 쓰겠다.
             Positional encoding(query_pos) 있으면 query에 더해준다.
        """
        # Key/Value가 주어지지 않으면 query 자체 사용
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first: # batch dimension이 맨 앞으로 오게 정렬
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            # batch dimension이 앞에 없으면 바꿔준다.

        #spatial_shapes 정보와 value 길이 일치하는지 확인
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value.float()) # value를 head에 맞게 변환 입력 value의 총 픽셀 수와 일치하는지 확인

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0) #padding된 부분은 feature를 0으로 만들어버린다. value를 head별로 나누기 전 변환
        
        # head 수 만큼 feature dimension 쪼갬 (multi-head 구조)
        value = value.view(bs, num_value, self.num_heads, -1) #multi-head 구조로 분할

        # query를 통해 어디를 샘플링할지 offset 예측
        # query를 통해 attention weight를 예측하고 softmax

        # offset & weight 계산
        # (query → offset 예측) → (B, Nq, H, L, P, 2)
        # sampling 위치마다 weight softmax

        # → reference point + offset → sampling 위치가 됨.
        # view는 Tensor의 shape(모양)을 바꿔주는 함수
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        #print(f" sampling_offsets {sampling_offsets.shape}") #  sampling_offsets torch.Size([3, 3814, 8, 4, 8, 2])
        attention_weights = self.attention_weights(query).view( # 각 query마다 샘플링된 포인트 수만큼 가중치(score) 를 예측함
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        #마지막 차원 (num_levels × num_points) 방향으로 softmax
        #→ 각 query/head마다 sampling된 포인트들의 확률(weight) 을 만듦
        #즉, sampling한 값들 중 어떤 곳에 더 집중할지를 softmax로 결정
        attention_weights = attention_weights.softmax(-1)
        #print(f" attention_weights {attention_weights.shape}") #   attention_weights torch.Size([3, 3814, 8, 32])

        #이제 attention 연산을 수행할 때는
        #각 head, level, point에 대해 정확히 align된 weight shape이 필요하니까!
        #어디를 볼까? → sampling_offsets
        #얼마나 집중할까? → attention_weights + softmax
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        #print(f" attention_weights {attention_weights.shape}") #    attention_weights torch.Size([3, 3814, 8, 4, 8])


        #reference_point + offset → sampling_locations 계산
        #sampling된 위치에서 bilinear interpolation 수행 원리
        #softmax를 쓰지 않고 weighted sum 하면 무슨 문제가 생기는지?
        '''
        Deformable Attention은 다음 구조로 작동해:
        각 query
            각 head
                각 level
                    각 point → 에 대해 (x, y) offset을 예측함.


        그래서 .view(...)를 통해 정확히 이렇게 나눠야
        뒤에서 sampling_locations 만들 때
        reference_point + offset 계산이 깔끔하게 맞아떨어져.+
        .view(...)는 1D 예측 결과를 (head, level, point, 2) 차원으로 재구성하여, 멀티헤드 attention 구조에 맞게 정렬하는 역할을 한다.
        '''

        if reference_points.shape[-1] == 2: #reference_points가 (x,y) 2개일 때
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.

            sampling_locations=reference_points+ (sampling_offsets/feature_size)
            (Normalized 좌표 기준 + offset 정규화)로 이동한 sampling 위치 계산
            """
            #정규화된 offset + 기준점 → 최종 sampling 위치
            #참고: reference_points shape이 (B, Nq, L, 2)
            #output = MSDeformAttnFunction.apply(...)

            #sampling_locations 계산 
            '''
            각 reference_point(정규화된 기준 위치)에 대해
            sampling_offsets를 정규화된 거리로 더해줘서
            최종적인 sampling 위치를 계산하는 과정이야.
            '''

            '''
            spatial_shapes: (num_levels, 2) → (H, W)

            [W, H] 순서로 쌓고 -1 (마지막 dim)에 붙임

            결과 shape: (num_levels, 2)
            → 이걸로 offset 정규화를 수행 (→ 나중에 나눠줄 것!)

            왜 정규화해?
            sampling_offsets는 raw 좌표 (pixel 단위)
            근데 reference_points는 [0, 1] normalized 좌표
            → 좌표계를 맞춰주기 위해 W, H로 나눠서 정규화!
                        
            '''

            '''
            offset_normalizer tensor([[448, 224],
            [224, 112],
            [112,  56],
            [224, 112]], device='cuda:0')

            offset_normalize.shaper torch.Size([4, 2])
            '''
            #print(f"offset_normalizer {offset_normalizer}")
            #print(f"offset_normalize.shaper {offset_normalizer.shape}")


            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            


            #reference_points.shape torch.Size([3, 3814, 8, 2])
            #print(f"reference_points.shape {reference_points.shape}") # reference_points.shape torch.Size([3, 3814, 8, 2]
            # → height(Z) 방향 anchor가 여러 개 있는 구조임.

            bs, num_query, num_Z_anchors, xy = reference_points.shape

            reference_points = reference_points[:, :, None, None, None, :, :] #이건 차원을 늘림
            #print(f"reference_points.shape {reference_points.shape}") # reference_points.shape torch.Size([3, 3814, 1, 1, 1, 8, 2])

            #broadcasting을 위해 차원을 맞춰줌 나중에 sampling_offsets랑 더할 때 shape 맞추려고!


            '''
            offset_normalizer shape: (num_levels, 2)
            나눠서 offset을 정규화된 좌표로 바꿔줌이걸로 (x, y) offset이 [0~1] range로 변함→ reference_points랑 더해줄 수 있어!
            '''

            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            
            #head-level-point-anchor 별로 분해
            #reference point에 정규화된 offset을 더함 → 최종 sampling 위치
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape

            assert num_all_points == num_points * num_Z_anchors  #여기서 num_all_points = num_Z_anchors × num_points
            #정규화된 기준 좌표(reference)에 offset을 더해서, sampling할 위치(sampling_locations)를 계산하는 과정이다.
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4: #(현재 코드에서는 지원 안 함)
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        # debug only:
        # output_debug = ms_deform_attn_core_pytorch(value=value, value_spatial_shapes=spatial_shapes,
        #                                            sampling_locations=sampling_locations,
        #                                            attention_weights=attention_weights)

        #shape 다시 (B, Nq, C) 형태로 맞춤
        output = MSDeformAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)
        
        #print(f"output shape {output.shape}") #output shape torch.Size([3, 3814, 128])
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        #여기서 핵심 custom CUDA 연산 호출
        #bilinear interpolation + weighted sum 수행
        #print(f"output shape {output.shape}") output shape torch.Size([3, 3814, 128])
        return output
    

