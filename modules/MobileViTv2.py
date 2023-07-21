import numpy as np
from torch import nn, Tensor
from typing import Optional, Tuple, Union, Sequence
from .Transformer import LinearAttnFFN
from torch.nn import functional as F

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1,1))
        # print(scale.shape)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scores = F.hardsigmoid(scale, inplace=True)
        return scores

class MobileViTBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ori_patch_size: int,
        re_patch_size: int,
        node_num: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 1.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        conv_ksize: Optional[int] = 3,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
    ) -> None:
        super().__init__()

        self.transformer_in_dim = in_channels
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

        self.global_rep = self._build_attn_layer(
            embed_channel=in_channels,
            ori_patch_size=ori_patch_size,
            re_patch_size=re_patch_size,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            node_num=node_num,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )
        self.ln = nn.LayerNorm(re_patch_size)

        self.layer_num = n_attn_blocks

    def _build_attn_layer(
        self,
        embed_channel: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        ori_patch_size: int,
        re_patch_size: int,
        node_num: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer: str,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * embed_channel
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * embed_channel] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * embed_channel] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]
        '''
        Input: :math:`(B, C, P, N)` where :math:
        # `B` is batch size, 
        # `C` is input embedding dim,
        # `P` is number of pixels in a patch,
        # `N` is number of patches,
        '''
        global_rep = []
        
        for _ in range(n_layers):
            global_rep.append(
                LinearAttnFFN(
                    embed_channel=embed_channel,
                    ori_patch_size=ori_patch_size,
                    re_patch_size=re_patch_size,
                    node_num=node_num,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                )
            )
            
        return nn.Sequential(*global_rep)#, d_model
    
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # learn global representations on all patches
        x = self.global_rep(x)
        return x