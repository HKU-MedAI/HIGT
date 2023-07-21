from typing import Optional, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        embed_channel: int,
        expand_channel: Optional[int]=2,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        ori_patch_size: int=1024,
        re_patch_size: int=16,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.embed_channel = embed_channel
        self.ori_patch_size = ori_patch_size
        self.re_patch_size = re_patch_size

        if not expand_channel:
            self.expand_channel = self.embed_channel
        else:
            self.expand_channel = expand_channel

        self.qkv_proj = ConvLayer(
            in_channels=self.embed_channel,
            out_channels=1 + (2 * self.expand_channel),
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            in_channels=self.expand_channel,
            out_channels=self.embed_channel,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )


    def __repr__(self):
        return "{}(embed_channel={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_channel, self.attn_dropout.p
        )
    
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.expand_channel, self.expand_channel], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        
        # Compute context vector
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        context_vector = context_vector.expand_as(value)

        out = F.relu(value) * context_vector
        out = self.out_proj(out)

        return out
        
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2)
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class LinearAttnFFN(nn.Module):
    def __init__(
        self,
        # opts,
        embed_channel: int,
        ori_patch_size: int,
        re_patch_size: int,
        node_num: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: float=0.1,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.re_patch_size = re_patch_size
        norm_dim = [ori_patch_size, node_num] if ori_patch_size==re_patch_size else [re_patch_size, int(node_num*ori_patch_size/re_patch_size)]
        self.attn = nn.Sequential(
            nn.LayerNorm(norm_dim),
            LinearSelfAttention( 
                embed_channel=embed_channel, 
                ori_patch_size=ori_patch_size,
                attn_dropout=attn_dropout, 
                re_patch_size=re_patch_size,
                bias=True,
            ),
            
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        
        
        ln_dim = ori_patch_size

        ffn_latent_dim = embed_channel*16

        self.ffn = nn.Sequential(
            ConvLayer(
                in_channels=embed_channel,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=True,
            ),
            nn.Dropout(p=ffn_dropout),
            ConvLayer(
                in_channels=ffn_latent_dim,
                out_channels=embed_channel,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=False,
            ),
            nn.Dropout(p=ffn_dropout),
        )

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        x_  = self.attn(x)
        x_ = self.attn_dropout(x_)
        x = x + x_
        x = x + self.ffn(x)

        return x