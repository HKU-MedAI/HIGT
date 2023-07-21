from torch import nn
from torch import Tensor
from .MobileViTv2 import MobileViTBlockV2, SqueezeExcitation
from torch_geometric.utils import scatter
import torch

class MobileHIT_Block(nn.Module):
    def __init__(
        self,
        channel: int=1,
        re_patch_size: int=1024,
        ori_patch_size: int=1024,
        region_node_num: int=5,
        patch_node_num: int=20,
        ) -> None:
        super().__init__()

        self.re_patch_size = re_patch_size
        self.ori_patch_size = ori_patch_size
        self.region_node_num = region_node_num
        self.patch_node_num = patch_node_num
        
        # Patch Level:
        self.patch_channel = channel
        self.patch_block = MobileViTBlockV2(
            in_channels = self.patch_channel,
            re_patch_size=self.re_patch_size,
            ori_patch_size = self.ori_patch_size,
            node_num = patch_node_num,
        )

        # Region Level:
        self.region_channel = channel
        self.region_block = MobileViTBlockV2(
            in_channels = self.region_channel,
            re_patch_size=self.re_patch_size,
            ori_patch_size = self.ori_patch_size,
            node_num = region_node_num,
        )
        
        # SE
        self.se_region = SqueezeExcitation(input_c=region_node_num)
        self.se_patch = SqueezeExcitation(input_c=patch_node_num)

    def forward(
        self, 
        region_nodes: Tensor, 
        patch_nodes: Tensor,
        tree: Tensor):

        # Patch block:
        patch_nodes = patch_nodes.reshape(-1, self.re_patch_size).permute(1,0).unsqueeze(0).unsqueeze(0)
        patch_nodes = self.patch_block(patch_nodes)
        patch_nodes = patch_nodes.squeeze(0).squeeze(0).permute(1,0).reshape(-1,self.ori_patch_size)
        
        # Hierachical Interaction:
        region_patch_nodes = torch.cat([region_nodes[i-1].unsqueeze(0) for i in tree], dim=0)
        patch_nodes = (patch_nodes+region_patch_nodes*self.se_patch(region_patch_nodes.unsqueeze(1).unsqueeze(0)).squeeze(0).squeeze(1))/2
        
        # Region block
        region_nodes = region_nodes.reshape(-1, self.re_patch_size).permute(1,0).unsqueeze(0).unsqueeze(0)
        region_nodes = self.region_block(region_nodes)
        region_nodes = region_nodes.squeeze(0).squeeze(0).permute(1,0).reshape(-1,self.ori_patch_size)

        return region_nodes, patch_nodes
