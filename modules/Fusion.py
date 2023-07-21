from torch import nn, Tensor
import torch

class Fusion_Block(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int
    ) -> None:
        super().__init__()
        self.conv_11 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 1//2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, local_patch: Tensor, global_patch: Tensor):
        
        patch_nodes = torch.cat((local_patch, global_patch), dim=0).unsqueeze(1).unsqueeze(0)
        patch_nodes = self.conv_11(patch_nodes).squeeze(0).squeeze(1)

        return patch_nodes