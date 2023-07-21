import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

from modules.Fusion import Fusion_Block
from modules.GCN import H2GCN
from modules.MobileHiT import MobileHIT_Block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HiGT(nn.Module):
    def __init__(
        self, 
        # GCN
        gcn_in_channels: int=1024,
        gcn_hid_channels: int=1024,
        gcn_out_channels: int=1024,
        gcn_drop_ratio: float=0.3,
        patch_ratio: int=4,
        pool_ratio: list=[0.5,5],

        # mhit
        re_patch_size: int=64,
        out_classes: int=2,
        mhit_num: int=3,

        # fusion
        fusion_exp_ratio: int=4,

        ) -> None:
        super().__init__()

        self.out_classes = out_classes
        self.ori_patch_size = gcn_out_channels
        self.re_patch_size = re_patch_size

        self.gcn = H2GCN(
            in_feats=gcn_in_channels, 
            n_hidden= gcn_hid_channels, 
            out_feats=gcn_out_channels, 
            drop_out_ratio=gcn_drop_ratio, 
            pool_ratio=pool_ratio,
        )

        global_rep = []

        self.last_pool_ratio = pool_ratio[-1]

        self.patch_ratio = patch_ratio

        for _ in range(mhit_num):
            global_rep.append(
                MobileHIT_Block(
                    channel = 1,
                    re_patch_size = re_patch_size,
                    ori_patch_size = gcn_out_channels,
                    region_node_num = self.last_pool_ratio,
                    patch_node_num = self.patch_ratio*self.last_pool_ratio
                )
            )

        self.mhit = nn.Sequential(*global_rep)

        fusion_in_channel = int(pool_ratio[-1]*patch_ratio*2)
        fusion_out_channel = fusion_in_channel*fusion_exp_ratio
        self.fusion = Fusion_Block(
            in_channel=fusion_in_channel,
            out_channel=fusion_out_channel
        )

        self.ln = nn.LayerNorm(gcn_out_channels)

        self.classifier = nn.Sequential(
            nn.Linear(gcn_out_channels, self.out_classes)
        )

        # init params
        self.apply(self.init_parameters)

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass
    

    def forward(self, data: Data):

        # HI_GCN
        x, edge_index, node_type, tree = self.gcn(data)

        # HI_ViT
        tumbnail_list = torch.where(node_type==0)[0].tolist()
        region_list = torch.where(node_type==1)[0].tolist()
        patch_list = torch.where(node_type==2)[0].tolist()

        region_nodes = x[region_list]
        patch_nodes = x[patch_list]
        thumbnail = x[tumbnail_list]
        patch_tree = tree[patch_list]

        n,c = patch_nodes.shape
        if n < self.last_pool_ratio*self.patch_ratio:
            patch_tree_values, patch_tree_counts = torch.unique(patch_tree, return_counts=True)
            value_add = []
            for i, value in enumerate(patch_tree_values):
                if patch_tree_counts[i]<4:
                    value_add += [int(value.item())]*int(4-patch_tree_counts[i].item())
            value_add = torch.tensor(value_add).to(patch_nodes.device)
            patch_tree = torch.cat((value_add, patch_tree)).long()
            e = torch.zeros((self.last_pool_ratio*self.patch_ratio-n,1024)).to(patch_nodes.device)
            patch_nodes = torch.cat((e,patch_nodes), dim=0)
        patch_nodes_ori = patch_nodes

        for mhit_ in self.mhit:
            region_nodes, patch_nodes = mhit_(
                region_nodes,
                patch_nodes,
                patch_tree.long()
            )

        # Fusion
        local_patch = self.ln(patch_nodes_ori+thumbnail)
        fusioned_patch_nodes = torch.mean(self.fusion(local_patch, patch_nodes), dim=0)

        # Classifier
        logits = self.classifier(fusioned_patch_nodes)
        prob = F.softmax(logits)
 
        return prob.squeeze(0)