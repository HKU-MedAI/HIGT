from .IHPool import IHPool
from .RAConv import RAConv

from torch import nn
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention,GCNConv, ChebConv, SAGEConv, GraphConv, LEConv, LayerNorm, GATConv

class H2GCN(nn.Module):
    def __init__(
        self, 
        in_feats: int, 
        n_hidden: int , 
        out_feats: int, 
        drop_out_ratio: float=0.2, 
        pool_ratio: list=[10], 
        ):
        super(H2GCN, self).__init__()

        self.pool_ratio = pool_ratio

        convs_list = []
        pools_list = []
        for i, ratio in enumerate(pool_ratio):
            if i == 0:
                convs_list.append(RAConv(in_channels=in_feats, out_channels=n_hidden))
                pools_list.append(IHPool(in_channels=n_hidden, ratio=ratio, select="inter", dis="ou"))
            elif i == len(pool_ratio)-1:
                convs_list.append(RAConv(in_channels=n_hidden, out_channels=out_feats))
                pools_list.append(IHPool(in_channels=out_feats, ratio=ratio, select="inter", dis="ou"))
            else:
                convs_list.append(RAConv(in_channels=n_hidden, out_channels=n_hidden))
                pools_list.append(IHPool(in_channels=n_hidden, ratio=ratio, select="inter", dis="ou"))

        self.convs = nn.Sequential(
            *convs_list
        )

        self.pools = nn.Sequential(
            *pools_list
        )

        self.norm = LayerNorm(in_feats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out_ratio)
        

    def forward(self, data):

        x, batch, edge_index, node_type, data_id, tree, x_y_index = data.x, data.batch, data.edge_index_tree_8nb, data.node_type, data.data_id, data.node_tree, data.x_y_index
        x_y_index = x_y_index * 2 - 1
        x = self.norm(x)

        for i, _ in enumerate(self.pool_ratio):
            x = self.convs[i](x, edge_index, node_type)
            x = self.norm(x)
            x = self.dropout(x)
            x, edge_index, edge_weight, batch, cluster, node_type, tree, score, x_y_index = self.pools[i](x,
                                                                                                            edge_index,
                                                                                                            node_type=node_type,
                                                                                                            tree=tree,
                                                                                                            x_y_index=x_y_index)
        

            batch = edge_index.new_zeros(x.size(0))

        return x, edge_index, node_type, tree #, x_y_index+1
