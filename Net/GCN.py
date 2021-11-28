import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph import *
#from torch_geometric.nn import GCNConv,GATConv

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
class GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, partition=3, dropout=0):
        super(GCN_Block, self).__init__()
        self.partition = partition
        self.skeleton_graph = Graph(strategy='spatial', max_hop=1, dilation=1)
        self.A = torch.tensor(self.skeleton_graph.A, dtype=torch.float32, requires_grad=False).to(device)
        self.edge_importance = nn.Parameter(torch.ones(self.A.size()))

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*partition, stride=1, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels*partition),
            nn.ReLU(inplace=True)
        )
        self.temporal_block = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=(stride, 1), kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        if out_channels==in_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1,0)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.conv_block(x)
        b, pc, f, j = x.size()
        x = x.view(b, self.partition, pc//self.partition, f, j)
        edge_weight = (self.A.cpu())*(self.edge_importance.cpu())
        x = torch.einsum('bpcfj,pjw->bcfw', (x.cpu(), edge_weight.cpu()))
        x = self.temporal_block(x.to(device)) + res
        return x


class GrahpNet(nn.Module):
    def __init__(self, in_channels, num_joints=14, out_dim=1024, partition=3):
        super(GrahpNet, self).__init__()

        #self.register_buffer('A', A)

        self.GCN_Block1 = GCN_Block(in_channels, 8, stride=1, partition=partition)
        self.GCN_Block2 = GCN_Block(8, 8, stride=1, partition=partition)
        self.GCN_Block3 = GCN_Block(8, 16, stride=2, partition=partition)
        self.GCN_Block4 = GCN_Block(16, 16, stride=1, partition=partition)
        self.GCN_Block5 = GCN_Block(16, 32, stride=2, partition=partition)
        self.GCN_Block6 = GCN_Block(32, 32, stride=1, partition=partition)
        self.GCN_Block7 = GCN_Block(32, 64, stride=2, partition=partition)

        # self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for i in range(7)])

        self.fc = nn.Linear(64, 1)
        self.attention = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5, inplace=True)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)


    def forward(self, x):
        batch, channle, clip_length, num_joints = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, num_joints*channle, clip_length)
        x = self.data_bn(x)
        x = x.view(batch, num_joints, channle, clip_length).permute(0, 2, 3, 1).contiguous()
        x = self.GCN_Block1(x)
        x = self.GCN_Block2(x)
        x = self.GCN_Block3(x)
        x = self.GCN_Block4(x)
        x = self.GCN_Block5(x)
        x = self.GCN_Block6(x)
        x = self.GCN_Block7(x)
        batch, channel, t, joints = x.size()
        #print(x.size())
        x = F.max_pool2d(x, (t, joints))
        x = x.view(batch, -1)
        x = self.fc(self.dropout(x))
        att = self.attention(x)
        #print(att)
        return x, att



if __name__ == '__main__':

    x = torch.ones([1, 2, 16, 14])#batch, channle, clips, num_joints]
    edge = torch.rand([3, 14, 14])

    net = GrahpNet(in_channels=2, out_dim=1024, partition=3)
    out, att = net(x)
    print(out, att)


'''
in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True,  negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs
'''