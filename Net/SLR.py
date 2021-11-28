from torchvision import transforms
from utils.VGGNET import *
from utils.P3DNet import *
from Net.GCN import *
from utils.LSTM import *
from config import *
from utils.VGGNET import *
from utils.tools import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class SLRNet(nn.Module):
    def __init__(self, cfg):
        super(SLRNet, self).__init__()
        self.feature_dim = cfg.VIDEO_FEATURE_DIM
        self.num_joints = cfg.NUM_JOINTS
        self.joints_channel = cfg.JOINTS_CHANNEL
        self.hidden_size = cfg.SEMANTIC_HIDDEN_DIM
        self.num_class = cfg.VOC_SZIE
        self.heatmap_size = cfg.HEATMAP_SIZE
        self.max_sentencelen = cfg.MAX_SENTENCELEN
        self.clip_length = cfg.CLIP_LENGTH
        self.sliding_stride = cfg.SLIDING_STRIDE

        self.nChannels = 4
        self.growthRate = 4
        self.SkeletonNet = Vgg(n_classes=2048)
        self.encoder = P3D63(layers=[3, 4, 6, 3],
                                    input_channel=4,  #(self.nChannels//2)+3*self.growthRate+1
                                    num_classes=self.feature_dim)

        self.graph_encoder = GrahpNet(in_channels=self.joints_channel, out_dim=self.feature_dim, partition=3)

        self.decoder = LSTM_seq(max_seq=self.max_sentencelen, input_size=self.feature_dim,
                                hidden_size=self.hidden_size, class_num=self.num_class)

        # for param in self.SkeletonNet.parameters():
        #     param.requires_grad = False

    def forward_2d(self, x):
        # x = torch.Size([4, 300, 200, 200, 3])
        # target = torch.Size([4, 300, 14, 100, 100])
        # joints = torch.Size([4, 300, 14, 2]
        x1, heatmap = checkpoint(self.SkeletonNet, x)
        # x1=[batch, time, feature_dim]
        # heatmap=[batch*video_len, 14, 100,100]
        return x1, heatmap

    def forward_3d(self, x, target):
        batch, video_length, h, w, channel = x.shape
        x = x.reshape(-1, h, w, channel).permute(0, 3, 1, 2).contiguous()
        target = target.reshape(-1, self.num_joints, self.heatmap_size[0], self.heatmap_size[1]).contiguous()
        attention_map = torch.sum(target, dim=1, keepdim=True)
        attention_map = transforms.functional.resize(attention_map, (h, w))
        x = torch.cat((x, attention_map), dim=1).unsqueeze(0)
        x = x.unfold(dimension=1, size=self.clip_length, step=self.sliding_stride).permute(0, 1, 2, 5, 3, 4).contiguous()
        batch, time_step, channel, clip_length, h, w = x.shape
        x = x.view(batch * time_step, channel, clip_length, h, w)
        x = checkpoint(self.encoder, x)
        x = x.view(batch, time_step, -1)
        return x

    def foward_grah(self, joints):
        # joints = [batch, video_len, 14, channel]
        joints = joints.unfold(dimension=1, size=self.clip_length, step=self.sliding_stride).permute(0, 1, 3, 4,
                                                                                                     2).contiguous()
        batch, time_step, channel, clip_length, num_joints = joints.shape
        joints = joints.view(batch * time_step, self.joints_channel, clip_length, self.num_joints)
        # [batch*time_step, channle, clips, num_joints]
        gx, gatt = checkpoint(self.graph_encoder, joints)
        # gx =[batch*time, feature_dim] gatt=[batch*time, 1]
        gx = gx.view(batch, time_step, -1)
        gatt = gatt.view(batch, time_step, -1)
        return gx, gatt

    def merge(self, x, gatt):
        x = torch.mul(x, gatt)
        return x

    def forward(self, x, label):
        x1, heatmap = self.forward_2d(x)
        x = self.forward_3d(x, heatmap.detach())
        joints, _ = get_max_preds(heatmap.detach().numpy())
        gx, gatt = self.foward_grah(joints)
        x = self.merge(x,  gatt=gatt)
        x = self.decoder(x, label)
        return x, x.argmax(2), heatmap, gatt


if __name__ == '__main__':
    cfg = CLR_config()
    edge = torch.rand([3, 14, 14])
    model = SLRNet(cfg).to(device)
    x = torch.rand([1,300, 200, 200, 3]).to(device)
    target = torch.rand([1, 300, 14, 100, 100]).to(device)
    joints = torch.rand([1, 300, 14, 2]).to(device)
    label = torch.tensor([[0, 1, 2, 2, 2, 3, 3, 3, 3, 3,4,4,5,5,5,5]])

    x,word, heatmap, gatt = model(x, label, target, joints)
    print("debug1==", x.shape, gatt.shape, heatmap.shape)




