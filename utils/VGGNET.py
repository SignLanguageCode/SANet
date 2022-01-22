import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from torchsummary import summary

Factor = 0.25


class Vgg(nn.Module):
    def __init__(self, num_joints=14, n_classes=10):
        super(Vgg, self).__init__()
        self.conv1_1 = nn.Conv2d(3, int(64*Factor), kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(int(64*Factor), int(64*Factor), kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(int(64*Factor), int(128*Factor), kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(int(128*Factor), int(128*Factor), kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(int(128*Factor), int(256*Factor), kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(int(256*Factor), int(256*Factor), kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(int(256*Factor), int(256*Factor), kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(int(256*Factor), int(512*Factor), kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(int(512*Factor), int(512*Factor), kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(int(512*Factor), int(512*Factor), kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(int(512*Factor), int(512*Factor), kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(int(512*Factor), int(512*Factor), kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(int(512*Factor), int(512*Factor), kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc6 = nn.Linear(6*6*int(512*Factor), n_classes)

        self.Tranconv1 = nn.ConvTranspose2d(int(512*Factor), int(256*Factor), 3, stride=2)
        self.Tranconv2 = nn.ConvTranspose2d(int(256*Factor), int(128*Factor), 3, stride=2, padding=1, output_padding=1)
        self.pointwise1 = nn.Conv2d(int(128*Factor), num_joints, 1)

        self.Tranconv3 = nn.ConvTranspose2d(int(256 * Factor), int(128 * Factor), 3, stride=2, padding=1,
                                            output_padding=1)

        self.pointwise2 = nn.Conv2d(int(128*Factor), num_joints, 1)

    def Conv1(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        return x

    def Conv2(self, x):
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        return x

    def Conv3(self, x):
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        return x

    def Conv4(self, x):
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        return x

    def Conv5(self, x):
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        # x =[batch, video_length, w, h, c]
        batch, video_length, w, h, channel = x.shape
        x = x.reshape(-1, w, h, channel).permute(0, 3, 1, 2).contiguous()

        # intput = [16,200,200,3] con2=[16,16,50,50]
        # conv3 =  [16, 32, 25, 25] conv4= [16, 64, 12, 12]
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        heatmap1 = self.Tranconv3(x)
        heatmap1 = self.pointwise2(heatmap1)

        x = self.Conv4(x)
        heatmap = self.Tranconv1(x)
        heatmap = self.Tranconv2(heatmap)
        heatmap = self.pointwise1(heatmap)
        heatmap = heatmap1 + heatmap

        x = self.Conv5(x)
        x = x.view(-1, 6 * 6 * int(512*Factor))
        x = self.fc6(x)
        x = x.view(batch, video_length, -1)

        return x, heatmap

if __name__ == '__main__':

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    model = Vgg(n_classes=2048).to(device)
    inputs = torch.rand(size=(1, 300, 200, 200, 3), dtype=torch.float32).to(device)
    x,heatmap = model(inputs)
    print(x.shape, heatmap.shape)
