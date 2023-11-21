import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expension = 1
    def __init__(self,in_ch,block_ch,stride=1,downsample=None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch,block_ch,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(block_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(block_ch,block_ch*self.expension,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(block_ch*self.expension)
        self.relu2 = nn.ReLU()


    def forward(self,x):
        identity = x

        if self.downsample:
            identity = self.downsample(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity

        return self.relu2(out)


class Bottleneck(nn.Module):
    expension = 4
    def __init__(self, in_ch, block_ch, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch, block_ch, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(block_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(block_ch, block_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_ch)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(block_ch, block_ch*self.expension, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(block_ch*self.expension)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        identity = x
        # print("identity shape:",identity.shape)

        if self.downsample:
            identity = self.downsample(x)
            # print("identity shape:", identity.shape)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # print("out shape:",out.shape)

        out += identity

        return self.relu3(out)


class Resnet(nn.Module):
    def __init__(self,ch_in=3,block=Bottleneck,num_class=1000,block_num=[3,4,6,3]):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in_ch = 64
        self.maxpooling1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,block_num[0],stride=1)
        self.layer2 = self._make_layer(block,128,block_num[1],stride=2)
        self.layer3 = self._make_layer(block,256,block_num[2],stride=2)
        self.layer4 = self._make_layer(block,512,block_num[3],stride=2)
        self.fc = nn.Sequential(
            nn.Linear(512*block.expension*7*7,num_class),
            nn.Softmax(dim=-1)
        )

    def _initializa_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,model='fan_out',nonlinearity='relu')

            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)

    def _make_layer(self,block,block_ch,block_num,stride):
        layer = []
        downsample = None
        if stride != 1 or self.in_ch != block_ch*block.expension:
            downsample = nn.Conv2d(self.in_ch,block_ch*block.expension,kernel_size=1,stride=stride)
        layer += [block(self.in_ch,block_ch,downsample=downsample,stride=stride)]

        self.in_ch = block_ch*block.expension
        for _ in range(1,block_num):
            layer.append(block(self.in_ch,block_ch))

        return nn.Sequential(*layer)


    def forward(self,x):
        out = self.maxpooling1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    x = torch.randn(1,3,224,224)
    model = Resnet(ch_in=3,block=Bottleneck,num_class=100,block_num=[3,4,6,3])
    y = model(x)
    print(y.shape)

