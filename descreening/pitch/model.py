from torch import nn
from torch.nn import Module


class PitchModel(Module):
    def __init__(self, channels=64, residual=False):
        super().__init__()
        #self.residual = residual
        self.layer1 = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=0)
        self.layer2 = self.conv_module(channels, channels)
        #self.layer3 = self.conv_module(channels, channels)
        self.layer4 = nn.Conv2d(channels, 64, kernel_size=2, stride=1, padding=0)
        self.layer5 = nn.Sequential(nn.LazyLinear(64), nn.ReLU(), nn.LazyLinear(1))

    def forward(self, x):
        x = self.layer1(x)
        x2 = self.layer2(x)
        #p = (x.shape[2] - x2.shape[2]) // 2
        #x2 = x2 #+ x[:, :, p:-p, p:-p]
        #x3 = x2[:,:, p:-p, p:-p] + self.layer3(x2)
        print(x2.shape)
        x = self.layer4(x2)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.layer5(x)
        x = x.view(
            -1,
        )  # Flatten
        return x

    def conv_module(self, in_channels, out_channels, kernel_size=19):
        return nn.Sequential(
            
            # nn.BatchNorm2d(in_channels),
            nn.InstanceNorm2d(in_channels),
            #nn.Conv2d(1, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, groups=in_channels),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
            #nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, groups=out_channels),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
        )

    def predict(self, x):
        pass
