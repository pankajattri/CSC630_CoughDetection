import torch
from torchvision import models

class Disc(torch.nn.Module):
    
    def __init__(self):
        
        # constructor of torch.nn.Module
        
        super(Disc, self).__init__()
        
       
        self.model = models.resnet18(pretrained=True)
        self.model.conv1=torch.nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size[0], 
                      stride=self.model.conv1.stride[0], padding=self.model.conv1.padding[0])
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model(x)
        
        return x

if __name__ == '__main__':
    
    net = Disc()
    # minimum input size is 128 x 128
    x = torch.randn(8,1,128,128)
    y = net(x)