import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(CNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((D_in[1] - 8)//4 + 1, (D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.fc = torch.nn.Linear(64*dim3[0]*dim3[1], D_hidden)
        
    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
        

import sys, inspect, re

header_dict = {}
for class_name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    naming_rule = lambda x: re.sub('([a-z])([A-Z])', r'\1_\2', x).lower()
    header_dict[naming_rule(class_name)] = _class
