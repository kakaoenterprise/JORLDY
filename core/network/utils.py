import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(CNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((D_in[1] - 8)//4 + 1, (D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
                
        self.D_head_out = 64*dim3[0]*dim3[1]
        
    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x

class MLP(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(MLP, self).__init__()

        self.l = torch.nn.Linear(D_in, D_hidden)
        self.D_head_out = D_hidden

    def forward(self, x):
        x = F.relu(self.l(x))
        return x

class CNN_LSTM(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(CNN_LSTM, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((D_in[1] - 8)//4 + 1, (D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.D_conv_out = 64*dim3[0]*dim3[1]
        
        self.lstm = torch.nn.LSTM(input_size=self.D_conv_out, hidden_size=D_hidden, batch_first=True)
        
        self.D_head_out = D_hidden

    def forward(self, x, hidden_in=None):
        x = (x-(255.0/2))/(255.0/2)
        
        seq_len = x.size(1)
        
        if hidden_in is None:
            hidden_in = (torch.zeros(1, x.size(0), self.D_head_out).to(x.device),
                         torch.zeros(1, x.size(0), self.D_head_out).to(x.device))
                
        x = x.reshape(-1, *x.shape[2:])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, seq_len, self.D_conv_out)
        x, hidden_out = self.lstm(x, hidden_in)
        
        return x, hidden_in, hidden_out
    
class MLP_LSTM(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(MLP_LSTM, self).__init__()

        self.l = torch.nn.Linear(D_in, D_hidden)
        self.lstm = torch.nn.LSTM(input_size=D_hidden, hidden_size=D_hidden, batch_first=True)
        self.D_head_out = D_hidden

    def forward(self, x, hidden_in=None):
        if hidden_in is None:
            hidden_in = (torch.zeros(1, x.size(0), self.D_head_out).to(x.device),
                         torch.zeros(1, x.size(0), self.D_head_out).to(x.device))
         
        x = F.relu(self.l(x))
        x, hidden_out = self.lstm(x, hidden_in)
        
        return x, hidden_in, hidden_out
    
import sys, inspect, re

head_dict = {}
for class_name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    naming_rule = lambda x: re.sub('([a-z])([A-Z])', r'\1_\2', x).lower()
    head_dict[naming_rule(class_name)] = _class
