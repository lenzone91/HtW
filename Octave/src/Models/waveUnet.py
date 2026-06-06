from torch import nn
import torch


class Decimate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # keep even indices
        y = x[..., ::2]
        # if original length was even, last sample was dropped -> append it
        if x.size(-1) % 2 == 0:
            y = torch.cat([y, x[..., -1:]], dim=-1)
        return y
    

class Downsampling(nn.Module):
    def __init__(self, Fc, fd, i):
        super().__init__()
        if i == 0:
            self.conv = nn.Conv1d(in_channels=1, out_channels=Fc*(i+1), kernel_size=fd, stride=1)
        else:
            self.conv = nn.Conv1d(in_channels=Fc*i, out_channels=Fc*(i+1), kernel_size=fd, stride=1)
        self.activation = nn.LeakyReLU()
        self.decimate = Decimate()

    def forward(self, x):
        z = self.conv(x)
        z = self.activation(z)
        x = self.decimate(z)
        return x, z
    

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.cat([torch.stack([x[..., :-1], 0.5*(x[..., :-1] + x[..., 1:])], dim=-1).flatten(-2), x[..., -1:]], dim=-1)
        return x
    


class Upsampling(nn.Module):
    def __init__(self, Fc, fu, i):
        super().__init__()
        self.upsample = Upsample()

        if i == 0:
            self.conv = nn.Conv1d(in_channels=2*Fc, out_channels=1, kernel_size=fu, stride=1)

        else:
            self.conv = nn.Conv1d(in_channels=2*Fc*(i+1), out_channels=Fc*i, kernel_size=fu, stride=1)

        self.activation = nn.LeakyReLU()

    def forward(self, x, z):
        x = self.upsample(x)
        x = self.upsample(x)

        x = self.center_crop(x, target_len=z.size(-1))


        

        x = torch.cat([x, z], dim = 1)
        x = self.conv(x)
        x = self.activation(x)
        return x

    @staticmethod
    def center_crop(z, target_len):
        diff = z.size(-1) - target_len 
        left = diff // 2
        right =  target_len + left

        # print(f'left\t:\t{left}\nright\t:\t{right}')
        
        return z[..., left:right]
    

class WaveUNet(nn.Module):
    def __init__(self, Fc, fd, fu, L):
        super().__init__()
        self.downsamplings = nn.ModuleList([Downsampling(Fc, fd, i) for i in range(L)])
        self.upsamplings = nn.ModuleList([Upsampling(Fc, fu, L-(i+1)) for i in range(L)])
        self.conv_low = nn.Conv1d(in_channels=Fc*L, out_channels=Fc*L, kernel_size=fd, stride=1)
        # self.conv_high = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1, stride=1)
        self.conv_high = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        self.activation = nn.LeakyReLU()
        self.upsample = Upsample()
        self.activation_last = nn.Tanh()


    def forward(self, x):
        z = []
        y = x.clone()
        for downsampling in self.downsamplings:
            y, z_i = downsampling(y)
            z.append(z_i)

        y = self.conv_low(y)
        y = self.activation(y)
        

        


        for i, upsampling in enumerate(self.upsamplings):
            y = upsampling(y, z[-(i+1)])
        y = self.upsample(y)
        y = self.center_crop(y, x.size(-1))


        x = torch.cat([x, y], dim = 1)
        x = self.conv_high(x)
        x = self.activation_last(x)

        return x
        # return x.split(1, dim=1)
    
    @staticmethod
    def center_crop(z, target_len):
        diff = z.size(-1) - target_len 
        left = diff // 2
        right =  target_len + left

        
        return z[..., left:right]