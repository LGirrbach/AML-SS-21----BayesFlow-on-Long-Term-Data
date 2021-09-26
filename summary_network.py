import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(MultiConvLayer, self).__init__()
        
        self.convs = [
            nn.Conv1d(in_channels, out_channels // 2, filter_size)
            for filter_size in range(2, 8)
        ]
        self.filter_sizes = list(range(2, 8))
        self.convs = nn.ModuleList(self.convs)
        self.dim_red = nn.Conv1d(len(self.filter_sizes) * (out_channels // 2), out_channels, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        out = [conv(F.pad(x, (f-1, 0))) for conv, f in zip(self.convs, self.filter_sizes)]
        out = torch.cat(out, dim=-2)
        out = F.relu(out)
        out = self.dim_red(out)
        out = F.relu(out)
        out = out.transpose(1, 2)
        
        return out


class SummaryNet3(nn.Module):
    """Dead, infected, + recovered"""
    def __init__(self, in_channels, n_summary):
        super(SummaryNet3, self).__init__()

        assert n_summary % 3 == 0
        assert in_channels % 3 == 0
        self.net_I = MultiConvNet(in_channels // 3, out_channels=n_summary // 3)
        self.net_D = MultiConvNet(in_channels // 3, out_channels=n_summary // 3)
        self.net_R = MultiConvNet(in_channels // 3, out_channels=n_summary // 3)

    def forward(self, x):
        split1 = x.shape[-1] // 3
        split2 = 2 * split1
        x_i = x[..., :split1].contiguous()
        x_r = x[..., split1:split2].contiguous()
        x_d = x[..., split2:].contiguous()

        x_i = self.net_I(x_i)
        x_r = self.net_R(x_r)
        x_d = self.net_D(x_d)

        return torch.cat([x_i, x_r, x_d], dim=-1)


class MultiConvNet(nn.Module):
    def __init__(self, in_channels, out_channels=64, n_layers=3):
        super(MultiConvNet, self).__init__()
        
        self.net = nn.Sequential(
            *[MultiConvLayer(in_channels if layer == 0 else out_channels, out_channels)
              for layer in range(n_layers)
             ]
        )
        self.lstm = nn.LSTM(out_channels, out_channels, batch_first=True)
    
    def forward(self, x):
        out = self.net(x)
        h, _ = self.lstm(out)
        return h[:, -1, :].contiguous()
    
    
class SummaryNet(nn.Module):
    def __init__(self, in_channels, n_summary, num_compartments):
        super(SummaryNet, self).__init__()
        self.num_compartments = num_compartments
        
        assert n_summary % num_compartments == 0
        assert in_channels % num_compartments == 0
        
        multiconvnets = [
            MultiConvNet(in_channels // num_compartments,
                         n_summary // num_compartments)
            for _ in range(self.num_compartments)
            ]
        self.conv_nets = nn.ModuleList(multiconvnets)
        
    
    def forward(self, x):
        compartment_size = x.shape[-1] // self.num_compartments
        compartments = torch.split(x, compartment_size, dim=-1)
        compartments = [conv_net(compartment) for compartment, conv_net
                        in zip(compartments, self.conv_nets)]
        return torch.cat(compartments, dim=-1)
