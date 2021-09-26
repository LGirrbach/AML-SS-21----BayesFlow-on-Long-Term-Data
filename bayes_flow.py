import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import numpy as np


class BayesFlow(nn.Module):
    """Implements a chain of conditional invertible blocks for Bayesian parameter inference."""
    
    def __init__(self, hyperparams, theta_dim, x_dim, summary_net=None, device='cpu'):
        super(BayesFlow, self).__init__()
        
        self.cINN = self.make_cinn(hyperparams, theta_dim, x_dim)
        self.summary_net = summary_net
        self.theta_dim = theta_dim
        self.device = device
        
    
    @staticmethod
    def make_cinn(hyperparams, theta_dim, x_dim):
        def subnet_fc(c_in, c_out):
            modules = []
            for layer in range(hyperparams['n_layers'] + 1):
                in_size = c_in if layer == 0 else hyperparams['n_units']
                out_size = c_out if layer == hyperparams['n_layers'] else hyperparams['n_units']
                modules.append(nn.Linear(in_size, out_size))
                modules.append(hyperparams['activation']())
        
            return nn.Sequential(*modules)
    
        cinn = Ff.SequenceINN(theta_dim)
        for _ in range(hyperparams['n_blocks']):
            cinn.append(Fm.AllInOneBlock, cond=0, cond_shape=(x_dim,), subnet_constructor=subnet_fc)
        return cinn
        
    def forward(self, theta, x, inverse=False):
        if self.summary_net is not None:
            x = self.summary_net(x)
        return self.cINN(theta, [x], rev=inverse)
    
    def sample(self, x, n_samples, to_numpy):
        with torch.no_grad():
            if x.shape[0] == 1:
                z_normal_samples = torch.randn(n_samples, self.theta_dim)
                theta_samples, _ = self.forward(
                    z_normal_samples.to(self.device),
                    x.tile((n_samples, 1, 1)).to(self.device),
                    inverse=True
                )
            else:
                z_normal_samples = torch.randn(n_samples, x.shape[0], self.theta_dim)
                theta_samples, _ = self.forward(
                    z_normal_samples.to(self.device),
                    x.tile((n_samples, 1, 1)).to(self.device),
                    inverse=True
                )
                #theta_samples = theta_samples.reshape(n_samples, x.shape[0], self.theta_dim)
                #theta_samples = theta_samples.transpose(0, 1)
            
            if to_numpy:
                return theta_samples.cpu().numpy()
            else:
                return theta_samples
