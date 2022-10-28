from .model import Model
import numpy as np
import torch

class RydbergModel(Model):

    def __init__(self, n, start_type = 'random'):
        super(RydbergModel, self).__init__()
        self.start_type = start_type
        self.n = n
        self.R = self.init_X()
        self.detuning = torch.unsqueeze(-1 * torch.autograd.Variable(torch.Tensor(np.random.normal(loc=0.0, scale=10, size=[self.n]))), dim=0)
        self.C_6 = 1

    def init_X(self):

        if self.start_type == 'random':
            R = torch.autograd.Variable(torch.Tensor(np.random.normal(loc=50, scale=0.25, size=[self.n, 2])))

        return R

    def forward(self, x):

        X = torch.matmul(torch.transpose(x, dim0=1, dim1=2), x)

        d = self.detuning * x
        d_sum = torch.squeeze(torch.sum(d, dim=2))
        
        diag = torch.diagflat(self.R)
    
        diag_x = diag[:self.n, :self.n]
        diag_y = diag[self.n:, self.n:]

        R_x = torch.matmul(x, diag_x)
        R_y = torch.matmul(x, diag_y)

        R_r = torch.cat((R_x, R_y), dim=1)
        R_r = torch.transpose(R_r, dim0=1, dim1 = 2)

        R_r1 = torch.unsqueeze(R_r, dim=1)
        R_r2 = torch.unsqueeze(R_r, dim=2)

        R_r = R_r1 - R_r2

        R_r = torch.square(R_r)

        R_r = torch.sqrt(0.5 * torch.sum(R_r, dim = 3))
        R_r = X * 0.5 * R_r
        V1 = torch.nan_to_num(torch.reciprocal(R_r), posinf=0)

        V2 = torch.pow(V1, 6) * self.C_6

        return torch.sum(V2, dim=[1, 2]) + d_sum

        

        



if __name__ == "__main__":

    r_model = RydbergModel(2)
    x = torch.Tensor([[0, 1], [1, 1], [0, 0], [1, 0]])
    r_model(x)