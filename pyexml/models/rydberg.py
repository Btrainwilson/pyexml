from .model import Model
import numpy as np
import torch

class RydbergModel(Model):

    def __init__(self, n, start_type = 'random'):
        super(RydbergModel, self).__init__()
        self.start_type = start_type
        self.n = n
        self.R = self.init_X()

    def init_X(self):

        if self.start_type == 'random':
            R = torch.autograd.Variable(torch.Tensor(np.random.normal(loc=0.5, scale=0.25, size=[self.n, 2])))

        return R

    def forward(self, x):
        diag = torch.diagflat(self.R)
    
        diag_x = torch.unsqueeze(diag[:self.n, :self.n], dim=0)
        diag_y = torch.unsqueeze(diag[self.n:, self.n:], dim=0)

        diag_new = torch.unsqueeze(torch.cat((diag_x, diag_y), 0), dim=0)

        R_p = torch.bmm(x, diag_new)

        print(R_p)



if __name__ == "__main__":

    r_model = RydbergModel(2)
    x = torch.Tensor([[0, 1], [1, 1], [0, 0], [1, 0]])
    r_model(x)