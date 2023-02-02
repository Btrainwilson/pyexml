import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from  torch.distributions import multivariate_normal
import math
from .model import Model
from ..geometry.kernels import MercerSPD


class VAE(Model):
    """
        Computes the loss function for a VAE
        x: Ideal data tensor of shape (batch_size, data_shape)
        x_hat: Generated tensor of shape (batch_size, data_shape)
    """
    def __init__(self, in_shape, latent_dim, prior, recon_fn):
        super(VAE, self).__init__()

        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.__buildencoder__()
        self.__builddecoder__()

        self.prior = prior
        self.recon_fn = recon_fn
        

    def loss(self, x, x_hat):
        """
        Computes the loss function for a VAE
        x: Ideal data tensor of shape (batch_size, data_shape)
        x_hat: Generated tensor of shape (batch_size, data_shape)
        """

        #Compute entropy of sampled points zeta
        #H(q) = -E_q[log q]
        entropy = self.entropy(self.zeta)

        #Compute cross entropy of sampled points compared to prior
        #H(q, p) = -E_q[log p]
        cross_entropy = self.cross_entropy(self.zeta)

        #Compute KL divergence between q and p
        #KL(q||p) = E_q[log q] - E_q[log p]
        D_kl = self.cross_entropy(self.zeta) - self.entropy(self.zeta)
        
        #Reconstruction loss
        #-E_q[log q]
        reconstruction = self.recon_fn(x_hat, x)

        total_loss = D_kl + reconstruction

        loss_dict = {}
        loss_dict['Reconstruction'] = reconstruction
        loss_dict['KL Divergence'] = D_kl
        loss_dict['Total'] = total_loss
        loss_dict['Entropy'] = entropy
        loss_dict['Cross Entropy'] = cross_entropy

        return loss_dict

    def __buildencoder__(self):
        pass

    def __builddecoder__(self):
        pass

    def reparameterize(self):
        pass

    def entropy(self, sample):
        pass

    def cross_entropy(self, sample):
        return -self.prior.log_prob(sample).sum(dim=-1)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        self.z = self.encode(x)
        self.z_p = self.reparameterize(self.z)
        return self.decode(self.z_p)

class KingVAE(VAE):

    def __init__(self, in_shape, latent_dim, graph_adj):

        self.graph_adj = graph_adj
        self.beta = 1

        super().__init__(in_shape, latent_dim)
        
    def __buildencoder__(self):

        #Computes probability of z0 given x, i.e., q(z_i | x)
        self.encoder = nn.Sequential(
            nn.Linear(int(np.prod(self.in_shape)), int(self.Hilbert_dim * 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.Hilbert_dim * 2), self.Hilbert_dim),
            torch.nn.Softmax(dim=1)
        )

        self.rnn = nn.Sequential(
            nn.Linear(2*self.latent_dim, 2*self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2*self.latent_dim, 2*self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2*self.latent_dim, 2*self.latent_dim),
            nn.Sigmoid()
        )
        
    def __builddecoder__(self):

        #Computes probability of x given zeta, i.e., q(x | zeta)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.Hilbert_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.Hilbert_dim, int(self.Hilbert_dim * 2)),
            nn.Linear(int(self.Hilbert_dim * 2), int(np.prod(self.in_shape))),
            nn.Sigmoid()
        )

    #Random hierarchy VAE
    def reparameterize(self):
        #Create random node list
        node_list = torch.randperm(self.latent_dim)

        #Initialize hidden state
        h_0 = torch.zeros_like(self.z)

        #Add running loss to ensure that hidden states behave like sequential probability distributions
        loss = 0

        #Presample uniform distribution
        rho = Variable(torch.tensor(np.random.uniform(0, 1, (self.z.size(0), self.latent_dim)),  dtype=self.dtype, device = self.device))

        for i in range(self.latent_dim):

            #Sample from current distribution
            zeta = torch.log(torch.div(F.relu(rho[:, i] + self.z[:, i] - 1), self.z[:, i]) * (math.exp(self.beta) - 1) + 1 ) / self.beta
            z = torch.sign(zeta)

            #Update z
            #torch.
            torch.sign(self.z[:, i], )

            #Update h

            #input_for_rnn = torch.cat((node_features, connected_nodes_hidden_states), dim=1)
            
                

        rho = Variable(torch.tensor(np.random.uniform(0, 1, (self.q_zeta_x.size(0), self.latent_dim)),  dtype=self.dtype, device = self.device))
        self.zeta = torch.log(torch.div(F.relu(rho + self.q_zeta_x - 1), self.q_zeta_x) * (math.exp(self.beta) - 1) + 1 ) / self.beta
        
        return self.zeta
    
class MultivariateNormalVAE(VAE):

    def __buildencoder__(self):

        #Computes probability of z0 given x, i.e., q(z_i | x)
        self.encoder = nn.Sequential(
            nn.Linear( int( np.prod( self.in_shape )), int(np.prod( self.in_shape ) / 2 )),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear( int( np.prod( self.in_shape ) / 2 ), 2 * self.latent_dim)
        )

    def __builddecoder__(self):

        #Computes probability of x given zeta, i.e., q(x | zeta)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim, int(self.latent_dim * 2)),
            nn.Linear(int(self.latent_dim * 2), int(np.prod(self.in_shape))),
            nn.Sigmoid()
        )

    def reparameterize(self, z):

        #Split output into mu and sig_z
        self.mu = z[:, self.latent_dim:]
        self.sig_z = z[:, :self.latent_dim]

        #Compute Covariance Matrix
        self.Sigma = MercerSPD(self.sig_z)

        #Cholesky Decomposition
        L = torch.linalg.cholesky(self.Sigma)

        #Generate independent random samples
        eps = Variable(torch.randn_like(self.mu))

        #Reparameterization equation
        self.zeta = torch.matmul(L, eps.unsqueeze(2)).squeeze() + self.mu

        return self.zeta

    def entropy(self, sample):

        dist = multivariate_normal.MultivariateNormal(loc=self.z[:, :self.latent_dim], covariance_matrix=self.Sigma)

        entropy = -dist.log_prob(sample)
        entropy_sum = entropy.sum(dim=-1)
        return entropy_sum

class BinaryMultivariateNormalVAE(MultivariateNormalVAE):

    def reparameterize(self, z):

        self.zeta = torch.sign(super().reparameterize(z))

        return self.zeta


class GaussianVAE(VAE):

    def __buildencoder__(self):

        #Computes probability of z0 given x, i.e., q(z_i | x)
        self.encoder = nn.Sequential(
            nn.Linear( int( np.prod( self.in_shape )), int(np.prod( self.in_shape ) / 2 )),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear( int( np.prod( self.in_shape ) / 2 ), 2 * self.latent_dim)
        )

    def __builddecoder__(self):

        #Computes probability of x given zeta, i.e., q(x | zeta)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim, int(self.latent_dim * 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.latent_dim * 2), int(np.prod(self.in_shape))),
            nn.Sigmoid()
        )

    def reparameterize(self, z):

        #Split output into mu and log_var
        self.mu = z[:, self.latent_dim:]
        self.log_var = z[:, :self.latent_dim]

        #Compute standard deviation
        self.std = torch.exp(0.5 * self.log_var)

        #Generate independent random samples
        eps = Variable(torch.randn_like(self.mu))

        #Reparameterization equation
        self.zeta = self.mu + eps * self.std

        return self.zeta

    def entropy(self, sample):
        #Compute entropy of Gaussian VAE
        #H(z) = 0.5 * log(2 * pi * e) + 0.5 * log(sigma^2) + 0.5 * (mu^2 + sigma^2)
        entr = 0.5 * (math.log(2*math.pi) + 1 + torch.sum(self.log_var))
        return entr

    def cross_entropy(self, sample):
        xentr = 0.5 * torch.sum((math.log(2*math.pi) + torch.pow(self.mu, 2) + torch.exp(self.log_var)))
        return xentr