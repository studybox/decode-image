import torch
import torch.nn as nn
import numpy as np
from torch import optim
import time 
from tqdm import tqdm
import math 
import copy

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, invert=False):
        """Coupling layer inside a normalizing flow.

        Args:
            network: A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask: Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in: Number of input channels
        """
        super().__init__()
        self.input_dim = input_dim
        # Define the scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim // 2)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim // 2)
        )
        self.invert = invert
        self.scaling_factor = nn.Parameter(torch.zeros(input_dim // 2))
        
    
    def forward(self, z, ldj, reverse=False):
        """Forward.

        Args:
            z: Latent input to the flow
            ldj:
                The current ldj of the previous flows. The ldj of this layer will be added to this tensor.
            reverse: If True, we apply the inverse of the layer.
            orig_img:
                Only needed in VarDeq. Allows external input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        if self.invert:
            z1, z2 = z[:, :self.input_dim // 2], z[:, self.input_dim // 2:]
        else:
            z2, z1 = z[:, :self.input_dim // 2], z[:, self.input_dim // 2:]
        
        s = self.scale_net(z1)
        t = self.translate_net(z1)
        
        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1)
        s = torch.tanh(s / s_fac) * s_fac


        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z2 = (z2 + t) * torch.exp(s)
            ldj += s.sum(dim=1)
        else:
            z2 = (z2 * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1, 2, 3])
        z = torch.cat([z1, z2], dim=1) if self.invert else torch.cat([z2, z1], dim=1)
        return z, ldj

class NormalizingFlow(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        # epoch stats
        self.start_epoch = 0
        self.start_time = None
        self.epoch_loss = []

    def forward(self, x):
        return self._get_likelihood(x)
    
    def encode(self, x):
        z, ldj = x, torch.zeros(x.size(0), device=x.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, x, return_ll=False):
        z, ldj = self.encode(x)
        log_pz = self.prior.log_prob(z).sum(dim=-1)
        log_px = log_pz + ldj
        nll = -log_px
        bpd = nll * np.log2(np.e) / np.prod(x.shape[1:])
        return bpd.mean() if not return_ll else nll
    
    def predict_ood_score(self, x):
        # with torch.no_grad():
        return self._get_likelihood(x, return_ll=True)

    def get_distance(self, x):
        # Encode x to get its latent representation z
        z, _ = self.encode(x)
        # Since the mean is 0, the difference is just z itself
        # And assuming an isotropic Gaussian, the covariance matrix is I, simplifying the computation
        # Mahalanobis distance in this context reduces to the Euclidean distance
        distance = torch.sqrt(torch.sum(z**2, dim=1))
        return distance
            
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.configs.train.lr)
    #     # An scheduler is optional, but can help in flows to get the last bpd improvement
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, self.configs.train.lr_scheduler.step_size, gamma=self.configs.train.lr_scheduler.gamma)
    #     return [optimizer], [scheduler]

    # def on_train_start(self) -> None:
    #     print('on_train_start')
    #     self.start_epoch = self.current_epoch
    #     self.start_time = time.time()

    # def training_step(self, batch, batch_idx):
    #     # Normalizing flows are trained by maximum likelihood => return bpd
    #     loss = self._get_likelihood(batch)
    #     self.log("train_bpd", loss, prog_bar=True, batch_size=self.configs.train.batch_size)
    #     return loss
    
    # def validation_step(self, batch, batch_idx):
    #     loss = self._get_likelihood(batch)
    #     self.log("val_bpd", loss, prog_bar=True, batch_size=self.configs.train.batch_size)
    #     self.epoch_loss.append(loss.item())

    # def on_validation_epoch_end(self):
    #     avg_loss = np.mean(self.epoch_loss)
    #     self.epoch_loss.clear()

    #     if self.start_time is not None:
    #         time_spent = time.time() - self.start_time
    #         epoch_left = self.configs.train.epochs - self.current_epoch
    #         time_ETA = int(time_spent / (self.current_epoch - self.start_epoch + 1) * epoch_left)
    #         hours, rem_seconds = divmod(time_ETA, 3600)
    #         minutes = rem_seconds // 60
    #         tqdm.write("Training ETA: {:02}h {:02}m | Val @ epoch {}: Loss {:.2f}".format(hours, minutes, self.current_epoch, avg_loss))

def build_flow(dim=768, depth=5, num_domains=10):
    flow_layers = []
    for i in range(depth):
        flow_layers.append(
            CouplingLayer(
                input_dim = dim,
                invert = (i % 2 == 1)
            )
        )
    flow_model = NormalizingFlow(flow_layers)
    flow_model_list = []
    for _ in range(num_domains):
        flow_model_list.append(copy.deepcopy(flow_model))
    flow_model_list = nn.ModuleList(flow_model_list)
    return flow_model_list