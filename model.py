import torch, torch.nn as nn, torch.nn.functional as F

class SpectrumNet(nn.Module):
    def __init__(self, n_in=6, n_bins=96, width=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.shape_head = nn.Linear(width, n_bins)   # logits over log-E bins
        self.norm_head  = nn.Linear(width, 1)         # predicts log(n_gamma)

    def forward(self, x):
        h = self.trunk(x)
        log_shape = F.log_softmax(self.shape_head(h), dim=-1)  # normalized spectrum
        log_ngamma = self.norm_head(h).squeeze(-1)
        return log_shape, log_ngamma

# inputs x: standardized [log N, log σx, log σy, log σz, E0, log Υ]
# target_shape: normalized histogram (sums to 1); target_logn: log of total photon count

def loss_fn(log_shape, log_ngamma, target_shape, target_logn):
    kl = F.kl_div(log_shape, target_shape, reduction='batchmean')  # shape
    mse = F.mse_loss(log_ngamma, target_logn)                       # normalization
    return kl + mse
