"""
Guinea: fast beam-beam collision emulator.

Three completely separate networks — one per particle species.
Photons (beamstrahlung radiation) and leptons (beam disruption + pair production)
are governed by different physics, so a shared trunk would force a single internal
representation to serve both processes and the tasks would interfere with each other.

Inputs (6, standardized before passing in):
    [log N,  log σx,  log σy,  log σz,  E0,  log Υ]

Outputs per species (photon / electron / positron):
    log_count   scalar              log(1 + particle count)
    log_energy  (n_energy_bins,)    log-prob histogram over particle energy
    log_theta   (n_angle_bins,)     log-prob histogram over polar angle θ
    log_phi     (n_angle_bins,)     log-prob histogram over azimuthal angle φ

Energy and angle use different bin counts because energy spans orders of magnitude
(needs finer resolution) while angles are bounded and more concentrated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Trunk builder ─────────────────────────────────────────────────────────────

def _make_trunk(n_in: int, width: int, depth: int) -> nn.Sequential:
    """
    MLP with `depth` hidden layers, LayerNorm + SiLU after each one.
    LayerNorm keeps activations well-scaled across layers, which matters
    especially when width is large (512) and depth > 3.
    """
    layers: list[nn.Module] = [nn.Linear(n_in, width), nn.LayerNorm(width), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU()]
    return nn.Sequential(*layers)


# ── Per-species network ───────────────────────────────────────────────────────

class ParticleNet(nn.Module):
    """
    Predicts the full output distribution for one particle species.

    Each output quantity has its own head so that gradients from
    (e.g.) the energy loss don't directly interfere with the angle heads —
    they only share the trunk representation.
    """

    def __init__(self, n_in: int, n_energy_bins: int, n_angle_bins: int,
                 width: int, depth: int):
        super().__init__()

        self.trunk = _make_trunk(n_in, width, depth)

        # Four output heads — each a single linear layer reading from the trunk.
        # The trunk does all the heavy lifting; the heads are just projections.
        self.count_head  = nn.Linear(width, 1)              # scalar log-count
        self.energy_head = nn.Linear(width, n_energy_bins)  # energy histogram
        self.theta_head  = nn.Linear(width, n_angle_bins)   # polar angle histogram
        self.phi_head    = nn.Linear(width, n_angle_bins)   # azimuthal angle histogram

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)

        # log_softmax ensures the predicted histogram is a valid probability
        # distribution (non-negative, sums to 1 in probability space) and
        # produces log-probabilities, which is what F.kl_div expects as input.
        log_energy = F.log_softmax(self.energy_head(h), dim=-1)
        log_theta  = F.log_softmax(self.theta_head(h),  dim=-1)
        log_phi    = F.log_softmax(self.phi_head(h),    dim=-1)

        # Count is predicted in log1p-space so the model never has to deal with
        # the enormous dynamic range of raw particle counts across beam configs.
        # squeeze(-1) removes the trailing size-1 dimension so shape is (batch,).
        log_count = self.count_head(h).squeeze(-1)

        return log_count, log_energy, log_theta, log_phi


# ── Top-level model ───────────────────────────────────────────────────────────

class Guinea(nn.Module):
    """
    Three separate ParticleNets — one each for photons, electrons, positrons.

    Default hyperparameters:
        n_energy_bins = 96   fine enough for the energy dynamic range
        n_angle_bins  = 64   angles are bounded, need fewer bins
        width         = 512  wider than the original 256 to match the larger
                             output space and the assumption of more training data
        depth         = 4    one extra hidden layer vs. original for more capacity
    """

    def __init__(
        self,
        n_in          : int = 6,
        n_energy_bins : int = 96,
        n_angle_bins  : int = 64,
        width         : int = 512,
        depth         : int = 4,
    ):
        super().__init__()

        # Separate networks — no shared parameters between species.
        # This means photon gradients never corrupt the lepton trunk and vice versa.
        self.photon   = ParticleNet(n_in, n_energy_bins, n_angle_bins, width, depth)
        self.electron = ParticleNet(n_in, n_energy_bins, n_angle_bins, width, depth)
        self.positron = ParticleNet(n_in, n_energy_bins, n_angle_bins, width, depth)

    def forward(self, x: torch.Tensor):
        # Each sub-network runs independently; PyTorch will parallelize on GPU.
        return self.photon(x), self.electron(x), self.positron(x)


# ── Loss ──────────────────────────────────────────────────────────────────────

def _particle_loss(
    pred   : tuple[torch.Tensor, ...],
    target : tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Loss for one particle species:
        3 × KL divergence  — energy, θ, φ histogram shape
        1 × MSE            — log particle count

    KL divergence is the natural choice for comparing two probability
    distributions. 'batchmean' divides by batch size so the loss scale
    is independent of batch size and comparable across training runs.

    KL expects (log-probs, probs) — pred provides log_softmax outputs,
    target histograms must be normalized (sum to 1) before being stored
    in the data file.
    """
    log_count, log_e, log_t, log_p  = pred
    t_count,   t_e,   t_t,   t_p   = target

    kl_energy = F.kl_div(log_e, t_e, reduction='batchmean')
    kl_theta  = F.kl_div(log_t, t_t, reduction='batchmean')
    kl_phi    = F.kl_div(log_p, t_p, reduction='batchmean')
    mse_count = F.mse_loss(log_count, t_count)

    return kl_energy + kl_theta + kl_phi + mse_count


def loss_fn(
    photon_pred   : tuple, electron_pred   : tuple, positron_pred   : tuple,
    photon_tgt    : tuple, electron_tgt    : tuple, positron_tgt    : tuple,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, photon_loss, electron_loss, positron_loss).

    Returning individual species losses lets the training loop log them
    separately, so you can see if one particle type is converging slower
    than the others — which is common since photon physics is simpler than
    lepton disruption physics.
    """
    l_pho = _particle_loss(photon_pred,   photon_tgt)
    l_ele = _particle_loss(electron_pred, electron_tgt)
    l_pos = _particle_loss(positron_pred, positron_tgt)

    return l_pho + l_ele + l_pos, l_pho, l_ele, l_pos
