"""
Train Guinea on data.npz.

Expected arrays in data.npz
────────────────────────────
X                                          (N, 6)        standardized beam inputs
photon_count   / electron_count   / positron_count       (N,)    raw particle counts
photon_energy  / electron_energy  / positron_energy      (N, n_energy_bins)
photon_theta   / electron_theta   / positron_theta       (N, n_angle_bins)
photon_phi     / electron_phi     / positron_phi         (N, n_angle_bins)

All histogram arrays must be normalized: each row sums to 1.
Count arrays store raw counts; log1p is applied here at load time.

Usage
─────
    python train.py
    python train.py --data data.npz --epochs 300 --lr 3e-4
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import Guinea, loss_fn


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> TensorDataset:
    """
    Load data.npz and pack everything into a TensorDataset.
    Counts are log1p-transformed here so the model works in a compact
    numerical range regardless of how large the raw counts are.
    """
    d = np.load(path)

    def hist(key):
        # Histograms are already normalized in the file — load as-is.
        return torch.tensor(d[key], dtype=torch.float32)

    def count(key):
        # log1p(x) = log(1 + x) handles zero counts without blowing up to -inf.
        return torch.tensor(np.log1p(d[key]), dtype=torch.float32)

    return TensorDataset(
        torch.tensor(d["X"], dtype=torch.float32),   # inputs

        # Photon targets
        count("photon_count"),  hist("photon_energy"),
        hist("photon_theta"),   hist("photon_phi"),

        # Electron targets
        count("electron_count"),  hist("electron_energy"),
        hist("electron_theta"),   hist("electron_phi"),

        # Positron targets
        count("positron_count"),  hist("positron_energy"),
        hist("positron_theta"),   hist("positron_phi"),
    )


def unpack(batch: list, device: str):
    """
    Split a flat batch list into (X, photon_targets, electron_targets, positron_targets).
    Each target is a tuple of (log_count, energy_hist, theta_hist, phi_hist).
    """
    batch   = [b.to(device) for b in batch]
    X       = batch[0]
    pho_tgt = tuple(batch[1:5])
    ele_tgt = tuple(batch[5:9])
    pos_tgt = tuple(batch[9:13])
    return X, pho_tgt, ele_tgt, pos_tgt


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    totals = {"total": 0.0, "pho": 0.0, "ele": 0.0, "pos": 0.0}

    for batch in loader:
        X, pho_tgt, ele_tgt, pos_tgt = unpack(batch, device)

        optimizer.zero_grad()
        pho_pred, ele_pred, pos_pred = model(X)

        # loss_fn returns total + per-species so we can track each separately
        total, l_pho, l_ele, l_pos = loss_fn(
            pho_pred, ele_pred, pos_pred,
            pho_tgt,  ele_tgt,  pos_tgt,
        )

        total.backward()
        optimizer.step()

        n = len(X)
        totals["total"] += total.item() * n
        totals["pho"]   += l_pho.item() * n
        totals["ele"]   += l_ele.item() * n
        totals["pos"]   += l_pos.item() * n

    n_total = len(loader.dataset)
    return {k: v / n_total for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    totals = {"total": 0.0, "pho": 0.0, "ele": 0.0, "pos": 0.0}

    for batch in loader:
        X, pho_tgt, ele_tgt, pos_tgt = unpack(batch, device)
        pho_pred, ele_pred, pos_pred = model(X)

        total, l_pho, l_ele, l_pos = loss_fn(
            pho_pred, ele_pred, pos_pred,
            pho_tgt,  ele_tgt,  pos_tgt,
        )

        n = len(X)
        totals["total"] += total.item() * n
        totals["pho"]   += l_pho.item() * n
        totals["ele"]   += l_ele.item() * n
        totals["pos"]   += l_pos.item() * n

    n_total = len(loader.dataset)
    return {k: v / n_total for k, v in totals.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data.npz")
    ap.add_argument("--epochs", type=int,   default=100)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--batch",  type=int,   default=512)
    ap.add_argument("--val",    type=float, default=0.1,
                    help="fraction of data held out for validation")
    ap.add_argument("--out",    default="model.pt",
                    help="path to save the best checkpoint")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset & split ───────────────────────────────────────────────────────
    dataset = load_data(args.data)
    n_val   = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),   # reproducible split
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Infer bin counts from the data so the model automatically matches.
    sample       = dataset[0]
    n_energy_bins = sample[2].shape[0]   # photon_energy tensor
    n_angle_bins  = sample[3].shape[0]   # photon_theta tensor

    model = Guinea(
        n_in          = dataset[0][0].shape[0],
        n_energy_bins = n_energy_bins,
        n_angle_bins  = n_angle_bins,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}  "
          f"(energy bins={n_energy_bins}, angle bins={n_angle_bins})")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ReduceLROnPlateau halves the learning rate if val loss doesn't improve
    # for 5 epochs — lets us start with a large lr and fine-tune automatically.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5, min_lr=1e-6,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(model, train_loader, opt, device)
        val_losses   = eval_epoch(model,  val_loader,   device)
        sched.step(val_losses["total"])

        marker = ""
        if val_losses["total"] < best_val:
            best_val = val_losses["total"]
            torch.save(model.state_dict(), args.out)
            marker = "  ← saved"

        # Print every 10 epochs; always print epoch 1 to confirm training started.
        # Per-species breakdown lets you spot if one particle type is lagging.
        if epoch % 10 == 0 or epoch == 1:
            tl = train_losses
            vl = val_losses
            print(
                f"Epoch {epoch:4d}  "
                f"train={tl['total']:.4f} "
                f"[γ={tl['pho']:.3f} e⁻={tl['ele']:.3f} e⁺={tl['pos']:.3f}]  "
                f"val={vl['total']:.4f} "
                f"[γ={vl['pho']:.3f} e⁻={vl['ele']:.3f} e⁺={vl['pos']:.3f}]"
                f"{marker}"
            )

    print(f"\nDone.  Best val loss: {best_val:.4f}  →  {args.out}")


if __name__ == "__main__":
    main()
