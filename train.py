"""
Train SpectrumNet on data.npz.

Usage:
    python train.py                        # uses data.npz, trains 100 epochs
    python train.py --data data.npz --epochs 200 --lr 1e-3
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import SpectrumNet, loss_fn


def load_data(path):
    d = np.load(path)
    X           = torch.tensor(d["X"],            dtype=torch.float32)
    target_shape = torch.tensor(d["target_shape"], dtype=torch.float32)
    target_logn  = torch.tensor(d["target_logn"],  dtype=torch.float32)
    return X, target_shape, target_logn


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for X, t_shape, t_logn in loader:
        X, t_shape, t_logn = X.to(device), t_shape.to(device), t_logn.to(device)
        optimizer.zero_grad()
        log_shape, log_ngamma = model(X)
        loss = loss_fn(log_shape, log_ngamma, t_shape, t_logn)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for X, t_shape, t_logn in loader:
        X, t_shape, t_logn = X.to(device), t_shape.to(device), t_logn.to(device)
        log_shape, log_ngamma = model(X)
        loss = loss_fn(log_shape, log_ngamma, t_shape, t_logn)
        total += loss.item() * len(X)
    return total / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="data.npz")
    ap.add_argument("--epochs",  type=int,   default=100)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--batch",   type=int,   default=512)
    ap.add_argument("--val",     type=float, default=0.1,  help="fraction held out for validation")
    ap.add_argument("--out",     default="model.pt",       help="where to save the best checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    X, target_shape, target_logn = load_data(args.data)
    dataset = TensorDataset(X, target_shape, target_logn)

    n_val   = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    n_bins = target_shape.shape[1]
    model  = SpectrumNet(n_in=X.shape[1], n_bins=n_bins).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device)
        val_loss   = eval_epoch(model,  val_loader,   device)
        sched.step(val_loss)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out)
            marker = "  ← saved"

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}  train={train_loss:.4f}  val={val_loss:.4f}{marker}")

    print(f"\nDone. Best val loss: {best_val:.4f}  →  {args.out}")


if __name__ == "__main__":
    main()
