"""
Step 5a: train the PINN on the seed dataset with the annealed combined loss.

Split is grouped by config_id (hold out entire configs) so validation
measures true generalization to unseen pendulums, mirroring the eval
protocol. Network weights are float32; the physics rollout inside the loss
casts to float64 internally.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pinn import config as C
from pinn import dataset as ds
from pinn import losses as L
from pinn.model import PINNPolicy, save_checkpoint, load_checkpoint


def _grouped_split(config_ids, val_frac, rng):
    """Hold out a fraction of *configs* (not rows) for validation."""
    uniq = np.unique(config_ids)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_configs = set(uniq[:n_val].tolist())
    val_mask = np.array([cid in val_configs for cid in config_ids])
    return ~val_mask, val_mask


def _make_loader(data, mask, batch_size, shuffle):
    states = torch.tensor(data["states"][mask], dtype=torch.float64)
    ml = torch.tensor(data["mlparams"][mask], dtype=torch.float64)
    u = torch.tensor(data["u"][mask], dtype=torch.float64)
    dsx = TensorDataset(states, ml, u)
    return DataLoader(dsx, batch_size=batch_size, shuffle=shuffle)


def train(dataset_path=None, epochs=None, out_ckpt=None, init_ckpt=None,
          weight_overrides=None, seed=None, verbose=True):
    """
    Train and checkpoint. weight_overrides (dict) can pin loss weights to a
    constant (e.g. {'w_phys':0,'w_bar':0,'w_el':0} for a data-only ablation)
    by monkeypatching config values -- returns the best-val model + history.
    """
    epochs = C.EPOCHS if epochs is None else epochs
    seed = C.SEED if seed is None else seed
    out_ckpt = out_ckpt or os.path.join(C.CKPT_DIR, "round0_best.pt")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # optional ablation: pin certain target weights to zero for the whole run
    saved = {}
    if weight_overrides:
        for k, v in weight_overrides.items():
            saved[k] = getattr(C, k)
            setattr(C, k, v)

    try:
        data = ds.load_dataset(dataset_path)
        stats = ds.load_norm_stats()

        tr_mask, va_mask = _grouped_split(data["config_id"], C.VAL_CONFIG_FRAC, rng)
        train_loader = _make_loader(data, tr_mask, C.BATCH_SIZE, shuffle=True)
        val_loader = _make_loader(data, va_mask, C.BATCH_SIZE, shuffle=False)

        if init_ckpt and os.path.exists(init_ckpt):
            model, _ = load_checkpoint(init_ckpt)
        else:
            model = PINNPolicy(stats)
        model.to(C.DEVICE)

        opt = torch.optim.Adam(model.parameters(), lr=C.LR,
                               weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=C.LR_MIN)

        best_val = float("inf")
        best_epoch = -1
        history = []
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                total, _ = L.combined_loss(model, batch, epoch, rng, epochs=epochs)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step()
            sched.step()

            val = _val_data_loss(model, val_loader)
            history.append(val)
            if val < best_val - 1e-6:
                best_val, best_epoch = val, epoch
                save_checkpoint(out_ckpt, model,
                                extra={"epoch": epoch, "val_data_mse": val})
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                w = L.loss_weights(epoch, epochs)
                print(f"[epoch {epoch:3d}] val_data_mse={val:.4f} "
                      f"best={best_val:.4f}@{best_epoch}  "
                      f"w=({w['w_data']:.2f},{w['w_phys']:.2f},{w['w_bar']:.2f},{w['w_el']:.2f})")
            min_epoch = int(C.EARLY_STOP_MIN_EPOCH_FRAC * epochs)
            if epoch >= min_epoch and epoch - best_epoch > C.EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"[early stop] no val improvement for "
                          f"{C.EARLY_STOP_PATIENCE} epochs")
                break

        if verbose:
            print(f"[train] best val_data_mse={best_val:.4f} @epoch {best_epoch} "
                  f"-> {out_ckpt}")
        return out_ckpt, history
    finally:
        for k, v in saved.items():
            setattr(C, k, v)


@torch.no_grad()
def _val_data_loss(model, loader):
    """Validation metric = pure imitation (data) MSE in voltage."""
    model.eval()
    tot, n = 0.0, 0
    for states, ml, u in loader:
        v = L.loss_data(model, states, ml, u)
        b = states.shape[0]
        tot += float(v) * b
        n += b
    return tot / max(1, n)


if __name__ == "__main__":
    train()
