"""
Step 5a: train the PINN on the seed dataset with the annealed combined loss.

Split is grouped by config_id (hold out entire configs) so validation
measures true generalization to unseen pendulums, mirroring the eval
protocol. Network weights are float32; the physics rollout inside the loss
casts to float64 internally.

Checkpoint selection / early stopping use the *combined* val loss evaluated
at the final (post-ramp) weights, fixed for the whole run -- not the live
annealed weights at the current epoch. Comparing epochs against a moving
yardstick means val_combined jumps upward the moment w_phys/w_bar/w_el
switch on, which looks identical to "no improvement" and starves early
stopping right as the physics-heavy phase begins. A fixed target metric
keeps "best so far" meaningful across the whole schedule.
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
          weight_overrides=None, seed=None, verbose=True,
          use_wandb=False, wandb_run_name=None, wandb_group=None,
          resume=False):
    """
    Train and checkpoint. weight_overrides (dict) can pin loss weights to a
    constant (e.g. {'w_phys':0,'w_bar':0,'w_el':0} for a data-only ablation)
    by monkeypatching config values -- returns the best-val model + history.

    use_wandb logs per-epoch val_data_mse, mean train loss components, and
    anneal weights to a Weights & Biases run (opt-in so smoke tests / CI
    don't need wandb installed or logged in).

    resume=True picks up a training run stopped mid-way (e.g. Ctrl+C):
    optimizer/scheduler state, epoch, rng stream and history are restored
    from a `<out_ckpt>.resume` sidecar written after every epoch, so the LR
    schedule and loss-anneal phase continue rather than restarting at
    epoch 0. Requires the same dataset_path/epochs/seed as the original
    call. No-op (returns immediately) if that run already finished.
    """
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(project="pinn-double-pendulum", name=wandb_run_name,
                          group=wandb_group, config={"epochs": epochs or C.EPOCHS,
                                                      "lr": C.LR, "seed": seed or C.SEED})
    epochs = C.EPOCHS if epochs is None else epochs
    seed = C.SEED if seed is None else seed
    out_ckpt = out_ckpt or os.path.join(C.CKPT_DIR, "round0_best.pt")
    resume_ckpt = out_ckpt + ".resume"
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

        do_resume = resume and os.path.exists(resume_ckpt)
        if do_resume:
            model, rckpt = load_checkpoint(resume_ckpt)
        elif init_ckpt and os.path.exists(init_ckpt):
            model, _ = load_checkpoint(init_ckpt)
        else:
            model = PINNPolicy(stats)
        model.to(C.DEVICE)

        opt = torch.optim.Adam(model.parameters(), lr=C.LR,
                               weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=C.LR_MIN)

        sel_epoch = epochs - 1  # fixed post-ramp weights for val selection, not the live epoch

        if do_resume:
            opt.load_state_dict(rckpt["opt_state"])
            sched.load_state_dict(rckpt["sched_state"])
            rng.bit_generator.state = rckpt["rng_state"]
            start_epoch = rckpt["epoch"] + 1
            best_val = rckpt["best_val"]
            best_epoch = rckpt["best_epoch"]
            history = list(rckpt["history"])
            if verbose:
                print(f"[resume] continuing from epoch {start_epoch} "
                      f"(best={best_val:.4f}@{best_epoch})")
        else:
            start_epoch = 0
            best_val = float("inf")
            best_epoch = -1
            history = []

        for epoch in range(start_epoch, epochs):
            model.train()
            comp_sums, n_batches = {}, 0
            for batch in train_loader:
                opt.zero_grad()
                total, comps = L.combined_loss(model, batch, epoch, rng, epochs=epochs)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step()
                n_batches += 1
                if run is not None:
                    for k, v in comps.items():
                        if k == "weights":
                            continue
                        comp_sums[k] = comp_sums.get(k, 0.0) + v
            sched.step()

            # fresh RNG per epoch: same collocation points every validation
            # pass, so val_combined noise comes only from the model, not from
            # which random L_EL points happened to get drawn this epoch
            val_rng = np.random.default_rng(seed + 1)
            val_data, val_combined = _val_metrics(model, val_loader, sel_epoch, epochs, val_rng)
            history.append(val_combined)
            if val_combined < best_val - 1e-6:
                best_val, best_epoch = val_combined, epoch
                save_checkpoint(out_ckpt, model,
                                extra={"epoch": epoch, "val_data_mse": val_data,
                                       "val_combined": val_combined})
            save_checkpoint(resume_ckpt, model, extra={
                "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch,
                "history": history, "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(), "rng_state": rng.bit_generator.state,
            })
            if run is not None:
                w = L.loss_weights(epoch, epochs)
                log = {"epoch": epoch, "val_data_mse": val_data, "val_combined": val_combined,
                       "best_val_combined": best_val, "lr": sched.get_last_lr()[0],
                       **{f"w_{k}": v for k, v in w.items()},
                       **{f"train_{k}": s / n_batches for k, s in comp_sums.items()}}
                run.log(log)
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                w = L.loss_weights(epoch, epochs)
                print(f"[epoch {epoch:3d}] val_data_mse={val_data:.4f} "
                      f"val_combined={val_combined:.4f} "
                      f"best={best_val:.4f}@{best_epoch}  "
                      f"w=({w['w_data']:.2f},{w['w_phys']:.2f},{w['w_bar']:.2f},{w['w_el']:.2f})")
            min_epoch = int(C.EARLY_STOP_MIN_EPOCH_FRAC * epochs)
            if epoch >= min_epoch and epoch - best_epoch > C.EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"[early stop] no val improvement for "
                          f"{C.EARLY_STOP_PATIENCE} epochs")
                break

        if verbose:
            print(f"[train] best val_combined={best_val:.4f} @epoch {best_epoch} "
                  f"-> {out_ckpt}")
        return out_ckpt, history
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
        if run is not None:
            run.finish()


@torch.no_grad()
def _val_metrics(model, loader, sel_epoch, epochs, rng):
    """
    Returns (val_data_mse, val_combined). val_combined is computed with the
    weights loss_weights(sel_epoch, epochs) resolves to -- callers pass a
    fixed sel_epoch (epochs - 1, i.e. the final post-ramp weights) so the
    metric is a constant yardstick across the whole run, not the live
    annealed weights at the current training epoch. Using the live weights
    would make val_combined jump the moment w_phys/w_bar/w_el ramp in,
    indistinguishable from the model getting worse.
    """
    model.eval()
    tot_data, tot_combined, n = 0.0, 0.0, 0
    for states, ml, u in loader:
        combined, comps = L.combined_loss(model, (states, ml, u), sel_epoch, rng, epochs=epochs)
        b = states.shape[0]
        tot_data += comps["data"] * b
        tot_combined += float(combined) * b
        n += b
    return tot_data / max(1, n), tot_combined / max(1, n)


if __name__ == "__main__":
    train()
