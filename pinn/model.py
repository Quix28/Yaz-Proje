"""
Step 3: the PINN policy network.

Inputs : 6 states + 4 params = 10, z-score standardized.
Hidden : 3 dense layers 128/128/64, tanh.
Output : a single VOLTAGE, tanh-scaled to [-V_MAX, V_MAX] (V_MAX = 24).

OUTPUT UNITS -- VOLTAGE, not force (user decision). tanh hard-bounds the
command to the actuator's +-24 V bus, a structural safety guarantee no soft
penalty can give. Convert to force via actuator.voltage_to_force wherever
the dynamics are needed (physics loss, closed-loop rollout, deployment).
The voltage->force map is provisional and refined at hardware Step 7; the
trained weights are unaffected by that refinement.

Normalization statistics are stored as buffers, so a checkpoint is
self-contained (no external stats file needed at inference) and moves with
.to(device).
"""
import torch
import torch.nn as nn

from pinn import config as C

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU, "swish": nn.SiLU}


class PINNPolicy(nn.Module):
    def __init__(self, norm_stats, hidden=None, activation=None, v_max=None):
        """
        norm_stats: dict with x_mean,x_std (6,), p_mean,p_std (4,).
        """
        super().__init__()
        hidden = hidden or C.HIDDEN
        activation = activation or C.ACTIVATION
        self.v_max = float(v_max if v_max is not None else C.V_MAX)

        act = _ACT[activation]
        dims = [10, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act()]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)

        # normalization as non-trainable buffers (travel in state_dict)
        self.register_buffer("x_mean", torch.as_tensor(norm_stats["x_mean"], dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(norm_stats["x_std"], dtype=torch.float32))
        self.register_buffer("p_mean", torch.as_tensor(norm_stats["p_mean"], dtype=torch.float32))
        self.register_buffer("p_std", torch.as_tensor(norm_stats["p_std"], dtype=torch.float32))

    def forward(self, state, mlparams):
        """
        state:    (B,6) tensor
        mlparams: (B,4) tensor [m1,m2,l1,l2]
        returns:  (B,) voltage in [-v_max, v_max]
        """
        state = torch.as_tensor(state)
        mlparams = torch.as_tensor(mlparams)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if mlparams.ndim == 1:
            mlparams = mlparams.unsqueeze(0)
        # match the network's parameter dtype (weights are float32)
        w_dtype = self.net[0].weight.dtype
        xs = (state.to(w_dtype) - self.x_mean) / self.x_std
        ps = (mlparams.to(w_dtype) - self.p_mean) / self.p_std
        z = torch.cat([xs, ps], dim=-1)
        raw = self.net(z).squeeze(-1)
        return torch.tanh(raw) * self.v_max


def save_checkpoint(path, model, extra=None):
    payload = {"state_dict": model.state_dict(),
               "hidden": tuple(C.HIDDEN),
               "activation": C.ACTIVATION,
               "v_max": model.v_max}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    """Reconstruct a PINNPolicy from a checkpoint (buffers carry norm stats)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    sd = ckpt["state_dict"]
    stats = dict(x_mean=sd["x_mean"].numpy(), x_std=sd["x_std"].numpy(),
                 p_mean=sd["p_mean"].numpy(), p_std=sd["p_std"].numpy())
    model = PINNPolicy(stats, hidden=ckpt["hidden"],
                       activation=ckpt["activation"], v_max=ckpt["v_max"])
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt
