# ppo_core.py
"""
ppo_core.py
PPO core for:
- TL-PPO (offloading policy/value): π^off_θ , V^off_ϕ
- TR-DP-AFRL (selection + aggregation weights policy/value): π^sel_ψ , V^sel_ξ

Implements:
- PPO clipped objective
- GAE advantage
- Explicit KL monitoring with early-stop:
  δ_off for offload policy
  δ_sel for selection+weights policy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bernoulli_kl(logits_old: torch.Tensor, logits_new: torch.Tensor) -> torch.Tensor:
    p_old = torch.sigmoid(logits_old)
    p_new = torch.sigmoid(logits_new)
    eps = 1e-8
    return p_old * (torch.log(p_old + eps) - torch.log(p_new + eps)) + (1 - p_old) * (
        torch.log(1 - p_old + eps) - torch.log(1 - p_new + eps)
    )


def dirichlet_kl(alpha_old: torch.Tensor, alpha_new: torch.Tensor) -> torch.Tensor:
    """
    KL( Dir(alpha_old) || Dir(alpha_new) ) for batch.
    """
    a0 = alpha_old.sum(dim=-1, keepdim=True)
    b0 = alpha_new.sum(dim=-1, keepdim=True)
    t1 = torch.lgamma(a0) - torch.lgamma(alpha_old).sum(dim=-1, keepdim=True)
    t2 = -torch.lgamma(b0) + torch.lgamma(alpha_new).sum(dim=-1, keepdim=True)
    t3 = ((alpha_old - alpha_new) * (torch.digamma(alpha_old) - torch.digamma(a0))).sum(dim=-1, keepdim=True)
    return (t1 + t2 + t3).squeeze(-1)


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    k_epochs: int = 4
    minibatch: int = 64
    gamma_rl: float = 0.99
    gae_lambda: float = 0.95
    delta_kl: float = 0.02

    # -----------------------------
    # PATCH knobs (do not break Algorithm 1/2)
    # -----------------------------
    gate_temp: float = 1.0              # >1 => softer gate probs early
    dir_alpha_floor: float = 0.10       # prevents near-zero concentration (degenerate Dirichlet)
    dir_alpha_cap: float = 50.0         # prevents extreme concentration


class OffloadPolicy(nn.Module):
    def __init__(self, state_dim: int, n_clients: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.logits = nn.Linear(hidden, n_clients)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.logits(self.net(s))


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class OffloadPPO:
    """
    Offload policy: per-client Bernoulli a_t^i ∈ {0,1}.
    """
    def __init__(self, state_dim: int, n_clients: int, cfg: PPOConfig):
        self.cfg = cfg
        self.pi = OffloadPolicy(state_dim, n_clients)
        self.v = ValueNet(state_dim)
        self.opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=cfg.lr)
        self.buf = []  # (s, a, r, s2, logp)

    @torch.no_grad()
    def act(self, s: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        logits = self.pi(st)
        dist = torch.distributions.Bernoulli(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a).sum(dim=-1)
        return a.squeeze(0).cpu().numpy().astype(np.int64), float(logp.item()), logits.squeeze(0).cpu().numpy()

    def store(self, s, a, r, s2, logp):
        self.buf.append((s, a, float(r), s2, float(logp)))

    def _gae(self):
        cfg = self.cfg
        s = torch.tensor(np.stack([b[0] for b in self.buf]), dtype=torch.float32)
        s2 = torch.tensor(np.stack([b[3] for b in self.buf]), dtype=torch.float32)
        r = torch.tensor([b[2] for b in self.buf], dtype=torch.float32)

        with torch.no_grad():
            v = self.v(s)
            v2 = self.v(s2)

        deltas = r + cfg.gamma_rl * v2 - v
        adv = torch.zeros_like(r)
        gae = 0.0
        for i in reversed(range(len(r))):
            gae = deltas[i] + cfg.gamma_rl * cfg.gae_lambda * gae
            adv[i] = gae
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        return s, adv, ret

    def update(self, tag="OFF"):
        if len(self.buf) < 8:
            self.buf.clear()
            return

        cfg = self.cfg
        s, adv, ret = self._gae()
        a = torch.tensor(np.stack([b[1] for b in self.buf]), dtype=torch.float32)
        logp_old = torch.tensor([b[4] for b in self.buf], dtype=torch.float32)

        with torch.no_grad():
            logits_old = self.pi(s)

        n = len(self.buf)
        idxs = np.arange(n)

        for _ in range(cfg.k_epochs):
            np.random.shuffle(idxs)
            for st_i in range(0, n, cfg.minibatch):
                mb = idxs[st_i : st_i + cfg.minibatch]
                sb, ab, advb, retb, lpob = s[mb], a[mb], adv[mb], ret[mb], logp_old[mb]

                logits = self.pi(sb)
                dist = torch.distributions.Bernoulli(logits=logits)
                logp = dist.log_prob(ab).sum(dim=-1)

                ratio = torch.exp(logp - lpob)
                clip = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pi_loss = -(torch.min(ratio * advb, clip * advb)).mean()

                v_loss = F.mse_loss(self.v(sb), retb)
                ent = dist.entropy().sum(dim=-1).mean()

                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    logits_new = self.pi(s)
                    kl = bernoulli_kl(logits_old, logits_new).sum(dim=-1).mean().item()

                print(f"[{tag}] KL={kl:.6f} (bound δ_off={cfg.delta_kl})")
                if kl > cfg.delta_kl:
                    print(f"[{tag}] Early-stop PPO update due to KL > δ_off")
                    self.buf.clear()
                    return

        self.buf.clear()


class SelectionPolicy(nn.Module):
    """
    Selection policy outputs:
    - gate logits -> Bernoulli for participation g_t^i
    - weight logits -> Dirichlet parameters for λ_t
    """
    def __init__(self, state_dim: int, n_clients: int, hidden: int = 128):
        super().__init__()
        self.bb = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.gate_logits = nn.Linear(hidden, n_clients)
        self.weight_logits = nn.Linear(hidden, n_clients)

    def forward(self, s: torch.Tensor):
        h = self.bb(s)
        return self.gate_logits(h), self.weight_logits(h)


class SelectionPPO:
    def __init__(self, state_dim: int, n_clients: int, cfg: PPOConfig):
        self.cfg = cfg
        self.pi = SelectionPolicy(state_dim, n_clients)
        self.v = ValueNet(state_dim)
        self.opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=cfg.lr)
        self.buf = []  # (s, g, lam_raw, r, s2, logp)

    @torch.no_grad()
    def act(self, s: np.ndarray):
        cfg = self.cfg
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        gate_logits, w_logits = self.pi(st)

        # PATCH: temperature to avoid saturated gate early
        gate_logits = gate_logits / max(1e-6, float(cfg.gate_temp))

        gate_dist = torch.distributions.Bernoulli(logits=gate_logits)
        g = gate_dist.sample()  # (1,N)

        # PATCH: alpha floor/cap to avoid degenerate Dirichlet
        alpha = F.softplus(w_logits).squeeze(0) + float(cfg.dir_alpha_floor)
        alpha = torch.clamp(alpha, min=float(cfg.dir_alpha_floor), max=float(cfg.dir_alpha_cap))

        dir_dist = torch.distributions.Dirichlet(alpha)
        lam_raw = dir_dist.sample()  # (N,)

        logp = gate_dist.log_prob(g).sum(dim=-1) + dir_dist.log_prob(lam_raw)
        return g.squeeze(0).cpu().numpy().astype(np.int64), lam_raw.cpu().numpy().astype(np.float32), float(logp.item())

    def store(self, s, g, lam_raw, r, s2, logp):
        self.buf.append((s, g, lam_raw, float(r), s2, float(logp)))

    def _gae(self):
        cfg = self.cfg
        s = torch.tensor(np.stack([b[0] for b in self.buf]), dtype=torch.float32)
        s2 = torch.tensor(np.stack([b[4] for b in self.buf]), dtype=torch.float32)
        r = torch.tensor([b[3] for b in self.buf], dtype=torch.float32)

        with torch.no_grad():
            v = self.v(s)
            v2 = self.v(s2)

        deltas = r + cfg.gamma_rl * v2 - v
        adv = torch.zeros_like(r)
        gae = 0.0
        for i in reversed(range(len(r))):
            gae = deltas[i] + cfg.gamma_rl * cfg.gae_lambda * gae
            adv[i] = gae
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        return s, adv, ret

    def update(self, tag="SEL"):
        if len(self.buf) < 8:
            self.buf.clear()
            return

        cfg = self.cfg
        s, adv, ret = self._gae()

        g = torch.tensor(np.stack([b[1] for b in self.buf]), dtype=torch.float32)
        lam = torch.tensor(np.stack([b[2] for b in self.buf]), dtype=torch.float32)
        logp_old = torch.tensor([b[5] for b in self.buf], dtype=torch.float32)

        with torch.no_grad():
            gate_logits_old, w_logits_old = self.pi(s)
            gate_logits_old = gate_logits_old / max(1e-6, float(cfg.gate_temp))
            alpha_old = F.softplus(w_logits_old) + float(cfg.dir_alpha_floor)
            alpha_old = torch.clamp(alpha_old, min=float(cfg.dir_alpha_floor), max=float(cfg.dir_alpha_cap))

        n = len(self.buf)
        idxs = np.arange(n)

        for _ in range(cfg.k_epochs):
            np.random.shuffle(idxs)
            for st_i in range(0, n, cfg.minibatch):
                mb = idxs[st_i : st_i + cfg.minibatch]
                sb, gb, lamb, advb, retb, lpob = s[mb], g[mb], lam[mb], adv[mb], ret[mb], logp_old[mb]

                gate_logits, w_logits = self.pi(sb)
                gate_logits = gate_logits / max(1e-6, float(cfg.gate_temp))
                gate_dist = torch.distributions.Bernoulli(logits=gate_logits)

                alpha = F.softplus(w_logits) + float(cfg.dir_alpha_floor)
                alpha = torch.clamp(alpha, min=float(cfg.dir_alpha_floor), max=float(cfg.dir_alpha_cap))
                dir_dist = torch.distributions.Dirichlet(alpha)
                logp_dir = dir_dist.log_prob(lamb)
                logp = gate_dist.log_prob(gb).sum(dim=-1) + logp_dir

                ratio = torch.exp(logp - lpob)
                clip = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pi_loss = -(torch.min(ratio * advb, clip * advb)).mean()

                v_loss = F.mse_loss(self.v(sb), retb)

                ent_gate = gate_dist.entropy().sum(dim=-1).mean()
                ent_dir = dir_dist.entropy().mean()

                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * (ent_gate + ent_dir)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    gate_logits_new, w_logits_new = self.pi(s)
                    gate_logits_new = gate_logits_new / max(1e-6, float(cfg.gate_temp))
                    alpha_new = F.softplus(w_logits_new) + float(cfg.dir_alpha_floor)
                    alpha_new = torch.clamp(alpha_new, min=float(cfg.dir_alpha_floor), max=float(cfg.dir_alpha_cap))

                    kl_gate = bernoulli_kl(gate_logits_old, gate_logits_new).sum(dim=-1).mean().item()
                    kl_dir = dirichlet_kl(alpha_old, alpha_new).mean().item()
                    kl_total = kl_gate + kl_dir

                print(
                    f"[{tag}] KL_gate={kl_gate:.6f} KL_dir={kl_dir:.6f} "
                    f"KL_total={kl_total:.6f} (bound δ_sel={cfg.delta_kl})"
                )
                if kl_total > cfg.delta_kl:
                    print(f"[{tag}] Early-stop PPO update due to KL_total > δ_sel")
                    self.buf.clear()
                    return

        self.buf.clear()
