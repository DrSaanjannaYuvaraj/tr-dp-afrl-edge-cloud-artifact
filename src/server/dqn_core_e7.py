"""Multi-head DQN / DDQN for {0,1}^N."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    target_update: int = 10
    batch: int = 64
    replay_size: int = 5000

class QNet(nn.Module):
    def __init__(self, state_dim: int, n_clients: int, hidden: int = 128):
        super().__init__()
        self.bb = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_clients*2)
        self.n_clients = n_clients
    def forward(self, s: torch.Tensor):
        q = self.head(self.bb(s))
        return q.view(q.size(0), self.n_clients, 2)

class Replay:
    def __init__(self, cap: int):
        self.cap = cap
        self.buf = []
    def add(self, s,a,r,s2,done):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append((s,a,r,s2,done))
    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        s,a,r,s2,d = zip(*[self.buf[i] for i in idx])
        return np.stack(s), np.stack(a), np.array(r, np.float32), np.stack(s2), np.array(d, np.float32)
    def __len__(self):
        return len(self.buf)

class MultiHeadDQN:
    def __init__(self, state_dim: int, n_clients: int, cfg: DQNConfig, ddqn: bool=False):
        self.cfg = cfg
        self.ddqn = ddqn
        self.q = QNet(state_dim, n_clients)
        self.tgt = QNet(state_dim, n_clients)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = Replay(cfg.replay_size)
        self.eps = cfg.eps_start
        self.step = 0
        self.n_clients = n_clients

    @torch.no_grad()
    def act(self, s: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.eps:
            return (np.random.rand(self.n_clients) < 0.5).astype(np.int64)
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        q = self.q(st)[0]
        return torch.argmax(q, dim=-1).cpu().numpy().astype(np.int64)

    def store(self, s,a,r,s2,done=False):
        self.replay.add(s,a,float(r),s2,float(done))

    def update(self):
        cfg = self.cfg
        if len(self.replay) < cfg.batch:
            self.eps = max(cfg.eps_end, self.eps*cfg.eps_decay)
            return
        s,a,r,s2,d = self.replay.sample(cfg.batch)
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        q_all = self.q(s)
        q_taken = torch.gather(q_all, 2, a.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.ddqn:
                a2 = torch.argmax(self.q(s2), dim=-1)
                q2 = torch.gather(self.tgt(s2), 2, a2.unsqueeze(-1)).squeeze(-1)
            else:
                q2 = torch.max(self.tgt(s2), dim=-1).values
            target = r.unsqueeze(-1) + cfg.gamma*(1-d.unsqueeze(-1))*q2

        loss = F.mse_loss(q_taken, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        self.step += 1
        if self.step % cfg.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())
        self.eps = max(cfg.eps_end, self.eps*cfg.eps_decay)
