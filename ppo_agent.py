"""
ppo_agent.py  —  PPO Agent for Lunar Lander
=============================================
Implements Proximal Policy Optimization (PPO) with:
  - Actor-Critic neural network
  - Clipped surrogate objective
  - Generalized Advantage Estimation (GAE)
  - Entropy bonus for exploration
  - Gradient clipping for stability
  - Save / Load model weights

Architecture:
  Shared backbone → Actor head (policy) + Critic head (value)

Usage:
    from ppo_agent import PPOAgent
    agent = PPOAgent(obs_dim=8, act_dim=6)
    action = agent.select_action(obs)        # during rollout
    agent.update(rollout_buffer)             # after collecting rollout
    agent.save("ppo_model.pth")
    agent = PPOAgent.load("ppo_model.pth")
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS  (all in one place for easy tuning)
# ══════════════════════════════════════════════════════════════════════
DEFAULT_HP = dict(
    obs_dim        = 8,
    act_dim        = 6,
    hidden_dim     = 256,       # neurons per hidden layer
    lr             = 3e-4,      # learning rate
    gamma          = 0.99,      # discount factor
    gae_lambda     = 0.95,      # GAE smoothing
    clip_eps       = 0.2,       # PPO clip range
    entropy_coef   = 0.02,      # raised from 0.01 → encourages more exploration early on
    value_coef     = 0.5,       # value loss weight
    max_grad_norm  = 0.5,       # gradient clipping
    ppo_epochs     = 8,         # raised from 5 → more updates per rollout (better sample efficiency)
    mini_batch     = 128,       # raised from 64 → more stable gradient estimates
    lr_anneal      = True,      # linearly anneal LR to 0 over training
)


# ══════════════════════════════════════════════════════════════════════
# NEURAL NETWORK  (Actor-Critic shared backbone)
# ══════════════════════════════════════════════════════════════════════
class ActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic network.

    Forward pass returns:
        action_logits  — raw logits for Categorical policy
        state_value    — scalar V(s) estimate
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head  → policy logits
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, act_dim),
        )

        # Critic head → state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Orthogonal weight initialisation (recommended for PPO)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller gain for the final actor layer → more uniform initial policy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        features      = self.backbone(x)
        action_logits = self.actor(features)
        state_value   = self.critic(features).squeeze(-1)
        return action_logits, state_value

    def get_action_and_value(self, x, action=None):
        """
        Sample an action (or evaluate a given one) and return
        (action, log_prob, entropy, value).
        """
        logits, value = self.forward(x)
        dist          = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob  = dist.log_prob(action)
        entropy   = dist.entropy()
        return action, log_prob, entropy, value


# ══════════════════════════════════════════════════════════════════════
# ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════════
class RolloutBuffer:
    """
    Stores one rollout (fixed number of steps across one or more episodes).
    Computes GAE advantages and returns before the PPO update.
    """

    def __init__(self, size, obs_dim, gamma=0.99, gae_lambda=0.95):
        self.size       = size
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.obs_dim    = obs_dim
        self.reset()

    def reset(self):
        self.obs        = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.actions    = np.zeros(self.size,                 dtype=np.int64)
        self.log_probs  = np.zeros(self.size,                 dtype=np.float32)
        self.rewards    = np.zeros(self.size,                 dtype=np.float32)
        self.values     = np.zeros(self.size,                 dtype=np.float32)
        self.dones      = np.zeros(self.size,                 dtype=np.float32)
        self.ptr        = 0
        self.full       = False

    def add(self, obs, action, log_prob, reward, value, done):
        i                = self.ptr % self.size
        self.obs[i]      = obs
        self.actions[i]  = action
        self.log_probs[i]= log_prob
        self.rewards[i]  = reward
        self.values[i]   = value
        self.dones[i]    = float(done)
        self.ptr        += 1
        if self.ptr >= self.size:
            self.full = True

    def is_full(self):
        return self.full

    def compute_gae(self, last_value, last_done):
        """
        Compute Generalised Advantage Estimation (GAE).
        Call this at the end of a rollout before calling get_batches().
        """
        advantages = np.zeros(self.size, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            delta         = (self.rewards[t]
                             + self.gamma * next_value * next_non_terminal
                             - self.values[t])
            last_gae      = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.returns    = advantages + self.values
        self.advantages = advantages

    def get_batches(self, mini_batch_size):
        """
        Yield randomised mini-batches of (obs, actions, log_probs,
        advantages, returns) as torch tensors.
        """
        indices = np.random.permutation(self.size)
        adv     = self.advantages
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalise

        for start in range(0, self.size, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            yield (
                torch.FloatTensor(self.obs[idx]),
                torch.LongTensor(self.actions[idx]),
                torch.FloatTensor(self.log_probs[idx]),
                torch.FloatTensor(adv[idx]),
                torch.FloatTensor(self.returns[idx]),
            )


# ══════════════════════════════════════════════════════════════════════
# PPO AGENT
# ══════════════════════════════════════════════════════════════════════
class PPOAgent:
    """
    PPO Agent wrapping ActorCritic network.

    Parameters
    ----------
    obs_dim : int   — observation vector size  (default 8)
    act_dim : int   — number of discrete actions (default 6)
    hp      : dict  — override any hyperparameter from DEFAULT_HP
    device  : str   — 'cpu' | 'cuda' | 'auto'
    """

    def __init__(self, obs_dim=8, act_dim=6, hp=None, device="auto"):
        self.hp = {**DEFAULT_HP, **(hp or {})}
        self.hp["obs_dim"] = obs_dim
        self.hp["act_dim"] = act_dim

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Network
        self.net = ActorCritic(
            obs_dim    = obs_dim,
            act_dim    = act_dim,
            hidden_dim = self.hp["hidden_dim"],
        ).to(self.device)

        # Optimiser
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.hp["lr"])

        # LR annealing state
        self._initial_lr    = self.hp["lr"]
        self._train_progress = 0.0   # fraction 0→1 of total training steps done

        # Training stats (logged per update)
        self.stats = dict(
            policy_loss = [],
            value_loss  = [],
            entropy     = [],
            total_loss  = [],
            clip_frac   = [],
        )

        print(f"[PPO] Device: {self.device}")
        print(f"[PPO] Network parameters: "
              f"{sum(p.numel() for p in self.net.parameters()):,}")

    # ──────────────────────────────────────────
    # ACTION SELECTION
    # ──────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs):
        """
        Sample an action given a numpy observation.
        Returns an int action index.
        Used during rollout collection.
        """
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.net.get_action_and_value(obs_t)
        return action.item()

    @torch.no_grad()
    def select_action_with_info(self, obs):
        """
        Like select_action but also returns log_prob and value.
        Used when filling the RolloutBuffer.
        """
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.net.get_action_and_value(obs_t)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_value(self, obs):
        """Return V(s) for a numpy observation."""
        obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        _, value = self.net(obs_t)
        return value.item()

    def eval(self):
        """Switch network to eval mode (disables dropout etc.)."""
        self.net.eval()
        return self

    def set_progress(self, fraction: float):
        """
        Call with fraction = steps_done / total_steps (0→1).
        Linearly decays the learning rate to near-zero if lr_anneal=True.
        """
        self._train_progress = max(0.0, min(1.0, fraction))
        if self.hp.get("lr_anneal", True):
            new_lr = self._initial_lr * (1.0 - self._train_progress)
            new_lr = max(new_lr, 1e-6)   # floor to avoid zero
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

    def train_mode(self):
        """Switch network to train mode."""
        self.net.train()
        return self

    # ──────────────────────────────────────────
    # PPO UPDATE
    # ──────────────────────────────────────────

    def update(self, buffer: RolloutBuffer):
        """
        Run PPO_EPOCHS passes over the rollout buffer,
        updating the network with the clipped objective.

        Call after buffer.compute_gae().
        """
        self.net.train()
        hp = self.hp

        ep_policy_loss = []
        ep_value_loss  = []
        ep_entropy     = []
        ep_total_loss  = []
        ep_clip_frac   = []

        for _ in range(hp["ppo_epochs"]):
            for obs_b, act_b, old_lp_b, adv_b, ret_b in buffer.get_batches(hp["mini_batch"]):

                obs_b    = obs_b.to(self.device)
                act_b    = act_b.to(self.device)
                old_lp_b = old_lp_b.to(self.device)
                adv_b    = adv_b.to(self.device)
                ret_b    = ret_b.to(self.device)

                # Evaluate actions under current policy
                _, log_prob, entropy, value = self.net.get_action_and_value(obs_b, act_b)

                # Probability ratio  π_new / π_old
                ratio = torch.exp(log_prob - old_lp_b)

                # Clipped surrogate objective
                surr1        = ratio * adv_b
                surr2        = torch.clamp(ratio, 1 - hp["clip_eps"],
                                                  1 + hp["clip_eps"]) * adv_b
                policy_loss  = -torch.min(surr1, surr2).mean()

                # Value loss  (clipped)
                value_loss   = 0.5 * ((value - ret_b) ** 2).mean()

                # Entropy bonus  (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss
                        + hp["value_coef"]   * value_loss
                        + hp["entropy_coef"] * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), hp["max_grad_norm"])
                self.optimizer.step()

                # Logging
                clip_frac = ((ratio - 1).abs() > hp["clip_eps"]).float().mean().item()
                ep_policy_loss.append(policy_loss.item())
                ep_value_loss.append(value_loss.item())
                ep_entropy.append(-entropy_loss.item())
                ep_total_loss.append(loss.item())
                ep_clip_frac.append(clip_frac)

        # Store mean stats
        self.stats["policy_loss"].append(np.mean(ep_policy_loss))
        self.stats["value_loss"].append(np.mean(ep_value_loss))
        self.stats["entropy"].append(np.mean(ep_entropy))
        self.stats["total_loss"].append(np.mean(ep_total_loss))
        self.stats["clip_frac"].append(np.mean(ep_clip_frac))

    # ──────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────

    def save(self, path="ppo_model.pth"):
        """Save model weights and hyperparameters."""
        torch.save({
            "net_state"  : self.net.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "hp"         : self.hp,
            "stats"      : self.stats,
        }, path)
        print(f"[PPO] Model saved → {path}")

    @classmethod
    def load(cls, path="ppo_model.pth", device="auto"):
        """Load a saved agent. Returns a PPOAgent ready for inference."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        hp         = checkpoint["hp"]
        agent      = cls(
            obs_dim = hp["obs_dim"],
            act_dim = hp["act_dim"],
            hp      = hp,
            device  = device,
        )
        agent.net.load_state_dict(checkpoint["net_state"])
        agent.optimizer.load_state_dict(checkpoint["optim_state"])
        agent.stats = checkpoint.get("stats", agent.stats)
        agent.net.to(agent.device)
        print(f"[PPO] Model loaded ← {path} (Running on {agent.device})")
        return agent

    # ──────────────────────────────────────────
    # STATS SUMMARY
    # ──────────────────────────────────────────

    def print_stats(self):
        """Print a summary of the last update's training stats."""
        if not self.stats["total_loss"]:
            print("[PPO] No updates yet.")
            return
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"[PPO] "
            f"PolicyLoss={self.stats['policy_loss'][-1]:.4f}  "
            f"ValueLoss={self.stats['value_loss'][-1]:.4f}  "
            f"Entropy={self.stats['entropy'][-1]:.4f}  "
            f"ClipFrac={self.stats['clip_frac'][-1]:.3f}  "
            f"LR={current_lr:.2e}"
        )


# ══════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running PPO agent sanity check...")

    agent  = PPOAgent(obs_dim=8, act_dim=6)
    buffer = RolloutBuffer(size=512, obs_dim=8,
                           gamma=agent.hp["gamma"],
                           gae_lambda=agent.hp["gae_lambda"])

    # Fill buffer with random data
    for _ in range(512):
        obs    = np.random.randn(8).astype(np.float32)
        action, log_prob, value = agent.select_action_with_info(obs)
        buffer.add(obs, action, log_prob,
                   reward=np.random.randn(),
                   value=value,
                   done=False)

    buffer.compute_gae(last_value=0.0, last_done=True)
    agent.update(buffer)
    agent.print_stats()

    # Save & reload
    agent.save("ppo_model_test.pth")
    agent2 = PPOAgent.load("ppo_model_test.pth")
    os.remove("ppo_model_test.pth")

    print("\nPPO agent looks good! Ready for training.")