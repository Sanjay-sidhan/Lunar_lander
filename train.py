"""
train.py  —  PPO Training Loop for Lunar Lander
=================================================
Trains the PPO agent from scratch using the LunarLanderEnv.
Runs completely headless (no pygame window) for maximum speed.

Features:
  - Rollout collection + PPO update loop
  - Episode reward tracking
  - Console progress logging
  - Auto-save best model + periodic checkpoints
  - Matplotlib reward curve plotted at end
  - Early stopping when solved

Usage:
    python train.py
    python train.py --steps 2000000
    python train.py --steps 1000000 --rollout 1024 --lr 0.0003

Trained model saved to:
    ppo_model.pth        ← best model (highest mean reward)
    checkpoints/         ← periodic saves every N updates
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from lunar_lander_env import LunarLanderEnv
from ppo_agent import PPOAgent, RolloutBuffer


# ══════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Train PPO on Lunar Lander")

parser.add_argument("--steps",       type=int,   default=1_000_000,
                    help="Total environment steps to train for (default 1M)")
parser.add_argument("--rollout",     type=int,   default=2048,
                    help="Steps per rollout buffer (default 2048)")
parser.add_argument("--lr",          type=float, default=3e-4,
                    help="Learning rate (default 3e-4)")
parser.add_argument("--gamma",       type=float, default=0.99,
                    help="Discount factor (default 0.99)")
parser.add_argument("--clip",        type=float, default=0.2,
                    help="PPO clip epsilon (default 0.2)")
parser.add_argument("--epochs",      type=int,   default=5,
                    help="PPO update epochs per rollout (default 5)")
parser.add_argument("--hidden",      type=int,   default=256,
                    help="Hidden layer size (default 256)")
parser.add_argument("--save",        type=str,   default="ppo_model.pth",
                    help="Path to save best model (default ppo_model.pth)")
parser.add_argument("--checkpoint",  type=int,   default=50,
                    help="Save checkpoint every N updates (default 50)")
parser.add_argument("--solve",       type=float, default=150.0,
                    help="Mean reward over 10 episodes to consider solved (default 150)")
parser.add_argument("--load",        type=str,   default=None,
                    help="Path to an existing model to resume training from")
parser.add_argument("--randompad",   action="store_true",
                    help="Randomise landing pad position each episode")
parser.add_argument("--wind",        action="store_true",
                    help="Enable wind disturbance")
parser.add_argument("--no-plot",     action="store_true",
                    help="Skip reward curve plot at the end")

args = parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
os.makedirs("checkpoints", exist_ok=True)

# Environment
env = LunarLanderEnv(
    render_mode   = None,
    randomize_pad = args.randompad,
    wind          = args.wind,
)

# Hyperparameters
hp = dict(
    lr            = args.lr,
    gamma         = args.gamma,
    clip_eps      = args.clip,
    ppo_epochs    = args.epochs,
    hidden_dim    = args.hidden,
    gae_lambda    = 0.95,
    entropy_coef  = 0.02,    # raised for more exploration
    value_coef    = 0.5,
    max_grad_norm = 0.5,
    mini_batch    = 128,     # larger batches = more stable gradients
    lr_anneal     = True,    # decay LR over training
)

# Agent
agent = PPOAgent(obs_dim=8, act_dim=6, hp=hp)

if args.load and os.path.exists(args.load):
    try:
        agent = PPOAgent.load(args.load)
        print(f"[INFO] Resuming training from existing model: {args.load}")
    except Exception as e:
        print(f"[WARN] Failed to load {args.load}: {e}. Starting fresh.")

agent.train_mode()

# Rollout buffer
buffer = RolloutBuffer(
    size       = args.rollout,
    obs_dim    = 8,
    gamma      = args.gamma,
    gae_lambda = hp["gae_lambda"],
)


# ══════════════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════
class TrainingLogger:
    def __init__(self):
        self.episode_rewards  = []
        self.episode_lengths  = []
        self.update_rewards   = []
        self.landed_count     = 0
        self.crashed_count    = 0
        self.total_episodes   = 0
        self.best_mean_reward = -np.inf
        self.start_time       = time.time()

    def log_episode(self, reward, length, status):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_episodes  += 1
        if status == "LANDED":
            self.landed_count += 1
        elif status == "CRASHED":
            self.crashed_count += 1

    def mean_reward(self, n=10):
        if len(self.episode_rewards) < 1:
            return -np.inf
        return float(np.mean(self.episode_rewards[-n:]))

    def success_rate(self, n=10):
        recent = self.episode_rewards[-n:]
        if not recent:
            return 0.0
        landed = self.landed_count
        return min(landed / max(self.total_episodes, 1), 1.0)

    def elapsed(self):
        return time.time() - self.start_time

    def print_update(self, step, n_updates, agent):
        mean_r   = self.mean_reward(10)
        elapsed  = self.elapsed()
        sps      = step / max(elapsed, 1)
        eta_sec  = max(0, (args.steps - step) / max(sps, 1))
        eta_min  = eta_sec / 60

        print(
            f"  Update {n_updates:4d} | "
            f"Steps {step:>8,} | "
            f"Episodes {self.total_episodes:>5} | "
            f"MeanR(10) {mean_r:>8.2f} | "
            f"Best {self.best_mean_reward:>8.2f} | "
            f"Landed {self.landed_count:>4} | "
            f"SPS {sps:>6,.0f} | "
            f"ETA {eta_min:.1f}m"
        )
        agent.print_stats()
        print()

    def plot(self, save_path="training_curve.png"):
        if len(self.episode_rewards) < 2:
            print("[PLOT] Not enough data to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("PPO Lunar Lander — Training Curve", fontsize=14, fontweight="bold")

        # Raw episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.4, color="steelblue", label="Episode reward")
        window = min(20, len(self.episode_rewards))
        if len(self.episode_rewards) >= window:
            smoothed = np.convolve(self.episode_rewards,
                                   np.ones(window) / window, mode="valid")
            axes[0, 0].plot(range(window - 1, len(self.episode_rewards)),
                            smoothed, color="royalblue", linewidth=2,
                            label=f"Moving avg ({window})")
        axes[0, 0].axhline(args.solve, color="green", linestyle="--", label="Solve threshold")
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.5, color="coral")
        axes[0, 1].set_title("Episode Length (steps)")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True, alpha=0.3)

        # Policy & value loss
        if agent.stats["policy_loss"]:
            axes[1, 0].plot(agent.stats["policy_loss"], label="Policy loss", color="tomato")
            axes[1, 0].plot(agent.stats["value_loss"],  label="Value loss",  color="orange")
            axes[1, 0].set_title("Losses per Update")
            axes[1, 0].set_xlabel("Update")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Entropy
        if agent.stats["entropy"]:
            axes[1, 1].plot(agent.stats["entropy"], color="mediumseagreen")
            axes[1, 1].set_title("Policy Entropy (exploration)")
            axes[1, 1].set_xlabel("Update")
            axes[1, 1].set_ylabel("Entropy")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"[PLOT] Training curve saved → {save_path}")
        plt.show()


logger = TrainingLogger()


# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PPO Lunar Lander — Training")
print("=" * 70)
print(f"  Total steps   : {args.steps:,}")
print(f"  Rollout size  : {args.rollout}")
print(f"  Learning rate : {args.lr}")
print(f"  Solve target  : {args.solve} mean reward over 10 episodes")
print(f"  Randomise pad : {args.randompad}")
print(f"  Wind          : {args.wind}")
print("=" * 70)
print()

obs, info     = env.reset()
ep_reward     = 0.0
ep_length     = 0
total_steps   = 0
n_updates     = 0
solved        = False

while total_steps < args.steps and not solved:

    # ── COLLECT ROLLOUT ─────────────────────────────────────────────
    buffer.reset()

    while not buffer.is_full():
        action, log_prob, value = agent.select_action_with_info(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, log_prob, reward, value, done)

        ep_reward   += reward
        ep_length   += 1
        total_steps += 1
        obs          = next_obs

        if done:
            logger.log_episode(ep_reward, ep_length, info["status"])
            ep_reward = 0.0
            ep_length = 0
            obs, info = env.reset()

    # ── COMPUTE GAE + UPDATE ─────────────────────────────────────────
    last_value = agent.get_value(obs)
    last_done  = False

    buffer.compute_gae(last_value=last_value, last_done=last_done)
    # Anneal LR based on training progress
    agent.set_progress(total_steps / args.steps)
    agent.update(buffer)
    n_updates += 1

    # ── LOGGING ──────────────────────────────────────────────────────
    mean_r = logger.mean_reward(10)

    # Save best model
    if mean_r > logger.best_mean_reward:
        logger.best_mean_reward = mean_r
        agent.save(args.save)

    # Periodic checkpoint
    if n_updates % args.checkpoint == 0:
        ckpt_path = f"checkpoints/ppo_step_{total_steps:08d}.pth"
        agent.save(ckpt_path)

    # Console log every update
    logger.print_update(total_steps, n_updates, agent)

    # Check solved
    if mean_r >= args.solve and logger.total_episodes >= 10:
        print(f"\n{'='*70}")
        print(f"  ✅  SOLVED at step {total_steps:,}!")
        print(f"  Mean reward (last 10 episodes): {mean_r:.2f}")
        print(f"{'='*70}\n")
        solved = True


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"  Total steps      : {total_steps:,}")
print(f"  Total episodes   : {logger.total_episodes:,}")
print(f"  Successful lands : {logger.landed_count}")
print(f"  Crashes          : {logger.crashed_count}")
print(f"  Best mean reward : {logger.best_mean_reward:.2f}")
print(f"  Time elapsed     : {logger.elapsed() / 60:.1f} minutes")
print(f"  Best model saved : {args.save}")
print("=" * 70)
print()
print("  To watch the trained agent:")
print(f"    python main.py --mode ppo --model {args.save}")
print()

env.close()

# ── PLOT ─────────────────────────────────────────────────────────────
if not args.no_plot:
    logger.plot("training_curve.png")