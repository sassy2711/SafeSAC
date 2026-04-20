import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# Imports
# =========================
from env import GridWorld
from replay_buffer import ReplayBuffer
from policy_net import PolicyNetwork
from q_net import QNetwork

# =========================
# Hyperparameters
# =========================
gamma = 0.99
tau = 0.005
batch_size = 64
lr = 3e-4
num_episodes = 500
max_steps = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Environment + Buffer
# =========================
env = GridWorld()
buffer = ReplayBuffer(10000)

state_dim = 2
action_dim = 2

# =========================
# Networks
# =========================
policy = PolicyNetwork(state_dim, action_dim).to(device)

q1 = QNetwork(state_dim, action_dim).to(device)
q2 = QNetwork(state_dim, action_dim).to(device)

q1_target = QNetwork(state_dim, action_dim).to(device)
q2_target = QNetwork(state_dim, action_dim).to(device)

q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())

# =========================
# Entropy (alpha) tuning
# =========================
log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
alpha_opt = optim.Adam([log_alpha], lr=lr)

target_entropy = -action_dim

# =========================
# Optimizers
# =========================
policy_opt = optim.Adam(policy.parameters(), lr=lr)
q1_opt = optim.Adam(q1.parameters(), lr=lr)
q2_opt = optim.Adam(q2.parameters(), lr=lr)

def is_unsafe(state):
    x, y = state
    return (1.0 <= x <= 3.0) and (1.0 <= y <= 3.0)

# =========================
# SAC Update
# =========================
def update():
    if len(buffer) < batch_size:
        return None

    s, a, r, s_next, d = buffer.sample(batch_size)

    s = torch.FloatTensor(s).to(device)
    a = torch.FloatTensor(a).to(device)
    r = torch.FloatTensor(r).unsqueeze(1).to(device)
    s_next = torch.FloatTensor(s_next).to(device)
    d = torch.FloatTensor(d).unsqueeze(1).to(device)

    # -------------------------
    # Q Target (Modern SAC)
    # -------------------------
    with torch.no_grad():
        a_next, log_pi_next = policy.sample(s_next)

        q1_next = q1_target(s_next, a_next)
        q2_next = q2_target(s_next, a_next)
        q_next = torch.min(q1_next, q2_next)

        alpha = log_alpha.exp()

        q_target = r + (1 - d) * gamma * (q_next - alpha * log_pi_next)

    # -------------------------
    # Q Loss
    # -------------------------
    q1_pred = q1(s, a)
    q2_pred = q2(s, a)

    J_Q1 = F.mse_loss(q1_pred, q_target)
    J_Q2 = F.mse_loss(q2_pred, q_target)

    q1_opt.zero_grad()
    J_Q1.backward()
    q1_opt.step()

    q2_opt.zero_grad()
    J_Q2.backward()
    q2_opt.step()

    # -------------------------
    # Policy Loss
    # -------------------------
    a_new, log_pi = policy.sample(s)

    q1_val = q1(s, a_new)
    q2_val = q2(s, a_new)
    q_min = torch.min(q1_val, q2_val)

    alpha = log_alpha.exp()

    J_pi = (alpha * log_pi - q_min).mean()
    alpha_log_pi = alpha * log_pi

    policy_opt.zero_grad()
    J_pi.backward()
    policy_opt.step()

    # -------------------------
    # Alpha Loss
    # -------------------------
    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()

    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    # -------------------------
    # Target Update
    # -------------------------
    for target_param, param in zip(q1_target.parameters(), q1.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(q2_target.parameters(), q2.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # -------------------------
    # Logging
    # -------------------------
    stats = {
        "J_Q": (J_Q1.item() + J_Q2.item()) / 2,
        "J_pi": J_pi.item(),
        "Q_mean": q_min.mean().item(),
        "log_pi_mean": log_pi.mean().item(),
        "alpha_log_pi_mean": alpha_log_pi.mean().item(),
        "entropy_term_ratio": (
            alpha_log_pi.abs().mean() / (q_min.abs().mean() + 1e-6)
        ).item()
    }

    return stats

# =========================
# Training Loop
# =========================
def train():
    violations_per_episode = []

    for ep in range(num_episodes):
        s = env.reset()
        total_reward = 0

        violation_count = 0   # 🔥 NEW

        losses = {
            "J_Q": [],
            "J_pi": [],
            "Q_mean": [],
            "log_pi_mean": [],
            "alpha_log_pi_mean": [],
            "entropy_term_ratio": []
        }

        for t in range(max_steps):
            s_tensor = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                a, _ = policy.sample(s_tensor)

            a_np = a.squeeze(0).cpu().numpy()

            s_next, r, done, _ = env.step(a_np)

            # 🔥 CHECK UNSAFE
            if is_unsafe(s_next*4.0):
                violation_count += 1

            buffer.push(s, a_np, r, s_next, done)

            s = s_next
            total_reward += r

            stats = update()

            if stats is not None:
                for k in losses:
                    losses[k].append(stats[k])

            if done:
                break

        violations_per_episode.append(violation_count)

        def avg(x):
            return np.mean(x) if len(x) > 0 else 0

        print(
            f"[Ep {ep:03d}] "
            f"R: {total_reward:7.2f} | "
            f"Viol: {violation_count:3d} | "
            f"Q: {avg(losses['Q_mean']):7.2f} | "
            f"Q_loss: {avg(losses['J_Q']):7.3f} | "
            f"Pi_loss: {avg(losses['J_pi']):7.3f} | "
            f"logπ: {avg(losses['log_pi_mean']):7.3f} | "
            f"αlogπ: {avg(losses['alpha_log_pi_mean']):7.3f} | "
            f"ratio: {avg(losses['entropy_term_ratio']):6.3f}"
        )

        # Save model
        torch.save({
            "policy": policy.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "log_alpha": log_alpha
        }, "sac_model.pth")

    # =========================
    # AFTER TRAINING: SAVE METRICS
    # =========================

    violations_per_episode = np.array(violations_per_episode)
    mean_violations = np.cumsum(violations_per_episode) / (np.arange(len(violations_per_episode)) + 1)

    # Save to CSV
    np.savetxt(
        "violations.csv",
        np.stack([violations_per_episode, mean_violations], axis=1),
        delimiter=",",
        header="violations,mean_violations",
        comments=""
    )

    # Plot
    plt.figure()
    plt.plot(violations_per_episode, label="Violations per Episode")
    plt.plot(mean_violations, label="Mean Violations", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Violations")
    plt.title("Unsafe State Violations During Training")
    plt.legend()
    plt.grid()

    plt.savefig("violations.png")
    plt.close()

if __name__ == "__main__":
    train()