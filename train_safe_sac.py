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
from safe_policy_generator import SafeLLMExplorerContinuous

# =========================
# Hyperparameters
# =========================
gamma = 0.99
tau = 0.005
batch_size = 64
lr = 3e-4
num_episodes = 500
max_steps = 100

# LLM settings
llm_update_freq = 5
num_past_episodes = 5
num_action_samples = 20

TASK_DESC = "Agent moves from (0,0) to (4,4)"
CONSTRAINTS = "Avoid square from (1,1) to (3,3)"

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
# Entropy (alpha)
# =========================
log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
alpha_opt = optim.Adam([log_alpha], lr=lr)
target_entropy = -action_dim

# Beta Curriculum Hyperparameters
BETA_PHASE1 = 50  # Episodes with beta = 0
BETA_PHASE2 = 100  # Episodes for linear ramp up from 0 to 1
BETA_TARGET = 0.2  # Final beta value

# =========================
# Optimizers
# =========================
policy_opt = optim.Adam(policy.parameters(), lr=lr)
q1_opt = optim.Adam(q1.parameters(), lr=lr)
q2_opt = optim.Adam(q2.parameters(), lr=lr)

# =========================
# Safe policy (LLM)
# =========================
safe_mu = torch.zeros(action_dim, device=device)

# =========================
# Helpers
# =========================
def is_unsafe(state):
    x, y = state
    return (1.0 <= x <= 3.0) and (1.0 <= y <= 3.0)

def kl_gaussian(mu, log_std, mu_safe, log_std_safe):
    std = log_std.exp()
    std_safe = log_std_safe.exp()

    kl = (
        log_std_safe - log_std
        + (std.pow(2) + (mu - mu_safe).pow(2)) / (2 * std_safe.pow(2))
        - 0.5
    )
    return kl.sum(dim=-1, keepdim=True)

# =========================
# SAC Update
# =========================
def update(ep):
    if len(buffer) < batch_size:
        return None

    s, a, r, s_next, d = buffer.sample(batch_size)

    s = torch.FloatTensor(s).to(device)
    a = torch.FloatTensor(a).to(device)
    r = torch.FloatTensor(r).unsqueeze(1).to(device)
    s_next = torch.FloatTensor(s_next).to(device)
    d = torch.FloatTensor(d).unsqueeze(1).to(device)

    # -------------------------
    # Q Target
    # -------------------------
    with torch.no_grad():
        a_next, log_pi_next = policy.sample(s_next)
        q_next = torch.min(q1_target(s_next, a_next), q2_target(s_next, a_next))

        alpha = log_alpha.exp()
        q_target = r + (1 - d) * gamma * (q_next - alpha * log_pi_next)

    # -------------------------
    # Q Loss
    # -------------------------
    J_Q1 = F.mse_loss(q1(s, a), q_target)
    J_Q2 = F.mse_loss(q2(s, a), q_target)

    q1_opt.zero_grad()
    J_Q1.backward()
    q1_opt.step()

    q2_opt.zero_grad()
    J_Q2.backward()
    q2_opt.step()

    # -------------------------
    # Policy Loss + KL
    # -------------------------
    a_new, log_pi = policy.sample(s)
    q_min = torch.min(q1(s, a_new), q2(s, a_new))

    mu, log_std = policy.forward(s)

    mu_safe_batch = safe_mu.unsqueeze(0).expand_as(mu)
    log_std_safe_batch = log_std.detach()  # IMPORTANT

    kl = kl_gaussian(mu, log_std, mu_safe_batch, log_std_safe_batch)
    kl = torch.clamp(kl, max=10.0)

    alpha = log_alpha.exp()

    # -------------------------
    # Beta Curriculum Logic
    # -------------------------
    if ep < BETA_PHASE1:
        effective_beta = 0.0
    elif ep < (BETA_PHASE1 + BETA_PHASE2):
        # Linear ramp from 0 to 1
        progress = (ep - BETA_PHASE1) / BETA_PHASE2
        effective_beta = progress * BETA_TARGET
    else:
        effective_beta = BETA_TARGET

    J_pi = (alpha * log_pi - q_min + effective_beta * kl).mean()
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
    # Target update
    # -------------------------
    for tp, p in zip(q1_target.parameters(), q1.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    for tp, p in zip(q2_target.parameters(), q2.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

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
        ).item(),
        "kl_mean": kl.mean().item(),
        "beta_kl": (effective_beta * kl).mean().item(),
        "kl_ratio": (
            (effective_beta * kl).abs().mean() / (q_min.abs().mean() + 1e-6)
        ).item()
    }

    return stats

def plot_trajectory(traj, save_path):
    traj = np.array(traj)

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], marker='o')

    # Start & Goal
    plt.scatter([0], [0], s=100, label="Start")
    plt.scatter([4], [4], s=100, label="Goal")

    # Unsafe region (optional but useful)
    plt.gca().add_patch(
        plt.Rectangle((1,1), 2, 2, fill=False)
    )

    plt.title("Training Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()

    plt.savefig(save_path)
    plt.close()

# =========================
# Training
# =========================
def train():
    global safe_mu

    past_episodes = []
    violations_per_episode = []
    rewards_per_episode = []
    safe_mu_history = []

    for ep in range(num_episodes):
        s = env.reset()
        trajectory = [(s * 4.0).copy()]
        total_reward = 0
        violation_count = 0
        episode_actions = []

        losses = {k: [] for k in [
            "J_Q","J_pi","Q_mean","log_pi_mean",
            "alpha_log_pi_mean","entropy_term_ratio",
            "kl_mean","beta_kl","kl_ratio"
        ]}

        for t in range(max_steps):
            s_tensor = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                a, _ = policy.sample(s_tensor)

            a_np = a.squeeze(0).cpu().numpy()
            s_next, r, done, _ = env.step(a_np)
            trajectory.append((s_next * 4.0).copy())

            if is_unsafe(s_next * 4.0):
                violation_count += 1

            episode_actions.append(a_np)

            buffer.push(s, a_np, r, s_next, done)

            s = s_next
            total_reward += r

            stats = update(ep)
            if stats:
                for k in losses:
                    losses[k].append(stats[k])

            if done:
                break

        # store episode
        past_episodes.append({
            "actions": episode_actions,
            "reward": total_reward,
            "violations": violation_count
        })

        violations_per_episode.append(violation_count)
        rewards_per_episode.append(total_reward)

        # -------------------------
        # LLM UPDATE
        # -------------------------
        if ep >= (num_past_episodes + BETA_PHASE1) and ep % llm_update_freq == 0:
            recent = past_episodes[-num_past_episodes:]

            action_sequences, rewards, violations = [], [], []

            for ep_data in recent:
                acts = ep_data["actions"]
                idx = np.random.choice(len(acts), min(len(acts), num_action_samples), replace=False)
                sampled = [acts[i] for i in idx]

                action_sequences.append(sampled)
                rewards.append(ep_data["reward"])
                violations.append(ep_data["violations"])

            explorer = SafeLLMExplorerContinuous()

            bias_vector, _, _ = explorer.get_safe_exploration_bias(
                action_sequences,
                rewards,
                violations,
                TASK_DESC,
                action_dim,
                CONSTRAINTS,
                num_action_samples
            )

            # smooth update
            new_mu = torch.tensor(bias_vector, device=device).float()
            safe_mu = 0.8 * safe_mu + 0.2 * new_mu

            print(f"[LLM UPDATE] safe_mu = {safe_mu.cpu().numpy()}")

        safe_mu_history.append(safe_mu.cpu().numpy())

        def avg(x): return np.mean(x) if x else 0

        print(
            f"[Ep {ep:03d}] R:{total_reward:7.2f} | Viol:{violation_count:3d} | "
            f"Q:{avg(losses['Q_mean']):6.2f} | "
            f"Q_loss:{avg(losses['J_Q']):6.3f} | "
            f"Pi:{avg(losses['J_pi']):6.3f} | "
            f"KL:{avg(losses['kl_mean']):6.3f} | "
            f"βKL:{avg(losses['beta_kl']):6.3f}"
        )
        if ep % 50 == 0:
            plot_trajectory(trajectory, f"trajectory_ep_{ep}.png")

    # =========================
    # SAVE EVERYTHING
    # =========================
    torch.save({
        "policy": policy.state_dict(),
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "log_alpha": log_alpha,
        "safe_mu": safe_mu
    }, "safe_sac_model.pth")

    np.savetxt("violations.csv", violations_per_episode, delimiter=",")
    np.savetxt("rewards.csv", rewards_per_episode, delimiter=",")
    np.savetxt("safe_mu.csv", np.array(safe_mu_history), delimiter=",")

    # plot violations
    plt.figure()
    plt.plot(violations_per_episode)
    plt.title("Violations per Episode")
    plt.savefig("violations.png")
    plt.close()

if __name__ == "__main__":
    train()