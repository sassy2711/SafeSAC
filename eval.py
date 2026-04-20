import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Imports (your modules)
# =========================
from env import GridWorld
from policy_net import PolicyNetwork
from q_net import QNetwork

import matplotlib
matplotlib.use('Agg')   # ← this is the fix
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
def load_model(path):
    checkpoint = torch.load(path, map_location=device)

    state_dim = 2
    action_dim = 2

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    q1 = QNetwork(state_dim, action_dim).to(device)
    q2 = QNetwork(state_dim, action_dim).to(device)

    policy.load_state_dict(checkpoint["policy"])
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint["q2"])

    policy.eval()
    q1.eval()
    q2.eval()

    return policy


# =========================
# Deterministic Evaluation
# =========================
def evaluate(policy, env, max_steps=100):
    s = env.reset()
    trajectory = [s.copy()]
    total_reward = 0

    for _ in range(max_steps):
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Mean action (deterministic)
            mean, _ = policy.forward(s_tensor)
            a = torch.tanh(mean)

        a_np = a.squeeze(0).cpu().numpy()

        s, r, done, _ = env.step(a_np)

        trajectory.append((s*4.0).copy())
        total_reward += r

        if done:
            break

    return np.array(trajectory), total_reward


# =========================
# Plot Trajectory
# =========================
def plot_trajectory(traj, save_path="trajectory.png"):
    plt.figure()

    # Path
    plt.plot(traj[:, 0], traj[:, 1], marker='o')

    # Start and Goal
    plt.scatter([0], [0], label="Start", s=100)
    plt.scatter([4], [4], label="Goal", s=100)

    plt.title("SAC Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()

    plt.savefig(save_path)
    plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    model_path = "safe_sac_model.pth"   # your saved file

    env = GridWorld()
    policy = load_model(model_path)

    traj, reward = evaluate(policy, env)

    print("Total Reward:", reward)
    print("Trajectory length:", len(traj))

    plot_trajectory(traj, "trajectory.png")