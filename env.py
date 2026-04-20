import numpy as np

class GridWorld:
    def __init__(self):
        self.low = 0.0
        self.high = 4.0

        self.start = np.array([0.0, 0.0])
        self.goal = np.array([4.0, 4.0])

        self.state = self.start.copy()

        self.step_size = 0.5   # controls movement scale
        self.goal_threshold = 0.3

    def reset(self):
        self.state = self.start.copy()
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1, 1)

        # continuous transition
        self.state = self.state + self.step_size * action
        self.state = np.clip(self.state, self.low, self.high)

        # distance-based reward
        dist = np.linalg.norm(self.state - self.goal)

        reward = -dist  # smooth shaping
        done = False

        if dist < self.goal_threshold:
            reward = 10.0
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # normalized state (important for NN stability)
        return self.state / 4.0