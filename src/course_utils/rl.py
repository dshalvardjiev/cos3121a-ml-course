"""A tiny, dependency-free grid world + tabular Q-learning for Session 3.
(AWS DeepRacer was retired from the console in Dec 2025; running our own
simulation is closer to how RL is actually practiced anyway.)"""
import numpy as np

# 4x4 grid: S start, G goal(+1), H hole(end, 0), . free
DEFAULT_MAP = ["S...", ".H.H", "...H", "H..G"]
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # up down left right
ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}


class GridWorld:
    def __init__(self, grid=None, step_reward: float = 0.0):
        self.grid = [list(r) for r in (grid or DEFAULT_MAP)]
        self.n = len(self.grid)
        self.step_reward = step_reward
        self.start = self._find("S")
        self.reset()

    def _find(self, ch):
        for i, row in enumerate(self.grid):
            for j, c in enumerate(row):
                if c == ch:
                    return (i, j)

    def reset(self):
        self.pos = self.start
        return self._sid(self.pos)

    def _sid(self, pos):
        return pos[0] * self.n + pos[1]

    def step(self, action: int):
        di, dj = ACTIONS[action]
        i = min(max(self.pos[0] + di, 0), self.n - 1)
        j = min(max(self.pos[1] + dj, 0), self.n - 1)
        self.pos = (i, j)
        cell = self.grid[i][j]
        if cell == "G":
            return self._sid(self.pos), 1.0, True
        if cell == "H":
            return self._sid(self.pos), 0.0, True
        return self._sid(self.pos), self.step_reward, False

    def render(self):
        out = [row[:] for row in self.grid]
        i, j = self.pos
        out[i][j] = "A"
        print("\n".join("".join(r) for r in out) + "\n")


def train_q_learning(env, episodes=2000, lr=0.1, gamma=0.95,
                     eps_start=1.0, eps_end=0.05, max_steps=100, seed=42):
    """Returns (Q-table, per-episode rewards). One line of learning, the rest is bookkeeping."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n * env.n, len(ACTIONS)))
    rewards = []
    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * ep / episodes
        s, done, total = env.reset(), False, 0.0
        for _ in range(max_steps):
            a = rng.integers(len(ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env.step(a)
            Q[s, a] += lr * (r + gamma * np.max(Q[s2]) - Q[s, a])   # <- the Q-learning update
            s, total = s2, total + r
            if done:
                break
        rewards.append(total)
    return Q, rewards


def greedy_run(env, Q, max_steps=30):
    s, done, path = env.reset(), False, [env.pos]
    for _ in range(max_steps):
        s, r, done = env.step(int(np.argmax(Q[s])))
        path.append(env.pos)
        if done:
            return path, r
    return path, 0.0


def print_policy(env, Q):
    for i in range(env.n):
        row = ""
        for j in range(env.n):
            cell = env.grid[i][j]
            row += cell if cell in "GH" else ARROWS[int(np.argmax(Q[i * env.n + j]))]
        print(row)
