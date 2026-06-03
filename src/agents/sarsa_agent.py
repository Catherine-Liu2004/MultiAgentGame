import random
import numpy as np

from config import ACTIONS, ALPHA, EPSILON, GAMMA


class IndependentSARSAAgents:
    def __init__(self, npc_count: int):
        self.npc_count = npc_count
        self.q_tables = [{} for _ in range(npc_count)]

    def _ensure_state(self, agent_idx: int, state):
        if state not in self.q_tables[agent_idx]:
            self.q_tables[agent_idx][state] = np.zeros(len(ACTIONS), dtype=float)

    @staticmethod
    def _argmax_random_tie(q_values: np.ndarray) -> int:
        max_q = np.max(q_values)
        candidates = np.where(q_values == max_q)[0]
        return int(random.choice(candidates))

    def act(self, states, epsilon=None):
        """
        states: 每个NPC的状态列表
        epsilon: 探索率；若为None则使用config.EPSILON
        """
        eps = EPSILON if epsilon is None else epsilon
        action_indices = []

        for i, state in enumerate(states):
            self._ensure_state(i, state)

            if random.random() < eps:
                a_idx = random.randint(0, len(ACTIONS) - 1)
            else:
                a_idx = self._argmax_random_tie(self.q_tables[i][state])

            action_indices.append(a_idx)

        return action_indices

    def update(self, states, action_indices, rewards, next_states, next_action_indices):
        for i in range(self.npc_count):
            s = states[i]
            a = action_indices[i]
            r = rewards[i]
            ns = next_states[i]
            na = next_action_indices[i]

            self._ensure_state(i, s)
            self._ensure_state(i, ns)

            td_target = r + GAMMA * self.q_tables[i][ns][na]
            td_error = td_target - self.q_tables[i][s][a]
            self.q_tables[i][s][a] += ALPHA * td_error