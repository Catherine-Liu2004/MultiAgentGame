import random
from config import ACTIONS


class RandomAgents:
    def __init__(self, npc_count: int):
        self.npc_count = npc_count
        self.q_tables = None  # 为统一接口保留

    def act(self, states):
        return [random.randint(0, len(ACTIONS) - 1) for _ in range(self.npc_count)]

    def update(self, *args, **kwargs):
        # 随机策略不学习
        pass