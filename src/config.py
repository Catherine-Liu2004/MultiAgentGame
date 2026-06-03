ACTIONS = [
    "up", "down", "left", "right",
    "up_left", "up_right", "down_left", "down_right"
]

# 训练参数
EPISODES = 1000
MAX_STEPS = 80
FRAME_DELAY_MS = 80

# NPC 学习参数（你们原来的）
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.05

# 规则升级参数
PLAYER_MAX_HP = 100
ATTACK_DAMAGE_ADJ = 20          # 邻近攻击伤害（距离=1）
ATTACK_DAMAGE_SAME = 35         # 同格攻击伤害（距离=0）
MULTI_ATTACK_BONUS = 10         # 同步多NPC围攻加成（每个命中者额外+10）
COOP_BONUS_DIST = 2             # 协同奖励判定阈值（所有NPC与玩家距离 <= 2）
REWARD_COOP = 2
REWARD_STEP = -1
REWARD_COLLISION = -5
REWARD_HIT_PLAYER = 8           # 命中玩家奖励
REWARD_KILL_PLAYER = 120        # 击杀奖励

# 视觉参数
TILE_SIZE = 64
HUD_WIDTH = 280

# 地图
MAP_FILE = "maps/map_01.txt"

# 动态障碍
ENABLE_MOVING_OBSTACLES = True
MOVING_OBS_STEP_INTERVAL = 2   # 每2步更新一次动态障碍

# =========================
# 玩家 DQN 对抗网络参数
# =========================
PLAYER_USE_DQN = True
PLAYER_EPSILON = 1.0
PLAYER_EPSILON_MIN = 0.05
PLAYER_EPSILON_DECAY = 0.995
PLAYER_GAMMA = 0.95
PLAYER_LR = 1e-3
PLAYER_BATCH_SIZE = 32
PLAYER_MEMORY_SIZE = 5000
PLAYER_TARGET_UPDATE = 50

# 玩家奖励（更偏向“活得更久”）
PLAYER_REWARD_STEP_SURVIVE = 2.0      # 每活一步奖励
PLAYER_REWARD_DAMAGE_SCALE = 0.2      # 受到伤害惩罚系数
PLAYER_REWARD_HIT_PENALTY = 5.0       # 每次被命中额外惩罚
PLAYER_REWARD_COOP_TRAP = 3.0         # 被围困惩罚
PLAYER_REWARD_SUCCESS_ESCAPE = 100.0  # 成功活到MAX_STEPS奖励