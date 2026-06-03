# test.py
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import MAP_FILE, MAX_STEPS
from map_loader import load_map_txt
from env import MultiNpcEnv
from train import build_agents, load_npc_tables
from player_dqn_agent import PlayerDQNAgent


def load_player_model(state_dim, checkpoint_path="checkpoints/player_dqn.pth"):
    """加载训练好的玩家模型"""
    player_agent = PlayerDQNAgent(state_dim=state_dim)
    player_agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=player_agent.device))
    player_agent.epsilon = 0.0  # 冻结，不探索
    player_agent.target_net.load_state_dict(player_agent.policy_net.state_dict())
    print(f"[OK] Player model loaded from: {checkpoint_path}")
    return player_agent


def _load_frozen_npc(algorithm):
    map_lines, walls, npc_spawns, player_spawn, h, w = load_map_txt(MAP_FILE)
    npc_count = len(npc_spawns)

    npc_ckpt = f"checkpoints/npc_{algorithm}_tables.pkl"
    agents = build_agents(algorithm, npc_count)
    agents.q_tables = load_npc_tables(npc_ckpt)

    if hasattr(agents, "epsilon"):
        agents.epsilon = 0.0
    if hasattr(agents, "agents"):
        for ag in agents.agents:
            if hasattr(ag, "epsilon"):
                ag.epsilon = 0.0

    env = MultiNpcEnv(npc_spawns, player_spawn, walls, h, w)
    return env, agents


def _run_fixed_player_baseline(algorithm="q", episodes=100, player_mode="rule"):
    """
    通用基线测试：训练好的NPC vs 固定玩家模式
    player_mode: random / rule
    """
    env, agents = _load_frozen_npc(algorithm)

    steps_list = []
    success_list = []
    first_hit_list = []
    multi_hit_list = []
    final_hp_list = []

    title = f"{algorithm.upper()} NPC vs {player_mode.upper()} Player"
    print(f"\n========== {title} ==========")
    print(f"Episodes: {episodes}")

    for ep in range(episodes):
        states = env.reset()
        done = False
        steps = 0
        first_hit_step = None
        multi_hit_count = 0
        info = {"player_hp": 100, "last_damage": 0, "last_hits": 0}

        while not done and steps < MAX_STEPS:
            action_indices = agents.act(states)
            next_states, _, _, done, info = env.step(
                action_indices,
                player_mode=player_mode,
            )

            if info["last_hits"] > 0 and first_hit_step is None:
                first_hit_step = steps + 1
            if info["last_hits"] >= 2:
                multi_hit_count += 1

            states = next_states
            steps += 1

        steps_list.append(steps)
        success_list.append(1 if done else 0)
        first_hit_list.append(first_hit_step if first_hit_step is not None else MAX_STEPS)
        multi_hit_list.append(multi_hit_count)
        final_hp_list.append(max(0, info["player_hp"]))

        if (ep + 1) % 20 == 0:
            avg = np.mean(steps_list[-20:])
            print(f"Episode {ep+1}/{episodes}, avg_steps={avg:.2f}")

    print(f"\n--- Results ({episodes} episodes) ---")
    print(f"Avg steps to kill: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
    print(f"Success rate: {np.mean(success_list):.2%}")
    print(f"Avg first hit step: {np.mean(first_hit_list):.2f}")
    print(f"Avg multi-hit count: {np.mean(multi_hit_list):.2f}")
    print(f"Avg final HP: {np.mean(final_hp_list):.2f}")

    return {
        "steps": steps_list,
        "success": success_list,
        "first_hit_step": first_hit_list,
        "multi_hit_count": multi_hit_list,
        "final_hp": final_hp_list,
    }


def test_random_player_baseline(algorithm="q", episodes=100):
    """基线测试：训练好的NPC vs 随机玩家"""
    return _run_fixed_player_baseline(algorithm=algorithm, episodes=episodes, player_mode="random")


def test_rule_player_baseline(algorithm="q", episodes=100):
    """基线测试：训练好的NPC vs 规则玩家"""
    return _run_fixed_player_baseline(algorithm=algorithm, episodes=episodes, player_mode="rule")


def test_frozen_battle(algorithm="q", episodes=100, render=False):
    """
    测试训练好的NPC vs 训练好的玩家
    两者都冻结，不学习
    """
    map_lines, walls, npc_spawns, player_spawn, h, w = load_map_txt(MAP_FILE)
    npc_count = len(npc_spawns)

    npc_ckpt = f"checkpoints/npc_{algorithm}_tables.pkl"
    agents = build_agents(algorithm, npc_count)
    agents.q_tables = load_npc_tables(npc_ckpt)
    if hasattr(agents, "epsilon"):
        agents.epsilon = 0.0
    if hasattr(agents, "agents"):
        for ag in agents.agents:
            if hasattr(ag, "epsilon"):
                ag.epsilon = 0.0

    env = MultiNpcEnv(npc_spawns, player_spawn, walls, h, w)
    dummy_state = env.get_player_state()
    player_agent = load_player_model(state_dim=len(dummy_state))

    steps_list = []
    success_list = []
    first_hit_list = []
    multi_hit_list = []
    final_hp_list = []

    print(f"\n========== Testing {algorithm.upper()} NPC vs ADV Player ==========")
    print(f"Episodes: {episodes}")

    for ep in range(episodes):
        states = env.reset()
        done = False
        steps = 0
        first_hit_step = None
        multi_hit_count = 0

        while not done and steps < MAX_STEPS:
            action_indices = agents.act(states)
            player_state = env.get_player_state()
            player_action = player_agent.act(player_state)
            next_states, _, _, done, info = env.step(
                action_indices,
                player_action_index=player_action,
                player_mode="adv"
            )

            if info["last_hits"] > 0 and first_hit_step is None:
                first_hit_step = steps + 1
            if info["last_hits"] >= 2:
                multi_hit_count += 1

            states = next_states
            steps += 1

        steps_list.append(steps)
        success_list.append(1 if done else 0)
        first_hit_list.append(first_hit_step if first_hit_step is not None else MAX_STEPS)
        multi_hit_list.append(multi_hit_count)
        final_hp_list.append(max(0, info["player_hp"]))

        if (ep + 1) % 20 == 0:
            avg = np.mean(steps_list[-20:])
            print(f"Episode {ep+1}/{episodes}, avg_steps={avg:.2f}")

    print(f"\n--- Results ({episodes} episodes) ---")
    print(f"Avg steps to kill: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
    print(f"Success rate: {np.mean(success_list):.2%}")
    print(f"Avg first hit step: {np.mean(first_hit_list):.2f}")
    print(f"Avg multi-hit count: {np.mean(multi_hit_list):.2f}")
    print(f"Avg final HP: {np.mean(final_hp_list):.2f}")

    return {
        "steps": steps_list,
        "success": success_list,
        "first_hit_step": first_hit_list,
        "multi_hit_count": multi_hit_list,
        "final_hp": final_hp_list,
    }


def plot_comparison(results_dict, title_prefix="NPC vs Player"):
    labels = list(results_dict.keys())
    x = np.arange(len(labels))
    width = 0.25

    def mean_last20(key):
        vals = []
        for label in labels:
            arr = np.array(results_dict[label][key], dtype=float)
            vals.append(arr[-20:].mean() if len(arr) >= 20 else arr.mean())
        return np.array(vals)

    steps = mean_last20("steps")
    first_hit = mean_last20("first_hit_step")
    multi_hit = mean_last20("multi_hit_count")
    final_hp = mean_last20("final_hp")
    success = mean_last20("success")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(labels, steps, color=["gray", "tab:blue", "tab:orange"][:len(labels)])
    axes[0, 0].set_title("Final-20 Avg Steps")
    axes[0, 0].set_ylabel("Steps")

    axes[0, 1].bar(labels, first_hit, color=["gray", "tab:blue", "tab:orange"][:len(labels)])
    axes[0, 1].set_title("Final-20 Avg First Hit Step")
    axes[0, 1].set_ylabel("Step")

    axes[1, 0].bar(labels, multi_hit, color=["gray", "tab:blue", "tab:orange"][:len(labels)])
    axes[1, 0].set_title("Final-20 Avg Multi-Hit Count")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].bar(labels, final_hp, color=["gray", "tab:blue", "tab:orange"][:len(labels)])
    axes[1, 1].set_title("Final-20 Avg Final HP")
    axes[1, 1].set_ylabel("HP")

    fig.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(labels, success, color=["gray", "tab:blue", "tab:orange"][:len(labels)])
    plt.title("Final-20 Avg Success Rate")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.show()


def print_summary(results_dict):
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Player':<12} {'Steps':<12} {'Success':<12} {'FirstHit':<12} {'MultiHit':<12} {'FinalHP':<12}")
    print("-" * 60)
    for name, res in results_dict.items():
        steps = np.mean(res["steps"][-20:]) if len(res["steps"]) >= 20 else np.mean(res["steps"])
        success = np.mean(res["success"][-20:]) if len(res["success"]) >= 20 else np.mean(res["success"])
        first_hit = np.mean(res["first_hit_step"][-20:]) if len(res["first_hit_step"]) >= 20 else np.mean(res["first_hit_step"])
        multi_hit = np.mean(res["multi_hit_count"][-20:]) if len(res["multi_hit_count"]) >= 20 else np.mean(res["multi_hit_count"])
        final_hp = np.mean(res["final_hp"][-20:]) if len(res["final_hp"]) >= 20 else np.mean(res["final_hp"])
        print(f"{name:<12} {steps:<12.2f} {success:<12.2%} {first_hit:<12.2f} {multi_hit:<12.2f} {final_hp:<12.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="q", choices=["q", "sarsa"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--player-mode",
        type=str,
        default="all",
        choices=["all", "random", "rule", "adv"],
        help="选择只跑某一种玩家，或者全部都跑",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只输出数字结果，不画图",
    )
    args = parser.parse_args()

    results = {}

    if args.player_mode in ["all", "random"]:
        results["random"] = test_random_player_baseline(algorithm=args.algo, episodes=args.episodes)
    if args.player_mode in ["all", "rule"]:
        results["rule"] = test_rule_player_baseline(algorithm=args.algo, episodes=args.episodes)
    if args.player_mode in ["all", "adv"]:
        results["adv"] = test_frozen_battle(algorithm=args.algo, episodes=args.episodes)

    print_summary(results)

    if not args.no_plot and len(results) > 0:
        plot_comparison(results, title_prefix=f"{args.algo.upper()} NPC vs Players")
