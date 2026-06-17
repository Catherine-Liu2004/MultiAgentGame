import os
import pickle
import pygame

from agents.q_learning_agent import IndependentQLearningAgents
from agents.random_agent import RandomAgents
from agents.sarsa_agent import IndependentSARSAAgents
from config import EPISODES, FRAME_DELAY_MS, MAX_STEPS
from env import MultiNpcEnv
from map_loader import load_map_txt
from player_dqn_agent import PlayerDQNAgent
from render_pygame import PygameRenderer


def build_agents(algorithm, npc_count):
    if algorithm == "q":
        return IndependentQLearningAgents(npc_count)
    if algorithm == "sarsa":
        return IndependentSARSAAgents(npc_count)
    if algorithm == "random":
        return RandomAgents(npc_count)
    raise ValueError("algorithm must be one of: q / sarsa / random")


def save_npc_tables(q_tables, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(q_tables, f)
    print(f"[OK] NPC q_tables saved to: {save_path}")


def load_npc_tables(load_path):
    with open(load_path, "rb") as f:
        q_tables = pickle.load(f)
    print(f"[OK] NPC q_tables loaded from: {load_path}")
    return q_tables


def train_agents(
    map_file,
    algorithm="q",
    render=True,
    player_mode="rule",
    freeze_npc=False,
    npc_q_tables=None,
):
    """
    player_mode:
        - "random": 随机玩家
        - "rule": 规则逃跑玩家
        - "adv": DQN 对抗玩家

    freeze_npc:
        - False: 正常训练 NPC
        - True : 冻结 NPC，只训练 player（若 player_mode=adv）

    npc_q_tables:
        - 传入预训练好的 NPC q_tables，用于冻结 NPC 或继续训练

    返回:
        q_tables, metrics
        或
        q_tables, metrics, player_agent
    """
    import torch
    import os
    
    map_lines, walls, npc_spawns, player_spawn, h, w = load_map_txt(map_file)
    npc_count = len(npc_spawns)

    env = MultiNpcEnv(npc_spawns, player_spawn, walls, h, w)
    agents = build_agents(algorithm, npc_count)

    # 如果传入了已有的 NPC q_tables，就直接加载
    if npc_q_tables is not None and hasattr(agents, "q_tables"):
        agents.q_tables = npc_q_tables
            # 如果冻结 NPC，则关闭探索，让对手真正稳定
        if freeze_npc:
            if hasattr(agents, "epsilon"):
                agents.epsilon = 0.0
            if hasattr(agents, "agents"):
                for ag in agents.agents:
                    if hasattr(ag, "epsilon"):
                        ag.epsilon = 0.0

    use_adversarial_player = (player_mode == "adv")
    player_agent = None
    if use_adversarial_player:
        dummy_state = env.get_player_state()
        player_agent = PlayerDQNAgent(state_dim=len(dummy_state))

    # 原始指标
    capture_steps_list = []
    success_list = []

    # 新增指标
    final_hp_list = []
    first_hit_step_list = []
    multi_hit_count_list = []

    renderer = PygameRenderer(map_lines, title=f"Coop Hunt - {algorithm.upper()}") if render else None

    for ep in range(EPISODES):
        states = env.reset()
        done = False
        steps = 0
        info = {"player_hp": 100, "last_damage": 0, "last_hits": 0}

        first_hit_step = None
        multi_hit_count = 0

        if algorithm == "sarsa":
            action_indices = agents.act(states)

        player_state = env.get_player_state() if use_adversarial_player else None

        while not done and steps < MAX_STEPS:
            if renderer:
                if not renderer.process_events():
                    renderer.close()
                    metrics = {
                        "steps": capture_steps_list,
                        "success": success_list,
                        "final_hp": final_hp_list,
                        "first_hit_step": first_hit_step_list,
                        "multi_hit_count": multi_hit_count_list,
                    }
                    if player_agent is not None:
                        # 保存玩家模型
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save(player_agent.policy_net.state_dict(), "checkpoints/player_dqn.pth")
                        print(f"[OK] Player model saved to: checkpoints/player_dqn.pth")
                        return agents.q_tables, metrics, player_agent
                    return agents.q_tables, metrics

                sr = (sum(success_list) / len(success_list)) if success_list else 0.0
                moving_obstacles = [o["pos"] for o in env.dynamic_obstacles] if hasattr(env, "dynamic_obstacles") else []
                title_algo = f"{algorithm.upper()} + {player_mode.upper()}_PLAYER"
                if freeze_npc:
                    title_algo += " (FROZEN_NPC)"

                renderer.draw(
                    npc_positions=env.npc_positions,
                    player_pos=env.player_position,
                    moving_obstacles=moving_obstacles,
                    episode=ep + 1,
                    step=steps,
                    algo=title_algo,
                    hp=info["player_hp"],
                    last_damage=info["last_damage"],
                    last_hits=info["last_hits"],
                    success_rate=sr,
                )

            player_action = None
            if use_adversarial_player:
                player_action = player_agent.act(player_state)

            if algorithm == "q":
                action_indices = agents.act(states)
                next_states, rewards, player_reward, done, info = env.step(
                    action_indices,
                    player_action_index=player_action,
                    player_mode=player_mode,
                )

                if not freeze_npc:
                    agents.update(states, action_indices, rewards, next_states)

            elif algorithm == "sarsa":
                next_states, rewards, player_reward, done, info = env.step(
                    action_indices,
                    player_action_index=player_action,
                    player_mode=player_mode,
                )
                next_action_indices = agents.act(next_states)

                if not freeze_npc:
                    agents.update(states, action_indices, rewards, next_states, next_action_indices)

                action_indices = next_action_indices

            else:  # random NPC
                action_indices = agents.act(states)
                next_states, rewards, player_reward, done, info = env.step(
                    action_indices,
                    player_action_index=player_action,
                    player_mode=player_mode,
                )
                agents.update()

            # 新增指标
            if info["last_hits"] > 0 and first_hit_step is None:
                first_hit_step = steps + 1

            if info["last_hits"] >= 2:
                multi_hit_count += 1

            # 只在 adv 模式下训练 player
            if use_adversarial_player:
                next_player_state = info["player_state"]

                if (steps + 1) >= MAX_STEPS and not done:
                    player_reward += 50.0

                player_agent.remember(player_state, player_action, player_reward, next_player_state, done)
                player_agent.replay()
                player_state = next_player_state

            states = next_states
            steps += 1

            if renderer:
                pygame.time.delay(FRAME_DELAY_MS)

        capture_steps_list.append(steps)
        success_list.append(1 if done else 0)

        final_hp_list.append(info["player_hp"])
        first_hit_step_list.append(first_hit_step if first_hit_step is not None else MAX_STEPS)
        multi_hit_count_list.append(multi_hit_count)

        if (ep + 1) % 20 == 0:
            recent_success = sum(success_list[-20:]) / len(success_list[-20:])
            recent_hp = sum(final_hp_list[-20:]) / len(final_hp_list[-20:])
            recent_first_hit = sum(first_hit_step_list[-20:]) / len(first_hit_step_list[-20:])
            recent_multi_hit = sum(multi_hit_count_list[-20:]) / len(multi_hit_count_list[-20:])

            msg = (
                f"[{algorithm.upper()} | {player_mode.upper()} | freeze_npc={freeze_npc}] "
                f"Episode {ep + 1}, "
                f"steps={steps}, hp={info['player_hp']}, "
                f"recent_success@20={recent_success:.2f}, "
                f"recent_final_hp@20={recent_hp:.2f}, "
                f"recent_first_hit@20={recent_first_hit:.2f}, "
                f"recent_multi_hit@20={recent_multi_hit:.2f}"
            )
            if use_adversarial_player and player_agent is not None:
                msg += f", player_epsilon={player_agent.epsilon:.3f}"
            print(msg)

    if renderer:
        renderer.close()

    metrics = {
        "steps": capture_steps_list,
        "success": success_list,
        "final_hp": final_hp_list,
        "first_hit_step": first_hit_step_list,
        "multi_hit_count": multi_hit_count_list,
    }

    if player_agent is not None:
        # 保存玩家模型
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(player_agent.policy_net.state_dict(), "checkpoints/player_dqn.pth")
        print(f"[OK] Player model saved to: checkpoints/player_dqn.pth")
        return agents.q_tables, metrics, player_agent
    return agents.q_tables, metrics


def train_npc_only(
    map_file,
    algorithm="q",
    render=False,
    save_path="checkpoints/npc_q_tables.pkl",
    player_mode="rule",
):
    """
    Stage 1:
    用指定玩家模式训练 NPC，并保存 NPC q_tables
    player_mode:
        - random: 随机玩家
        - rule: 规则逃跑玩家
        - adv: DQN 对抗玩家（一般不建议用于 NPC 初训）
    """
    q_tables, metrics = train_agents(
        map_file=map_file,
        algorithm=algorithm,
        render=render,
        player_mode=player_mode,
        freeze_npc=False,
        npc_q_tables=None,
    )

    if hasattr(q_tables, "__class__"):
        save_npc_tables(q_tables, save_path)

    return q_tables, metrics


def train_adv_with_frozen_npc(
    map_file,
    algorithm="q",
    render=False,
    load_path="checkpoints/npc_q_tables.pkl",
):
    """
    Stage 2:
    读取已训练好的 NPC q_tables，冻结 NPC，只训练 ADV player
    """
    npc_q_tables = load_npc_tables(load_path)

    q_tables, metrics, player_agent = train_agents(
        map_file=map_file,
        algorithm=algorithm,
        render=render,
        player_mode="adv",
        freeze_npc=True,
        npc_q_tables=npc_q_tables,
    )

    return q_tables, metrics, player_agent


def train_adversarial_two_stage(
    map_file,
    algorithm="q",
    render=False,
    save_path="checkpoints/npc_q_tables.pkl",
):
    """
    一键两阶段：
    Stage 1 训练 NPC
    Stage 2 冻结 NPC，训练 ADV player
    """
    print("\n========== Stage 1: Train NPC with RULE player ==========")
    q_tables, npc_metrics = train_npc_only(
        map_file=map_file,
        algorithm=algorithm,
        render=False,
        save_path=save_path,
    )

    print("\n========== Stage 2: Freeze NPC and train ADV player ==========")
    q_tables, adv_metrics, player_agent = train_adv_with_frozen_npc(
        map_file=map_file,
        algorithm=algorithm,
        render=render,
        load_path=save_path,
    )

    return q_tables, npc_metrics, adv_metrics, player_agent


def train_self_play(
        map_file,
        algorithm="q",
        render=False,
        num_iterations=3,  # 自我博弈交互轮数
        episodes_per_phase=500,  # 每个子阶段的训练 Episode 数
        save_dir="checkpoints/self_play"
):
    """
    Stage 3: 自我博弈 (Self-Play) 交替迭代训练
    Phase A: 冻结 DQN 玩家，训练 NPC 针对聪明玩家
    Phase B: 冻结 NPC，训练 DQN 玩家对抗变强的 NPC
    """
    import numpy as np
    import torch
    from map_loader import load_map_txt
    from player_dqn_agent import PlayerDQNAgent

    os.makedirs(save_dir, exist_ok=True)

    # 1. 基础环境参数准备
    map_lines, walls, npc_spawns, player_spawn, h, w = load_map_txt(map_file)
    npc_count = len(npc_spawns)
    state_dim = npc_count * 2 + 2 

    # 2. 初始化或加载已有的智能体
    print("\n[Self-Play] 正在初始化智能体架构...")
    npc_agents = build_agents(algorithm, npc_count)
    player_agent = PlayerDQNAgent(state_dim=state_dim)

    # 聚合记录总指标
    sp_metrics = {
        "iteration_bounds": [],  # 记录每次迭代切换的边界
        "steps": [],
        "success": [],
        "final_hp": [],
        "first_hit_step": [],
        "multi_hit_count": []
    }

    total_episodes_counter = 0

    for iter_idx in range(1, num_iterations + 1):
        print(f"\n========================================================")
        print(f" 开启自我博弈第 {iter_idx} / {num_iterations} 轮大迭代")
        print(f"========================================================")

        # ----------------------------------------------------
        # Phase A: 冻结 DQN 玩家策略，解冻并单独训练 NPC
        # ----------------------------------------------------
        print(f"\n>>> [Phase A] 训练 NPC (当前大轮次: {iter_idx}) | DQN玩家策略已锁定")
       

        _, metrics_a, player_agent = train_agents(
            map_file=map_file,
            algorithm=algorithm,
            render=render,
            player_mode="adv",  # 使用学习型 DQN 玩家作为对手
            freeze_npc=False,  # 解冻 NPC
            npc_q_tables=npc_agents.q_tables if hasattr(npc_agents, 'q_tables') else None,
            
        )

        # 同步当前的 NPC 策略
        if hasattr(npc_agents, 'q_tables') and hasattr(metrics_a, '__class__'):
            npc_agents.q_tables = _get_q_tables_from_result(metrics_a, algorithm)

            # 保存本阶段成果
        npc_path = os.path.join(save_dir, f"npc_{algorithm}_iter{iter_idx}.pkl")
        save_npc_tables(npc_agents.q_tables, npc_path)

        # 拼接指标
        _append_metrics(sp_metrics, metrics_a)
        total_episodes_counter += len(metrics_a["steps"])
        sp_metrics["iteration_bounds"].append((total_episodes_counter, f"Iter{iter_idx}_NPC_Done"))

        # ----------------------------------------------------
        # Phase B: 冻结 NPC 策略，解冻并单独训练 DQN 玩家
        # ----------------------------------------------------
        print(f"\n>>> [Phase B] 训练 DQN 玩家 (当前大轮次: {iter_idx}) | NPC策略已锁定")

        _, metrics_b, player_agent = train_agents(
            map_file=map_file,
            algorithm=algorithm,
            render=render,
            player_mode="adv",
            freeze_npc=True,  # 锁定刚才的 NPC 策略
            npc_q_tables=npc_agents.q_tables if hasattr(npc_agents, 'q_tables') else None,
        )

        # 保存本阶段玩家模型
        player_path = os.path.join(save_dir, f"player_dqn_iter{iter_idx}.pth")
        torch.save(player_agent.policy_net.state_dict(), player_path)

        # 拼接指标
        _append_metrics(sp_metrics, metrics_b)
        total_episodes_counter += len(metrics_b["steps"])
        sp_metrics["iteration_bounds"].append((total_episodes_counter, f"Iter{iter_idx}_Player_Done"))

    print(f"\n[OK] 自我博弈训练全部结束！模型已保存在 {save_dir}/")
    return sp_metrics


def _append_metrics(target, source):
    for key in ["steps", "success", "final_hp", "first_hit_step", "multi_hit_count"]:
        if key in source:
            target[key].extend(source[key])


def _get_q_tables_from_result(train_result, algorithm):
    if isinstance(train_result, dict):
        return None
    return train_result
