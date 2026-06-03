import matplotlib
matplotlib.use('TkAgg')
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from config import MAP_FILE
from train import (
    train_npc_only,
    train_adv_with_frozen_npc,
    train_adversarial_two_stage,
    train_self_play,
)


def smooth(arr, window=10):
    arr = np.array(arr, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def print_metrics(prefix, metrics):
    steps = metrics["steps"]
    success = metrics["success"]
    final_hp = metrics["final_hp"]
    first_hit_step = metrics["first_hit_step"]
    multi_hit_count = metrics["multi_hit_count"]

    print(f"\n===== {prefix} =====")
    print(f"success rate: {sum(success)/len(success):.3f}")
    print(f"final-20 avg steps: {sum(steps[-20:])/len(steps[-20:]):.2f}")
    print(f"final-20 avg final hp: {sum(final_hp[-20:])/len(final_hp[-20:]):.2f}")
    print(f"final-20 avg first hit step: {sum(first_hit_step[-20:])/len(first_hit_step[-20:]):.2f}")
    print(f"final-20 avg multi-hit count: {sum(multi_hit_count[-20:])/len(multi_hit_count[-20:]):.2f}")


def plot_single_stage(metrics, title_prefix):
    steps = metrics["steps"]
    success = metrics["success"]
    first_hit_step = metrics["first_hit_step"]

    y_steps = smooth(steps, 10)
    y_success = smooth(success, 10)
    y_first_hit = smooth(first_hit_step, 10)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(y_steps) + 1), y_steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps to kill player")
    plt.title(f"{title_prefix} - Steps Curve")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(y_success) + 1), y_success)
    plt.xlabel("Episode")
    plt.ylabel("Success rate (moving avg)")
    plt.title(f"{title_prefix} - Success Curve")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(y_first_hit) + 1), y_first_hit)
    plt.xlabel("Episode")
    plt.ylabel("First hit step (moving avg)")
    plt.title(f"{title_prefix} - First Hit Curve")
    plt.tight_layout()
    plt.show()


def plot_two_stage_compare(npc_metrics, adv_metrics, algo_name):
    npc_steps = smooth(npc_metrics["steps"], 10)
    adv_steps = smooth(adv_metrics["steps"], 10)

    npc_success = smooth(npc_metrics["success"], 10)
    adv_success = smooth(adv_metrics["success"], 10)

    npc_first_hit = smooth(npc_metrics["first_hit_step"], 10)
    adv_first_hit = smooth(adv_metrics["first_hit_step"], 10)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(npc_steps) + 1), npc_steps, label="Stage1 NPC_ONLY")
    plt.plot(range(1, len(adv_steps) + 1), adv_steps, label="Stage2 ADV_ONLY")
    plt.xlabel("Episode")
    plt.ylabel("Steps to kill player")
    plt.title(f"{algo_name} Two-Stage Training: Steps Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(npc_success) + 1), npc_success, label="Stage1 NPC_ONLY")
    plt.plot(range(1, len(adv_success) + 1), adv_success, label="Stage2 ADV_ONLY")
    plt.xlabel("Episode")
    plt.ylabel("Success rate (moving avg)")
    plt.title(f"{algo_name} Two-Stage Training: Success Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(npc_first_hit) + 1), npc_first_hit, label="Stage1 NPC_ONLY")
    plt.plot(range(1, len(adv_first_hit) + 1), adv_first_hit, label="Stage2 ADV_ONLY")
    plt.xlabel("Episode")
    plt.ylabel("First hit step (moving avg)")
    plt.title(f"{algo_name} Two-Stage Training: First Hit Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_self_play_results(metrics, title):
    """
    Stage 3 自我博弈绘图函数。
    训练结束时自动弹出双子图，并标出NPC训练和DQN玩家训练的切换边界。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    steps_smoothed = smooth(metrics["steps"], window=20)
    hp_smoothed = smooth(metrics["final_hp"], window=20)

    # 1. 绘制击杀步数曲线
    ax1.plot(steps_smoothed, color="royalblue", label="Steps to Capture (Smoothed)")
    ax1.set_ylabel("Steps")
    ax1.set_title(f"{title} - Self-Play Evolutionary Process", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. 绘制玩家剩余生命值曲线
    ax2.plot(hp_smoothed, color="crimson", label="Player Final HP (Smoothed)")
    ax2.set_ylabel("Player HP")
    ax2.set_xlabel("Total Combined Episodes (Cumulative)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. 动态绘制大轮次交替的分界线
    for bound, label in metrics["iteration_bounds"]:
        # 确保分界线索引不超出平滑后的数组长度
        if bound < len(steps_smoothed):
            ax1.axvline(x=bound, color="darkorange", linestyle=":", linewidth=2, alpha=0.8)
            ax2.axvline(x=bound, color="darkorange", linestyle=":", linewidth=2, alpha=0.8)

            # 在图表上方标注当前属于哪一轮的什么阶段
            y_pos = max(steps_smoothed) * 0.85 if len(steps_smoothed) > 0 else 40
            ax1.text(bound + 5, y_pos, label, color="darkorange", fontsize=9, rotation=90, weight='bold')

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="two_stage",
        choices=["npc_only", "adv_only", "two_stage", "self_play"],  # 增加了 self_play 模式
        help="选择训练模式: npc_only(阶段1), adv_only(阶段2), two_stage(一键两阶段), self_play(阶段3自我博弈)",
    )
    parser.add_argument("--algo", type=str, default="q", choices=["q", "sarsa"])
    parser.add_argument("--map_file", type=str, default=MAP_FILE)
    parser.add_argument("--no-render", action="store_true", help="禁用Pygame图形渲染以加速训练")
    parser.add_argument("--no-plot", action="store_true", help="训练后不展示曲线图")
    parser.add_argument("--ckpt", type=str, default="checkpoints/npc_q_tables.pkl")
    parser.add_argument("--npc_player_mode", type=str, default="rule", choices=["random", "rule"])

    # 新增：专用于自我博弈模式的超参数（大迭代轮数）
    parser.add_argument("--sp_loops", type=int, default=3, help="自我博弈(军备竞赛)交替迭代的大轮次")

    args = parser.parse_args()
    render = not args.no_render
    plot_enabled = not args.no_plot

    map_file = args.map_file
    map_name = os.path.splitext(os.path.basename(map_file))[0]

    # ========================================================
    # 新增分支：一键启动 Stage 3 自我博弈
    # ========================================================
    if args.mode == "self_play":
        print(f"\n[启动] 开始执行 Stage 3 交互式自我博弈（Self-Play）演化流水线...")
        print(f"总计交替大轮次: {args.sp_loops} 轮 | 当前算法: {args.algo.upper()} | 地图: {map_name}")

        # 调用 train.py 中的新扩展函数
        sp_metrics = train_self_play(
            map_file=map_file,
            algorithm=args.algo,
            render=render,
            num_iterations=args.sp_loops,
            episodes_per_phase=500,  # 每一小阶段的训练 Episode 数量
            save_dir="checkpoints/self_play"
        )

        # 打印全流程汇总数据
        print_metrics(f"STAGE3 SELF-PLAY TOTAL SUMMARY ({args.algo.upper()})", sp_metrics)

        # 自动化绘制带有边界线的军备竞赛交替图
        if plot_enabled:
            plot_self_play_results(sp_metrics, f"MARL Encirclement ({args.algo.upper()})")

    # ========================================================
    # 保留原有的三种传统模式分支
    # ========================================================
    elif args.mode == "npc_only":
        _, metrics = train_npc_only(
            map_file=map_file,
            algorithm=args.algo,
            render=render,
            save_path=args.ckpt,
            player_mode=args.npc_player_mode,
        )
        print_metrics(f"STAGE1 NPC_ONLY ({args.algo.upper()} | {map_name})", metrics)

        if plot_enabled:
            plot_single_stage(metrics, f"{args.algo.upper()} NPC_ONLY - {map_name}")

    elif args.mode == "adv_only":
        _, metrics, _ = train_adv_with_frozen_npc(
            map_file=map_file,
            algorithm=args.algo,
            render=render,
            load_path=args.ckpt,
        )
        print_metrics(f"STAGE2 ADV_ONLY with FROZEN NPC ({args.algo.upper()} | {map_name})", metrics)

        if plot_enabled:
            plot_single_stage(metrics, f"{args.algo.upper()} ADV_ONLY - {map_name}")

    else:
        # 默认的 two_stage 模式
        _, npc_metrics, adv_metrics, _ = train_adversarial_two_stage(
            map_file=map_file,
            algorithm=args.algo,
            render=render,
            save_path=args.ckpt,
        )
        print_metrics(f"STAGE1 NPC ({args.algo.upper()} | {map_name})", npc_metrics)
        print_metrics(f"STAGE2 ADV PLAYER ({args.algo.upper()} | {map_name})", adv_metrics)

        if plot_enabled:
            # 兼容原有逻辑，分别展示单阶段图
            plot_single_stage(npc_metrics, f"{args.algo.upper()} STAGE1 NPC - {map_name}")
            plot_single_stage(adv_metrics, f"{args.algo.upper()} STAGE2 ADV PLAYER - {map_name}")