import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

from config import EPISODES, MAP_FILE
from train import train_agents


def run_once(algo, seed, player_mode):
    random.seed(seed)
    np.random.seed(seed)

    result = train_agents(
        map_file=MAP_FILE,
        algorithm=algo,
        render=False,
        player_mode=player_mode,
    )

    if len(result) == 3:
        _, metrics, _ = result
    else:
        _, metrics = result

    return {
        "steps": np.array(metrics["steps"], dtype=float),
        "success": np.array(metrics["success"], dtype=float),
        "final_hp": np.array(metrics["final_hp"], dtype=float),
        "first_hit_step": np.array(metrics["first_hit_step"], dtype=float),
        "multi_hit_count": np.array(metrics["multi_hit_count"], dtype=float),
    }


def summarize(arr_2d):
    mean = arr_2d.mean(axis=0)
    std = arr_2d.std(axis=0)
    return mean, std


def moving_avg(x, window=20):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def evaluate_mode(algos, seeds, player_mode):
    all_metrics = {
        a: {
            "steps": [],
            "success": [],
            "final_hp": [],
            "first_hit_step": [],
            "multi_hit_count": [],
        }
        for a in algos
    }

    mode_name = f"{player_mode.upper()}_PLAYER"

    for a in algos:
        for s in seeds:
            print(f"Running seed={s} for {a.upper()} under {mode_name}...")
            metrics = run_once(a, s, player_mode)

            for k in all_metrics[a].keys():
                all_metrics[a][k].append(metrics[k])

    for a in algos:
        for k in all_metrics[a].keys():
            all_metrics[a][k] = np.vstack(all_metrics[a][k])

    return all_metrics


def plot_metrics(player_mode, metrics_dict, algos, colors):
    x = np.arange(1, EPISODES + 1)

    plt.figure(figsize=(10, 5))
    for a in algos:
        m, sd = summarize(metrics_dict[a]["steps"])
        plt.plot(x, m, label=a.upper(), color=colors[a])
        plt.fill_between(x, m - sd, m + sd, color=colors[a], alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Steps to kill player")
    plt.title(f"Algorithm Comparison under {player_mode.upper()}_PLAYER (mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player-mode",
        type=str,
        default="all",
        choices=["all", "random", "rule", "adv"],
        help="Run all player modes or only one mode",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default="random,q,sarsa",
        help="Comma-separated algorithms to compare, e.g. random,q,sarsa",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated random seeds, e.g. 0,1,2",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all matplotlib plots and only print summaries",
    )
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    colors = {"random": "gray", "q": "tab:blue", "sarsa": "tab:orange"}
    player_modes = [args.player_mode] if args.player_mode != "all" else ["random", "rule", "adv"]

    mode_results = {}
    for player_mode in player_modes:
        mode_results[player_mode] = evaluate_mode(algos, seeds, player_mode=player_mode)

    if not args.no_plots:
        if args.player_mode == "all":
            # 三档玩家
            random_player_metrics = mode_results["random"]
            rule_player_metrics = mode_results["rule"]
            adv_player_metrics = mode_results["adv"]

            x = np.arange(1, EPISODES + 1)

            # 图1：规则玩家下 steps 曲线
            plt.figure(figsize=(10, 5))
            for a in algos:
                m, sd = summarize(rule_player_metrics[a]["steps"])
                plt.plot(x, m, label=a.upper(), color=colors.get(a, None))
                plt.fill_between(x, m - sd, m + sd, color=colors.get(a, None), alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Steps to kill player")
            plt.title("Algorithm Comparison under RULE_PLAYER (mean ± std)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 图2：对抗玩家下 steps 曲线
            plt.figure(figsize=(10, 5))
            for a in algos:
                m, sd = summarize(adv_player_metrics[a]["steps"])
                plt.plot(x, m, label=a.upper(), color=colors.get(a, None))
                plt.fill_between(x, m - sd, m + sd, color=colors.get(a, None), alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Steps to kill player")
            plt.title("Algorithm Comparison under ADV_PLAYER (mean ± std)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 图3：Q-learning 在三档玩家下的对比
            plt.figure(figsize=(10, 5))
            for player_mode, metrics_dict in [
                ("RANDOM", random_player_metrics),
                ("RULE", rule_player_metrics),
                ("ADV", adv_player_metrics),
            ]:
                if "q" not in metrics_dict:
                    continue
                m, _ = summarize(metrics_dict["q"]["steps"])
                plt.plot(x, m, label=f"Q + {player_mode}_PLAYER")
            plt.xlabel("Episode")
            plt.ylabel("Steps to kill player")
            plt.title("Q-learning under Different Player Difficulties")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 图4：成功率滑动平均（Q-learning 三档对比）
            plt.figure(figsize=(10, 5))
            win = 20
            for label, metrics_dict in [
                ("Q + RANDOM", random_player_metrics),
                ("Q + RULE", rule_player_metrics),
                ("Q + ADV", adv_player_metrics),
            ]:
                if "q" not in metrics_dict:
                    continue
                m, _ = summarize(metrics_dict["q"]["success"])
                sm = moving_avg(m, window=win)
                xs = np.arange(win, EPISODES + 1) if len(m) >= win else np.arange(1, len(sm) + 1)
                plt.plot(xs, sm, label=label)
            plt.xlabel("Episode")
            plt.ylabel("Success rate (moving avg)")
            plt.title("Q-learning Success Rate under Different Player Difficulties")
            plt.legend()
            plt.tight_layout()
            plt.show()

        else:
            # 单一玩家模式：只画该模式下的算法对比
            metrics_dict = mode_results[args.player_mode]
            plot_metrics(args.player_mode, metrics_dict, algos, colors)

    # 汇总表
    print("\n=== Final 20 Episodes Summary ===")
    for mode_name, metrics_dict in mode_results.items():
        print(f"\n--- {mode_name.upper()}_PLAYER ---")
        for a in algos:
            mean_steps = metrics_dict[a]["steps"].mean(axis=0)
            mean_success = metrics_dict[a]["success"].mean(axis=0)
            mean_final_hp = metrics_dict[a]["final_hp"].mean(axis=0)
            mean_first_hit = metrics_dict[a]["first_hit_step"].mean(axis=0)
            mean_multi_hit = metrics_dict[a]["multi_hit_count"].mean(axis=0)

            final_steps_20 = mean_steps[-20:].mean()
            final_success_20 = mean_success[-20:].mean()
            final_hp_20 = mean_final_hp[-20:].mean()
            final_first_hit_20 = mean_first_hit[-20:].mean()
            final_multi_hit_20 = mean_multi_hit[-20:].mean()

            print(
                f"{a.upper():7s} | "
                f"final-20 avg steps: {final_steps_20:6.2f} | "
                f"final-20 success: {final_success_20:.2%} | "
                f"final-20 avg hp: {final_hp_20:6.2f} | "
                f"final-20 first hit: {final_first_hit_20:6.2f} | "
                f"final-20 multi-hit: {final_multi_hit_20:6.2f}"
            )
