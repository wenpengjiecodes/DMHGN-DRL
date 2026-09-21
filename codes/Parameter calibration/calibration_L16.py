

import os
import sys
import csv
import glob
import time
import random
import importlib.util

import numpy as np
import torch


# ============================================================
# 1. 路径配置
# ============================================================

ALGO_FILE = r"../my/V1_01.py"
TRAIN_DIR = r"../TestData/train_canshu"
VAL_DIR = r"../TestData/yanzheng1"
SAVE_ROOT = r"./param_tuning_results_topk"   # 与第一轮共用结果目录

EPISODES_PER_EXP = 200
SAVE_INTERVAL = 50
EVAL_INTERVAL = 50
UPDATE_INTERVAL = 8
VAL_RUNS = 1
BASE_SEED = 2026


# ============================================================
# 2. 工具函数（与第一轮相同）
# ============================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_algo_module(algo_file: str):
    algo_file = os.path.abspath(algo_file)
    if not os.path.exists(algo_file):
        raise FileNotFoundError(f"找不到原始训练代码文件: {algo_file}")
    module_name = "algo_module_for_param_tuning"
    spec = importlib.util.spec_from_file_location(module_name, algo_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Config"):
        raise AttributeError("原始训练代码中未找到 Config 类。")
    if not hasattr(module, "PPOAgent"):
        raise AttributeError("原始训练代码中未找到 PPOAgent 类。")
    return module


def get_json_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"目录下没有找到 json 文件: {data_dir}")
    return files


def make_exp_name(exp_id, lr, gamma, entropy_coef, topk_ratio):
    lr_str = f"{lr:.0e}".replace("+", "")
    gamma_str = str(gamma).replace(".", "p")
    ent_str = str(entropy_coef).replace(".", "p")
    topk_str = str(topk_ratio).replace(".", "p")
    return (
        f"Exp{exp_id:02d}"
        f"_lr{lr_str}"
        f"_g{gamma_str}"
        f"_ent{ent_str}"
        f"_topk{topk_str}"
    )


def safe_float(x, default=float("inf")):
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# 3. L16(4^4) 正交实验设计表
# ============================================================

def build_l16_design():
    """
    L16(4^4) 正交实验表。
    四个因子各 4 个水平，共 16 组实验。
    """
    # 水平取值（基于第一轮结果调整）
    lr_levels = {
        1: 1e-4,
        2: 2e-4,
        3: 1e-3,
        4: 2e-3,
    }

    gamma_levels = {
        1: 0.90,
        2: 0.95,
        3: 0.99,
        4: 1.00,
    }

    entropy_levels = {
        1: 0.001,
        2: 0.010,
        3: 0.050,
        4: 0.1,
    }

    topk_levels = {
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 0.4,
    }

    # 标准 L16(4^4) 正交表
    # 列顺序：[A, B, C, D]
    l16_level_table = [
        [1, 1, 1, 1],   # 1
        [1, 2, 2, 2],   # 2
        [1, 3, 3, 3],   # 3
        [1, 4, 4, 4],   # 4
        [2, 1, 2, 3],   # 5
        [2, 2, 1, 4],   # 6
        [2, 3, 4, 1],   # 7
        [2, 4, 3, 2],   # 8
        [3, 1, 3, 4],   # 9
        [3, 2, 4, 3],   # 10
        [3, 3, 1, 2],   # 11
        [3, 4, 2, 1],   # 12
        [4, 1, 4, 2],   # 13
        [4, 2, 3, 1],   # 14
        [4, 3, 2, 4],   # 15
        [4, 4, 1, 3],   # 16
    ]

    design = []
    for i, row in enumerate(l16_level_table, start=1):
        a, b, c, d = row
        design.append({
            "exp_id": i,
            "A_level": a,
            "B_level": b,
            "C_level": c,
            "D_level": d,
            "lr": lr_levels[a],
            "gamma": gamma_levels[b],
            "entropy_coef": entropy_levels[c],
            "top_k_ratio": topk_levels[d],
        })

    return design


def save_design_table(design, out_path):
    fieldnames = [
        "exp_id",
        "A_level", "B_level", "C_level", "D_level",
        "lr", "gamma", "entropy_coef", "top_k_ratio",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in design:
            writer.writerow(row)


# ============================================================
# 4. 以下与 JiaoZhun.py 完全相同：阶段模型评分、agent 封装、日志、评估
# ============================================================

def init_checkpoint_score_files(exp_dir):
    checkpoint_scores_csv = os.path.join(exp_dir, "checkpoint_scores.csv")
    checkpoint_detail_csv = os.path.join(exp_dir, "checkpoint_validation_details.csv")

    score_fieldnames = [
        "saved_episode", "model_path", "eval_runs", "checkpoint_validation_time_sec",
        "mean_Cmax", "mean_Total_Energy", "mean_Avg_Fatigue",
        "mean_Cmax_ratio", "mean_TEC_ratio", "mean_Favg_ratio", "mean_score",
        "exp_id", "A_level", "B_level", "C_level", "D_level",
        "lr", "gamma", "entropy_coef", "top_k_ratio",
    ]
    detail_fieldnames = [
        "saved_episode", "model_path", "instance", "eval_runs", "validation_time_sec",
        "Cmax", "Total_Energy", "Avg_Fatigue",
        "Cmax_ratio", "TEC_ratio", "Favg_ratio", "score",
        "exp_id", "A_level", "B_level", "C_level", "D_level",
        "lr", "gamma", "entropy_coef", "top_k_ratio",
    ]

    with open(checkpoint_scores_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=score_fieldnames)
        writer.writeheader()
    with open(checkpoint_detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()

    return checkpoint_scores_csv, checkpoint_detail_csv


def evaluate_saved_checkpoint_model(agent, saved_episode, model_path):
    if not getattr(agent, "enable_checkpoint_eval", False):
        return None

    val_files = getattr(agent, "checkpoint_eval_val_files", None)
    eval_runs = int(getattr(agent, "checkpoint_eval_runs", 1))
    exp_info = getattr(agent, "checkpoint_exp_info", None)
    checkpoint_scores_csv = getattr(agent, "checkpoint_scores_csv", None)
    checkpoint_detail_csv = getattr(agent, "checkpoint_detail_csv", None)

    if not val_files or exp_info is None or checkpoint_scores_csv is None or checkpoint_detail_csv is None:
        print("⚠️ 阶段模型评分配置不完整，跳过该模型评分。")
        return None

    detail_rows = []
    all_start_time = time.time()

    for json_path in val_files:
        instance_name = os.path.basename(json_path)
        cmax_runs, tec_runs, favg_runs = [], [], []
        ref_cmax = ref_tec = ref_favg = None

        instance_eval_start = time.time()
        for _ in range(eval_runs):
            agent.reset_instance(json_path)
            ref_cmax = getattr(agent.config, "ref_cmax", None)
            ref_tec = getattr(agent.config, "ref_tec", None)
            ref_favg = getattr(agent.config, "ref_favg", None)
            with torch.no_grad():
                _, metrics = agent.run_one_episode(episode=0, deterministic=True, collect_experience=False)
            cmax_runs.append(safe_float(metrics.get("makespan")))
            tec_runs.append(safe_float(metrics.get("energy")))
            favg_runs.append(safe_float(metrics.get("fatigue_mean")))

        validation_time_sec = time.time() - instance_eval_start
        mean_cmax = float(np.mean(cmax_runs))
        mean_tec = float(np.mean(tec_runs))
        mean_favg = float(np.mean(favg_runs))

        cmax_ratio = mean_cmax / max(1e-6, float(ref_cmax)) if ref_cmax is not None else mean_cmax / max(1e-6, float(getattr(agent, "cmax_norm_factor", 1.0)))
        tec_ratio = mean_tec / max(1e-6, float(ref_tec)) if ref_tec is not None else mean_tec / max(1e-6, float(getattr(agent, "energy_norm_factor", 1.0)))
        favg_ratio = mean_favg / max(1e-6, float(ref_favg)) if ref_favg is not None else mean_favg

        score = 0.5 * cmax_ratio + 0.3 * tec_ratio + 0.2 * favg_ratio

        detail_rows.append({
            "saved_episode": saved_episode, "model_path": model_path,
            "instance": instance_name, "eval_runs": eval_runs,
            "validation_time_sec": validation_time_sec,
            "Cmax": mean_cmax, "Total_Energy": mean_tec, "Avg_Fatigue": mean_favg,
            "Cmax_ratio": cmax_ratio, "TEC_ratio": tec_ratio, "Favg_ratio": favg_ratio,
            "score": score,
            "exp_id": exp_info["exp_id"], "A_level": exp_info["A_level"],
            "B_level": exp_info["B_level"], "C_level": exp_info["C_level"],
            "D_level": exp_info["D_level"],
            "lr": exp_info["lr"], "gamma": exp_info["gamma"],
            "entropy_coef": exp_info["entropy_coef"], "top_k_ratio": exp_info["top_k_ratio"],
        })

    total_checkpoint_validation_time = time.time() - all_start_time

    mean_cmax = float(np.mean([r["Cmax"] for r in detail_rows]))
    mean_tec = float(np.mean([r["Total_Energy"] for r in detail_rows]))
    mean_favg = float(np.mean([r["Avg_Fatigue"] for r in detail_rows]))
    mean_cmax_ratio = float(np.mean([r["Cmax_ratio"] for r in detail_rows]))
    mean_tec_ratio = float(np.mean([r["TEC_ratio"] for r in detail_rows]))
    mean_favg_ratio = float(np.mean([r["Favg_ratio"] for r in detail_rows]))
    mean_score = float(np.mean([r["score"] for r in detail_rows]))

    detail_fieldnames = list(detail_rows[0].keys())
    with open(checkpoint_detail_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        for row in detail_rows:
            writer.writerow(row)

    score_fieldnames = [
        "saved_episode", "model_path", "eval_runs", "checkpoint_validation_time_sec",
        "mean_Cmax", "mean_Total_Energy", "mean_Avg_Fatigue",
        "mean_Cmax_ratio", "mean_TEC_ratio", "mean_Favg_ratio", "mean_score",
        "exp_id", "A_level", "B_level", "C_level", "D_level",
        "lr", "gamma", "entropy_coef", "top_k_ratio",
    ]
    summary_row = {
        "saved_episode": saved_episode, "model_path": model_path,
        "eval_runs": eval_runs,
        "checkpoint_validation_time_sec": total_checkpoint_validation_time,
        "mean_Cmax": mean_cmax, "mean_Total_Energy": mean_tec, "mean_Avg_Fatigue": mean_favg,
        "mean_Cmax_ratio": mean_cmax_ratio, "mean_TEC_ratio": mean_tec_ratio,
        "mean_Favg_ratio": mean_favg_ratio, "mean_score": mean_score,
        "exp_id": exp_info["exp_id"], "A_level": exp_info["A_level"],
        "B_level": exp_info["B_level"], "C_level": exp_info["C_level"],
        "D_level": exp_info["D_level"],
        "lr": exp_info["lr"], "gamma": exp_info["gamma"],
        "entropy_coef": exp_info["entropy_coef"], "top_k_ratio": exp_info["top_k_ratio"],
    }

    with open(checkpoint_scores_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=score_fieldnames)
        writer.writerow(summary_row)

    print(f"阶段模型评分已保存: episode={saved_episode}, mean_score={mean_score:.6f}")
    return summary_row


def build_tuned_agent_class(algo_module):
    BasePPOAgent = algo_module.PPOAgent
    Config = algo_module.Config

    class TunedPPOAgent(BasePPOAgent):
        def set_tuning_params(self, lr, gamma, entropy_coef, top_k_ratio):
            self.tuning_lr = float(lr)
            self.tuning_gamma = float(gamma)
            self.tuning_entropy_coef = float(entropy_coef)
            self.tuning_top_k_ratio = float(top_k_ratio)
            self._apply_tuning_params_to_config(self.config)
            if hasattr(self, "return_calc"):
                self.return_calc.gamma = self.tuning_gamma
            if hasattr(self, "optimizer"):
                for group in self.optimizer.param_groups:
                    group["lr"] = self.tuning_lr

        def _apply_tuning_params_to_config(self, config):
            config.actor_lr = self.tuning_lr
            config.critic_lr = self.tuning_lr
            config.hdgat_lr = self.tuning_lr
            config.gamma = self.tuning_gamma
            config.entropy_coef = self.tuning_entropy_coef
            config.top_k_ratio = self.tuning_top_k_ratio

        def reset_instance(self, json_path):
            new_config = Config(json_path=json_path)
            self._apply_tuning_params_to_config(new_config)
            self.config = new_config
            self.hdgat.config = new_config
            self.actor.config = new_config
            self.critic.config = new_config
            self.current_instance_name = os.path.basename(json_path)
            self._compute_dynamic_parameters()
            self._initialize_from_json()
            if hasattr(self, "return_calc"):
                self.return_calc.gamma = self.tuning_gamma

        def _compute_top_k_actions(self):
            top_k_ratio = getattr(self.config, "top_k_ratio", 0.2)
            base_top_k = int(self.total_ops * top_k_ratio)
            min_top_k, max_top_k = 5, 100
            top_k = max(min_top_k, base_top_k)
            top_k = min(max_top_k, top_k)
            return top_k

        def save_models(self, path="saved_models/", episode_num=None):
            os.makedirs(path, exist_ok=True)
            history_episode_max = max(self.training_history.get("episode", [0]) or [0])
            trained_episodes = max(int(getattr(self, "trained_episodes", 0)), int(history_episode_max))

            if episode_num is None or str(episode_num) == "latest":
                episode_tag = str(trained_episodes)
            elif str(episode_num) in ["best_val", "final"]:
                print(f"跳过额外模型保存: {episode_num}。当前参数校准只保存每 {SAVE_INTERVAL} 代模型。")
                return None
            else:
                episode_tag = str(episode_num)

            try:
                episode_int = int(episode_tag)
            except ValueError:
                print(f"跳过非数字模型标签: {episode_tag}")
                return None

            if episode_int <= 0 or episode_int % SAVE_INTERVAL != 0:
                return None

            if not hasattr(self, "_saved_interval_episodes"):
                self._saved_interval_episodes = set()
            if episode_int in self._saved_interval_episodes:
                return None
            self._saved_interval_episodes.add(episode_int)

            save_path = os.path.join(path, f"ppo_agent_episode_{episode_int}.pth")
            if os.path.exists(save_path):
                idx = 1
                while True:
                    alt_save_path = os.path.join(path, f"ppo_agent_episode_{episode_int}_dup{idx}.pth")
                    if not os.path.exists(alt_save_path):
                        save_path = alt_save_path
                        break
                    idx += 1

            checkpoint = {
                "hdgat_state_dict": self.hdgat.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config_dict": dict(self.config.__dict__),
                "best_makespan": getattr(self, "best_makespan", float("inf")),
                "best_energy": getattr(self, "best_energy", float("inf")),
                "best_fatigue_mean": getattr(self, "best_fatigue_mean", float("inf")),
                "trained_episodes": trained_episodes,
                "saved_episode": episode_int,
                "best_val_score": getattr(self, "best_val_score", float("inf")),
                "best_val_metrics": getattr(self, "best_val_metrics", None),
                "best_metrics": getattr(self, "best_metrics", None),
                "tuning_params": {
                    "lr": getattr(self, "tuning_lr", None),
                    "gamma": getattr(self, "tuning_gamma", None),
                    "entropy_coef": getattr(self, "tuning_entropy_coef", None),
                    "top_k_ratio": getattr(self, "tuning_top_k_ratio", None),
                }
            }
            torch.save(checkpoint, save_path)
            print(f"模型已保存到 {save_path}, episode: {episode_int}")

            evaluate_saved_checkpoint_model(agent=self, saved_episode=episode_int, model_path=save_path)
            return save_path

    return TunedPPOAgent


def create_agent_for_exp(algo_module, TunedPPOAgent, first_json, lr, gamma, entropy_coef, top_k_ratio):
    config = algo_module.Config(json_path=first_json)
    config.actor_lr = float(lr)
    config.critic_lr = float(lr)
    config.hdgat_lr = float(lr)
    config.gamma = float(gamma)
    config.entropy_coef = float(entropy_coef)
    config.top_k_ratio = float(top_k_ratio)
    agent = TunedPPOAgent(config)
    agent.set_tuning_params(lr=lr, gamma=gamma, entropy_coef=entropy_coef, top_k_ratio=top_k_ratio)
    return agent


def install_training_time_logger(agent, exp_dir, exp_info):
    training_log_csv = os.path.join(exp_dir, "training_log.csv")
    fieldnames = [
        "exp_id", "A_level", "B_level", "C_level", "D_level",
        "lr", "gamma", "entropy_coef", "top_k_ratio",
        "instance_name", "episode", "duration_seconds", "cumulative_train_time_sec",
        "cmax", "total_energy", "avg_fatigue", "total_reward",
        "weight_cmax", "weight_energy", "weight_fatigue",
    ]

    with open(training_log_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    cumulative_time = {"value": 0.0}

    def patched_log_episode_to_csv(episode, duration, cmax, energy, fatigue_mean, reward, weights):
        try:
            if isinstance(weights, torch.Tensor):
                weights = weights.cpu().detach().tolist()
            elif weights is None:
                weights = [0.0, 0.0, 0.0]
            else:
                weights = list(weights)
            if len(weights) < 3:
                weights = weights + [0.0] * (3 - len(weights))

            duration_value = safe_float(duration, default=0.0)
            cumulative_time["value"] += duration_value

            row = {
                "exp_id": exp_info["exp_id"], "A_level": exp_info["A_level"],
                "B_level": exp_info["B_level"], "C_level": exp_info["C_level"],
                "D_level": exp_info["D_level"],
                "lr": exp_info["lr"], "gamma": exp_info["gamma"],
                "entropy_coef": exp_info["entropy_coef"], "top_k_ratio": exp_info["top_k_ratio"],
                "instance_name": getattr(agent, "current_instance_name", ""),
                "episode": episode,
                "duration_seconds": f"{duration_value:.4f}",
                "cumulative_train_time_sec": f"{cumulative_time['value']:.4f}",
                "cmax": f"{safe_float(cmax, default=0.0):.4f}",
                "total_energy": f"{safe_float(energy, default=0.0):.4f}",
                "avg_fatigue": f"{safe_float(fatigue_mean, default=0.0):.6f}",
                "total_reward": f"{safe_float(reward, default=0.0):.6f}",
                "weight_cmax": f"{safe_float(weights[0], default=0.0):.6f}",
                "weight_energy": f"{safe_float(weights[1], default=0.0):.6f}",
                "weight_fatigue": f"{safe_float(weights[2], default=0.0):.6f}",
            }
            with open(training_log_csv, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️ 参数校准训练日志写入失败: {e}")

    agent._log_episode_to_csv = patched_log_episode_to_csv
    agent.csv_log_path = training_log_csv
    return training_log_csv


def evaluate_each_val_instance(agent, val_files, eval_runs, out_csv_path, exp_info):
    rows = []
    for json_path in val_files:
        instance_name = os.path.basename(json_path)
        cmax_runs, tec_runs, favg_runs = [], [], []
        ref_cmax = ref_tec = ref_favg = None
        eval_start_time = time.time()

        for _ in range(eval_runs):
            agent.reset_instance(json_path)
            ref_cmax = getattr(agent.config, "ref_cmax", None)
            ref_tec = getattr(agent.config, "ref_tec", None)
            ref_favg = getattr(agent.config, "ref_favg", None)
            with torch.no_grad():
                _, metrics = agent.run_one_episode(episode=0, deterministic=True, collect_experience=False)
            cmax_runs.append(safe_float(metrics.get("makespan")))
            tec_runs.append(safe_float(metrics.get("energy")))
            favg_runs.append(safe_float(metrics.get("fatigue_mean")))

        validation_time_sec = time.time() - eval_start_time
        mean_cmax = float(np.mean(cmax_runs))
        mean_tec = float(np.mean(tec_runs))
        mean_favg = float(np.mean(favg_runs))

        cmax_ratio = mean_cmax / max(1e-6, float(ref_cmax)) if ref_cmax is not None else mean_cmax / max(1e-6, float(getattr(agent, "cmax_norm_factor", 1.0)))
        tec_ratio = mean_tec / max(1e-6, float(ref_tec)) if ref_tec is not None else mean_tec / max(1e-6, float(getattr(agent, "energy_norm_factor", 1.0)))
        favg_ratio = mean_favg / max(1e-6, float(ref_favg)) if ref_favg is not None else mean_favg
        score = 0.5 * cmax_ratio + 0.3 * tec_ratio + 0.2 * favg_ratio

        rows.append({
            "exp_id": exp_info["exp_id"], "A_level": exp_info["A_level"],
            "B_level": exp_info["B_level"], "C_level": exp_info["C_level"],
            "D_level": exp_info["D_level"],
            "lr": exp_info["lr"], "gamma": exp_info["gamma"],
            "entropy_coef": exp_info["entropy_coef"], "top_k_ratio": exp_info["top_k_ratio"],
            "instance": instance_name, "eval_runs": eval_runs,
            "validation_time_sec": validation_time_sec,
            "Cmax": mean_cmax, "Total_Energy": mean_tec, "Avg_Fatigue": mean_favg,
            "Cmax_ratio": cmax_ratio, "TEC_ratio": tec_ratio, "Favg_ratio": favg_ratio,
            "score": score,
        })

    fieldnames = list(rows[0].keys())
    with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def summarize_validation_rows(rows):
    if not rows:
        return {"mean_Cmax": float("inf"), "mean_Total_Energy": float("inf"),
                "mean_Avg_Fatigue": float("inf"), "mean_Cmax_ratio": float("inf"),
                "mean_TEC_ratio": float("inf"), "mean_Favg_ratio": float("inf"),
                "mean_score": float("inf"), "total_validation_time_sec": float("inf")}
    return {
        "mean_Cmax": float(np.mean([r["Cmax"] for r in rows])),
        "mean_Total_Energy": float(np.mean([r["Total_Energy"] for r in rows])),
        "mean_Avg_Fatigue": float(np.mean([r["Avg_Fatigue"] for r in rows])),
        "mean_Cmax_ratio": float(np.mean([r["Cmax_ratio"] for r in rows])),
        "mean_TEC_ratio": float(np.mean([r["TEC_ratio"] for r in rows])),
        "mean_Favg_ratio": float(np.mean([r["Favg_ratio"] for r in rows])),
        "mean_score": float(np.mean([r["score"] for r in rows])),
        "total_validation_time_sec": float(np.sum([r["validation_time_sec"] for r in rows])),
    }


def save_summary(summary_rows, out_path):
    fieldnames = [
        "rank", "exp_id", "A_level", "B_level", "C_level", "D_level",
        "lr", "gamma", "entropy_coef", "top_k_ratio",
        "train_episodes", "train_time_sec", "total_validation_time_sec",
        "mean_Cmax", "mean_Total_Energy", "mean_Avg_Fatigue",
        "mean_Cmax_ratio", "mean_TEC_ratio", "mean_Favg_ratio", "mean_score",
        "training_log_csv", "checkpoint_scores_csv", "checkpoint_detail_csv",
        "validation_csv", "exp_dir",
    ]
    sorted_rows = sorted(summary_rows, key=lambda x: x["mean_score"])
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            out_row = dict(row)
            out_row["rank"] = rank
            writer.writerow(out_row)


# ============================================================
# 5. 主流程
# ============================================================

def run_orthogonal_tuning():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    set_global_seed(BASE_SEED)

    algo_module = load_algo_module(ALGO_FILE)
    TunedPPOAgent = build_tuned_agent_class(algo_module)

    train_files = get_json_files(TRAIN_DIR)
    val_files = get_json_files(VAL_DIR)

    design = build_l16_design()

    design_csv = os.path.join(SAVE_ROOT, "orthogonal_design_L16.csv")
    save_design_table(design, design_csv)

    summary_rows = []

    print("\n================ 第二轮参数校准 — L16(4^4) ================")
    print(f"训练集实例数: {len(train_files)}")
    print(f"验证集实例数: {len(val_files)}")
    print(f"实验组数: {len(design)}")
    print(f"每组训练轮数: {EPISODES_PER_EXP}")
    print(f"结果目录: {os.path.abspath(SAVE_ROOT)}")
    print("==========================================================\n")

    for exp in design:
        exp_id = exp["exp_id"]
        lr = exp["lr"]
        gamma = exp["gamma"]
        entropy_coef = exp["entropy_coef"]
        top_k_ratio = exp["top_k_ratio"]

        exp_name = make_exp_name(exp_id, lr, gamma, entropy_coef, top_k_ratio)
        exp_dir = os.path.join(SAVE_ROOT, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"开始实验 {exp_name}")
        print(f"A 学习率 lr = {lr}")
        print(f"B 折扣因子 gamma = {gamma}")
        print(f"C 熵系数 entropy_coef = {entropy_coef}")
        print(f"D Top-K 剪枝比例 top_k_ratio = {top_k_ratio}")
        print("=" * 80)

        set_global_seed(BASE_SEED + exp_id)

        agent = create_agent_for_exp(
            algo_module=algo_module, TunedPPOAgent=TunedPPOAgent,
            first_json=train_files[0],
            lr=lr, gamma=gamma, entropy_coef=entropy_coef, top_k_ratio=top_k_ratio,
        )

        training_log_csv = install_training_time_logger(agent, exp_dir, exp)
        checkpoint_scores_csv, checkpoint_detail_csv = init_checkpoint_score_files(exp_dir)

        agent.enable_checkpoint_eval = True
        agent.checkpoint_eval_val_files = val_files
        agent.checkpoint_eval_runs = VAL_RUNS
        agent.checkpoint_exp_info = exp
        agent.checkpoint_scores_csv = checkpoint_scores_csv
        agent.checkpoint_detail_csv = checkpoint_detail_csv

        start_time = time.time()

        agent.train_multi_instance(
            train_files=train_files, val_files=val_files,
            total_episodes=EPISODES_PER_EXP, update_interval=UPDATE_INTERVAL,
            eval_interval=EVAL_INTERVAL, save_interval=SAVE_INTERVAL,
            save_dir=exp_dir, debug_interval=10,
        )

        train_time = time.time() - start_time

        validation_csv = os.path.join(exp_dir, "validation_results.csv")
        val_rows = evaluate_each_val_instance(
            agent=agent, val_files=val_files, eval_runs=VAL_RUNS,
            out_csv_path=validation_csv, exp_info=exp,
        )
        val_summary = summarize_validation_rows(val_rows)

        summary_row = {
            "exp_id": exp_id, "A_level": exp["A_level"], "B_level": exp["B_level"],
            "C_level": exp["C_level"], "D_level": exp["D_level"],
            "lr": lr, "gamma": gamma, "entropy_coef": entropy_coef, "top_k_ratio": top_k_ratio,
            "train_episodes": EPISODES_PER_EXP, "train_time_sec": train_time,
            "total_validation_time_sec": val_summary["total_validation_time_sec"],
            "mean_Cmax": val_summary["mean_Cmax"], "mean_Total_Energy": val_summary["mean_Total_Energy"],
            "mean_Avg_Fatigue": val_summary["mean_Avg_Fatigue"],
            "mean_Cmax_ratio": val_summary["mean_Cmax_ratio"], "mean_TEC_ratio": val_summary["mean_TEC_ratio"],
            "mean_Favg_ratio": val_summary["mean_Favg_ratio"], "mean_score": val_summary["mean_score"],
            "training_log_csv": training_log_csv, "checkpoint_scores_csv": checkpoint_scores_csv,
            "checkpoint_detail_csv": checkpoint_detail_csv, "validation_csv": validation_csv, "exp_dir": exp_dir,
        }
        summary_rows.append(summary_row)

        summary_csv = os.path.join(SAVE_ROOT, "orthogonal_summary_L16.csv")
        save_summary(summary_rows, summary_csv)

        print(f"\n实验 {exp_name} 完成, 训练耗时: {train_time:.2f} 秒, "
              f"mean_score={val_summary['mean_score']:.6f}")

        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = sorted(summary_rows, key=lambda x: x["mean_score"])
    summary_csv = os.path.join(SAVE_ROOT, "orthogonal_summary_L16.csv")
    save_summary(summary_rows, summary_csv)

    best = summary_rows[0]
    print("\n================ 第二轮参数校准完成 ================")
    print(f"实验结果: {design_csv}")
    print(f"总汇总表: {summary_csv}")
    print("\n本轮最优参数组合：")
    print(f"Exp{best['exp_id']:02d}")
    print(f"学习率 lr = {best['lr']}")
    print(f"折扣因子 gamma = {best['gamma']}")
    print(f"熵系数 entropy_coef = {best['entropy_coef']}")
    print(f"Top-K 剪枝比例 top_k_ratio = {best['top_k_ratio']}")
    print(f"mean_score = {best['mean_score']:.6f}")
    print("==================================================\n")


if __name__ == "__main__":
    run_orthogonal_tuning()
