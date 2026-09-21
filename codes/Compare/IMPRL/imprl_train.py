# IMPRL 对比算法训练入口
# 用法：D:/anaconda/envs/pytorch_env/python.exe baseline_imprl/imprl_train.py
#   --train_dir ../trainData --val_dir ../trainData --num_episodes 2000
#
# 数据目录默认与自研一致（../trainData），可通过命令行覆盖。

import os
import sys
import glob
import argparse

# 把自研 train 目录和本目录加入 path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_TRAIN_DIR = os.path.join(os.path.dirname(_HERE), 'my', 'train')
sys.path.insert(0, _TRAIN_DIR)

from V1_01 import Config
from imprl_agent import IMPRLAgent


def main():
    parser = argparse.ArgumentParser(description='IMPRL 对比算法训练')
    parser.add_argument('--train_dir', default='../../TestData/trainForImpal', help='训练实例目录')
    parser.add_argument('--val_dir', default='../../TestData/yanzheng1', help='验证实例目录（可为空串禁用）')
    parser.add_argument('--num_episodes', type=int, default=5000, help='本次新增训练轮数')
    parser.add_argument('--update_interval', type=int, default=8)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--save_dir', default='saved_models_imprl')
    parser.add_argument('--debug_interval', type=int, default=10)
    args = parser.parse_args()

    train_dir = os.path.normpath(os.path.join(_HERE, args.train_dir))
    val_dir = args.val_dir
    if val_dir:
        val_dir = os.path.normpath(os.path.join(_HERE, val_dir))

    train_files = sorted(glob.glob(os.path.join(train_dir, '*.json')))
    if not train_files:
        raise FileNotFoundError(f"未找到训练实例: {train_dir}")
    val_files = sorted(glob.glob(os.path.join(val_dir, '*.json'))) if val_dir and os.path.exists(val_dir) else None

    print(f"[IMPRL] 训练实例数: {len(train_files)} | 验证实例数: {len(val_files) if val_files else 0}")

    init_config = Config(json_path=train_files[0])
    agent = IMPRLAgent(init_config)

    # 断点续训：优先加载 best_val
    for name in ['ppo_agent_episode_best_val.pth', 'ppo_agent_episode_latest.pth']:
        p = os.path.join(args.save_dir, name)
        if os.path.exists(p):
            print(f"[IMPRL] 载入已有模型: {p}")
            agent.load_models(p)
            break

    agent.train_multi_instance(
        train_files=train_files,
        val_files=val_files,
        total_episodes=args.num_episodes,
        update_interval=args.update_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        debug_interval=args.debug_interval,
    )


if __name__ == '__main__':
    main()
