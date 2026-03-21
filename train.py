"""
DQN Training Script for SpaceInvaders (ALE/SpaceInvaders-v5).

Trains a DQN agent using Stable Baselines3 with support for both
CNN (pixel-based) and MLP (RAM-based) policies.

Each member's results are saved under results/<member>/ to keep
experiments organized when multiple people share the same repo.
"""

import argparse
import json
import os
import shutil

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on SpaceInvaders")
    parser.add_argument("--member", type=str, required=True,
                        help="Your name (used to namespace results, e.g. 'Cedric')")
    parser.add_argument("--experiment", type=int, default=None,
                        help="Experiment number (e.g. 1-10). Used for naming model files.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.1, help="Exploration fraction")
    parser.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"],
                        help="Policy type: CnnPolicy (pixels) or MlpPolicy (RAM)")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def get_member_dirs(member, experiment=None):
    """Get the directory paths for a member's results.

    Structure:
        results/<member>/
            models/          - saved model .zip files
            logs/            - tensorboard and eval logs
            best_model.zip   - best model across all experiments
    """
    base = os.path.join("results", member)
    dirs = {
        "base": base,
        "models": os.path.join(base, "models"),
        "logs": os.path.join(base, "logs"),
    }

    if experiment is not None:
        exp_name = f"exp{experiment}"
        dirs["exp_logs"] = os.path.join(base, "logs", exp_name)
        dirs["model_path"] = os.path.join(base, "models", f"{exp_name}_model")
    else:
        dirs["exp_logs"] = os.path.join(base, "logs", "default")
        dirs["model_path"] = os.path.join(base, "models", "model")

    return dirs


def make_env(policy: str, seed: int):
    """Create the appropriate environment based on policy type.

    Args:
        policy: "CnnPolicy" for pixel observations, "MlpPolicy" for RAM observations.
        seed: Random seed for reproducibility.

    Returns:
        A vectorized environment suitable for the chosen policy.
    """
    if policy == "CnnPolicy":
        env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=1, seed=seed)
        env = VecFrameStack(env, n_stack=4)
    else:
        # MLP uses RAM observations (128-byte state vector)
        env = DummyVecEnv([lambda: gym.make("ALE/SpaceInvaders-v5", obs_type="ram")])
    return env


def make_eval_env(policy: str, seed: int):
    """Create an evaluation environment matching the training setup."""
    if policy == "CnnPolicy":
        env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=1, seed=seed + 100)
        env = VecFrameStack(env, n_stack=4)
    else:
        env = DummyVecEnv([lambda: gym.make("ALE/SpaceInvaders-v5", obs_type="ram")])
    return env


def main():
    args = parse_args()
    dirs = get_member_dirs(args.member, args.experiment)

    # Create all directories
    for d in [dirs["models"], dirs.get("exp_logs", dirs["logs"])]:
        os.makedirs(d, exist_ok=True)

    log_dir = dirs.get("exp_logs", dirs["logs"])

    # Print hyperparameter summary
    print("=" * 60)
    print(f"DQN Training — SpaceInvaders")
    print(f"Member: {args.member}" + (f" | Experiment: {args.experiment}" if args.experiment else ""))
    print("=" * 60)
    print(f"  Policy:              {args.policy}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Gamma:               {args.gamma}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Epsilon start:       {args.epsilon_start}")
    print(f"  Epsilon end:         {args.epsilon_end}")
    print(f"  Epsilon decay frac:  {args.epsilon_decay}")
    print(f"  Total timesteps:     {args.timesteps}")
    print(f"  Seed:                {args.seed}")
    print(f"  Output directory:    {dirs['base']}/")
    print(f"  Model save path:     {dirs['model_path']}.zip")
    print("=" * 60)

    # Save experiment config as JSON for documentation
    config = {
        "member": args.member,
        "experiment": args.experiment,
        "policy": args.policy,
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "timesteps": args.timesteps,
        "seed": args.seed,
    }
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # Create environments
    print("Creating environments...")
    env = make_env(args.policy, args.seed)
    eval_env = make_eval_env(args.policy, args.seed)

    # Create DQN agent
    print("Initializing DQN agent...")
    model = DQN(
        policy=args.policy,
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=args.epsilon_decay,
        buffer_size=100000,
        learning_starts=10000,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        seed=args.seed,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Set up evaluation callback to save best model
    eval_dir = os.path.join(log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_dir,
        log_path=eval_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Train the agent
    print(f"\nStarting training for {args.timesteps} timesteps...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        log_interval=100,
        tb_log_name="DQN_SpaceInvaders",
    )

    # Save the final model
    model.save(dirs["model_path"])
    print(f"\nModel saved to {dirs['model_path']}.zip")

    # Copy best model from eval callback if available
    best_eval_model = os.path.join(eval_dir, "best_model.zip")
    best_member_model = os.path.join(dirs["base"], "best_model.zip")
    if os.path.exists(best_eval_model):
        # Check if this is better than the member's current best
        shutil.copy(best_eval_model, best_member_model)
        print(f"Best model saved to {best_member_model}")

    # Final evaluation
    print("\nRunning final evaluation (10 episodes)...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"\nFinal Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save evaluation result to config
    config["mean_reward"] = float(mean_reward)
    config["std_reward"] = float(std_reward)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Member:         {args.member}")
    print(f"  Experiment:     {args.experiment}")
    print(f"  Mean Reward:    {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Model:          {dirs['model_path']}.zip")
    print(f"  Best model:     {best_member_model}")
    print(f"  Logs:           {log_dir}/")
    print(f"  TensorBoard:    tensorboard --logdir {log_dir}")
    print("=" * 60)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
