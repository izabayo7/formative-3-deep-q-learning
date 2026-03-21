"""
DQN Evaluation/Play Script for SpaceInvaders (ALE/SpaceInvaders-v5).

Loads a trained DQN model and runs episodes with GUI display.
Supports per-member model loading via --member flag.
"""

import argparse
import os

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


def parse_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Play SpaceInvaders with a trained DQN agent")
    parser.add_argument("--member", type=str, default=None,
                        help="Member name — loads their best model from results/<member>/best_model.zip")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Direct path to a model .zip file (overrides --member)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"],
                        help="Policy type used during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def resolve_model_path(args):
    """Determine the model path from --model-path or --member flag."""
    if args.model_path:
        return args.model_path

    if args.member:
        member_model = os.path.join("results", args.member, "best_model.zip")
        if os.path.exists(member_model):
            return member_model
        else:
            print(f"Error: No best model found for member '{args.member}'.")
            print(f"Expected at: {member_model}")
            print(f"Run train.py --member {args.member} first.")
            return None

    # Default fallback
    if os.path.exists("dqn_model.zip"):
        return "dqn_model.zip"

    print("Error: No model found. Provide --model-path or --member, or place dqn_model.zip in the project root.")
    return None


def make_play_env(policy: str):
    """Create an environment for playing/evaluation with GUI display.

    Args:
        policy: "CnnPolicy" or "MlpPolicy" — must match the trained model.

    Returns:
        A vectorized environment matching the training setup.
    """
    if policy == "CnnPolicy":
        base_env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
        base_env = AtariWrapper(base_env)
        env = DummyVecEnv([lambda: base_env])
        env = VecFrameStack(env, n_stack=4)
    else:
        base_env = gym.make("ALE/SpaceInvaders-v5", obs_type="ram", render_mode="human")
        env = DummyVecEnv([lambda: base_env])

    return env


def main():
    args = parse_args()

    model_path = resolve_model_path(args)
    if model_path is None:
        return

    print("=" * 60)
    print("DQN SpaceInvaders — Play Mode")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Policy:   {args.policy}")
    print(f"  Episodes: {args.episodes}")
    if args.member:
        print(f"  Member:   {args.member}")
    print("=" * 60)

    # Load the trained model
    print("\nLoading model...")
    model = DQN.load(model_path)

    # Create environment
    env = make_play_env(args.policy)

    # Run episodes
    scores = []
    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

        scores.append(total_reward)
        print(f"  Episode {episode}: Score = {total_reward:.0f} ({steps} steps)")

    # Print summary
    print("\n" + "-" * 40)
    print(f"  Average Score: {np.mean(scores):.1f}")
    print(f"  Best Score:    {np.max(scores):.0f}")
    print(f"  Worst Score:   {np.min(scores):.0f}")
    print("-" * 40)

    env.close()


if __name__ == "__main__":
    main()
