import argparse
import csv
import os
import time

import numpy as np
import torch

from browser_env import BrowserDinoEnv
from dino_observation import OBS_DIM
from ppo import PPOAgent, RolloutBuffer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train PPO on the original Chrome Dino game")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=str, default="checkpoints/ppo_browser.pt")
    parser.add_argument("--log", type=str, default="training_log.csv")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--eval-max-steps", type=int, default=8000)
    parser.add_argument("--reset-best", action="store_true")
    parser.add_argument("--save-history", action="store_true")
    return parser.parse_args()


def evaluate(agent, environment, episodes, max_steps):
    previous_max = environment.max_steps
    environment.max_steps = max_steps
    lengths = []
    scores = []
    for _ in range(episodes):
        observation, _ = environment.reset()
        info = {"score": 0}
        steps = 0
        while steps < max_steps:
            action, _, _ = agent.act(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            steps += 1
            if terminated or truncated:
                break
        lengths.append(environment.steps)
        scores.append(info["score"])
    environment.max_steps = previous_max
    return float(np.mean(lengths)), float(np.std(lengths)), float(np.mean(scores))


def main():
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    os.makedirs(os.path.dirname(arguments.out) or ".", exist_ok=True)
    environment = BrowserDinoEnv(headless=True, max_steps=arguments.max_steps)
    observation, _ = environment.reset()

    if arguments.resume:
        agent, extra = PPOAgent.load(arguments.resume, observation_dim=OBS_DIM, action_dim=3)
        start_timestep = extra.get("total_timesteps", 0)
        best_length = 0.0 if arguments.reset_best else extra.get("best_length", 0.0)
        print(f"Resumed from {arguments.resume} at step {start_timestep}")
    else:
        agent = PPOAgent(OBS_DIM, 3)
        start_timestep = 0
        best_length = 0.0

    log_file = open(arguments.log, "w" if not arguments.resume else "a", newline="")
    log_writer = csv.writer(log_file)
    if log_file.tell() == 0:
        log_writer.writerow(
            ["timestep", "episode_reward", "episode_length", "eval_length", "eval_std", "eval_score", "best_length", "policy_loss", "value_loss", "entropy"]
        )
        log_file.flush()

    buffer = RolloutBuffer(arguments.rollout, OBS_DIM, agent.device)
    total_timesteps = start_timestep
    training_target = start_timestep + arguments.timesteps
    episode_rewards = []
    episode_lengths = []
    episode_reward = 0.0
    episode_length = 0
    rollout_index = 0
    started_at = time.time()

    try:
        while total_timesteps < training_target:
            relative_progress = (total_timesteps - start_timestep) / max(1, training_target - start_timestep)
            buffer.clear()
            for _ in range(arguments.rollout):
                action, log_probability, value = agent.act(observation)
                next_observation, reward, terminated, truncated, _ = environment.step(action)
                buffer.store(observation, action, log_probability, value, reward, terminated or truncated)
                episode_reward += reward
                episode_length += 1
                observation = next_observation
                total_timesteps += 1
                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episode_reward = 0.0
                    episode_length = 0
                    observation, _ = environment.reset()

            _, _, last_value = agent.act(observation, deterministic=True)
            losses = agent.update(buffer, last_value, progress=relative_progress)
            rollout_index += 1

            mean_reward = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
            mean_length = float(np.mean(episode_lengths[-20:])) if episode_lengths else 0.0
            elapsed = time.time() - started_at
            speed = (total_timesteps - start_timestep) / max(elapsed, 1.0)
            print(
                f"step {total_timesteps:>8} | episode {mean_length:6.1f} frames, reward {mean_reward:7.2f} "
                f"| {speed:6.0f} steps/s | policy {losses['policy_loss']:7.4f} value {losses['value_loss']:7.3f} "
                f"entropy {losses['entropy']:.3f}",
                flush=True,
            )
            log_writer.writerow(
                [total_timesteps, mean_reward, mean_length, "", "", "", best_length, losses["policy_loss"], losses["value_loss"], losses["entropy"]]
            )
            log_file.flush()

            if rollout_index % arguments.eval_every == 0 or total_timesteps >= training_target:
                eval_length, eval_std, eval_score = evaluate(
                    agent,
                    environment,
                    arguments.eval_episodes,
                    arguments.eval_max_steps,
                )
                metadata = {"total_timesteps": total_timesteps, "best_length": best_length, "eval_score": eval_score}
                if eval_length > best_length:
                    best_length = eval_length
                    metadata["best_length"] = best_length
                    agent.save(arguments.out, extra=metadata)
                agent.save(arguments.out.replace(".pt", "_last.pt"), extra=metadata)
                if arguments.save_history:
                    history_path = os.path.join(
                        os.path.dirname(arguments.out) or ".", "history", f"ppo_browser_{total_timesteps}.pt"
                    )
                    os.makedirs(os.path.dirname(history_path), exist_ok=True)
                    agent.save(history_path, extra=metadata)
                print(
                    f"  eval: {eval_length:6.1f} +/- {eval_std:5.1f} frames, score {eval_score:6.1f}, best {best_length:6.1f}",
                    flush=True,
                )
                log_writer.writerow([total_timesteps, "", "", eval_length, eval_std, eval_score, best_length, "", "", ""])
                log_file.flush()
                observation, _ = environment.reset()
                episode_reward = 0.0
                episode_length = 0
    finally:
        environment.close()
        log_file.close()
    print(f"Done. Best checkpoint: {arguments.out}")


if __name__ == "__main__":
    main()
