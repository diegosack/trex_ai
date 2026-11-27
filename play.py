import argparse
import time

from browser_dino import BrowserDino
from dino_observation import OBS_DIM
from dino_shield import shield_action
from ppo import PPOAgent


def parse_arguments():
    parser = argparse.ArgumentParser(description="Watch the trained PPO agent play the original Chrome Dino")
    parser.add_argument("--model", type=str, default="checkpoints/ppo_browser_tuned.pt")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fps", type=float, default=60.0)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    agent, metadata = PPOAgent.load(arguments.model, observation_dim=OBS_DIM, action_dim=3)
    print(f"Loaded {arguments.model}: {metadata}")

    browser = BrowserDino(headless=arguments.headless)
    scores = []
    try:
        for episode in range(arguments.episodes):
            browser.reset()
            if not arguments.headless:
                browser.page.bring_to_front()
            print(f"episode {episode + 1}: started")
            started_at = time.perf_counter()
            next_frame = started_at
            last_report = 0.0
            survived = False
            state = None
            while True:
                now = time.perf_counter()
                if now < next_frame:
                    time.sleep(next_frame - now)
                next_frame += 1.0 / arguments.fps

                state = browser.state()
                if state is None:
                    continue
                if state["crashed"] or not state["playing"]:
                    break

                observation = browser.observation(state)
                action, _, _ = agent.act(observation, deterministic=True)
                browser.apply_action(shield_action(state, action))

                elapsed = time.perf_counter() - started_at
                if elapsed - last_report >= 1.0:
                    score = round(state["distance"] * 0.025)
                    print(f"  t {elapsed:5.1f}s  speed {state['speed']:5.2f}  score {score:5.0f}", flush=True)
                    last_report = elapsed

                if elapsed >= arguments.max_seconds:
                    survived = True
                    break

            browser.release_all()
            elapsed = time.perf_counter() - started_at
            score = round(state["distance"] * 0.025) if state else 0
            scores.append(score)
            outcome = "survived the limit" if survived else "crashed"
            print(f"episode {episode + 1}: {outcome} after {elapsed:.1f}s, score {score}")
    finally:
        browser.close()

    if scores:
        print(f"mean score {sum(scores) / len(scores):.1f}")


if __name__ == "__main__":
    main()
