import numpy as np

GROUND_Y = 93.0
BASE_SPEED = 6.0
MAX_SPEED = 13.0
OBS_DIM = 21
ACTION_NONE = 0
ACTION_JUMP = 1
ACTION_DUCK = 2
OBSTACLE_TYPES = ("CACTUS_SMALL", "CACTUS_LARGE", "PTERODACTYL")


def observation_from_state(state):
    dino = state["dino"]
    dino_x = float(dino["x"])
    obstacles = sorted(
        (
            {
                "kind": obstacle["type"],
                "x": float(obstacle["x"]),
                "y": float(obstacle["y"]),
                "width": float(obstacle["width"]),
                "height": float(obstacle["height"]),
            }
            for obstacle in state["obstacles"]
            if float(obstacle["x"]) + float(obstacle["width"]) > dino_x
        ),
        key=lambda obstacle: obstacle["x"],
    )
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0] = np.clip((GROUND_Y - float(dino["y"])) / 70.0, 0.0, 2.0)
    observation[1] = np.clip(float(dino["jumpVelocity"]) / 12.0, -2.0, 2.0)
    observation[2] = 1.0 if not dino["jumping"] else 0.0
    observation[3] = 1.0 if dino["ducking"] else 0.0
    observation[4] = np.clip((float(state["speed"]) - BASE_SPEED) / (MAX_SPEED - BASE_SPEED), 0.0, 2.0)

    def fill(offset, obstacle):
        if obstacle is None:
            return
        dx = max(0.0, obstacle["x"] - dino_x)
        observation[offset] = np.clip(dx / 600.0, 0.0, 2.0)
        observation[offset + 1] = np.clip(obstacle["y"] / 150.0, 0.0, 2.0)
        observation[offset + 2] = np.clip(obstacle["height"] / 60.0, 0.0, 2.0)
        observation[offset + 3] = np.clip(dx / max(float(state["speed"]), 0.1) / 100.0, 0.0, 2.0)
        index = OBSTACLE_TYPES.index(obstacle["kind"])
        observation[offset + 4 + index] = 1.0
        observation[offset + 7] = np.clip(obstacle["width"] / 100.0, 0.0, 2.0)

    fill(5, obstacles[0] if obstacles else None)
    fill(13, obstacles[1] if len(obstacles) > 1 else None)
    return observation
