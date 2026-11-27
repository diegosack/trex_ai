from browser_dino import BrowserDino
from dino_observation import ACTION_DUCK, ACTION_JUMP, OBS_DIM, observation_from_state


class BrowserDinoEnv:
    observation_dim = OBS_DIM
    action_dim = 3

    def __init__(self, headless=True, max_steps=8000, viewport=(800, 300)):
        self.browser = BrowserDino(headless=headless, viewport=viewport)
        self.browser.fast_setup()
        self.max_steps = max_steps
        self.steps = 0
        self.passed = 0
        self.jump_held = False
        self.duck_held = False
        self.state = None
        self.observation = None

    def _info(self, action=0, crashed=False):
        return {
            "score": int(round(float(self.state["distance"]) * 0.025)),
            "distance": float(self.state["distance"]),
            "speed": float(self.state["speed"]),
            "crashed": crashed,
            "action": action,
            "passed": self.passed,
        }

    def reset(self):
        self.state = self.browser.fast_reset()
        self.steps = 0
        self.passed = 0
        self.jump_held = False
        self.duck_held = False
        self.observation = observation_from_state(self.state)
        return self.observation, self._info()

    @staticmethod
    def _nearest_obstacle(state):
        dino_x = float(state["dino"]["x"])
        ahead = [obstacle for obstacle in state["obstacles"] if float(obstacle["x"]) + float(obstacle["width"]) > dino_x]
        if not ahead:
            return None
        return min(ahead, key=lambda obstacle: float(obstacle["x"]))

    def step(self, action):
        action = int(action)
        previous = self.state
        previous_dino = previous["dino"]
        was_on_ground = not previous_dino["jumping"]
        was_ducking = bool(previous_dino["ducking"])
        speed = max(float(previous["speed"]), 0.1)
        nearest = self._nearest_obstacle(previous)
        nearest_dx = (
            max(0.0, float(nearest["x"]) - float(previous_dino["x"])) if nearest is not None else 1000.0
        )
        frames_to_hit = nearest_dx / speed
        nearest_type = nearest["type"] if nearest is not None else None
        nearest_y = float(nearest["y"]) if nearest is not None else 0.0
        jump_pressed = action == ACTION_JUMP and not self.jump_held
        duck_pressed = action == ACTION_DUCK and not self.duck_held
        self.jump_held = action == ACTION_JUMP
        self.duck_held = action == ACTION_DUCK

        self.state = self.browser.fast_step(action)
        self.steps += 1
        self.observation = observation_from_state(self.state)

        reward = 0.01
        passed_delta = int(self.state["passed"]) - self.passed
        if passed_delta > 0:
            reward += 0.1 * passed_delta
        self.passed = int(self.state["passed"])
        if jump_pressed and not was_on_ground:
            reward -= 0.02
        if jump_pressed and was_ducking:
            reward -= 0.03
        if jump_pressed and nearest_type in ("CACTUS_SMALL", "CACTUS_LARGE"):
            if frames_to_hit < 14.0:
                reward -= 0.08
            elif frames_to_hit <= 32.0:
                reward += 0.04
            elif frames_to_hit > 45.0:
                reward -= 0.02
        if jump_pressed and nearest_type == "PTERODACTYL":
            if nearest_y >= 95.0:
                if frames_to_hit <= 35.0:
                    reward += 0.04
            elif frames_to_hit <= 40.0:
                reward -= 0.08
        if duck_pressed and nearest_type == "PTERODACTYL":
            if nearest_y < 95.0:
                if frames_to_hit <= 35.0:
                    reward += 0.05
            elif frames_to_hit <= 25.0:
                reward -= 0.08
        if duck_pressed and nearest_type in ("CACTUS_SMALL", "CACTUS_LARGE") and frames_to_hit < 20.0:
            reward -= 0.02
        if duck_pressed and not was_on_ground and nearest_dx < 300.0:
            reward -= 0.01

        crashed = bool(self.state["crashed"])
        if crashed:
            reward = -1.0
        terminated = crashed
        truncated = self.steps >= self.max_steps
        return self.observation, float(reward), terminated, truncated, self._info(action, crashed)

    def close(self):
        self.browser.close()
