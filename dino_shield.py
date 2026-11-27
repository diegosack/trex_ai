from dino_observation import ACTION_DUCK, ACTION_JUMP

CACTUS_TYPES = ("CACTUS_SMALL", "CACTUS_LARGE")
LOW_PTERODACTYL_Y = 95.0
PTERODACTYL_DUCK_FRAMES = 25.0
PTERODACTYL_JUMP_FRAMES = 18.0
CACTUS_JUMP_FRAMES = 12.0


def shield_action(state, action):
    dino = state["dino"]
    dino_x = float(dino["x"])
    speed = max(float(state["speed"]), 0.1)
    ahead = [
        obstacle
        for obstacle in state["obstacles"]
        if float(obstacle["x"]) + float(obstacle["width"]) > dino_x - 10.0
    ]
    if not ahead:
        if not dino["jumping"] and action in (ACTION_DUCK, ACTION_JUMP):
            return 0
        return action
    nearest = min(ahead, key=lambda obstacle: float(obstacle["x"]))
    kind = nearest["type"]
    dx = float(nearest["x"]) - dino_x
    frames_to_hit = max(0.0, dx) / speed

    if kind == "PTERODACTYL":
        height = float(nearest["y"])
        if height >= LOW_PTERODACTYL_Y:
            if frames_to_hit <= PTERODACTYL_JUMP_FRAMES and not dino["jumping"]:
                return ACTION_JUMP
        elif frames_to_hit <= PTERODACTYL_DUCK_FRAMES or (
            dx < 60.0 and dx + float(nearest["width"]) > -10.0
        ):
            return ACTION_DUCK
    elif kind in CACTUS_TYPES and frames_to_hit <= CACTUS_JUMP_FRAMES and not dino["jumping"]:
        return ACTION_JUMP
    if action == ACTION_DUCK:
        if dino["jumping"]:
            return action
        if kind in CACTUS_TYPES and frames_to_hit <= 20.0:
            return ACTION_JUMP
        return 0
    return action
