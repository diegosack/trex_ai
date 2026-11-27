import time
from pathlib import Path

from dino_observation import ACTION_DUCK, ACTION_JUMP, observation_from_state

GAME_DIR = Path(__file__).resolve().parent / "game"
GAME_URL = (GAME_DIR / "index.html").as_uri()

STATE_SCRIPT = """
() => {
    const runner = window.Runner && Runner.instance_;
    if (!runner) return null;
    const tRex = runner.tRex;
    return {
        playing: runner.playing,
        crashed: runner.crashed,
        intro: runner.playingIntro,
        speed: runner.currentSpeed,
        distance: runner.distanceRan,
        dino: {
            x: tRex.xPos,
            y: tRex.yPos,
            jumping: tRex.jumping,
            ducking: tRex.ducking,
            speedDrop: tRex.speedDrop,
            jumpVelocity: tRex.jumpVelocity
        },
        obstacles: runner.horizon.obstacles.map((obstacle) => ({
            type: obstacle.typeConfig.type,
            x: obstacle.xPos,
            y: obstacle.yPos,
            width: obstacle.width,
            height: obstacle.typeConfig.height
        }))
    };
}
"""

FAST_SETUP_SCRIPT = """
() => {
    const runner = Runner.instance_;
    window.__dinoClock = performance.now();
    performance.now = () => window.__dinoClock;
    cancelAnimationFrame(runner.raqId);
    runner.updatePending = false;
    runner.raqId = 1;
    runner.activated = true;
    runner.playing = false;
    runner.crashed = false;
    runner.playingIntro = false;
    runner.paused = false;
    window.__dinoPassed = 0;
    window.__dinoKeys = {jump: false, duck: false};
    window.__dinoEvent = (code) => ({
        keyCode: code,
        target: document.body,
        preventDefault() {},
    });
    window.__dinoState = () => {
        const current = Runner.instance_;
        const tRex = current.tRex;
        return {
            playing: current.playing,
            crashed: current.crashed,
            speed: current.currentSpeed,
            distance: current.distanceRan,
            passed: window.__dinoPassed,
            dino: {
                x: tRex.xPos,
                y: tRex.yPos,
                jumping: tRex.jumping,
                ducking: tRex.ducking,
                jumpVelocity: tRex.jumpVelocity,
            },
            obstacles: current.horizon.obstacles.map((obstacle) => ({
                type: obstacle.typeConfig.type,
                x: obstacle.xPos,
                y: obstacle.yPos,
                width: obstacle.width,
                height: obstacle.typeConfig.height,
            })),
        };
    };
    window.__dinoKeyDown = (code) => runner.onKeyDown(window.__dinoEvent(code));
    window.__dinoKeyUp = (code) => runner.onKeyUp(window.__dinoEvent(code));
    window.__dinoStep = (action) => {
        const current = Runner.instance_;
        const keys = window.__dinoKeys;
        if (action === 1) {
            if (keys.duck || current.tRex.ducking) {
                window.__dinoKeyUp(40);
                keys.duck = false;
            }
            if (!keys.jump) {
                window.__dinoKeyDown(32);
                keys.jump = true;
            }
        } else if (action === 2) {
            if (keys.jump) {
                window.__dinoKeyUp(32);
                keys.jump = false;
            }
            if (!current.tRex.ducking && (!current.tRex.jumping || !keys.duck)) {
                window.__dinoKeyDown(40);
            }
            keys.duck = true;
        } else {
            if (keys.jump) {
                window.__dinoKeyUp(32);
                keys.jump = false;
            }
            if (keys.duck || current.tRex.ducking) {
                window.__dinoKeyUp(40);
                keys.duck = false;
            }
        }
        window.__dinoClock += 1000 / 60;
        current.update();
        cancelAnimationFrame(current.raqId);
        current.updatePending = false;
        current.raqId = 1;
        const dinoX = current.tRex.xPos;
        for (const obstacle of current.horizon.obstacles) {
            if (!obstacle.__aiPassed && obstacle.xPos + obstacle.width < dinoX + 44) {
                obstacle.__aiPassed = true;
                window.__dinoPassed += 1;
            }
        }
        return window.__dinoState();
    };
    window.__dinoReset = () => {
        const current = Runner.instance_;
        current.raqId = 0;
        current.activated = true;
        current.playing = true;
        current.crashed = false;
        current.playingIntro = false;
        current.paused = false;
        current.tRex.xPos = current.tRex.config.START_X_POS;
        window.__dinoKeys.jump = false;
        window.__dinoKeys.duck = false;
        window.__dinoPassed = 0;
        current.restart();
        cancelAnimationFrame(current.raqId);
        current.updatePending = false;
        current.raqId = 1;
        return window.__dinoState();
    };
}
"""


class BrowserDino:
    def __init__(self, headless=True, slow_mo=0, viewport=(800, 300)):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(headless=headless, slow_mo=slow_mo)
        self.page = self.browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        self.page.goto(GAME_URL, wait_until="load")
        self.page.wait_for_function("() => window.Runner && window.Runner.instance_ && window.Runner.instance_.canvas")
        self.jump_down = False
        self.duck_down = False
        self.frames = 0

    def reset(self, wait_seconds=1.0):
        self.page.reload(wait_until="load")
        self.page.wait_for_function("() => window.Runner && window.Runner.instance_ && window.Runner.instance_.canvas")
        self.jump_down = False
        self.duck_down = False
        self.frames = 0
        self.page.keyboard.press("Space")
        self.page.evaluate(
            "() => { const box = document.getElementById('messageBox'); if (box) box.style.visibility = 'hidden'; }"
        )
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            state = self.state()
            if state and state["playing"] and not state["intro"]:
                break
            time.sleep(0.02)
        return self.state()

    def state(self):
        return self.page.evaluate(STATE_SCRIPT)

    def observation(self, state):
        return observation_from_state(state)

    def fast_setup(self):
        self.page.evaluate(FAST_SETUP_SCRIPT)

    def fast_reset(self):
        return self.page.evaluate("() => window.__dinoReset()")

    def fast_step(self, action):
        return self.page.evaluate("(action) => window.__dinoStep(action)", int(action))

    def _key_down(self, key_code):
        self.page.evaluate(
            "(code) => Runner.instance_.onKeyDown({keyCode: code, target: document.body, preventDefault() {}})",
            key_code,
        )

    def _key_up(self, key_code):
        self.page.evaluate(
            "(code) => Runner.instance_.onKeyUp({keyCode: code, target: document.body, preventDefault() {}})",
            key_code,
        )

    def apply_action(self, action):
        if action == ACTION_JUMP:
            if self.duck_down:
                self._key_up(40)
                self.duck_down = False
            if not self.jump_down:
                self._key_down(32)
                self.jump_down = True
        elif action == ACTION_DUCK:
            if self.jump_down:
                self._key_up(32)
                self.jump_down = False
            if not self.duck_down:
                self._key_down(40)
                self.duck_down = True
        else:
            if self.jump_down:
                self._key_up(32)
                self.jump_down = False
            if self.duck_down:
                self._key_up(40)
                self.duck_down = False
        self.frames += 1

    def release_all(self):
        try:
            if self.jump_down:
                self._key_up(32)
            if self.duck_down:
                self._key_up(40)
        finally:
            self.jump_down = False
            self.duck_down = False

    def close(self):
        try:
            self.browser.close()
        finally:
            self._playwright.stop()
