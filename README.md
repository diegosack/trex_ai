# Chrome Dino PPO

Минимальная версия проекта: оригинальная игра Chrome Dino запускается в Chrome, а обученный PPO-агент играет в неё. Код обучения оставлен.

## Запуск

```bash
uv sync
uv run python play.py
```

Откроется окно Chrome с оригинальной игрой, в которой играет модель `checkpoints/ppo_browser_tuned.pt`.

Полезные параметры:

```bash
uv run python play.py --episodes 3        # несколько попыток
uv run python play.py --max-seconds 1200  # ограничение на эпизод
uv run python play.py --headless          # без окна
```

## Обучение

```bash
uv run python train.py --timesteps 1000000
```

Продолжить с сохранённой модели:

```bash
uv run python train.py --resume checkpoints/ppo_browser_tuned.pt --timesteps 1000000
```

Чекпоинты сохраняются в `checkpoints/`, лог — в `training_log.csv`.

## Структура

| Файл | Назначение |
| --- | --- |
| `play.py` | запуск оригинальной игры в Chrome с агентом |
| `train.py` | обучение PPO на оригинальной игре |
| `browser_dino.py` | управление Chrome и оригинальной игрой |
| `browser_env.py` | среда Gym-подобного интерфейса для обучения |
| `dino_observation.py` | признаки состояния игры |
| `dino_shield.py` | страховка действий агента |
| `ppo.py` | PPO-агент |
| `game/` | оригинальная игра wayou/t-rex-runner |

Оригинальная игра распространяется по лицензии из `game/LICENSE`.
