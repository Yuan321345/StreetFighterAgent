import os
import time

import numpy as np
import retro
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

NUM_EPISODES = 20
MODEL_DIR = "trained_models"
VIDEO_DIR = "demo_videos"

GAME = "StreetFighterIISpecialChampionEdition-Genesis"
STATE = "Champion.Level12.RyuVsMBison"

# 可选值：
# "deterministic_fixed_state"
# "stochastic_randomized_state"
DEMO_MODE_NAME = "stochastic_randomized_state"

# 可选值：["PPO"]、["A2C"]、["RecurrentPPO"] 或三者组合
ALGORITHMS_TO_RUN = ["PPO", "A2C", "RecurrentPPO"]

RENDERING = True
RECORD_VIDEO = True
SLOW_MOTION = True
SLOW_MOTION_DELAY = 0.02

EVAL_MODES = [
    {
        "name": "deterministic_fixed_state",
        "label": "Deterministic Fixed-State",
        "deterministic": True,
        "reset_round": True,
        "rendering": RENDERING,
        "random_start_max_noops": 0,
        "seed_base": 20260627,
    },
    {
        "name": "stochastic_randomized_state",
        "label": "Stochastic Randomized-State",
        "deterministic": False,
        "reset_round": True,
        "rendering": RENDERING,
        "random_start_max_noops": 6,
        "seed_base": 20260727,
    },
]

MODEL_SPECS = [
    {
        "name": "PPO",
        "loader": PPO,
        "model_path": "ppo_sf2_ryu_vs_mbison_final.zip",
        "recurrent": False,
    },
    {
        "name": "A2C",
        "loader": A2C,
        "model_path": "a2c_sf2_ryu_vs_mbison_final.zip",
        "recurrent": False,
    },
    {
        "name": "RecurrentPPO",
        "loader": RecurrentPPO,
        "model_path": "recurrent_ppo_sf2_ryu_vs_mbison_final.zip",
        "recurrent": True,
    },
]


def get_mode(mode_name):
    for mode in EVAL_MODES:
        if mode["name"] == mode_name:
            return mode
    raise ValueError("Unknown DEMO_MODE_NAME: {}".format(mode_name))


def make_env(game, state, reset_round, rendering):
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
    )
    return StreetFighterCustomWrapper(env, reset_round=reset_round, rendering=rendering)


def reset_with_mode(env, mode, episode_idx):
    episode_seed = mode["seed_base"] + episode_idx
    env.seed(episode_seed)
    obs = env.reset()

    randomized_steps = 0
    if mode["random_start_max_noops"] > 0:
        rng = np.random.default_rng(episode_seed)
        randomized_steps = int(rng.integers(0, mode["random_start_max_noops"] + 1))
        for _ in range(randomized_steps):
            obs, reward, done, info = env.step([0] * 12)
            if info["enemy_hp"] < 0 or info["agent_hp"] < 0 or done:
                break

    return obs, randomized_steps


def run_demo(spec, mode):
    model_path = os.path.join(MODEL_DIR, spec["model_path"])
    if not os.path.exists(model_path):
        print("Skip {}: missing model {}".format(spec["name"], model_path))
        return

    video_path = None
    video_recorder = None
    if RECORD_VIDEO:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(
            VIDEO_DIR,
            "{}_{}.mp4".format(spec["name"], mode["name"]),
        )

    env = make_env(
        GAME,
        STATE,
        reset_round=mode["reset_round"],
        rendering=mode["rendering"],
    )
    model = spec["loader"].load(model_path, env=env)

    if RECORD_VIDEO:
        video_recorder = VideoRecorder(env, path=video_path, enabled=True)

    total_wins = 0
    total_rewards = []

    print("\n=== {} | {} ===".format(spec["name"], mode["label"]))
    print("Model: {}".format(model_path))
    if RECORD_VIDEO:
        print("Video output: {}".format(os.path.abspath(video_path)))

    for episode_idx in range(NUM_EPISODES):
        obs, randomized_steps = reset_with_mode(env, mode, episode_idx)
        done = False
        episode_reward = 0.0
        info = {}

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        print(
            "\nEpisode {} starts. Opening no-ops: {}".format(
                episode_idx + 1, randomized_steps
            )
        )

        if video_recorder is not None:
            video_recorder.capture_frame()

        while not done:
            if spec["recurrent"]:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=mode["deterministic"],
                )
            else:
                action, _states = model.predict(
                    obs,
                    deterministic=mode["deterministic"],
                )

            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if video_recorder is not None:
                video_recorder.capture_frame()

            if spec["recurrent"]:
                episode_starts = np.array([done], dtype=bool)

            if reward != 0:
                print(
                    "Reward: {:.3f}, playerHP: {}, enemyHP: {}".format(
                        reward, info["agent_hp"], info["enemy_hp"]
                    )
                )

            if info["enemy_hp"] < 0 or info["agent_hp"] < 0:
                done = True

            if SLOW_MOTION:
                time.sleep(SLOW_MOTION_DELAY)

        won = info.get("enemy_hp", 0) < 0
        if won:
            total_wins += 1

        total_rewards.append(episode_reward)
        print(
            "Episode {} result: {}, total_reward={:.3f}".format(
                episode_idx + 1,
                "Victory" if won else "Defeat",
                episode_reward,
            )
        )

    if video_recorder is not None:
        video_recorder.close()
    env.close()

    avg_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    win_rate = total_wins / NUM_EPISODES if NUM_EPISODES > 0 else 0.0

    print("\nSummary for {} [{}]".format(spec["name"], mode["label"]))
    print("Win rate: {:.2%}".format(win_rate))
    print("Average reward: {:.4f}".format(avg_reward))


def main():
    mode = get_mode(DEMO_MODE_NAME)
    selected_specs = [spec for spec in MODEL_SPECS if spec["name"] in ALGORITHMS_TO_RUN]

    if not selected_specs:
        raise ValueError("ALGORITHMS_TO_RUN does not match any known algorithm.")

    for spec in selected_specs:
        run_demo(spec, mode)


if __name__ == "__main__":
    main()
