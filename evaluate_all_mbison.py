import os

import numpy as np
import retro
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

NUM_EPISODES = 20
MODEL_DIR = "trained_models"

GAME = "StreetFighterIISpecialChampionEdition-Genesis"
STATE = "Champion.Level12.RyuVsMBison"

EVAL_MODES = [
    {
        "name": "deterministic_fixed_state",
        "label": "Deterministic Fixed-State",
        "deterministic": True,
        "reset_round": True,
        "rendering": False,
        "random_start_max_noops": 0,
        "seed_base": 20260627,
    },
    {
        "name": "stochastic_randomized_state",
        "label": "Stochastic Randomized-State",
        "deterministic": False,
        "reset_round": True,
        "rendering": False,
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


def evaluate_model(model_name, model_cls, model_path, mode, recurrent=False):
    full_model_path = os.path.join(MODEL_DIR, model_path)
    if not os.path.exists(full_model_path):
        return {
            "algorithm": model_name,
            "status": "missing",
            "model_path": full_model_path,
            "mode": mode["name"],
        }

    env = make_env(
        GAME,
        STATE,
        reset_round=mode["reset_round"],
        rendering=mode["rendering"],
    )
    model = model_cls.load(full_model_path, env=env)

    num_victory = 0
    episode_rewards = []
    opening_noops = []

    for episode_idx in range(NUM_EPISODES):
        obs, randomized_steps = reset_with_mode(env, mode, episode_idx)
        done = False
        total_reward = 0.0
        info = {}

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        opening_noops.append(randomized_steps)

        while not done:
            if recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=mode["deterministic"],
                )
            else:
                action, _states = model.predict(obs, deterministic=mode["deterministic"])

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if recurrent:
                episode_starts = np.array([done], dtype=bool)

            if info["enemy_hp"] < 0 or info["agent_hp"] < 0:
                done = True

        if info.get("enemy_hp", 0) < 0:
            num_victory += 1

        episode_rewards.append(total_reward)

        if not mode["reset_round"]:
            while info["enemy_hp"] < 0 or info["agent_hp"] < 0:
                obs, reward, done, info = env.step([0] * 12)
                if recurrent:
                    lstm_states = None
                    episode_starts = np.ones((1,), dtype=bool)

    env.close()

    rewards = np.array(episode_rewards, dtype=np.float32)
    return {
        "algorithm": model_name,
        "status": "ok",
        "model_path": full_model_path,
        "mode": mode["name"],
        "episodes": NUM_EPISODES,
        "wins": num_victory,
        "win_rate": num_victory / NUM_EPISODES,
        "avg_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "avg_opening_noops": float(np.mean(opening_noops)),
    }


def print_summary(mode, results):
    print("\nMBison evaluation summary [{}]".format(mode["label"]))
    print(
        "Config: episodes={}, deterministic={}, reset_round={}, rendering={}, random_start_max_noops={}\n".format(
            NUM_EPISODES,
            mode["deterministic"],
            mode["reset_round"],
            mode["rendering"],
            mode["random_start_max_noops"],
        )
    )

    header = "{:<15} {:<10} {:>8} {:>10} {:>12} {:>12} {:>14}".format(
        "Algorithm",
        "Status",
        "Wins",
        "WinRate",
        "AvgReward",
        "StdReward",
        "AvgOpenNoops",
    )
    print(header)
    print("-" * len(header))

    for result in results:
        if result["status"] != "ok":
            print(
                "{:<15} {:<10} {:>8} {:>10} {:>12} {:>12} {:>14}".format(
                    result["algorithm"], "missing", "-", "-", "-", "-", "-"
                )
            )
            print("  missing file: {}".format(result["model_path"]))
            continue

        print(
            "{:<15} {:<10} {:>8} {:>9.2%} {:>12.4f} {:>12.4f} {:>14.2f}".format(
                result["algorithm"],
                "ok",
                result["wins"],
                result["win_rate"],
                result["avg_reward"],
                result["std_reward"],
                result["avg_opening_noops"],
            )
        )

    available_results = [result for result in results if result["status"] == "ok"]
    if available_results:
        best_win_rate = max(available_results, key=lambda item: item["win_rate"])
        best_reward = max(available_results, key=lambda item: item["avg_reward"])

        print("\nBest by win rate : {} ({:.2%})".format(best_win_rate["algorithm"], best_win_rate["win_rate"]))
        print("Best by avg reward: {} ({:.4f})".format(best_reward["algorithm"], best_reward["avg_reward"]))


def main():
    for mode in EVAL_MODES:
        results = []
        for spec in MODEL_SPECS:
            print(
                "Evaluating {} using {} [{}]".format(
                    spec["name"],
                    spec["model_path"],
                    mode["label"],
                )
            )
            results.append(
                evaluate_model(
                    model_name=spec["name"],
                    model_cls=spec["loader"],
                    model_path=spec["model_path"],
                    mode=mode,
                    recurrent=spec["recurrent"],
                )
            )

        print_summary(mode, results)


if __name__ == "__main__":
    main()
