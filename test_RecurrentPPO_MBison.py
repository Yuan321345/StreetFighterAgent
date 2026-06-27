import os
import time

import numpy as np
import retro
from sb3_contrib import RecurrentPPO

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

RESET_ROUND = False
RENDERING = True

MODEL_NAME = r"recurrent_ppo_sf2_ryu_vs_mbison_final.zip"

RANDOM_ACTION = False
NUM_EPISODES = 30
MODEL_DIR = r"trained_models/"


def make_env(game, state):
    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
        )
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING)
        return env

    return _init


game = "StreetFighterIISpecialChampionEdition-Genesis"
env = make_env(game, state="Champion.Level12.RyuVsMBison")()

if not RANDOM_ACTION:
    model = RecurrentPPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)

obs = env.reset()
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")

for _ in range(num_episodes):
    done = False

    if RESET_ROUND:
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

    total_reward = 0

    while not done:
        _timestamp = time.time()

        if RANDOM_ACTION:
            obs, reward, done, info = env.step(env.action_space.sample())
        else:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
            )
            obs, reward, done, info = env.step(action)

        if reward != 0:
            total_reward += reward
            print(
                "Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(
                    reward, info["agent_hp"], info["enemy_hp"]
                )
            )

        if info["enemy_hp"] < 0 or info["agent_hp"] < 0:
            done = True

        episode_starts = np.array([done], dtype=bool)

    if info["enemy_hp"] < 0:
        print("Victory!")
        num_victory += 1

    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward

    if not RESET_ROUND:
        while info["enemy_hp"] < 0 or info["agent_hp"] < 0:
            obs, reward, done, info = env.step([0] * 12)
            env.render()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum / num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum / num_episodes))
