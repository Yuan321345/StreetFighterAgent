import os
import sys

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

NUM_ENV = 16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(game, state, seed=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
        env = StreetFighterCustomWrapper(env)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def main():
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = SubprocVecEnv([make_env(game, state="Champion.Level12.RyuVsMBison", seed=i) for i in range(NUM_ENV)])
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)


    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log="logs"
    )

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # model_path = "trained_models/ppo_ryu_500000_steps.zip"
    # custom_objects = {
    #     "learning_rate": lr_schedule,
    #     "clip_range": clip_range_schedule,
    #     "n_steps": 512
    # }
    # model = PPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    checkpoint_interval = 31250 
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_ryu_vs_mbison")

    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=int(5000000), 
            callback=[checkpoint_callback]
        )
        env.close()

    sys.stdout = original_stdout

    model.save(os.path.join(save_dir, "ppo_sf2_ryu_vs_mbison_final.zip"))

if __name__ == "__main__":
    main()
