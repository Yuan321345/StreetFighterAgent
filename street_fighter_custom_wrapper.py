import math
import time
import collections

import gym
import numpy as np

# 自定义环境包装器
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # 使用双端队列保存最近 9 帧画面
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering
    
    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # 清空帧缓存，并将初始观测重复压入 [num_frames] 次
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def step(self, action):
        custom_done = False

        obs, _reward, _done, info = self.env.step(action)
        self.frame_stack.append(obs[::2, ::2, :])

        # 若 rendering 标志为 True，则渲染游戏画面
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        for _ in range(self.num_step_frames - 1):
            
            # 将当前动作持续按住 (num_step_frames - 1) 帧
            obs, _reward, _done, info = self.env.step(action)
            self.frame_stack.append(obs[::2, ::2, :])
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        
        self.total_timesteps += self.num_step_frames
        
        # 对局结束且玩家失败
        if curr_player_health < 0:
            custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # 将对手剩余生命值作为惩罚项。
                                                   # 如果对手生命值也小于 0，则视为平局，此时奖励为 +1。
            custom_done = True

        # 对局结束且玩家获胜
        elif curr_oppont_health < 0:
            # custom_reward = curr_player_health * self.reward_coeff # 将玩家剩余生命值作为奖励。
                                                                   # 乘以 reward_coeff，使胜利奖励大于失败惩罚，避免智能体出现过度保守行为。

            # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # 将剩余时间步作为奖励。
            custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
            custom_done = True

        # 对战仍在继续时
        else:
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # 当 reset_round 为 False 时，不在回合结束后重置，对局持续进行
        if not self.reset_round:
            custom_done = False
             
        # 最大奖励约为 6 * full_hp = 1054（伤害奖励 * 3 + 胜利奖励 * 3），归一化系数为 0.001
        return self._stack_observation(), 0.001 * custom_reward, custom_done, info 
    
