import miniworld
import gymnasium
import numpy as np
from PIL import Image
### based on memorymaze.py


class MiniWorld(gymnasium.Env):
    def __init__(self, task="MiniWorld-Hallway-v0", obs_key="image",
                 act_key="action", size=(64, 64), seed=0):
        super().__init__()
        self._env = gymnasium.make(task)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = size
        self._gray = False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)
    
    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: gymnasium.spaces.Box(
                0, 255, (64, 64, 3), dtype=np.uint8
            ),}
            spaces["is_first"] = gymnasium.spaces.Box(0, 1, (), dtype=bool)
            spaces["is_last"] = gymnasium.spaces.Box(0, 1, (), dtype=bool)
            spaces["is_terminal"] = gymnasium.spaces.Box(0, 1, (), dtype=bool)
        return gymnasium.spaces.Dict(spaces)
    
    def _resize_obs(self, obs):
        img = obs[self._obs_key]
        img = Image.fromarray(img).resize(self._size, Image.NEAREST)
        obs[self._obs_key] = np.array(img)
        return obs
    
    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs = self._resize_obs(obs)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = terminated
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs = self._resize_obs(obs)
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs
