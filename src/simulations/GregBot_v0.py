import gym
import mujoco_py
import numpy as np
import os


class GregBot(gym.Env):
    def __init__(self, xml_path='car210.xml', simend=50):
        # Initialize MuJoCo data structures
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None  # Viewer for rendering
        self.simend = simend

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32)

    def reset(self):
        # Reset the environment to its initial state
        self.sim.reset()
        return self._get_observation()

    def step(self, action):
        # Take a step in the environment based on the given action
        self._apply_action(action)
        self.sim.step()
        observation = self._get_observation()
        reward = 0.0  # You can define your own reward function
        done = self.sim.data.time >= self.simend
        info = {}  # Additional information if needed

        return observation, reward, done, info

    def render(self, mode='human'):
        # Render the current state of the environment
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()

    def close(self):
        # Clean up resources when the environment is closed
        if self.viewer is not None:
            self.viewer = None

    def _apply_action(self, action):
        # Apply the action to the MuJoCo model
        self.sim.data.ctrl[:] = [action[0], action[1]]
        

    def _get_observation(self):
        # Return the current observation
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel])

# Register the environment
# gym.envs.register(
#     id='GregBot-v0',
#     entry_point='your_script_name:MyMujocoEnv',  # Replace 'your_script_name' with the actual script name
# )

# Create and use the environment
# env = gym.make('GregBot-v0')

env = GregBot()
observation = env.reset()

for _ in range(10000000000000000000000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        observation = env.reset()

env.close()
