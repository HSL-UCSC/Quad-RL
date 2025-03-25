import os
import torch as th
import matplotlib.pyplot as plt
from obstacleavoidance_env import ObstacleAvoidance
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

load = True
train = True
save = False

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = ObstacleAvoidance()
env = Monitor(env, log_dir)

if load:
    # loading in the model
    model = DQN.load("dqn_obstacleavoidance", env)
else:
    # building the model
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=[64, 64])
    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                learning_rate=0.0001946, gamma= 0.9664)


if train:
    # Separate evaluation env
    eval_env = ObstacleAvoidance()
    
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=96, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
    timesteps = 1000000
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    if save:
        model.save("dqn_obstacleavoidance")

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "DQN Obstacle Avoidance")
plt.show()