import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from obstacleavoidance_env import ObstacleAvoidance
matplotlib.rcParams['text.usetex'] = True

def compute_observation(x, y, x_obst=1.5, y_obst=0., radius_obst=0.75, 
                        x_goal=3.0, y_goal=0.):
    dist_obst = max(np.sqrt((x-x_obst)**2 + \
                        (y-y_obst)**2) - radius_obst, 0.)
    dist_goal = np.sqrt((x-x_goal)**2 + \
                        (y-y_goal)**2)

    observation = np.array([dist_obst, dist_goal, y], 
                           dtype=np.float32)
    return observation

def plot_policy(model, resolution=150, figure_number=1):
    plt.figure(figure_number)
    x_ = np.linspace(0, 3, resolution)
    y_ = np.linspace(-1.5, 1.5, resolution)
    actions = np.zeros((resolution,resolution))
    for idy in range(resolution):
        for idx in range(resolution):
            obs = compute_observation(x_[idx], y_[idy])
            action, _ = model.predict(obs, deterministic=True)
            actions[idy, idx] = (action-2)/2
    x, y = np.meshgrid(x_, y_)
    plt.scatter(x, y, s=15, c=actions)
    obstacle = matplotlib.patches.Circle((1.5,0.), radius=.75, color='gray')
    critical = matplotlib.patches.Circle((0.375,0.), radius=.375, color='red', 
                                         fill=None)
    plt.gca().add_patch(obstacle)
    plt.gca().add_patch(critical)
    plt.text(1.42, -0.1, '$\mathcal{O}$', fontsize=22)
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('$u$', rotation=270, fontsize=22, labelpad=22)
    plt.grid()
    plt.xticks(np.linspace(0, 3, num=7, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1.5, 1.5, num=7, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()

plt.close('all')
env = ObstacleAvoidance()
model = DQN.load("dqn_obstacleavoidance", env)

plot_policy(model)
plt.savefig('ObstAvoid_criticPoint_policyMap.eps', format='eps')