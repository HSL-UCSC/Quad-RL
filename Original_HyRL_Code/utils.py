import os
import numpy as np
import torch as th
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.cluster import KMeans
from obstacleavoidance_env import ObstacleAvoidance
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, \
    StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
matplotlib.rcParams['text.usetex'] = True

def find_critical_points(initial_points, state_difference, model, Env,
                         min_state_difference, steps, threshold,
                         n_clusters=8, custom_state_init=None,
                         custom_state_to_observation=None,
                         get_state_from_env=None, verbose=False):
    
    def generate_rod(center_point, dimension, state_difference):
        ndims = center_point.ndim+1
        if ndims > 1:
            displacement = np.zeros(ndims)
            displacement[dimension] = state_difference
        else:
            displacement = state_difference
        rod = [center_point + displacement, center_point - displacement]
        return rod

    def get_rod_length(rod_points):
        return LA.norm(rod_points[0]-rod_points[1])
    # initialize the set of points to consider
    next_points = initial_points
    while state_difference > min_state_difference:
        new_points = []
        if verbose:
            print('number of points: ', len(next_points))
        for center_point in next_points: 
            for dim in range(center_point.ndim+1):
                # along each dimension of the point, do the following:
                    # 1) create a "rod", i.e., consider two points close to the
                    # original point. 
                    # 2) compute the length of the rod before simulation, i.e.,
                    # the distance between the two points
                    # 3) simulate the system for n_steps
                    # 4) compute the distance between the rod after simulation
                    # 5) if the ending length of the rod is greater than the 
                    # starting length, the points diverged from each other.
                    # Then the original center point is potentially a critical
                    # point.
                # creating the rod
                start_points = generate_rod(center_point, dim, state_difference)
                # compute the length of the starting rod
                rod_length_start = get_rod_length(start_points)
                end_points = []
                steps_left_total = 0
                # simulate the system for both points in the rod
                for start in start_points:
                    if custom_state_init is not None:
                        start = custom_state_init(start)
                    env = Env(steps=steps, random_init=False, 
                                            state_init=start)
                    if custom_state_to_observation is None:
                        obs = np.copy(start)
                    else:
                        obs = custom_state_to_observation(np.copy(start))
                    done = False
                    while done == False:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, done, _ = env.step(action)
                    steps_left = env.steps_left
                    if get_state_from_env is None:
                        end_points.append(env.state)
                    else:
                        end_points.append(get_state_from_env(env))
                    steps_left_total += steps_left
                if steps_left_total == 0:
                    # compute the final length of the rod (after simulation)
                    rod_length_end = get_rod_length(end_points)
                    # if the length of the rod increased, create new points for
                    # the next loop
                    if (rod_length_end-rod_length_start) > threshold:
                        # the new points for the next loop are taken slightly
                        # spaced from the orignal point
                        new_rod = generate_rod(center_point, dim, 
                                               state_difference/4)
                        new_points.extend(new_rod)
        # finally, K-means clustering is used to find n_disconnected sets of
        # critical points
        next_points = new_points
        state_difference = state_difference/2
        cluster_array = np.array(next_points)
        if center_point.ndim+1 == 1:
            cluster_array = cluster_array.reshape(-1,1)
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=0).fit(cluster_array)
        cluster_centers = kmeans.cluster_centers_
        if center_point.ndim+1 == 1:
            cluster_centers = cluster_centers[0]
    return cluster_centers

def state_to_observation_OA(state, x_obst=1.5, y_obst=0., radius_obst=0.75, 
                         x_goal=3., y_goal=0):
    x, y = state[0], state[1]
    dist_obst = max(np.sqrt((x-x_obst)**2 + \
                        (y-y_obst)**2) - radius_obst, 0.)
    dist_goal = np.sqrt((x-x_goal)**2 + \
                        (y-y_goal)**2)
    return np.array([dist_obst, dist_goal, y], dtype=np.float32)

def get_state_from_env_OA(env):
    return np.array([env.x, env.y], dtype=np.float32)

def find_X_i(M_i, model, horizon=0.3, n_sims=100, t_sampling=0.05):
    
    def generate_start_point(center, mag=0.05):
        return center + mag * (np.random.rand(2)-0.5)
    
    def get_point_from_env(env):
        return np.array([env.x, env.y], dtype=np.float32)
    
    steps = int(horizon/t_sampling)        
    extension = []
    for center in M_i.M_star:
        X_i = [center]        
        for _ in range(n_sims):
            start_point = generate_start_point(center)
            while M_i.in_M(start_point) == False:
                start_point = generate_start_point(center)
            env_bw = ObstacleAvoidance(steps=steps, random_init=False,
                                       state_init=start_point, backwards=True)
            done = False
            obs = env_bw.state
            while done == False:
                point = get_point_from_env(env_bw)
                if M_i.in_M(point) == False:
                    X_i.append(point)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env_bw.step(action)
                
        distance = []
        for entry in X_i:
            distance.append(LA.norm(center-entry))
        indx = distance.index(max(distance))
        extension.append(X_i[indx])
    return extension
 
def train_hybrid_agent(env, save_name, M_exti, load_agent=None, timesteps=200000):
    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    
    # wrap the environment
    env = Monitor(env, log_dir)
    
    # Separate evaluation env
    eval_env = ObstacleAvoidance(hybridlearning=True, M_ext=M_exti)
    
    if load_agent is not None:
        # loading the existing model
        model = DQN.load(load_agent, env)
    else:
        # building the model
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=[64, 64])
        model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                    learning_rate=0.0001946, gamma= 0.9664)
    
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=110, 
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env, 
                                 callback_on_new_best=callback_on_best, 
                                 verbose=1)
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(save_name)
    return model

class M_i():
    def __init__(self, M_star, index):
        self.M_star = M_star
        self.len_M = len(M_star)
        self.index = index
    
    def get_linecoeff(self, points):
        A = np.array([[points[0,0], 1],
                      [points[1,0], 1]], dtype=np.float32)
        B = np.array([points[0,1],
                     points[1,1]], dtype=np.float32)
        self.a, self.b = np.linalg.solve(A, B)

    def check_ab_line(self, points):
        self.get_linecoeff(points)
        if self.sign*self.y >= self.sign*(self.a*self.x + self.b):
            return True
        else:
            return False

    def check_x(self, points):
        if self.x >= points[0,0] and self.x <= points[1,0]:
            return True
        else:
            return False
    
    def in_M(self, state):
        self.x, self.y = state[0], state[1]
        if self.index == 0:
            self.sign = 1
            if self.y >= 0 and self.x >= self.M_star[-1][0]:
                return True
            for idx in range(self.len_M-1):
                points = self.M_star[self.len_M-2-idx:self.len_M-idx]
                if self.check_x(points) and \
                    self.check_ab_line(points):
                    return True
            return False
        elif self.index == 1:
            self.sign = -1
            if self.y < 0 and self.x >= self.M_star[-1][0]:
                return True
            for idx in range(self.len_M-1):
                points = self.M_star[self.len_M-2-idx:self.len_M-idx]
                if self.check_x(points) and \
                    self.check_ab_line(points):
                    return True
            return False
        else:
            print('Warning! Index out of bounds!')

class M_ext():
    def __init__(self, M_i, X_i):
        self.M_i = M_i
        X_i.append(self.M_i.M_star[-1,:])
        self.X_i = np.array(X_i)
        self.len_X = len(X_i)
        
    def get_linecoeff(self, points):
        A = np.array([[points[0,0], 1],
                      [points[1,0], 1]], dtype=np.float32)
        B = np.array([points[0,1],
                     points[1,1]], dtype=np.float32)
        self.a, self.b = np.linalg.solve(A, B)

    def check_ab_line(self, points):
        self.get_linecoeff(points)
        if self.sign*self.y >= self.sign*(self.a*self.x + self.b):
            return True
        else:
            return False

    def check_x(self, points):
        if self.x >= points[0,0] and self.x <= points[1,0]:
            return True
        else:
            return False
    
    def in_M_ext(self, state):
        self.x, self.y = state[0], state[1]
        if self.M_i.index == 0:
            self.sign = 1
            if self.M_i.in_M(state):
                return True
            for idx in range(self.len_X-1):
                points = self.X_i[self.len_X-2-idx:self.len_X-idx]
                if self.check_x(points) and \
                    self.check_ab_line(points):
                    return True
            return False
        
        elif self.M_i.index == 1:
            self.sign = -1
            if self.M_i.in_M(state):
                return True
            for idx in range(self.len_X-1):
                points = self.X_i[self.len_X-2-idx:self.len_X-idx]
                if self.check_x(points) and \
                    self.check_ab_line(points):
                    return True
            return False

        else:
            print('Warning! Index out of bounds!')

class HyRL_agent():
    def __init__(self, agent_0, agent_1, M_ext0, M_ext1, q_init=0):
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.M_ext0 = M_ext0
        self.M_ext1 = M_ext1
        self.q = q_init
        
    def predict(self, state):
        switch = -10
        if self.q == 0:
            if self.M_ext0.in_M_ext(state):
                active_agent = self.agent_0
            else:
                switch = 1
                self.q = 1
                active_agent = self.agent_1
                
        elif self.q == 1:
            if self.M_ext1.in_M_ext(state):
                active_agent = self.agent_1
            else:
                switch = 1
                self.q = 0
                active_agent = self.agent_0
        action, _ = active_agent.predict(state_to_observation_OA(state),
                                         deterministic=True)
        return action, switch

def simulate_obstacleavoidance(hybrid_agent, original_agent, state_init, 
                               noise_mag=0.1, figure_number=3, 
                               show_switches=False):
    env_or = ObstacleAvoidance(state_init=state_init, random_init=False)
    env_hyb = ObstacleAvoidance(state_init=state_init, random_init=False)
    done = False
    state_or, state_hyb = state_init, state_init
    states_or, states_hyb = [state_or], [state_hyb]
    _, switch = hybrid_agent.predict(state_hyb)
    switches = []
    score_or, score_hyb = 0, 0
    sign = 1
    stop_appending_or, stop_appending_hyb = False, False
    while done == False:
        noise = noise_mag * sign
        disturbance = np.array([0, noise], dtype=np.float32)

        action_or, _ = original_agent.predict(
                            state_to_observation_OA(state_or+disturbance),
                            deterministic=True)
        action_hyb, switch = hybrid_agent.predict(state_hyb+disturbance)
        
        env_or.state = state_or
        env_hyb.state = state_hyb
        
        _, reward_or, done, _ = env_or.step(action_or)
        _, reward_hyb, done, _ = env_hyb.step(action_hyb)
        state_or = get_state_from_env_OA(env_or)
        state_hyb = get_state_from_env_OA(env_hyb)
        
        if env_or.terminate:
            stop_appending_or = True
        if stop_appending_or == False:
            states_or.append(state_or)
            score_or += reward_or
            
        if env_hyb.terminate:
            stop_appending_hyb = True
        if stop_appending_hyb == False:
            states_hyb.append(state_hyb)
            score_hyb += reward_hyb
            switches.append(switch*states_hyb[-1])
        
        sign *= -1

    plt.figure(figure_number)
    plt.plot(np.array(states_hyb)[:,0], np.array(states_hyb)[:,1], 'blue',
             linewidth=3)
    plt.plot(np.array(states_or)[:,0], np.array(states_or)[:,1], 'red', 
             linestyle='--', linewidth=3)
    plt.plot(np.array(states_or)[0,0], np.array(states_or)[0,1], 'o', 
             color='red', markersize=15, fillstyle='none')
    plt.plot(np.array(states_or)[-1,0], np.array(states_or)[-1,1], 'x', 
             color='red', markersize=22, fillstyle='none')
    plt.plot(np.array(states_hyb)[-1,0], np.array(states_hyb)[-1,1], 'x', 
             color='blue', markersize=22, fillstyle='none')
    if show_switches:
        plt.plot(np.array(switch)[:,0], np.array(switches)[:,1], 'x', markersize=15, color='black', linewidth=2,
                  fillstyle='none')
    obstacle = matplotlib.patches.Circle((1.5,0.), radius=.75, color='gray')
    plt.gca().add_patch(obstacle)
    plt.text(1.42, -0.1, '$\mathcal{O}$', fontsize=22)
    plt.grid(visible=True)
    plt.xticks(np.linspace(0, 3, num=7, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1.5, 1.5, num=7, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()
    print('reward original', score_or, ' reward hybrid', score_hyb)    
    
def visualize_M_ext(M_ext, figure_number, resolution=50):
    plt.figure(figure_number)
    x_ = np.linspace(0, 3, resolution)
    y_ = np.linspace(-1.5, 1.5, resolution)
    x, y = np.meshgrid(x_, y_)
    in_M = np.zeros((resolution,resolution))
    
    for idy in range(resolution):
        for idx in range(resolution):
            in_M[idy, idx] = M_ext.in_M_ext(np.array([x_[idx], y_[idy]]))
    plt.scatter(x, y, s=15, c=in_M)
    plt.plot(M_ext.M_i.M_star[:,0], M_ext.M_i.M_star[:,1], color='red')
    obstacle = matplotlib.patches.Circle((1.5,0.), radius=.75, color='gray')
    plt.gca().add_patch(obstacle)
    plt.text(1.42, -0.1, '$\mathcal{C}$', fontsize=22)
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    plt.clim(-1,1)
    cbar.ax.tick_params(labelsize=18)
    plt.grid()
    plt.xticks(np.linspace(0, 3, num=7, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1.5, 1.5, num=7, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()