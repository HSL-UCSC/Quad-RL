import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from obstacleavoidance_env import ObstacleAvoidance
from stable_baselines3 import DQN
from utils import find_critical_points, state_to_observation_OA, \
    get_state_from_env_OA, find_X_i, \
        train_hybrid_agent, M_i, M_ext, HyRL_agent, \
            simulate_obstacleavoidance, visualize_M_ext

if __name__ == '__main__':
    # Loading in the trained agent
    model = DQN.load("dqn_obstacleavoidance")
    
    # finding the set of critical points
    resolution = 30
    x_ = np.linspace(0, 3, resolution)
    y_ = np.linspace(-1.5, 1.5, resolution)
    state_difference = LA.norm(np.array([x_[1]-x_[0], y_[1]-y_[0]]))
    initial_points = []

    for idx in range(resolution):
        for idy in range(resolution):
            initial_points.append(np.array([x_[idx], y_[idy]], 
                                           dtype=np.float32))
    
    M_star = find_critical_points(initial_points, state_difference, model, 
                                  ObstacleAvoidance, min_state_difference=1e-2, 
                                  steps=5, threshold=1e-1, n_clusters=8, 
                                  custom_state_to_observation=state_to_observation_OA,
                                  get_state_from_env=get_state_from_env_OA,
                                  verbose=False)
    M_star = M_star[np.argsort(M_star[:,0])]
    
    # building sets M_0 and M_1
    M_0 = M_i(M_star, index=0)
    M_1 = M_i(M_star, index=1)
    
    # finding the extension sets
    X_0 = find_X_i(M_0, model)
    X_1 = find_X_i(M_1, model)
    M_ext0 = M_ext(M_0, X_0)
    M_ext1 = M_ext(M_1, X_1)
    
    # visualizing the extended sets
    visualize_M_ext(M_ext0, figure_number=1)
    visualize_M_ext(M_ext1, figure_number=2)
    
    # building the environment for hybrid learning
    env_0 = ObstacleAvoidance(hybridlearning=True, M_ext=M_ext0)
    env_1 = ObstacleAvoidance(hybridlearning=True, M_ext=M_ext1)
    
    # training the new agents
    training2 = False
    if training2:
        agent_0 = train_hybrid_agent(env_0, load_agent='dqn_obstacleavoidance', 
                                      save_name='dqn_obstacleavoidance_0',
                                      M_exti=M_ext0, timesteps=300000)
        agent_1 = train_hybrid_agent(env_1, load_agent='dqn_obstacleavoidance', 
                                      save_name='dqn_obstacleavoidance_1',
                                      M_exti=M_ext1, timesteps=300000)
    else:
        agent_0 = DQN.load('dqn_obstacleavoidance_0')
        agent_1 = DQN.load('dqn_obstacleavoidance_1')
    
    # simulation the hybrid agent compared to the original agent    
    starting_conditions = [np.array([0., 0.0], dtype=np.float32),
                           np.array([0., 0.055], dtype=np.float32),
                           np.array([0., -.055], dtype=np.float32),
                           np.array([0., 0.15], dtype=np.float32),
                           np.array([0., -.15], dtype=np.float32),]
    for q in range(2):
        for state_init in starting_conditions:
            hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, 
                                      q_init=q)
            simulate_obstacleavoidance(hybrid_agent, model, state_init, 
                                       figure_number=3+q)
        save_name = 'OA_HyRLDQN_Sim_q'+str(q)+'.eps'
        plt.savefig(save_name, format='eps')