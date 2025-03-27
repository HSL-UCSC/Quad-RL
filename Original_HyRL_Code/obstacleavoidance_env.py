import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class ObstacleAvoidance(gym.Env):
    def __init__(self, steps=60, random_init=True, 
                 state_init=np.array([0.,0.], dtype=np.float32), 
                 spread=np.array([0.5,1.], dtype=np.float32),
                 backwards=False, hybridlearning=False, M_ext=None):

        self.steps = steps
        self.random_init = random_init
        self.state_init = state_init
        self.spread = spread
        self.backwards = backwards
        self.hybridlearning = hybridlearning
        self.M_ext = M_ext
        
        self.min_x, self.max_x = 0., 3.
        self.min_y, self.max_y = -1.5, 1.5
        self.t_sampling = 0.05
        
        self.x_obst, self.y_obst = 1.5, 0.
        self.x_goal, self.y_goal = 3., 0.
        self.radius_obst = 0.75
        
        self.low_state = np.array(
            [0., 0., self.min_y], dtype=np.float32
        )
        self.high_state = np.array(
            [3, 4.5, self.max_y], dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(3,),
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def update_observation(self):
        dist_obst = max(np.sqrt((self.x-self.x_obst)**2 + \
                            (self.y-self.y_obst)**2) - self.radius_obst, 0.)
        dist_goal = np.sqrt((self.x-self.x_goal)**2 + \
                            (self.y-self.y_goal)**2)

        self.state = np.array([dist_obst, dist_goal, self.y], 
                              dtype=np.float32)
    def check_terminate(self):
        self.terminate = self.state[0] <= 1e-3 or abs(self.y) >= self.max_y \
                                        or self.x > self.max_x
    
    def step(self, action):        
        force = (action-2)/2
        if self.backwards==False:
            sign = 1
        else:
            sign = -1
            
        if self.hybridlearning == True:
            if self.M_ext.in_M_ext(np.array([self.x, self.y], 
                                            dtype=np.float32)) == False:
                # this is to prevet leaving the extended set during training 
                sign = 0
            pass
        
        self.x += sign * self.t_sampling
        self.y += sign * force * self.t_sampling

        
        self.update_observation()
        
        # Calculate reward
        barrier = (self.state[0] - 2*self.radius_obst)**2 - \
                    np.log(max(self.state[0], 1e-6))
        reward = max(0, -self.state[1] - 0.1*barrier + 3.5)
        
        # Update steps left
        self.steps_left -= 1
        
        # Check if illegal states are reached
        self.check_terminate()
        
        # Check if simulation is done
        if self.steps_left <= 0 or self.terminate:
            done = True
        else:
            done = False 
            
        # Set placeholder for info
        info = {}
        return self.state, reward, done, info
    
    def reset(self):
        self.terminate = False
        self.x, self.y = self.state_init[0], self.state_init[1]
        if self.random_init==True:
            # initialize system around the initial state
            self.x = np.float32(self.state_init[0] + \
                                self.spread[0]*np.random.uniform(0, 1))
            self.y = np.float32(self.state_init[1] + \
                                self.spread[1]*np.random.uniform(-1., 1.))
            if self.hybridlearning:
                # ensure that the initial state is in the extended set
                while self.M_ext.in_M_ext(np.array([self.x, self.y], 
                                                dtype=np.float32)) == False:
                    self.x = np.float32(self.state_init[0] + \
                                    self.spread[0]*np.random.uniform(0, 1))
                    self.y = np.float32(self.state_init[1] + \
                                    self.spread[1]*np.random.uniform(-1., 1.))
        self.update_observation()
        # set the total number of episode steps
        self.steps_left = self.steps
        return self.state