import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class LinDynEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.states_dim = 2
    self.extra_dim = 0
    self.salient_states_dim = self.states_dim - self.extra_dim

    A_all = {}

    A_all[10] = np.array([[0.9 , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                   [0.01, 0.2 , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                   [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                   [0.01, 0.01, 0.01, 0.6 , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                   [0.01, 0.01, 0.01, 0.01, 0.8 , 0.01, 0.01, 0.01, 0.01, 0.01],
                   [0.01, 0.01, 0.01, 0.01, 0.01, 0.7 , 0.01, 0.01, 0.01, 0.01],
                   [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1 , 0.01, 0.01, 0.01],
                   [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.4 , 0.01, 0.01],
                   [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.3 , 0.01],
                   [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.5 ]])

    A_all[5] = np.array([[-0.2,  0.1,  0.1,  0.1,  0.1],
                  [ 0.1,  0.1,  0.1,  0.1,  0.1],
                  [ 0.1,  0.1,  0.5,  0.1,  0.1],
                  [ 0.1,  0.1,  0.1,  0.8,  0.1],
                  [ 0.1,  0.1,  0.1,  0.1, -0.9]])
    
    A_all[4] = np.array([[-0.2,  0.3,  0.3,  0.3],
                  [ 0.3, -0.4,  0.3,  0.3],
                  [ 0.3,  0.3,  0.3,  0.3],
                  [ 0.3,  0.3,  0.3, -0.1]])
    
    A_all[3] = np.array([[-0.5, -0.5, -0.5],
                  [ 0.3, -0.2,  0.3],
                  [ 0.3,  0.3,  0.4]])

    A_all[2] = np.array([[0.9, 0.4], [-0.4, 0.9]])

    A_all[1] = np.array([0.4])


    A_numpy = A_all[self.salient_states_dim]

    self.A = A_numpy
    self.discount = 0.9

    self.seed()
    self.viewer = None
    self.x = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def step(self, action):
    reward = -(np.dot(self.x.T, self.x) + np.dot(action.T,action))
    if self.extra_dim > 0:
      x_prime = np.asarray(A.dot(self.x[:-extra_dim])) #in case only one dimension is relevant
    else:
      x_prime = A.dot(self.x)

    x_prime = add_irrelevant_features(x_prime, extra_dim=self.extra_dim, noise_level=0.4)
    x_prime = x_prime + action
     
    self.x = x_prime
    return x_prime, reward, 0, {}


  def reset(self):
    self.x = 2*np.random.random(size = (self.salient_states_dim,)) - 0.5
    return self.x   

  def render(self, mode='human'):
    pass

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None


  def add_irrelevant_features(x, extra_dim, noise_level = 0.4):
    if isinstance(x, np.ndarray):
      x_irrel= noise_level*np.random.randn(1, extra_dim).reshape(-1,)
      return np.hstack([x, x_irrel])

    elif isinstance(x, torch.Tensor):
      x_irrel= noise_level*torch.randn(x.shape[0],x.shape[1],extra_dim).double().to(device)   
      return torch.cat((x, x_irrel),2)


