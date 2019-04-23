
from gym.envs.registration import register

register(
    id='lin-dyn-v0',
    entry_point='gym_linear_dynamics.envs:LinDynEnv',
)

