import gym
from gym_minigrid.wrappers import VectorObsWrapper


def make_env(env_name):
    # TODO Set horizon from config file
    def _make_env():
        env = gym.make(env_name)
        env.max_steps = min(env.max_steps, 20)
        return VectorObsWrapper(env)        
    return _make_env


class Sampler(object):
    def __init__(self,
                 env_name,
                 env_kwargs,
                 batch_size,
                 policy,
                 seed=None,
                 env=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        self.seed = seed

        if env is None:
            env = gym.make(env_name, **env_kwargs)
        self.env = env
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        self.env.close()
        self.closed = False

    def sample_async(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        return self.sample_async(*args, **kwargs)
