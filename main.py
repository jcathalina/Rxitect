from fragment_env import FragmentEnv
from sampler import Sampler
from gflownet import FragmentGFlowNet
import numpy as np


if __name__ == "__main__":
    env = FragmentEnv()
    gfn = FragmentGFlowNet(env=env, num_emb=256, out_per_stem=env.num_frags, out_per_mol=1, num_conv_steps=10)
    sampler = Sampler(train_rng=np.random.RandomState(seed=1234), sampling_model=gfn, env=env)

    x = sampler.sample_from_model()
    print(x)
