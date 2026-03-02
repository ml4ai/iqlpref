import os
import sys

sys.path.insert(0, os.path.abspath("gp_reward-priors"))
sys.path.insert(0, os.path.abspath("../"))
from optbnn.bnn.likelihoods import LikCE
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.priors import OptimGaussianPrior
from optbnn.sgmcmc_bayes_net.pref_net import PrefNet
from optbnn.utils import util

util.set_seed(6)
# Initialize BNN Priors
width = 64  # Number of units in each hidden layer
depth = 4  # Number of hidden layers
transfer_fn = "relu"  # Activation function

X_train, y_train, _, _ = util.load_pref_data(
    "./gp_reward-priors/data/antmaze/antmaze-large-play-v2_pref.hdf5", 0.3
)

prior_dir = "./gp_reward-priors/exp/reward_learning/antmaze_tuning_star/br-antmaze_large_play-64-4/ckpts/it-1000.ckpt"

# Initialize the prior
prior = OptimGaussianPrior(prior_dir)

# Setup likelihood
net = MLP(37, 1, [width] * depth, transfer_fn)
likelihood = LikCE()

# Initialize the sampler
saved_dir = os.path.join(
    "./antmaze_models/antmaze_br/optim_star/antmaze-large-play-v2/reduce_70_br",
    "sampling_optim",
)
util.ensure_dir(saved_dir)
bayes_net_std = PrefNet(net, likelihood, prior, saved_dir, n_gpu=1, name="optim")

bayes_net_std.sampled_weights = bayes_net_std._load_sampled_weights(
    os.path.join(saved_dir, "sampled_weights", "sampled_weights_0000003")
)

bayes_net_std.find_map(X_train, y_train)
bayes_net_std.save_map()
