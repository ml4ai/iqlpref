import os
import sys

sys.path.insert(0, os.path.abspath("gp_reward-priors"))
sys.path.insert(0, os.path.abspath("../"))
from optbnn.bnn.likelihoods import LikCE
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.priors import FixedGaussianPrior
from optbnn.sgmcmc_bayes_net.pref_net import PrefNet
from optbnn.utils import util

util.set_seed(7)
# Initialize BNN Priors
width = 64  # Number of units in each hidden layer
depth = 2  # Number of hidden layers
transfer_fn = "relu"  # Activation function

X_train, y_train, _, _ = util.load_pref_data(
    "./gp_reward-priors/data/antmaze/antmaze-medium-diverse-v2_pref.hdf5", 0.3
)

# Initialize the prior
prior = FixedGaussianPrior(std=1.0)

# Setup likelihood
net = MLP(37, 1, [width] * depth, transfer_fn)
likelihood = LikCE()

# Initialize the sampler
saved_dir = os.path.join(
    "./antmaze_models/antmaze_br/FG/antmaze-medium-diverse-v2/reduce_70_br",
    "sampling_std",
)
util.ensure_dir(saved_dir)
bayes_net_std = PrefNet(net, likelihood, prior, saved_dir, n_gpu=1, name="FG")

bayes_net_std.sampled_weights = bayes_net_std._load_sampled_weights(
    os.path.join(saved_dir, "sampled_weights", "sampled_weights_0000003")
)

bayes_net_std.find_map(X_train, y_train)
bayes_net_std.save_map()
