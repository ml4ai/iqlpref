import jax.numpy as jnp
import jax
from flax import nnx
import orbax.checkpoint as ocp
from CORL.reward_models.utils import prng_to_raw, raw_to_prng


class Identity(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, X: jax.Array, *args, **kwargs):
        return X


class Q_MLP(nnx.Module):
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 3,
        hidden_dims: list[int] = [256, 256],
        orthogonal_init: bool = False,
        activations: str = "relu",
        activation_final: str = "none",
        rngs=nnx.Rngs(0, params=1, dropout=2),
    ):
        # Setup activation function
        options = {
            "cos": jnp.cos,
            "tanh": nnx.tanh,
            "relu": nnx.relu,
            "softplus": nnx.softplus,
            "sin": jnp.sin,
            "leaky_relu": nnx.leaky_relu,
            "swish": nnx.swish,
            "none": Identity(),
        }
        self.activations = options[activations]

        self.activation_final = options[activation_final]

        # Initialize layers
        if orthogonal_init:
            self.layers = [
                nnx.Linear(
                    state_dim + action_dim,
                    hidden_dims[0],
                    kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            ]

            for i in range(1, len(hidden_dims)):
                self.layers.append(
                    nnx.Linear(
                        hidden_dims[i - 1],
                        hidden_dims[i],
                        kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                        bias_init=nnx.initializers.zeros_init(),
                        rngs=rngs,
                    )
                )

            self.output_layer = nnx.Linear(
                hidden_dims[-1],
                1,
                kernel_init=nnx.initializers.orthogonal(1e-2),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
        else:
            self.layers = [
                nnx.Linear(
                    state_dim + action_dim,
                    hidden_dims[0],
                    rngs=rngs,
                )
            ]

            for i in range(1, len(hidden_dims)):
                self.layers.append(
                    nnx.Linear(
                        hidden_dims[i - 1],
                        hidden_dims[i],
                        rngs=rngs,
                    )
                )

            self.output_layer = nnx.Linear(
                hidden_dims[-1],
                1,
                kernel_init=nnx.initializers.variance_scaling(
                    1e-2, "fan_in", "uniform"
                ),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)

        for linear_layer in self.layers:
            x = self.activations(linear_layer(x))

        return jnp.squeeze(self.activation_final(self.output_layer(x)), -1)


def load_QMLP(model_dir, chkptr, on_cpu=False):
    model_args = chkptr.restore(
        model_dir,
        args=ocp.args.Composite(
            model_args=ocp.args.ArrayRestore(),
        ),
    )
    model_args = model_args["model_args"]
    rng_key = jax.random.key(int(model_args[5]))
    rng_key, _ = jax.random.split(rng_key, 2)
    rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
    rngs = nnx.Rngs(rng_subkey1, params=rng_subkey2, dropout=rng_subkey3)
    options = [
        "cos",
        "tanh",
        "relu",
        "softplus",
        "sin",
        "leaky_relu",
        "swish",
        "none",
    ]
    activations = None
    activation_final = None
    for i, j in enumerate(options):
        if model_args[3] == i:
            activations = j
        if model_args[4] == i:
            activation_final = j

    hidden_dims = [int(k) for k in model_args[6:]]
    model = Q_MLP(
        state_dim=int(model_args[0]),
        action_dim=int(model_args[1]),
        hidden_dims=hidden_dims,
        orthogonal_init=bool(model_args[2]),
        activations=activations,
        activation_final=activation_final,
        rngs=rngs,
    )
    prng_to_raw(model)
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    # Loads onto first cpu found
    if on_cpu:

        def set_sharding(var):
            var.sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
            return var

        abstract_state = jax.tree.map(set_sharding, abstract_state)
    model_state = chkptr.restore(
        model_dir,
        args=ocp.args.Composite(
            model_state=ocp.args.StandardRestore(abstract_state),
        ),
    )
    model = nnx.merge(graphdef, model_state["model_state"])
    raw_to_prng(model)
    return model
