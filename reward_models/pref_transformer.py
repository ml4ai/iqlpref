import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import CORL.reward_models.ops as ops
from flax import nnx
from jax import lax
from CORL.reward_models.utils import prng_to_raw, raw_to_prng


class GPT2MLP(nnx.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        intermediate_dim: int = 256,
        resid_dropout: float = 0.1,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.in_linear = nnx.Linear(embd_dim, intermediate_dim, rngs=rngs)
        self.out_linear = nnx.Linear(intermediate_dim, embd_dim, rngs=rngs)
        self.resid_dropout = nnx.Dropout(resid_dropout, rngs=rngs)

    def __call__(self, x, training=False):
        x = self.in_linear(x)
        x = nnx.relu(x)
        x = self.out_linear(x)
        x = self.resid_dropout(x, deterministic=not training)
        return x


class GPT2SelfAttention(nnx.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        max_pos: int = 1024,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads
        self.max_pos = max_pos

        self.in_linear = nnx.Linear(embd_dim, 3 * embd_dim, rngs=rngs)
        self.attn_dropout = nnx.Dropout(attn_dropout, rngs=rngs)

        self.out_linear = nnx.Linear(embd_dim, embd_dim, rngs=rngs)
        self.resid_dropout = nnx.Dropout(resid_dropout, rngs=rngs)

    def __call__(self, x, attn_mask, training=False):
        x = self.in_linear(x)

        query, key, value = jnp.split(x, 3, axis=2)

        query = ops.split_heads(query, self.num_heads, self.head_dim)
        value = ops.split_heads(value, self.num_heads, self.head_dim)
        key = ops.split_heads(key, self.num_heads, self.head_dim)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.tril(jnp.ones((1, 1, self.max_pos, self.max_pos)))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        casual_mask = casual_mask.astype(bool)

        out, _attn_weights = ops.attention(
            query,
            key,
            value,
            casual_mask,
            -1e4,
            self.attn_dropout,
            True,
            training,
            attn_mask,
        )
        out = ops.merge_heads(out, self.num_heads, self.head_dim)

        out = self.out_linear(out)

        out = self.resid_dropout(out, deterministic=not training)
        return out, _attn_weights


class GPT2Block(nnx.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        intermediate_dim: int = 256,
        max_pos: int = 1024,
        eps: float = 1e-05,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.layer_norm_0 = nnx.LayerNorm(embd_dim, epsilon=eps, rngs=rngs)
        self.attention = GPT2SelfAttention(
            embd_dim=embd_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            max_pos=max_pos,
            rngs=rngs,
        )
        self.layer_norm_1 = nnx.LayerNorm(embd_dim, epsilon=eps, rngs=rngs)
        self.mlp = GPT2MLP(
            embd_dim=embd_dim,
            intermediate_dim=intermediate_dim,
            resid_dropout=resid_dropout,
            rngs=rngs,
        )

    def __call__(self, x, attn_mask, training=False):
        residual = x
        x = self.layer_norm_0(x)
        x, _attn_weights = self.attention(x, attn_mask, training)
        x += residual
        residual = x
        x = self.layer_norm_1(x)
        x = self.mlp(x, training)
        x += residual
        return x, _attn_weights


class GPT2Model(nnx.Module):
    def __init__(
        self,
        embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        intermediate_dim: int = 256,
        num_layers: int = 1,
        embd_dropout: float = 0.1,
        max_pos: int = 1024,
        eps: float = 1e-05,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.dropout = nnx.Dropout(embd_dropout, rngs=rngs)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                GPT2Block(
                    embd_dim=embd_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    intermediate_dim=intermediate_dim,
                    max_pos=max_pos,
                    eps=eps,
                    rngs=rngs,
                )
            )
        self.layer_norm = nnx.LayerNorm(embd_dim, epsilon=eps, rngs=rngs)

    def __call__(self, input_embds, attn_mask, training=False):
        x = self.dropout(input_embds, deterministic=not training)
        batch_size = input_embds.shape[0]
        attn_weights_list = []
        attn_mask = ops.get_attention_mask(attn_mask, batch_size)
        for m in self.layers:
            x, attn_weights = m(x, attn_mask, training)
            attn_weights_list.append(attn_weights)
        x = self.layer_norm(x)
        return {
            "last_hidden_state": x,
            "attn_weights_list": attn_weights_list,
        }


class PT(nnx.Module):
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 3,
        max_episode_steps: int = 1219,
        embd_dim: int = 64,
        pref_attn_embd_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        intermediate_dim: int = 256,
        num_layers: int = 1,
        embd_dropout: float = 0.1,
        max_pos: int = 1024,
        eps: float = 1e-05,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.embd_dim = embd_dim
        self.pref_attn_embd_dim = pref_attn_embd_dim

        self.state_linear = nnx.Linear(state_dim, embd_dim, rngs=rngs)
        self.action_linear = nnx.Linear(action_dim, embd_dim, rngs=rngs)
        self.timestep_embed = nnx.Embed(max_episode_steps + 1, embd_dim, rngs=rngs)
        self.stacked_layer_norm = nnx.LayerNorm(embd_dim, epsilon=eps, rngs=rngs)
        self.gpt = GPT2Model(
            embd_dim=embd_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            embd_dropout=embd_dropout,
            max_pos=max_pos,
            eps=eps,
            rngs=rngs,
        )
        self.pref_linear = nnx.Linear(embd_dim, 2 * pref_attn_embd_dim + 1, rngs=rngs)
        self.attn_dropout = nnx.Dropout(0.0, rngs=rngs)

    def __call__(self, states, actions, timesteps, attn_mask, training=False):
        batch_size, seq_length = states.shape[0], states.shape[1]

        embd_states = self.state_linear(states)
        embd_actions = self.action_linear(actions)

        embd_timesteps = self.timestep_embed(timesteps)

        embd_states = embd_states + embd_timesteps
        embd_actions = embd_actions + embd_timesteps

        stacked_inputs = (
            jnp.stack([embd_states, embd_actions], axis=1)
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.embd_dim)
        )

        stacked_inputs = self.stacked_layer_norm(stacked_inputs)

        stacked_attn_mask = (
            jnp.stack([attn_mask, attn_mask], axis=1)
            .transpose(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )

        transformer_outputs = self.gpt(
            input_embds=stacked_inputs, attn_mask=stacked_attn_mask, training=training
        )

        x = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]
        x = x.reshape(batch_size, seq_length, 2, self.embd_dim).transpose(0, 2, 1, 3)
        hidden_output = x[:, 1]

        x = self.pref_linear(hidden_output)

        num_heads = 1

        query, key, value = jnp.split(
            x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim * 2], axis=2
        )
        query = ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
        key = ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
        value = ops.split_heads(value, num_heads, 1)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.ones((1, 1, seq_length, seq_length))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        casual_mask = casual_mask.astype(bool)

        new_attn_mask = ops.get_attention_mask(attn_mask, batch_size)
        out, last_attn_weights = ops.attention(
            query,
            key,
            value,
            casual_mask,
            -1e-4,
            self.attn_dropout,
            scale_attn_weights=True,
            training=training,
            attn_mask=new_attn_mask,
        )
        attn_weights_list.append(last_attn_weights)

        output = ops.merge_heads(out, num_heads, 1)

        return {"weighted_sum": output, "value": value}, attn_weights_list


def load_PT(model_dir, chkptr, on_cpu=False):
    model_args = chkptr.restore(
        model_dir,
        args=ocp.args.Composite(
            model_args=ocp.args.ArrayRestore(),
        ),
    )
    model_args = model_args["model_args"]
    rng_key = jax.random.key(int(model_args[13]))
    rng_key, _ = jax.random.split(rng_key, 2)
    rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
    rngs = nnx.Rngs(rng_subkey1, params=rng_subkey2, dropout=rng_subkey3)
    model = PT(
        state_dim=int(model_args[0]),
        action_dim=int(model_args[1]),
        max_episode_steps=int(model_args[2]),
        embd_dim=int(model_args[3]),
        pref_attn_embd_dim=int(model_args[4]),
        num_heads=int(model_args[5]),
        attn_dropout=float(model_args[6]),
        resid_dropout=float(model_args[7]),
        intermediate_dim=int(model_args[8]),
        num_layers=int(model_args[9]),
        embd_dropout=float(model_args[10]),
        max_pos=int(model_args[11]),
        eps=float(model_args[12]),
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
