import jax
import jax.numpy as jnp
from flax import nnx


def get_attention_mask(attn_mask, batch_size):
    assert batch_size > 0, "batch_size should be > 0."
    attn_mask = jnp.reshape(attn_mask, (batch_size, -1))
    attn_mask = jnp.expand_dims(attn_mask, axis=(1, 2))
    attn_mask = (1.0 - attn_mask) * -10000.0
    return attn_mask


def split_heads(x, num_heads, head_dim):
    """
    Splits embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
    """
    newshape = x.shape[:-1] + (num_heads, head_dim)
    x = jnp.reshape(x, newshape)
    if x.ndim == 5:
        # [batch, blocks, head, block_len, head_dim]
        return jnp.transpose(x, axes=(0, 1, 3, 2, 4))
    elif x.ndim == 4:
        # [batch, head, seq_len, head_dim]
        return jnp.transpose(x, axes=(0, 2, 1, 3))
    else:
        raise ValueError(
            f"Input tensor should have rank 4 or 5, but has rank {x.ndim}."
        )


def attention(
    query,
    key,
    value,
    casual_mask,
    masked_bias,
    dropout,
    scale_attn_weights,
    training,
    attn_mask=None,
    feedback=None,
):
    """
    Computes Dot-Product Attention for the given query, key and value.

    Args:
        query (tensor): Query, shape [B, num_heads, seq_len, embd_dim].
        key (tensor): Key, shape [B, num_heads, seq_len, embd_dim].
        value (tensor): Value, shape [B, num_heads, seq_len, embd_dim].
        casual_mask (tensor): Mask to ensure that attention is only applied to the left of the input sequence,
                              shape [1, 1, key_len - query_len :key_len, :key_len].
        masked_bias (float): Value to insert for masked part of the sequence.
        dropout (nn.Dropout): Dropout module that is applied to the attention output.
        scale_attn_weights (bool): If True, scale the attention weights.
        training (bool): Training mode.
        attn_mask (tensor): Mask to avoid performing attention on padded tokens indices, shape [B, seq_len].
        head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads,] or [num_layers, num_heads].
        feedback (tensor): external feedback with marked points.

    Returns:
        (tensor): Attention output, shape [B, num_heads, seq_len, embd_dim].
        (tensor): Attention weights, shape [B, num_heads, seq_len, seq_len].
        (tensor): KLD loss with external feedback, float.
    """
    query = query.astype(jnp.bfloat16)
    key = key.astype(jnp.bfloat16)
    attn_weights = jnp.matmul(query, jnp.swapaxes(key, -1, -2))

    if scale_attn_weights:
        attn_weights = attn_weights / (float(value.shape[-1]) ** 0.5)

    attn_weights = jnp.where(casual_mask, attn_weights, masked_bias)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    _attn_weights = nnx.softmax(attn_weights, axis=-1)
    attn_weights = _attn_weights.astype(value.dtype)
    attn_weights = dropout(attn_weights, deterministic=not training)

    out = jnp.matmul(attn_weights, value)
    return out, _attn_weights


def merge_heads(x, num_heads, head_dim):
    """
    Merge embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
    """
    if x.ndim == 5:
        x = jnp.transpose(x, axes=(0, 1, 3, 2, 4))
    elif x.ndim == 4:
        x = jnp.transpose(x, axes=(0, 2, 1, 3))
    else:
        raise ValueError(
            f"Input tensor should have rank 4 or 5, but has rank {x.ndim}."
        )

    newshape = x.shape[:-2] + (num_heads * head_dim,)
    x = jnp.reshape(x, newshape)
    return x


def apply_activation(x, activation="linear"):
    if activation == "linear":
        return x
    elif activation == "gelu_new":
        return (
            0.5
            * x
            * (
                1.0
                + nnx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0)))
            )
        )
    elif activation == "gelu_fast":
        return 0.5 * x * (1.0 + nnx.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    elif activation == "gelu":
        return nnx.gelu(x)
    elif activation == "relu":
        return nnx.relu(x)
    elif activation == "leaky_relu":
        return nnx.leaky_relu(x)
    elif activation == "sigmoid":
        return nnx.sigmoid(x)
    elif activation == "tanh":
        return nnx.tanh(x)
    else:
        raise ValueError(f"Unknown activation function: {activation}.")
