# src/sim_env/maniskill_env/scripts/final/miras.py

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from functools import partial

# -----------------------------------------------------------------------------
# 1) Simple RMSNorm
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        # x: (…, dim)
        norm2 = jnp.mean(x**2, axis=-1, keepdims=True)
        inv_rms = jax.lax.rsqrt(norm2 + self.eps)
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        return x * inv_rms * scale

# -----------------------------------------------------------------------------
# 2) The MLP memory M(·)
# -----------------------------------------------------------------------------
class MemoryMLP(nn.Module):
    d_model: int
    d_hidden: int = 512

    @nn.compact
    def __call__(self, k):
        x = nn.Dense(self.d_hidden)(k)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)
        return x

# -----------------------------------------------------------------------------
# 3) One chunk of Miras: vectorized grad and single update
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("p"))
def miras_chunk_update(
    params,                # pytree of MemoryMLP params *before* this chunk
    keys,                  # shape (chunk_size, d_model)
    vals,                  # shape (chunk_size, d_model)
    etas,                  # shape (chunk_size,)  inner-loop LRs
    betas,                 # shape (chunk_size,)  beta decay products
    p: float = 2.0,
    retention_lambda: float = 1e-2,
):
    """
    Runs one parallel chunk of length b = chunk_size:
      - computes per-token grad = ∇_param ℓ_p( M(params, k_i), v_i )
      - weights each grad by w_i = etas[i] * (betas[-1]/betas[i])
      - sums them → single aggregate grad
      - returns updated_params
    """
    # define per-token loss
    def loss_per_token(param_pytree, k, v):
        pred = MemoryMLP(d_model=k.shape[-1]).apply({"params": param_pytree}, k)
        return jnp.sum(jnp.abs(pred - v) ** p)  # ℓ_p^p

    # compute per-token grads in parallel
    grad_fn = jax.vmap(jax.grad(loss_per_token), in_axes=(None, 0, 0))
    grads = grad_fn(params, keys, vals)
    # grads is a pytree with leaves of shape (chunk_size, ...)

    # form weights w_i = etas[i] * (betas[-1]/betas[i])
    final_beta = betas[-1]
    weights = etas * (final_beta / betas)    # shape (chunk_size,)

    # weighted sum of grads
    def weighted_sum(tree):
        # tree has shape (chunk_size, *param_shape)
        return jnp.tensordot(weights, tree, axes=((0,), (0,)))
    agg_grad = jax.tree.map(weighted_sum, grads)

    # single GD step
    updated = jax.tree.map(lambda p, g: p - g, params, agg_grad)
    return updated

# -----------------------------------------------------------------------------
# 4) The full Miras-block over T timesteps (divided into chunks)
# -----------------------------------------------------------------------------
def miras_sequence_apply(
    init_params,
    all_keys,         # (T, d_model)
    all_vals,         # (T, d_model)
    alpha: float,
    eta0: float,
    chunk_size: int,
    p: float,
):
    """
    Splits T into non-overlapping chunks of size chunk_size.
    Uses miras_chunk_update on each chunk in parallel, then recalls.
    """
    T, d_model = all_keys.shape
    n_chunks = T // chunk_size

    # Pre-chunk into static blocks of shape (n_chunks, b, d_model)
    keys_chunks = all_keys.reshape(n_chunks, chunk_size, d_model)
    vals_chunks = all_vals.reshape(n_chunks, chunk_size, d_model)
    print(n_chunks)
    def chunk_body(carry, chunk):
        params = carry
        K, V = chunk  # each is (chunk_size, d_model)

        # build etas and betas
        etas = eta0 * (alpha ** jnp.arange(chunk_size))
        betas = alpha ** jnp.arange(chunk_size)

        # 1) update memory in parallel for this chunk
        new_params = miras_chunk_update(params, K, V, etas, betas, p=p)

        # 2) recall outputs for this chunk using the updated memory
        recall_fn = lambda k: MemoryMLP(d_model=d_model).apply({"params": new_params}, k)
        Y_chunk = jax.vmap(recall_fn)(K)  # (chunk_size, d_model)

        return new_params, Y_chunk

    # scan over all chunks
    (final_params, _), Y_chunks = jax.lax.scan(
        chunk_body,
        (init_params),
        (keys_chunks, vals_chunks),
    )

    # flatten back to (T, d_model)
    Y = Y_chunks.reshape(T, d_model)
    return final_params, Y

# -----------------------------------------------------------------------------
# 5) Plug into a tiny Flax model with Conv → RMSNorm → Miras → reshape
# -----------------------------------------------------------------------------
class MirasModel(nn.Module):
    d_model: int = 28*28*4
    chunk_size: int = 64
    alpha: float = 0.9
    eta0: float = 0.1
    p: float = 2.0

    @nn.compact
    def __call__(self, x):
        # x: (T, 4, 28, 28)
        T = x.shape[0]
        # conv encoders
        x = jnp.transpose(x, (0, 2, 3, 1)) 
        ck = nn.Conv(features=4, kernel_size=(3, 3), padding="SAME")(x)
        cv = nn.Conv(features=4, kernel_size=(3, 3), padding="SAME")(x)
        nk = RMSNorm(4)(ck).reshape(T, -1)         # (T, d_model)
        nv = RMSNorm(4)(cv).reshape(T, -1)
        print(nk.shape, x.shape, ck.shape)

        # init memory params
        mem = MemoryMLP(d_model=self.d_model)
        params = mem.init(self.make_rng("params"), nk[:1])["params"]

        # run the parallel Miras sequence
        final_params, Y = miras_sequence_apply(
            params, nk, nv,
            alpha=self.alpha,
            eta0=self.eta0,
            chunk_size=self.chunk_size,
            p=self.p,
        )

        # reshape outputs back to (T,4,28,28)
        return Y.reshape(T, 4, 28, 28)

# -----------------------------------------------------------------------------
# 6) Dummy outer training loop
# -----------------------------------------------------------------------------
def train():
    import numpy as np

    T = 64
    X = np.random.randn(T, 4, 28, 28).astype(np.float32)
    Y = np.random.randn(T, 4, 28, 28).astype(np.float32)

    model = MirasModel()
    rng   = jax.random.PRNGKey(0)
    # split rng for init vs for forward‐rng
    rng, init_rng = jax.random.split(rng)
    vars  = model.init({"params": init_rng}, X)
    params= vars["params"]

    tx       = optax.adam(1e-3)
    opt_state= tx.init(params)

    @jax.jit
    def step(params, opt_state, x, y, rng):
        def loss_fn(p):
            # pass the same rng in so that make_rng("params") is non‐None
            preds = model.apply({"params": p}, x, rngs={"params": rng})
            return jnp.mean((preds - y)**2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    for epoch in range(50):
        # split rng per step
        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss = step(params, opt_state, X, Y, step_rng)
        print(f"Epoch {epoch}  MSE: {loss:.4f}")

if __name__ == "__main__":
    train()
