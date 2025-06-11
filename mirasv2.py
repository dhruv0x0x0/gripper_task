# src/sim_env/maniskill_env/scripts/final/mirasv2_fixed.py

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from functools import partial

# -----------------------------------------------------------------------------
# 1) Simple RMSNorm (unchanged)
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
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
        x = RMSNorm(dim=self.d_model)(x)  # Apply RMSNorm here
        return x
# -----------------------------------------------------------------------------
# 3) Refactored: One chunk of Miras, with a pre-instantiated MemoryMLP
# -----------------------------------------------------------------------------
def make_miras_chunk_update(memory_mlp: MemoryMLP):
    """
    Returns a jitted chunk-update function that closes over `memory_mlp`.
    memory_mlp: an instance MemoryMLP(d_model=...) instantiated once.
    """
    @partial(jax.jit, static_argnames=("p",))
    def miras_chunk_update(params, keys, vals, etas, betas, p: float = 2.0):
        """
        params: pytree of MemoryMLP params before this chunk
        keys, vals: (chunk_size, d_model)
        etas, betas: (chunk_size,)
        """
        # define per-token loss using the closed-over memory_mlp
        def loss_per_token(param_pytree, k, v):
            # use the pre-made instance:
            pred = memory_mlp.apply({"params": param_pytree}, k)
            return jnp.sum((pred - v) ** 2)  # ℓ_p^p

        # vectorized gradient
        grad_fn = jax.vmap(jax.grad(loss_per_token), in_axes=(None, 0, 0))
        grads = grad_fn(params, keys, vals)
        # form weights w_i = etas[i] * (betas[-1]/betas[i])
        final_beta = betas[-1]
        weights = etas * (final_beta / betas)  # (chunk_size,)

        # weighted sum of grads
        def weighted_sum(tree):
            # tree: (chunk_size, *param_shape)
            return 1e-4*jnp.tensordot(weights, tree, axes=((0,), (0,)))
        agg_grad = jax.tree_util.tree_map(weighted_sum, grads)

        # single GD step
        updated = jax.tree_util.tree_map(lambda p, g: p - g, params, agg_grad)
        return updated

    return miras_chunk_update

# -----------------------------------------------------------------------------
# 4) Refactored: Full Miras-block over T timesteps, closing over same MemoryMLP
# -----------------------------------------------------------------------------
def make_miras_sequence_apply(memory_mlp: MemoryMLP):
    """
    Returns a function miras_sequence_apply that closes over memory_mlp.
    """
    def miras_sequence_apply(init_params, all_keys, all_vals, alpha, eta0, chunk_size, p: float):
        T, d_model = all_keys.shape
       # assert T % chunk_size == 0, "T must be divisible by chunk_size"
        #n_chunks = T #// chunk_size

        keys_chunks = all_keys#.reshape(n_chunks, chunk_size, d_model)
        vals_chunks = all_vals#.reshape(n_chunks, chunk_size, d_model)

        # get the chunk update fn
        miras_chunk_update = make_miras_chunk_update(memory_mlp)
        K_chunk, V_chunk = all_keys, all_vals

            # build etas and betas
        etas = eta0 * (alpha ** jnp.arange(chunk_size))
        betas = alpha ** jnp.arange(chunk_size)

            # 1) update memory in parallel for this chunk
        final_params = miras_chunk_update(init_params, K_chunk, V_chunk, etas, betas, p=p)

            # 2) recall outputs using updated memory
            # use memory_mlp closed over
        recall_fn = lambda k: memory_mlp.apply({"params": final_params}, k)
        Y_chunks = jax.vmap(recall_fn)(K_chunk)  # (chunk_size, d_model)


        # final_params, Y_chunks = jax.lax.scan(
        #     chunk_body,
        #     init_params,
        #     (keys_chunks, vals_chunks),
        # )

        # def chunk_body(carry, chunk):
        #     params = carry  # current memory params
        #     K_chunk, V_chunk = chunk  # each (chunk_size, d_model)

        #     # build etas and betas
        #     etas = eta0 * (alpha ** jnp.arange(chunk_size))
        #     betas = alpha ** jnp.arange(chunk_size)

        #     # 1) update memory in parallel for this chunk
        #     new_params = miras_chunk_update(params, K_chunk, V_chunk, etas, betas, p=p)

        #     # 2) recall outputs using updated memory
        #     # use memory_mlp closed over
        #     recall_fn = lambda k: memory_mlp.apply({"params": new_params}, k)
        #     Y_chunk = jax.vmap(recall_fn)(K_chunk)  # (chunk_size, d_model)

        #     return new_params, Y_chunk

        # final_params, Y_chunks = jax.lax.scan(
        #     chunk_body,
        #     init_params,
        #     (keys_chunks, vals_chunks),
        # )
        Y = Y_chunks.reshape(T, d_model)
        return final_params, Y

    return miras_sequence_apply

# -----------------------------------------------------------------------------
# 5) MirasModel, accepting external mem_params, with projections
# -----------------------------------------------------------------------------
class MirasModel(nn.Module):
    d_model: int = 28*28*4
    chunk_size: int = 10
    alpha: float = 0.9
    eta0: float = 0.1
    p: float = 2.0

    @nn.compact
    def __call__(self, x, mem_params):
        """
        x: (T, 4, 28, 28)
        mem_params: pytree of MemoryMLP params, initial memory
        Returns:
          - outputs Y: (T, 4, 28, 28)
          - mem_final: final memory params after inner updates
        """
        T = x.shape[0]
        x_t = jnp.transpose(x, (0, 2, 3, 1))  # (T,28,28,4)
        ck = nn.Conv(features=4, kernel_size=(3,3), padding="SAME")(x_t)
        cv = nn.Conv(features=4, kernel_size=(3,3), padding="SAME")(x_t)
        nk = RMSNorm(4)(ck).reshape(T, -1)  # (T, d_model)
        nv = RMSNorm(4)(cv).reshape(T, -1)  # (T, d_model)

        # Linear projections
        nk_proj = nn.Dense(self.d_model)(nk)
        nv_proj = nn.Dense(self.d_model)(nv)

        # Prepare a MemoryMLP instance closed-over by the sequence fn
        memory_mlp = MemoryMLP(d_model=self.d_model)
    

        # Get the sequence apply fn closing over this memory_mlp
        miras_sequence_apply = make_miras_sequence_apply(memory_mlp)

        # Run Miras
        final_mem_params, Y_flat = miras_sequence_apply(
            mem_params, nk_proj, nv_proj,
            alpha=self.alpha,
            eta0=self.eta0,
            chunk_size=self.chunk_size,
            p=self.p,
        )
        Y_flat = RMSNorm(dim=Y_flat.shape[-1])(Y_flat)
        Y = Y_flat.reshape(T, 4, 28, 28)
        return Y, final_mem_params

# -----------------------------------------------------------------------------
# 6) Training loop (same pattern as before)
# -----------------------------------------------------------------------------
import numpy as np

def _make_trajectories(total_T, height, width, channels):
    """
    For each channel, pick one trajectory type and compute (y_t, x_t) for t=0..total_T-1.
    """
    trajectories = np.zeros((channels, total_T, 2), dtype=np.float32)
    ts = np.arange(total_T)

    for c in range(channels):
        traj_type = c % 5  # cycles through 5 types
        if traj_type == 0:
            # 0) Linear bouncing
            vy, vx = (0.07 + 0.03*c), (0.05 + 0.02*c)
            y = (vy * ts) % (2*height)
            x = (vx * ts) % (2*width)
            # reflect
            y = np.where(y>height, 2*height - y, y)
            x = np.where(x>width,  2*width  - x, x)
        elif traj_type == 1:
            # 1) Circular motion around center
            R = min(height,width)*0.3 + 2*c
            cy, cx = height/2 + (c-1)*2, width/2 - (c-2)*2
            theta = ts * (0.1 + 0.02*c)
            y = cy + R * np.sin(theta)
            x = cx + R * np.cos(theta)
        elif traj_type == 2:
            # 2) Sinusoidal horizontal with vertical drift + wall reflect
            A = width * 0.3 + c*2
            y = (0.05 + 0.01*c)*ts
            y = y % (2*height)
            y = np.where(y>height, 2*height - y, y)
            x = width/2 + A * np.sin(ts * (0.15 + 0.01*c))
        elif traj_type == 3:
            # 3) Lissajous figure
            ay = 0.2 + 0.02*c
            ax = 0.15 + 0.015*c
            delta = np.pi * c / channels
            y = height/2 + (height*0.3) * np.sin(ay * ts + delta)
            x = width/2  + (width *0.3) * np.sin(ax * ts)
        else:
            # 4) Outward spiral from center
            omega = 0.2 + 0.02*c
            r = (min(height,width)*0.02) * ts
            theta = omega * ts
            y = height/2 + r * np.sin(theta)
            x = width/2  + r * np.cos(theta)

        trajectories[c, :, 0] = y
        trajectories[c, :, 1] = x

    return trajectories  # shape (channels, total_T, 2)


def generate_moving_gaussian_sequence(total_T, height=28, width=28, channels=4, sigma=3.0):
    """
    Generate a sequence of shape (total_T, channels, height, width) where each channel
    contains a Gaussian blob moving along a (deterministic) but non-trivial trajectory.
    """
    seq = np.zeros((total_T, channels, height, width), dtype=np.float32)
    ys = np.arange(height).reshape(-1, 1)
    xs = np.arange(width).reshape(1, -1)

    # get per-channel trajectories
    trajs = _make_trajectories(total_T, height, width, 4)
    c =0
    for t in range(total_T):
        
            y0, x0 = trajs[c, t]
            # wrap & reflect into [0,height) and [0,width)
            y0_mod = np.clip(y0, 0, height-1)
            x0_mod = np.clip(x0, 0, width-1)
            dist2 = (ys - y0_mod)**2 + (xs - x0_mod)**2
            seq[t, c] = np.exp(-dist2 / (2 * sigma**2))

    # Normalize to [0,1] per channel
    seq_min = seq.min(axis=(0,2,3), keepdims=True)
    seq_max = seq.max(axis=(0,2,3), keepdims=True)
    seq = (seq - seq_min) / (seq_max - seq_min + 1e-8)
    return seq


def data_generator(num_batches, chunk_size, height=28, width=28, channels=4, sigma=3.0):
    """
    Generate a long moving-Gaussian sequence with varied trajectories,
    then yield (X, Y) where Y is the same sequence shifted by one time step.
    Each X, Y has shape (chunk_size, channels, height, width).
    """
    total_T = num_batches * chunk_size + 1
    seq = generate_moving_gaussian_sequence(total_T, height, width, channels, sigma)

    for i in range(num_batches):
        start = i * chunk_size
        X = seq[start : start + chunk_size]
        Y = seq[start + 1 : start + chunk_size + 1]
        yield X, Y
# def generate_moving_gaussian_sequence(total_T, height=28, width=28, channels=4, sigma=3.0):
#     """
#     Generate a sequence of shape (total_T, channels, height, width) where each channel
#     contains a Gaussian blob moving in a distinct direction over time.
#     """
#     seq = np.zeros((total_T, channels, height, width), dtype=np.float32)
#     init_positions = [
#         (height * 0.25, width * 0.25),
#         (height * 0.75, width * 0.25),
#         (height * 0.25, width * 0.75),
#         (height * 0.75, width * 0.75),
#     ]
#     velocities = [
#         (0.1, 0.1),
#         (-0.1, 0.1),
#         (0.1, -0.1),
#         (-0.1, -0.1),
#     ]
#     ys = np.arange(height).reshape(-1, 1)
#     xs = np.arange(width).reshape(1, -1)

#     for t in range(total_T):
#         for c in range(channels):
#             y0 = init_positions[c][0] + velocities[c][0] * t
#             x0 = init_positions[c][1] + velocities[c][1] * t
#             y0_mod = y0 % height
#             x0_mod = x0 % width
#             dist2 = (ys - y0_mod)**2 + (xs - x0_mod)**2
#             blob = np.exp(-dist2 / (2 * sigma**2))
#             seq[t, c] = blob

#     # Normalize to [0,1] per channel
#     seq_min = seq.min(axis=(0,2,3), keepdims=True)
#     seq_max = seq.max(axis=(0,2,3), keepdims=True)
#     seq = (seq - seq_min) / (seq_max - seq_min + 1e-8)
#     return seq

# def data_generator(num_batches, chunk_size, height=28, width=28, channels=4, sigma=3.0):
#     """
#     Generate a long moving-Gaussian sequence of length num_batches*chunk_size + 1,
#     then yield (X, Y) where Y is the same sequence shifted by one time step.
#     Each X, Y has shape (chunk_size, channels, height, width).
#     """
#     total_T = num_batches * chunk_size + 1
#     seq = generate_moving_gaussian_sequence(total_T, height, width, channels, sigma)
#     for i in range(num_batches):
#         start = i * chunk_size
#         X = seq[start:start + chunk_size]           # shape (chunk_size, channels, height, width)
#         Y = seq[start + 1:start + chunk_size + 1]   # next time-shifted chunk
#         yield X, Y

# Example usage and sanity check:
num_batches = 4
chunk_size = 64
# for idx, (X, Y) in enumerate(gen):
#     print(f"Batch {idx}: X.shape={X.shape}, Y.shape={Y.shape}")

# def train(num_epochs: int = 5, batch_size: int = 1):
#     import numpy as np

#     T = 10
    

#     # Dummy data
#     # def data_generator(num_batches):
#     #     for _ in range(num_batches):
#     #         X = np.random.randn(T, 4, 28, 28).astype(np.float32)
#     #         Y = np.random.randn(T, 4, 28, 28).astype(np.float32)
#     #         yield X, Y

#     model = MirasModel()

#     rng = jax.random.PRNGKey(0)
#     rng, init_rng = jax.random.split(rng)

#     # Initialize memory params once
#     dummy_k = jnp.zeros((model.d_model,))
#     rng, mem_init_rng = jax.random.split(rng)
#     init_mem_params = MemoryMLP(d_model=model.d_model).init({"params": mem_init_rng}, dummy_k)["params"]
#     def zeros_like_tree(tree):
#         return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)
#     accum_mem_params = zeros_like_tree(init_mem_params)

#     # Initialize model params, passing dummy memory total
#     dummy_mem_total = init_mem_params  # accum is zero initially
#     dummy_x = jnp.zeros((T, 4, 28, 28), jnp.float32)
#     variables = model.init({"params": init_rng}, dummy_x, dummy_mem_total)
#     model_params = variables["params"]

#     outer_params = {"model": model_params, "init_mem": init_mem_params}
#     tx = optax.adam(1e-3)
#     opt_state = tx.init(outer_params)

#     @jax.jit
#     def step(outer_params, opt_state, accum_mem_params, x, y, rng):
#         def loss_fn(outer_params):
#             model_params = outer_params["model"]
#             init_mem = outer_params["init_mem"]
#             mem_total = jax.tree_util.tree_map(
#                 lambda i, a: i + jax.lax.stop_gradient(a),
#                 init_mem, accum_mem_params
#             )
#             preds, mem_final = model.apply(
#                 {"params": model_params},
#                 x,
#                 mem_total,
#                 rngs={"params": rng}
#             )
#             loss = jnp.mean((preds - y) ** 2)
#             return loss, mem_final

#         (loss, mem_final), grads = jax.value_and_grad(loss_fn, has_aux=True)(outer_params)
#         updates, opt_state = tx.update(grads, opt_state)
#         new_outer_params = optax.apply_updates(outer_params, updates)

#         old_init_mem = outer_params["init_mem"]
#         mem_total_old = jax.tree_util.tree_map(
#             lambda i, a: i + jax.lax.stop_gradient(a),
#             old_init_mem, accum_mem_params
#         )
#         delta_mem = jax.tree_util.tree_map(lambda nf2, mt2: jax.lax.stop_gradient(nf2 - mt2),
#                                            mem_final, mem_total_old)
#         new_accum_mem_params = jax.tree_util.tree_map(lambda a, d: a + d,
#                                                       accum_mem_params, delta_mem)

#         return new_outer_params, opt_state, new_accum_mem_params, loss

#     num_batches = 10
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         cnt = 0
#         gen = data_generator(4, T)
#         for idx, (X_np, Y_np) in enumerate(gen):
#             rng, step_rng = jax.random.split(rng)
#             x = jnp.array(X_np)
#             y = jnp.array(Y_np)
#             outer_params, opt_state, accum_mem_params, loss = step(
#                 outer_params, opt_state, accum_mem_params, x, y, step_rng
#             )
#             total_loss += loss
#             print(loss)
#             cnt += 1
#         print(f"Epoch {epoch}: avg loss = { (total_loss) }", cnt)

#     return outer_params, accum_mem_params
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import imageio
import os

# Ensure directory for videos
os.makedirs("epoch_videos", exist_ok=True)

def train(num_epochs: int = 5, batch_size: int = 1):
    import numpy as np

    T = 10

    model = MirasModel()

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # Initialize memory params once
    dummy_k = jnp.zeros((model.d_model,))
    rng, mem_init_rng = jax.random.split(rng)
    init_mem_params = MemoryMLP(d_model=model.d_model).init({"params": mem_init_rng}, dummy_k)["params"]
    def zeros_like_tree(tree):
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)
    accum_mem_params = zeros_like_tree(init_mem_params)

    # Initialize model params, passing dummy memory total
    dummy_mem_total = init_mem_params  # accum is zero initially
    dummy_x = jnp.zeros((T, 4, 28, 28), jnp.float32)
    variables = model.init({"params": init_rng}, dummy_x, dummy_mem_total)
    model_params = variables["params"]

    outer_params = {"model": model_params, "init_mem": init_mem_params}
    tx = optax.adam(1e-3)
    opt_state = tx.init(outer_params)

    @jax.jit
    def step(outer_params, opt_state, accum_mem_params, x, y, rng):
        def loss_fn(outer_params):
            model_params = outer_params["model"]
            init_mem = outer_params["init_mem"]
            mem_total = jax.tree_util.tree_map(
                lambda i, a: i + jax.lax.stop_gradient(a),
                init_mem, accum_mem_params
            )
            preds, mem_final = model.apply(
                {"params": model_params},
                x,
                mem_total,
                rngs={"params": rng}
            )
            loss = jnp.mean((preds - y) ** 2)
            return loss, mem_final

        (loss, mem_final), grads = jax.value_and_grad(loss_fn, has_aux=True)(outer_params)
        updates, opt_state = tx.update(grads, opt_state)
        new_outer_params = optax.apply_updates(outer_params, updates)

        old_init_mem = outer_params["init_mem"]
        mem_total_old = jax.tree_util.tree_map(
            lambda i, a: i + jax.lax.stop_gradient(a),
            old_init_mem, accum_mem_params
        )
        delta_mem = jax.tree_util.tree_map(lambda nf2, mt2: jax.lax.stop_gradient(nf2 - mt2),
                                           mem_final, mem_total_old)
        new_accum_mem_params = jax.tree_util.tree_map(lambda a, d: a + d,
                                                      accum_mem_params, delta_mem)

        return new_outer_params, opt_state, new_accum_mem_params, loss

    num_batches = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        cnt = 0
        gen = data_generator(20, T)
        for idx, (X_np, Y_np) in enumerate(gen):
            rng, step_rng = jax.random.split(rng)
            x = jnp.array(X_np)
            y = jnp.array(Y_np)
            outer_params, opt_state, accum_mem_params, loss = step(
                outer_params, opt_state, accum_mem_params, x, y, step_rng
            )
            total_loss += loss
            print(loss)
            cnt += 1
        print(f"Epoch {epoch}: avg loss = {total_loss} over {cnt} batches")

        # --- Validation visualization: take one chunk from data_generator and make a GIF ---
        val_mem_total = outer_params["init_mem"]
        val_accum_mem_params = zeros_like_tree(init_mem_params)
    # If you want to carry accumulated memory too:
    # val_accum = accum_mem_params
    # Otherwise, skip using accum_mem_params in validation.

        gif_frames = []
        val_gen = data_generator(3, T)  # 7 sequential chunks of length T
        for idx, (X_np, Y_np) in enumerate(val_gen):
            rng, val_rng = jax.random.split(rng)
            # form memory total for this validation chunk:
            # mem_total = jax.tree_util.tree_map(
            #     lambda i, a: i + jax.lax.stop_gradient(a),
            #     val_mem, val_accum_mem_params  # or zero if you don't want to use accum
            # )
            X_val = jnp.array(X_np)
            preds_val, val_mem_total = model.apply(
                {"params": outer_params["model"]},
                X_val,
                val_mem_total,
                rngs={"params": val_rng}
            )
            preds_val = np.array(preds_val)  # shape (T, 4, 28, 28)
            Y_val = Y_np  # shape (T, 4, 28, 28)
            abs_err = np.abs(Y_val - preds_val)

            # append frames for this chunk
            for t in range(T):
                fig, axes = plt.subplots(3, 4, figsize=(4, 3))
                fig.suptitle(f"Epoch {epoch} Chunk {idx} Frame {t}")
                for c in range(4):
                    ax = axes[0, c]
                    ax.imshow(Y_val[t, c], cmap='viridis')
                    if c == 0:
                        ax.set_ylabel("Actual")
                    ax.axis('off')

                    ax = axes[1, c]
                    ax.imshow(preds_val[t, c], cmap='viridis')
                    if c == 0:
                        ax.set_ylabel("Pred")
                    ax.axis('off')

                    ax = axes[2, c]
                    ax.imshow(abs_err[t, c], cmap='inferno')
                    if c == 0:
                        ax.set_ylabel("Err")
                    ax.axis('off')
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                gif_frames.append(img)
                plt.close(fig)

            # carry memory forward if desired:
            #val_mem = mem_final

        # Save GIF
        gif_path = f"epoch_videos/epoch_{epoch}.gif"
        imageio.mimsave(gif_path, gif_frames, fps=4)
        print(f"Saved validation GIF: {gif_path}")

    return outer_params, accum_mem_params
if __name__ == "__main__":
    outer_params, accum_mem = train(num_epochs=100)
    print("Done")
