"""Four uncertainty-aware regressors behind one interface.

Every model implements::

    model.fit(X_train, y_train)
    mean, std = model.predict_dist(X_test)   # both shape (n,)

so the evaluation code never needs to know which model produced a prediction.
``std`` is the standard deviation of the predictive distribution over a *new
observation* (i.e. it includes observation noise), which is what calibration
and proper scoring rules should be measured against.

The four models:

* ``NNGPRegressor``      -- exact Gaussian-process posterior of the
  infinite-width limit of a fully-connected network (via ``neural_tangents``).
  No training loop: the architecture defines a kernel, and inference is closed
  form. Cost is O(n^3) in training points.

* ``DeepEnsembleRegressor`` -- the finite-width counterpart. An ensemble of
  small MLPs of the *same* architecture, each predicting a mean and a variance
  (Lakshminarayanan et al., 2017 style). This is the standard strong baseline
  for "what you'd actually train", and the fair finite-vs-infinite contrast.

* ``MCDropoutRegressor`` -- a single MLP with dropout left on at test time;
  predictive moments come from stochastic forward passes (Gal & Ghahramani,
  2016). A cheap approximate-Bayesian baseline.

* ``RBFGPRegressor``     -- a textbook RBF-kernel Gaussian process
  (scikit-learn). A non-neural GP reference whose uncertainty, like the NNGP's,
  grows with distance from the training data by construction.

The two neural models are implemented in plain JAX (no optimiser library) to
keep the dependency surface identical to neural_tangents and to make the
training step legible.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from neural_tangents import stax
import neural_tangents as nt

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# Shared standardisation helper
# --------------------------------------------------------------------------- #
class _Standardiser:
    """Fit on train only; standardise X and y, and rescale predictions back."""

    def fit(self, X, y):
        self.x_mu = X.mean(0, keepdims=True)
        self.x_sd = X.std(0, keepdims=True) + 1e-8
        self.y_mu = float(y.mean())
        self.y_sd = float(y.std() + 1e-8)
        return self

    def x(self, X):
        return (X - self.x_mu) / self.x_sd

    def y(self, y):
        return (y - self.y_mu) / self.y_sd

    def inv_mean(self, m):
        return m * self.y_sd + self.y_mu

    def inv_std(self, s):
        return s * self.y_sd


# --------------------------------------------------------------------------- #
# 1. NNGP (infinite-width network as a Gaussian process)
# --------------------------------------------------------------------------- #
_ACTIVATIONS = {"erf": stax.Erf, "relu": stax.Relu}


class NNGPRegressor:
    """Exact NNGP posterior for a fully-connected architecture.

    ``neural_tangents`` turns the architecture into an analytic infinite-width
    kernel; this class then does textbook GP regression with that kernel. Two
    hyperparameters -- a signal-variance ``amplitude`` that scales the kernel
    and an observation ``noise`` variance -- are selected by maximising the GP
    log marginal likelihood on the training set (``calibrate=True``). This is
    the same model-selection criterion the neural-tangents authors use, and it
    is what makes the comparison against an RBF-GP (whose hyperparameters are
    likewise tuned) a fair one. Cost is O(n^3) in training points.

    Parameters
    ----------
    depth : number of hidden nonlinear layers.
    width : nominal hidden width; the analytic kernel is width-independent, so
        this only mirrors the finite ensemble.
    activation : "erf" or "relu".
    W_std, b_std : prior weight / bias standard deviations (kernel shape).
    calibrate : tune (amplitude, noise) by marginal likelihood if True.
    """

    def __init__(self, depth=2, width=512, activation="erf",
                 W_std=1.5, b_std=0.05, calibrate=True, seed=0):
        self.depth = depth
        self.width = width
        self.activation = activation
        self.W_std = W_std
        self.b_std = b_std
        self.calibrate = calibrate

    def _build_kernel_fn(self):
        act = _ACTIVATIONS[self.activation]
        layers = []
        for _ in range(self.depth):
            layers += [stax.Dense(self.width, W_std=self.W_std, b_std=self.b_std),
                       act()]
        layers += [stax.Dense(1, W_std=self.W_std, b_std=self.b_std)]
        _, _, kernel_fn = stax.serial(*layers)
        return kernel_fn

    def _kernel(self, X1, X2):
        K = self.kernel_fn_(jnp.asarray(X1), jnp.asarray(X2), "nngp")
        return np.asarray(K, dtype=np.float64)

    @staticmethod
    def _log_marg_likelihood(K, y, amplitude, noise):
        """GP log marginal likelihood for cov = amplitude*K + noise*I."""
        n = len(y)
        C = amplitude * K + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return float(-0.5 * y @ alpha - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi))

    def fit(self, X, y):
        self.std_ = _Standardiser().fit(X, y)
        self.X_train_ = self.std_.x(X)
        self.y_train_ = self.std_.y(y).reshape(-1)
        self.kernel_fn_ = self._build_kernel_fn()
        K = self._kernel(self.X_train_, self.X_train_)

        if self.calibrate:
            amps = np.logspace(-1.0, 1.0, 9)
            noises = np.logspace(-2.0, 0.0, 9)
            best = (-np.inf, 1.0, 0.1)
            for a in amps:
                for s2 in noises:
                    ll = self._log_marg_likelihood(K, self.y_train_, a, s2)
                    if ll > best[0]:
                        best = (ll, a, s2)
            _, self.amplitude_, self.noise_ = best
        else:
            self.amplitude_, self.noise_ = 1.0, 1e-2

        n = len(self.y_train_)
        C = self.amplitude_ * K + self.noise_ * np.eye(n)
        self.L_ = np.linalg.cholesky(C)
        self.alpha_ = np.linalg.solve(self.L_.T,
                                      np.linalg.solve(self.L_, self.y_train_))
        return self

    def predict_dist(self, X):
        Xs = self.std_.x(X)
        Ks = self.amplitude_ * self._kernel(Xs, self.X_train_)      # (m, n)
        kss = self.amplitude_ * np.diag(self._kernel(Xs, Xs))       # (m,)
        mean = Ks @ self.alpha_
        v = np.linalg.solve(self.L_, Ks.T)                          # (n, m)
        var = kss - np.sum(v ** 2, axis=0) + self.noise_            # predictive over y*
        var = np.clip(var, 1e-12, None)
        return self.std_.inv_mean(mean), self.std_.inv_std(np.sqrt(var))


# --------------------------------------------------------------------------- #
# Minimal JAX MLP + Adam (shared by the two neural baselines)
# --------------------------------------------------------------------------- #
def _init_mlp(key, d_in, width, depth, out_dim):
    keys = jax.random.split(key, depth + 1)
    sizes = [d_in] + [width] * depth + [out_dim]
    params = []
    for i, k in enumerate(keys):
        fan_in = sizes[i]
        w = jax.random.normal(k, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros((sizes[i + 1],))
        params.append((w, b))
    return params


def _mlp_forward(params, x, dropout_key=None, p_drop=0.0):
    h = x
    for i, (w, b) in enumerate(params[:-1]):
        h = jnp.tanh(h @ w + b)
        if dropout_key is not None and p_drop > 0.0:
            dropout_key, sub = jax.random.split(dropout_key)
            mask = (jax.random.uniform(sub, h.shape) > p_drop) / (1.0 - p_drop)
            h = h * mask
    w, b = params[-1]
    return h @ w + b  # last layer linear, out_dim columns


def _adam(grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * gi * gi, v, grads)
    mhat = jax.tree_util.tree_map(lambda mi: mi / (1 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda vi: vi / (1 - b2 ** t), v)
    updates = jax.tree_util.tree_map(
        lambda mh, vh: lr * mh / (jnp.sqrt(vh) + eps), mhat, vhat
    )
    return updates, (m, v, t)


def _zeros_like_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


# --------------------------------------------------------------------------- #
# 2. Deep ensemble (finite-width network with predictive variance)
# --------------------------------------------------------------------------- #
class DeepEnsembleRegressor:
    """Ensemble of heteroscedastic MLPs; the finite-width counterpart of NNGP.

    Each member outputs a mean and a log-variance and is trained with the
    Gaussian negative-log-likelihood. The ensemble predictive distribution is a
    Gaussian matched to the mixture's first two moments (law of total variance).
    """

    def __init__(self, n_members=5, depth=2, width=64, epochs=400,
                 lr=1e-2, seed=0):
        self.n_members = n_members
        self.depth = depth
        self.width = width
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        self.std_ = _Standardiser().fit(X, y)
        Xs = jnp.asarray(self.std_.x(X))
        ys = jnp.asarray(self.std_.y(y)).reshape(-1, 1)
        d_in = Xs.shape[1]

        def nll_loss(params, x, t):
            out = _mlp_forward(params, x)
            mu, log_var = out[:, :1], out[:, 1:]
            log_var = jnp.clip(log_var, -8.0, 8.0)
            inv_var = jnp.exp(-log_var)
            return jnp.mean(0.5 * log_var + 0.5 * inv_var * (t - mu) ** 2)

        grad_fn = jax.jit(jax.value_and_grad(nll_loss))

        self.members_ = []
        key = jax.random.PRNGKey(self.seed)
        for _ in range(self.n_members):
            key, ik = jax.random.split(key)
            params = _init_mlp(ik, d_in, self.width, self.depth, out_dim=2)
            opt_state = (_zeros_like_tree(params), _zeros_like_tree(params), 0)
            for _ in range(self.epochs):
                _, grads = grad_fn(params, Xs, ys)
                updates, opt_state = _adam(grads, opt_state, lr=self.lr)
                params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
            self.members_.append(params)
        return self

    def predict_dist(self, X):
        Xs = jnp.asarray(self.std_.x(X))
        means, variances = [], []
        for params in self.members_:
            out = _mlp_forward(params, Xs)
            mu = np.asarray(out[:, 0])
            log_var = np.clip(np.asarray(out[:, 1]), -8.0, 8.0)
            means.append(mu)
            variances.append(np.exp(log_var))
        means = np.stack(means)            # (M, n)
        variances = np.stack(variances)    # (M, n)
        mean = means.mean(0)
        # Total variance of an equally-weighted Gaussian mixture.
        var = (variances + means ** 2).mean(0) - mean ** 2
        var = np.clip(var, 1e-12, None)
        return self.std_.inv_mean(mean), self.std_.inv_std(np.sqrt(var))


# --------------------------------------------------------------------------- #
# 3. MC-dropout
# --------------------------------------------------------------------------- #
class MCDropoutRegressor:
    """Single MLP with dropout active at test time (Gal & Ghahramani, 2016).

    Predictive moments are estimated from ``n_samples`` stochastic forward
    passes; a homoscedastic noise term (estimated from training residuals) is
    added to form the predictive variance over a new observation.
    """

    def __init__(self, depth=2, width=64, p_drop=0.1, epochs=600,
                 lr=1e-2, n_samples=100, seed=0):
        self.depth = depth
        self.width = width
        self.p_drop = p_drop
        self.epochs = epochs
        self.lr = lr
        self.n_samples = n_samples
        self.seed = seed

    def fit(self, X, y):
        self.std_ = _Standardiser().fit(X, y)
        Xs = jnp.asarray(self.std_.x(X))
        ys = jnp.asarray(self.std_.y(y)).reshape(-1, 1)
        d_in = Xs.shape[1]
        key = jax.random.PRNGKey(self.seed)
        key, ik = jax.random.split(key)
        params = _init_mlp(ik, d_in, self.width, self.depth, out_dim=1)

        def mse_loss(params, x, t, dk):
            pred = _mlp_forward(params, x, dropout_key=dk, p_drop=self.p_drop)
            return jnp.mean((pred - t) ** 2)

        grad_fn = jax.jit(jax.value_and_grad(mse_loss))
        opt_state = (_zeros_like_tree(params), _zeros_like_tree(params), 0)
        for _ in range(self.epochs):
            key, dk = jax.random.split(key)
            _, grads = grad_fn(params, Xs, ys, dk)
            updates, opt_state = _adam(grads, opt_state, lr=self.lr)
            params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)

        self.params_ = params
        self.key_ = key
        # Homoscedastic noise estimate from training residuals (deterministic pass).
        train_pred = np.asarray(_mlp_forward(params, Xs)).reshape(-1)
        resid = np.asarray(ys).reshape(-1) - train_pred
        self.noise_var_ = float(np.var(resid)) + 1e-6
        return self

    def predict_dist(self, X):
        Xs = jnp.asarray(self.std_.x(X))
        key = self.key_
        samples = []
        for _ in range(self.n_samples):
            key, dk = jax.random.split(key)
            pred = _mlp_forward(self.params_, Xs, dropout_key=dk, p_drop=self.p_drop)
            samples.append(np.asarray(pred).reshape(-1))
        samples = np.stack(samples)        # (S, n)
        mean = samples.mean(0)
        var = samples.var(0) + self.noise_var_
        return self.std_.inv_mean(mean), self.std_.inv_std(np.sqrt(var))


# --------------------------------------------------------------------------- #
# 4. RBF-kernel Gaussian process (scikit-learn)
# --------------------------------------------------------------------------- #
class RBFGPRegressor:
    """Textbook RBF + white-noise Gaussian process (scikit-learn)."""

    def __init__(self, n_restarts=4, seed=0):
        self.n_restarts = n_restarts
        self.seed = seed

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

        self.std_ = _Standardiser().fit(X, y)
        Xs = self.std_.x(X)
        ys = self.std_.y(y)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(0.1)
        self.gp_ = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=self.n_restarts,
            random_state=self.seed, normalize_y=False,
        ).fit(Xs, ys)
        return self

    def predict_dist(self, X):
        Xs = self.std_.x(X)
        mean, std = self.gp_.predict(Xs, return_std=True)
        return self.std_.inv_mean(mean), self.std_.inv_std(std)


MODEL_REGISTRY = {
    "NNGP": NNGPRegressor,
    "DeepEnsemble": DeepEnsembleRegressor,
    "MCDropout": MCDropoutRegressor,
    "RBF-GP": RBFGPRegressor,
}
