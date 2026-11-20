"""Logit demand primitives.

The outside good has mean utility normalised to zero; ``shares`` returns
only inside shares, so sum(s) < 1 and the remainder is the outside share.
"""

import numpy as np


def shares(delta):
    """Multinomial logit inside shares from mean utilities ``delta``."""
    delta = np.asarray(delta, dtype=float)
    z = delta - np.max(delta)
    ex = np.exp(z)
    return ex / (np.exp(-np.max(delta)) + ex.sum())


def price_jacobian(s, alpha):
    """(J, J) matrix G with G[j, k] = ds_j / dp_k under logit demand.

    With delta_j = x_j - alpha * p_j:
        ds_j / dp_j = -alpha * s_j * (1 - s_j)
        ds_j / dp_k =  alpha * s_j * s_k   (k != j)

    where (1 - s_j) includes the outside good.
    """
    s = np.asarray(s, dtype=float)
    return alpha * (np.outer(s, s) - np.diag(s))


def invert_shares(s_obs):
    """Recover mean utilities delta from observed inside shares.

    Inverts s_j = exp(delta_j) / (1 + sum_k exp(delta_k))
    with the outside good pinned at delta_0 = 0.
    """
    s_obs = np.asarray(s_obs, dtype=float)
    return np.log(s_obs) - np.log(1.0 - s_obs.sum())


def ownership_matrix(owner_ids):
    """Binary (J, J) matrix: 1 where products i and j share an owner."""
    owner_ids = np.asarray(owner_ids)
    return (owner_ids[:, None] == owner_ids[None, :]).astype(float)
