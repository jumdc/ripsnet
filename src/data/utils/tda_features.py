import numpy as np
import gudhi as gd
from sklearn.metrics import pairwise_distances
from gudhi.representations import PersistenceImage


def alpha_pi(X, max_d=None, sigma=None, im_bnds=None, sp_bnds=None):
    """
    Compute the PI based on the alpha complex of the point cloud X.

    Parameters
    ----------
    X : np.array
        Point cloud.
    max_d : float
        Maximum distance to compute the alpha complex.
    sigma : float
        Bandwidth of the Gaussian kernel.
    im_bnds : list
        Image bounds.
    sp_bnds : list
        Spatial bounds.

    Returns
    -------
    np.array
        Persistence Image.
    dict
        Hyperparameters used if needed.
    """
    pd, hparams = [], {}

    if max_d is None:
        # use the first 30 samples to compute the max_d
        ds = [pairwise_distances(X).flatten() for X in X[:30]]
        max_d = np.max(np.concatenate(ds))
        hparams["max_d"] = max_d

    # - Create the Complex
    st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=max_d)
    st.persistence()
    dg = st.persistence_intervals_in_dimension(1)
    if len(dg) == 0:
        dg = np.empty([0, 2])
    pd.append(dg)

    # - Clean the persistence diagram - rm the infinities
    clean_pd = gd.representations.DiagramSelector(use=True).fit_transform(pd)

    # - if train, compute the bandwidth
    if sigma is None:
        vpdtr = np.vstack(pd)
        pers = vpdtr[:, 1] - vpdtr[:, 0]
        bps_pairs = pairwise_distances(
            np.hstack([vpdtr[:, 0:1], vpdtr[:, 1:2] - vpdtr[:, 0:1]])[:200]
        ).flatten()
        ppers = bps_pairs[np.argwhere(bps_pairs > 1e-5).ravel()]
        sigma = np.quantile(ppers, 0.2)
        im_bnds = [
            np.quantile(vpdtr[:, 0], 0.0),
            np.quantile(vpdtr[:, 0], 1.0),
            np.quantile(pers, 0.0),
            np.quantile(pers, 1.0),
        ]
        sp_bnds = [np.quantile(vpdtr[:, 0], 0.0), np.quantile(vpdtr[:, 1], 1.0)]
        hparams.update({"sigma": sigma, "im_bnds": im_bnds, "sp_bnds": sp_bnds})

    # - Compute the Persistence Image
    PI_params = {
        "bandwidth": sigma,
        "weight": lambda x: 10 * np.tanh(x[1]),
        "resolution": [50, 50],
        "im_range": im_bnds,
    }
    pi = PersistenceImage(**PI_params).fit_transform(clean_pd)
    return pi, hparams