import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import stats

def energy_test(X, Y, n_perm=1000, seed=None, precompute=True):
    """
    Two-sample Energy Distance permutation test (any dimension).
    Returns (statistic, pvalue).
    precompute=True: build full distance matrix once (fast perms, more RAM).
    precompute=False: recompute distances each perm (less RAM, slower).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
    nx, ny = len(X), len(Y)
    n = nx + ny

    def e_stat_from_blocks(Daa, Dbb, Dab):
        # means excluding diagonals for within; simple mean for between
        def mean_upper(M):
            if M.shape[0] < 2: 
                return 0.0
            iu = np.triu_indices_from(M, k=1)
            return M[iu].mean()
        Daa_m = mean_upper(Daa)
        Dbb_m = mean_upper(Dbb)
        Dab_m = Dab.mean() if Dab.size else 0.0
        return 2.0 * Dab_m - Daa_m - Dbb_m

    if precompute:
        XY = np.vstack([X, Y])
        D = squareform(pdist(XY))  # (n x n)
        ix = np.arange(nx)
        iy = np.arange(nx, n)

        def e_stat_ix(ix_, iy_):
            Daa = D[np.ix_(ix_, ix_)]
            Dbb = D[np.ix_(iy_, iy_)]
            Dab = D[np.ix_(ix_, iy_)]
            return e_stat_from_blocks(Daa, Dbb, Dab)

        observed = e_stat_ix(ix, iy)
        perm_stats = np.empty(n_perm)
        labels = np.arange(n)
        for b in range(n_perm):
            rng.shuffle(labels)
            ixp = labels[:nx]; iyp = labels[nx:]
            perm_stats[b] = e_stat_ix(ixp, iyp)

    else:
        def e_stat_raw(A, B):
            return e_stat_from_blocks(
                cdist(A, A), cdist(B, B), cdist(A, B)
            )
        observed = e_stat_raw(X, Y)
        XY = np.vstack([X, Y])
        perm_stats = np.empty(n_perm)
        labels = np.array([0]*nx + [1]*ny)
        for b in range(n_perm):
            rng.shuffle(labels)
            A = XY[labels == 0]
            B = XY[labels == 1]
            perm_stats[b] = e_stat_raw(A, B)

    pval = (np.sum(perm_stats >= observed) + 1) / (n_perm + 1)
    return observed, pval
