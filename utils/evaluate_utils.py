import math
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


def _entropy_mm(counts):
    """Miller-Madow bias-corrected entropy (bits) of a count vector."""
    counts = np.asarray(counts, float)
    N = counts.sum()
    if N <= 0:
        return 0.0
    p = counts[counts > 0] / N
    H = -(p * np.log2(p)).sum()
    K = int((counts > 0).sum())
    return H + (K - 1) / (2.0 * N * np.log(2))

def compute_d_intra(B, y):
    """
    Intra-class bit inconsistency (lower is better):
        d_intra = (1/C) * sum_c (2/D) * sum_j p_cj * (1 - p_cj)
    p_cj is the activation rate of bit j within class c.

    # Arguments
        B: binary codes, numpy.array with shape `(n_samples, D)`
        y: labels, numpy.array with shape `(n_samples,)`
    """
    B = B.astype(np.float64)
    D = B.shape[1]
    vals = []
    for c in np.unique(y):
        p_cj = B[y == c].mean(axis=0)
        vals.append((2.0 / D) * (p_cj * (1.0 - p_cj)).sum())
    return float(np.mean(vals)) if vals else 0.0

def compute_anmi(B, y):
    """
    Activation-normalized mutual information (bits, higher is better):
        anMI = sum_j I(Y; B_j) / sum_j f_j,   f_j = P(B_j = 1)
    I(Y; B_j) is the two-sided MI with Miller-Madow correction, clipped at 0.

    # Arguments
        B: binary codes, numpy.array with shape `(n_samples, D)`
        y: labels, numpy.array with shape `(n_samples,)`
    """
    _, y = np.unique(y, return_inverse=True)
    C = y.max() + 1
    B = B.astype(int)
    HY = _entropy_mm(np.bincount(y, minlength=C))
    f = B.mean(axis=0)
    I = np.zeros(B.shape[1])
    for j in range(B.shape[1]):
        b = B[:, j]
        Hb = _entropy_mm(np.bincount(b, minlength=2))
        if Hb <= 1e-9:  # constant bit carries no class information
            continue
        joint = np.bincount(b * C + y, minlength=2 * C)
        I[j] = max(Hb + HY - _entropy_mm(joint), 0.0)
    return float(I.sum() / f.sum()) if f.sum() > 0 else 0.0


def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def get_dis_max(args):
    Y_U, L = args.unlabeled_nums, args.hash_code_length
    target = (2 ** L) / Y_U

    for d in range(2, L + 1):
        lower_sum = binomial_coefficient(L, d - 2)
        upper_sum = binomial_coefficient(L, d - 1)

        if lower_sum <= target <= upper_sum:
            print("d_max:", d)
            return d

    return 0
