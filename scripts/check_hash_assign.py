"""
Check that train_eval.hash_assign gives exactly the same predictions as the
original list-based Hamming matching loop in evaluate().

Run from the repo root:
    python scripts/check_hash_assign.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_eval import hash_assign


# Original implementation from evaluate() (commit 79f7d56), kept verbatim
def compute_hamming_distance_list(list1, list2):
    differences = [x != y for x, y in zip(list1, list2)]
    hamming_distance = sum(differences)
    return hamming_distance

def original_assign(all_feats, centers, radius):
    feats_hash = torch.Tensor(all_feats > 0).float().tolist()
    hash_dict = centers.numpy().tolist()
    preds1 = []
    for feat in feats_hash:
        found = False
        if feat in hash_dict:
            preds1.append(hash_dict.index(feat))
            found = True
        if not found:
            distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
            min_distance = min(distances)
            min_index = distances.index(min_distance)
            if min_distance <= radius:
                preds1.append(min_index)
                found = True
        if not found:
            hash_dict.append(feat)
            preds1.append(len(hash_dict) - 1)
    return np.array(preds1)


def make_case(rng, L, n_cls, n, noise, half):
    # Initial centers are hash_centers.cpu().sign(): values in {-1, 0, 1}
    hc = torch.from_numpy(rng.standard_normal((n_cls, L)).astype(np.float32))
    hc[torch.from_numpy(rng.random((n_cls, L)) < 0.05)] = 0
    if half:
        hc = hc.half()  # hash centers come out of autocast
    centers = hc.sign()
    # Features: noisy copies of hidden prototypes, so both matches and new categories occur
    protos = rng.standard_normal((n_cls * 2, L)).astype(np.float32)
    feats = protos[rng.integers(0, len(protos), n)] + noise * rng.standard_normal((n, L)).astype(np.float32)
    # Exact duplicates of earlier samples
    dup = rng.random(n) < 0.1
    feats[dup] = feats[rng.integers(0, n, dup.sum())]
    # Codes equal to the 0/1 version of a center
    m = rng.random(n) < 0.05
    feats[m] = (centers[rng.integers(0, n_cls, m.sum())].float().numpy() > 0) - 0.5
    return feats, centers


def main():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    elif torch.backends.mps.is_available():
        devices.append('mps')

    rng = np.random.default_rng(0)
    n_ok = n_all = 0
    for L in [12, 16, 32, 64, 128, 256, 512, 1024]:
        for radius in sorted({1, max(L // 8, 1), max(L // 4, 1), L // 2}):
            for noise, half in [(0.3, False), (1.0, True), (3.0, False)]:
                feats, centers = make_case(rng, L, n_cls=20, n=400, noise=noise, half=half)
                ref = original_assign(feats, centers, radius)
                for device in devices:
                    out = hash_assign(torch.from_numpy(feats > 0), centers, radius, device=device)
                    n_all += 1
                    if np.array_equal(ref, out):
                        n_ok += 1
                    else:
                        diff = int((ref != out).sum())
                        print(f'MISMATCH L={L} radius={radius} noise={noise} half={half} '
                              f'device={device}: {diff}/{len(ref)} preds differ')
    print(f'{n_ok}/{n_all} cases identical (devices: {devices})')
    sys.exit(0 if n_ok == n_all else 1)


if __name__ == '__main__':
    main()
