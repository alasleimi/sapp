"""Compiled evaluation of the accompanying MPUrge-MAP implementation."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def subset(longer, target):
    n, m = len(longer), len(target)
    cost = np.full((m + 1, n + 1), np.inf)
    take = np.zeros((m + 1, n + 1), np.bool_)
    cost[0, :] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            choose = cost[i - 1, j - 1] + abs(longer[j - 1] - target[i - 1])
            skip = cost[i, j - 1]
            if choose <= skip:
                cost[i, j] = choose
                take[i, j] = True
            else:
                cost[i, j] = skip
    selected = np.empty(m, np.int64)
    i, j = m, n
    while i:
        if take[i, j]:
            selected[i - 1] = j - 1
            i -= 1
        j -= 1
    return longer[selected]


@njit(cache=True)
def windows(a, p):
    n = len(a)
    scale = max(a[-1] - a[0], np.max(np.abs(a)), 1.0)
    tiny = np.finfo(np.float64).eps * scale * 16.0
    padded = np.empty(n + 2 * p)
    padded[p : p + n] = a
    for j in range(p):
        padded[j] = a[0] - tiny * (p - j)
        padded[p + n + j] = a[-1] + tiny * (j + 1)
    w = np.empty((n, 2 * p + 1))
    for i in range(n):
        for j in range(2 * p + 1):
            w[i, j] = padded[i + j] - a[i]
    return w


@njit(cache=True)
def pair_score(query, reference, p=2, alpha=0.7):
    original_max = max(len(query), len(reference))
    if min(len(query), len(reference)) == 0:
        return np.inf
    a, b = query.copy(), reference.copy()
    if len(a) > len(b):
        a = subset(a, b)
    elif len(b) > len(a):
        b = subset(b, a)
    accepted_a = np.empty(len(a))
    accepted_b = np.empty(len(a))
    accepted_d = np.empty(len(a))
    accepted_n = 0
    while len(a):
        n = len(a)
        wa, wb = windows(a, p), windows(b, p)
        delta = np.empty((n, n))
        normalizer = (2 * p) * (2 * p + 1) / 2
        for i in range(n):
            for j in range(n):
                pattern = 0.0
                for u in range(2 * p + 1):
                    for v in range(u + 1, 2 * p + 1):
                        pattern += abs(abs(wa[i, u] - wa[i, v]) - abs(wb[j, u] - wb[j, v]))
                if normalizer:
                    pattern /= normalizer
                delta[i, j] = alpha * abs(a[i] - b[j]) + (1 - alpha) * pattern
        row_best = np.empty(n, np.int64)
        col_best = np.empty(n, np.int64)
        for i in range(n):
            row_best[i] = np.argmin(delta[i, :])
            col_best[i] = np.argmin(delta[:, i])
        pi = np.empty(n, np.int64)
        pj = np.empty(n, np.int64)
        pn = 0
        keep_a, keep_b = np.ones(n, np.bool_), np.ones(n, np.bool_)
        for i in range(n):
            j = row_best[i]
            if col_best[j] == i:
                keep_a[i] = False
                keep_b[j] = False
                pi[pn], pj[pn] = i, j
                pn += 1
        # Stable insertion sort gives the source's (cost, index_a, index_b) order.
        for k in range(1, pn):
            ii, jj = pi[k], pj[k]
            h = k - 1
            while h >= 0 and delta[pi[h], pj[h]] > delta[ii, jj]:
                pi[h + 1], pj[h + 1] = pi[h], pj[h]
                h -= 1
            pi[h + 1], pj[h + 1] = ii, jj
        for k in range(pn):
            i, j = pi[k], pj[k]
            valid = True
            for h in range(accepted_n):
                if (a[i] - accepted_a[h]) * (b[j] - accepted_b[h]) < 0:
                    valid = False
                    break
            if valid:
                accepted_a[accepted_n] = a[i]
                accepted_b[accepted_n] = b[j]
                accepted_d[accepted_n] = delta[i, j]
                accepted_n += 1
        a, b = a[keep_a], b[keep_b]
    if accepted_n == 0:
        return np.inf
    return np.mean(accepted_d[:accepted_n]) / (accepted_n / original_max)


@njit(cache=True)
def predict(queries, query_counts, survey, survey_counts, xy):
    out = np.empty((len(queries), 2))
    for q in range(len(queries)):
        scores = np.empty(len(survey))
        for r in range(len(survey)):
            scores[r] = pair_score(queries[q, : query_counts[q]], survey[r, : survey_counts[r]])
        selected = np.empty(3, np.int64)
        number = 0
        for k in range(min(3, len(survey))):
            chosen = -1
            best = np.inf
            for r in range(len(survey)):
                already = False
                for h in range(number):
                    if selected[h] == r:
                        already = True
                if not already and scores[r] < best:
                    chosen, best = r, scores[r]
            if chosen >= 0:
                selected[number] = chosen
                number += 1
        if number == 0:
            out[q, 0] = np.mean(xy[:, 0])
            out[q, 1] = np.mean(xy[:, 1])
        else:
            weights = np.empty(number)
            for k in range(number):
                weights[k] = 1 / max(scores[selected[k]], np.finfo(np.float64).eps)
            weights /= weights.sum()
            out[q, :] = 0.0
            for k in range(number):
                out[q, :] += weights[k] * xy[selected[k], :]
    return out
