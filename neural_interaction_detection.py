import bisect
import operator
import numpy as np
import torch
from torch.utils import data
from multilayer_perceptron import *
from utils import *


def preprocess_weights(weights):
    w_later = np.abs(weights[-1])
    w_input = np.abs(weights[0])

    for i in range(len(weights) - 2, 0, -1):
        w_later = np.matmul(w_later, np.abs(weights[i]))

    return w_input, w_later


def make_one_indexed(interaction_ranking):
    return [(tuple(np.array(i) + 1), s) for i, s in interaction_ranking]


def interpret_interactions(w_input, w_later, get_main_effects=False):
    interaction_strengths = {}
    for i in range(w_later.shape[1]):
        sorted_hweights = sorted(
            enumerate(w_input[i]), key=lambda x: x[1], reverse=True
        )
        interaction_candidate = []
        candidate_weights = []
        for j in range(w_input.shape[1]):
            bisect.insort(interaction_candidate, sorted_hweights[j][0])
            candidate_weights.append(sorted_hweights[j][1])

            if not get_main_effects and len(interaction_candidate) == 1:
                continue
            interaction_tup = tuple(interaction_candidate)
            if interaction_tup not in interaction_strengths:
                interaction_strengths[interaction_tup] = 0
            interaction_strength = (min(candidate_weights)) * (np.sum(w_later[:, i]))
            interaction_strengths[interaction_tup] += interaction_strength

    interaction_ranking = sorted(
        interaction_strengths.items(), key=operator.itemgetter(1), reverse=True
    )

    return interaction_ranking


def interpret_pairwise_interactions(w_input, w_later):
    p = w_input.shape[1]

    interaction_ranking = []
    for i in range(p):
        for j in range(p):
            if i < j:
                strength = (np.minimum(w_input[:, i], w_input[:, j]) * w_later).sum()
                interaction_ranking.append(((i, j), strength))

    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    return interaction_ranking


def get_interactions(weights, pairwise=False, one_indexed=False):
    w_input, w_later = preprocess_weights(weights)

    if pairwise:
        interaction_ranking = interpret_pairwise_interactions(w_input, w_later)
    else:
        interaction_ranking = interpret_interactions(w_input, w_later)
        interaction_ranking = prune_redundant_interactions(interaction_ranking)

    if one_indexed:
        return make_one_indexed(interaction_ranking)
    else:
        return interaction_ranking


def prune_redundant_interactions(interaction_ranking, max_interactions=100):
    interaction_ranking_pruned = []
    current_superset_inters = []
    for inter, strength in interaction_ranking:
        set_inter = set(inter)
        if len(interaction_ranking_pruned) >= max_interactions:
            break
        subset_inter_skip = False
        update_superset_inters = []
        for superset_inter in current_superset_inters:
            if set_inter < superset_inter:
                subset_inter_skip = True
                break
            elif not (set_inter > superset_inter):
                update_superset_inters.append(superset_inter)
        if subset_inter_skip:
            continue
        current_superset_inters = update_superset_inters
        current_superset_inters.append(set_inter)
        interaction_ranking_pruned.append((inter, strength))

    return interaction_ranking_pruned


def detect_interactions(
    Xd,
    Yd,
    arch=[256, 128, 64],
    batch_size=100,
    device=torch.device("cpu"),
    seed=None,
    **kwargs
):

    if seed is not None:
        set_seed(seed)

    data_loaders = convert_to_torch_loaders(Xd, Yd, batch_size)

    model = create_mlp([feats.shape[1]] + arch + [1]).to(device)

    model, mlp_loss = train(model, data_loaders, device=device, **kwargs)
    inters = get_interactions(get_weights(model))

    return inters, mlp_loss
