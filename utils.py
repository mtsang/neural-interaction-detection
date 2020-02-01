import torch
from torch.utils import data
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def force_float(X_numpy):
    return torch.from_numpy(X_numpy.astype(np.float32))


def convert_to_torch_loaders(Xd, Yd, batch_size):
    if type(Xd) != dict and type(Yd) != dict:
        Xd = {"train": Xd}
        Yd = {"train": Yd}

    data_loaders = {}
    for k in Xd:
        if k == "scaler":
            continue
        feats = force_float(Xd[k])
        targets = force_float(Yd[k])
        dataset = data.TensorDataset(feats, targets)
        data_loaders[k] = data.DataLoader(dataset, batch_size, shuffle=(k == "train"))

    return data_loaders


def preprocess_data(
    X,
    Y,
    valid_size=500,
    test_size=500,
    std_scale=False,
    get_torch_loaders=False,
    batch_size=100,
):

    n, p = X.shape
    ## Make dataset splits
    ntrain, nval, ntest = n - valid_size - test_size, valid_size, test_size

    Xd = {
        "train": X[:ntrain],
        "val": X[ntrain : ntrain + nval],
        "test": X[ntrain + nval : ntrain + nval + ntest],
    }
    Yd = {
        "train": np.expand_dims(Y[:ntrain], axis=1),
        "val": np.expand_dims(Y[ntrain : ntrain + nval], axis=1),
        "test": np.expand_dims(Y[ntrain + nval : ntrain + nval + ntest], axis=1),
    }

    for k in Xd:
        if len(Xd[k]) == 0:
            assert k != "train"
            del Xd[k]
            del Yd[k]

    if std_scale:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(Xd["train"])
        scaler_y.fit(Yd["train"])

        for k in Xd:
            Xd[k] = scaler_x.transform(Xd[k])
            Yd[k] = scaler_y.transform(Yd[k])

        Xd["scaler"] = scaler_x
        Yd["scaler"] = scaler_y

    if get_torch_loaders:
        return convert_to_torch_loaders(Xd, Yd, batch_size)

    else:
        return Xd, Yd


def get_pairwise_auc(interactions, ground_truth):
    strengths = []
    gt_binary_list = []
    for inter, strength in interactions:
        inter_set = set(inter)  # assume 1-indexed
        strengths.append(strength)
        if any(inter_set <= gt for gt in ground_truth):
            gt_binary_list.append(1)
        else:
            gt_binary_list.append(0)

    auc = roc_auc_score(gt_binary_list, strengths)
    return auc


def get_anyorder_R_precision(interactions, ground_truth):

    R = len(ground_truth)
    recovered_gt = []
    counter = 0

    for inter, strength in interactions:
        if counter == R:
            break

        inter_set = set(inter)  # assume 1-indexed

        if any(inter_set < gt for gt in ground_truth):
            continue
        counter += 1
        if inter_set in ground_truth:
            recovered_gt.append(inter_set)

    R_precision = len(recovered_gt) / R

    return R_precision


def print_rankings(pairwise_interactions, anyorder_interactions, top_k=10, spacing=14):
    print(
        justify(["Pairwise interactions", "", "Arbitrary-order interactions"], spacing)
    )
    for i in range(top_k):
        p_inter, p_strength = pairwise_interactions[i]
        a_inter, a_strength = anyorder_interactions[i]
        print(
            justify(
                [
                    p_inter,
                    "{0:.4f}".format(p_strength),
                    "",
                    a_inter,
                    "{0:.4f}".format(a_strength),
                ],
                spacing,
            )
        )


def justify(row, spacing=14):
    return "".join(str(item).ljust(spacing) for item in row)
