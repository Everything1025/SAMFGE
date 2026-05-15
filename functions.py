import scipy.io as sio
import math
from sklearn import preprocessing
from munkres import Munkres
from sklearn import metrics
import torch
import os
import numpy as np

device = torch.device("cpu")


def get_model_class(model_name):
    if model_name == 'my_model':
        from my_model import UnifiedHSIClusteringModel
        return UnifiedHSIClusteringModel
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_superpixel_segmentation(dataset_name, input_numpy, gt_hsi, num_classes, scale):
    if dataset_name == 'PaviaU':
        import LDA_SLIC_PU as SLIC_Module
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    print(f"Executing SLIC segmentation for {dataset_name}...")

    ls = SLIC_Module.LDA_SLIC(input_numpy, gt_hsi, num_classes - 1)
    Q, S, A, Edge_index, Edge_atter, Seg, A_ones = ls.simple_superpixel(scale)

    return Q, S, A, Edge_index, Edge_atter, Seg, A_ones


def normalize(data):
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])
    return data


def load_dataset(Dataset):
    hyper_dir = r"D:\DataSets_eus"

    if Dataset == 'PaviaU':
        uPavia = sio.loadmat(os.path.join(hyper_dir, 'PaviaU.mat'))
        gt_uPavia = sio.loadmat(os.path.join(hyper_dir, 'PaviaU_gt.mat'))
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT


def sampling(proportion, ground_truth, CLASSES_NUM):
    train = {}
    test = {}
    train_num = []
    test_num = []
    labels_loc = {}
    for i in range(CLASSES_NUM):
        indexes = np.argwhere(ground_truth == (i + 1))
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            if indexes.shape[0] <= 60:
                nb_val = 15
            else:
                nb_val = 30
        else:
            nb_val = 0

        train_num.append(nb_val)
        test_num.append(len(indexes) - nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = train[0]
    test_indexes = test[0]
    for i in range(CLASSES_NUM - 1):
        train_indexes = np.concatenate((train_indexes, train[i + 1]), axis=0)
        test_indexes = np.concatenate((test_indexes, test[i + 1]), axis=0)
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes, train_num, test_num


def get_label(indices, gt_hsi):
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label = gt_hsi[dim_0, dim_1]
    return label


def get_data(dataset):
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(dataset)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
    CLASSES_NUM = max(gt)
    _, total_indices, _, total_num = sampling(1, gt_hsi, CLASSES_NUM)

    y_true = get_label(total_indices, gt_hsi) - 1

    return data_hsi, CLASSES_NUM, y_true, gt, gt_hsi


def spixel_to_pixel_labels(sp_level_label, association_mat):
    sp_level_label = np.reshape(sp_level_label, (-1, 1))
    pixel_level_label = np.matmul(association_mat, sp_level_label).reshape(-1)
    return pixel_level_label.astype('int')


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def class_acc(y_true, y_pre):
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = metrics.accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca


def cluster_accuracy(y_true, y_pre, return_aligned=False):
    y_true = y_true.astype('float32')
    y_pre = y_pre.astype('float32')
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]

    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ca = class_acc(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    fscore = metrics.f1_score(y_true, y_best, average='micro')
    pur = purity_score(y_true, y_best)
    if return_aligned:
        return y_best, acc, kappa, nmi, ari, pur, ca
    return acc, kappa, nmi, ari, pur, ca


def scipy_to_torch_sparse(coo_mat, device):
    coo_mat = coo_mat.tocoo()
    indices = torch.tensor(
        np.vstack((coo_mat.row, coo_mat.col)),
        dtype=torch.long,
        device=device
    )
    values = torch.tensor(coo_mat.data, dtype=torch.float32, device=device)
    L = torch.sparse_coo_tensor(indices, values, size=coo_mat.shape).coalesce()
    return L
