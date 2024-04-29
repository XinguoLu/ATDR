from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import torch

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, auc


def parameter_setting():
    parser = argparse.ArgumentParser(description='A deep cycle attention model to in silico drug repositioning')

    parser.add_argument('--lr1', type=float, default=0.01, help='Learning rate1')
    parser.add_argument('--flr1', type=float, default=0.001, help='Final learning rate1')
    parser.add_argument('--lr2', type=float, default=0.002, help='Learning rate2')
    parser.add_argument('--flr2', type=float, default=0.0002, help='Final learning rate2')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--eps', type=float, default=0.01, help='eps')

    parser.add_argument('--sf1', type=float, default=2.0, help='scale_factor_1 for supervision signal from scRNA-seq')
    parser.add_argument('--sf2', type=float, default=1.0,
                        help='scale_factor_2 for supervision signal from scEpigenomics')
    parser.add_argument('--cluster1', '-clu1', type=int, default=2, help='predefined cluster for scRNA')
    parser.add_argument('--cluster2', '-clu2', type=int, default=2, help='predefined cluster for other epigenomics')
    parser.add_argument('--geneClu', '-gClu', type=list, default=None, help='predefined gene cluster for scRNA')

    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--batch', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true',
                        help=" whether use cuda(default: True)")

    parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')
    parser.add_argument('--latent', '-l', type=int, default=10, help='latent layer dim')
    parser.add_argument('--max_epoch', '-me', type=int, default=1000, help='Max epoches')
    parser.add_argument('--max_iteration', '-mi', type=int, default=3000, help='Max iteration')
    parser.add_argument('--anneal_epoch', '-ae', type=int, default=200, help='Anneal epoch')
    parser.add_argument('--epoch_per_test', '-ept', type=int, default=10, help='Epoch per test')
    parser.add_argument('--max_ARI', '-ma', type=int, default=-200, help='initial ARI')

    parser.add_argument('--gpu', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--learn_rate', help='learning rate', type=float, default=0.001)  # 用到了
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    return parser


def read_dataset(File1=None, File2=None, transpose=True,
                 format_drug="table", format_info="table"):
    adata = adata1 = None

    if File1 is not None:
        if format_drug == "table":
            adata = sc.read(File1)
        else:
            adata1 = sc.read_mtx(File2)

        if transpose:
            adata = adata.transpose()

    if File2 is not None:
        if format_info == "table":
            adata1 = sc.read(File2)
        else:
            adata1 = sc.read_mtx(File2)

        if transpose:
            adata1 = adata1.transpose()

    print('Successfully preprocessed {} drugs and {} diseases.'.format(adata.n_obs, adata.n_vars))
    print('Successfully preprocessed {} drugs and {} features.'.format(adata1.n_obs, adata1.n_vars))

    return adata, adata1


def get_data_set(adata_drug):
    whole_positive_index = []  # 创建全正和全负的索引
    whole_negative_index = []

    for i in range(np.shape(adata_drug.X)[0]):
        for j in range(np.shape(adata_drug.X)[1]):
            if int(adata_drug.X[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(adata_drug.X[i][j]) == 0:
                whole_negative_index.append([i, j])

    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(whole_positive_index), replace=False)

    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)

    count = 0

    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    return data_set


def normalize(adata, filter_min_counts=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    else:
        adata.raw = adata

    return adata


def calculate_log_library_size(Dataset):
    ### Dataset is raw read counts, and should be cells * features
    Nsamples = np.shape(Dataset)[0]
    library_sum = np.log(np.sum(Dataset, axis=1))

    lib_mean = np.full((Nsamples, 1), np.mean(library_sum))
    lib_var = np.full((Nsamples, 1), np.var(library_sum))

    return lib_mean, lib_var


def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):
    lr = max(init_lr * (0.9 ** (iteration // adjust_epoch)), max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def save_checkpoint(model, filename='model_best.pth.tar'):
    torch.save(model.state_dict(), filename)


def load_checkpoint(file_path, model, use_cuda=False):
    if use_cuda:
        device = torch.device("cuda")
        model.load_state_dict(torch.load(file_path))
        model.to(device)

    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(file_path, map_location=device))
    model.eval()

    return model


def Normalized_0_1(Data):
    ## here ,Data is cell * genes
    adata = sc.AnnData(Data)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1,
                             key_n_counts='n_counts2')
    return adata


def estimate_cluster_numbers(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (n,p) matrix, n is feature, p is sample
    """
    n, p = data.shape
    if type(data) is not np.ndarray:
        data = data.toarray()

    x = scale(data)  # normalization for each sample
    muTW = (np.sqrt(n - 1) + np.sqrt(p)) ** 2
    sigmaTW = (np.sqrt(n - 1) + np.sqrt(p)) * (1 / np.sqrt(n - 1) + 1 / np.sqrt(p)) ** (1 / 3)
    sigmaHatNaive = x.T.dot(x)

    bd = np.sqrt(p) * sigmaTW + muTW
    evals = np.linalg.eigvalsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k


def pearson(data):
    print('Start for pearson similarity ')
    df = pd.DataFrame(data.T)
    pear_ = df.corr(method='pearson')
    return np.where(pear_ >= 0, pear_, np.zeros(shape=(pear_.shape)))


def evaluate(model, drugnet, DTItrain, DTItest, state):
    output = model.inference(drugnet)
    score = output["recon_x"]

    Zscore = score.detach().numpy()

    # train auc aupr
    pred_list = []
    ground_truth = []
    for ele in DTItrain:
        pred_list.append(Zscore[ele[0], ele[1]])
        ground_truth.append(ele[2])
    train_auc = roc_auc_score(ground_truth, pred_list)
    train_aupr = average_precision_score(ground_truth, pred_list)
    print('train auc aupr,', train_auc, train_aupr)

    # test auc aupr
    pred_list = []
    ground_truth = []
    for ele in DTItest:
        pred_list.append(Zscore[ele[0], ele[1]])
        ground_truth.append(ele[2])
    test_auc = roc_auc_score(ground_truth, pred_list)
    test_aupr = average_precision_score(ground_truth, pred_list)
    print('test auc aupr', test_auc, test_aupr)

    if state==1:
        f = open("./result/ATDR_predicted.txt", "a")
        f.write(str(pred_list))
        f.close()
        f = open("./result/ATDR_ground_truth.txt", "a")
        f.write(str(ground_truth))
        f.close()

    return test_auc, test_aupr
