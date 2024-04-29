import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from AT.utilities import read_dataset, normalize, parameter_setting, get_data_set
from AT.MVAE_cycleVAE import ATDR
from AT.MVAE_cycleVAE import VAE


def train_with_argas(args):
    args.batch_size = 64
    args.epoch_per_test = 10
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    args.alpha = 15
    args.b = 3

    args.workdir = './dataset/'  # Work path
    args.outdir = './result/'  # Output path

    args.File1 = os.path.join(args.workdir, 'drugDisease.txt')
    args.File2 = os.path.join(args.workdir, 'drugmdaFeatures.txt')

    adata_drug, adata_info = read_dataset(File1=args.File1, File2=args.File2, transpose=False)

    adata_drug = normalize(adata_drug, filter_min_counts=True)
    adata_info = normalize(adata_info, filter_min_counts=False)

    Nsample1, Nfeature1 = np.shape(adata_drug.X)
    Nsample2, Nfeature2 = np.shape(adata_info.X)

    data_set = get_data_set(adata_drug)

    model = ATDR(layer_e_1=[Nfeature1, 800, 400, 100], hidden1_1=100, Zdim_1=30,
                 layer_d_1=[30, 100, 400, 800], hidden2_1=800,
                 layer_e_2=[Nfeature2, 200, 100], hidden1_2=100, Zdim_2=30,
                 layer_d_2=[30, 400], hidden2_2=400,
                 args=args, Type_1='Bernoulli', Type_2='Gaussian',
                 attention_loss='Eucli_dis').to(args.device)

    test_auc_fold = []
    test_aupr_fold = []

    rs = np.random.randint(0, 1000, 1)[0]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    for train_index, test_index in kf.split(np.zeros(len(data_set)), data_set[:, 2]):
        DTItrain, DTItest = data_set[train_index], data_set[test_index]
        Xtrain = np.zeros((np.shape(adata_drug.X)[0], np.shape(adata_drug.X)[1]))
        for ele in DTItrain:
            Xtrain[ele[0], ele[1]] = ele[2]

        drugnet = torch.from_numpy(Xtrain.astype('float32')).to(args.device)
        drug_loader = DataLoader(drugnet, args.batch, shuffle=True)

        model.model1 = VAE(layer_e=[Nfeature1, 800, 400, 100], hidden1=100, Zdim=30,
                           layer_d=[30, 100, 400, 800], hidden2=800,
                           Type='Bernoulli', penality='Gaussian', droprate=0.1)

        infonet = torch.from_numpy(adata_info.X).to(args.device)
        info_loader = DataLoader(infonet, args.batch, shuffle=True)

        model.model2 = VAE(layer_e=[Nfeature2, 200, 100], hidden1=100, Zdim=30,
                           layer_d=[30, 400], hidden2=400,
                           Type='Gaussian', penality='Gaussian', droprate=0.1)

        test_auc, test_aupr=model.fit_model(args, drugnet, infonet, DTItrain, DTItest,
                                drug_loader, info_loader, model.model1, model.model2,)

        test_auc_fold.append(test_auc)
        test_aupr_fold.append(test_aupr)


    avg_auc = np.mean(test_auc_fold)
    avg_pr = np.mean(test_aupr_fold)
    print('mean auc aupr', avg_auc, avg_pr)

    print('Finish Drug-Disease Prediction')


if __name__ == "__main__":
    parser = parameter_setting()
    args = parser.parse_args()

    # whether to ran with cuda
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    train_with_argas(args)
