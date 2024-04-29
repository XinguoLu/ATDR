import time

from torch.autograd import Variable
from torch import optim

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence as kl

from AT.layers import Encoder, Decoder_logNorm_ZINB, Decoder_logNorm_NB, Decoder
from AT.loss_function import log_zinb_positive, log_nb_positive, binary_cross_entropy, mse_loss, KL_diver, \
    regularization
from AT.loss_function import NSTLoss, FactorTransfer, Similarity, Correlation, Attention, Eucli_dis, L1_dis
from AT.utilities import evaluate

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class VAE(nn.Module):
    # def __init__( self, layer_e, hidden1, hidden2, layer_l, layer_d, hidden ):
    def __init__(self, layer_e, hidden1, Zdim, layer_d, hidden2,
                 Type='NB', penality='Gaussian', droprate=0.3):

        super(VAE, self).__init__()

        ###  encoder
        self.encoder = Encoder(layer_e, hidden1, Zdim, droprate=droprate)
        self.activation = nn.Softmax(dim=-1)

        ### the decoder
        if Type == 'ZINB':
            self.decoder = Decoder_logNorm_ZINB(layer_d, hidden2, layer_e[0], droprate=droprate)

        elif Type == 'NB':
            self.decoder = Decoder_logNorm_NB(layer_d, hidden2, layer_e[0], droprate=droprate)

        else:  ## Bernoulli, or Gaussian
            self.decoder = Decoder(layer_d, hidden2, layer_e[0], Type, droprate=droprate)

        ### parameters
        self.Type = Type
        self.penality = penality

    def inference(self, X=None, scale_factor=1.0):
        # encoder
        mean_1, logvar_1, latent_1, hidden = self.encoder.return_all_params(X)

        ### decoder
        if self.Type == 'ZINB':
            output = self.decoder(latent_1, scale_factor)
            norm_x = output["normalized"]
            disper_x = output["disperation"]
            recon_x = output["scale_x"]
            dropout_rate = output["dropoutrate"]

        elif self.Type == 'NB':
            output = self.decoder(latent_1, scale_factor)
            norm_x = output["normalized"]
            disper_x = output["disperation"]
            recon_x = output["scale_x"]
            dropout_rate = None

        else:
            recons_x = self.decoder(latent_1)
            recon_x = recons_x
            norm_x = recons_x
            disper_x = None
            dropout_rate = None

        return dict(norm_x=norm_x, disper_x=disper_x, dropout_rate=dropout_rate,
                    recon_x=recon_x, latent_z1=latent_1, mean_1=mean_1,
                    logvar_1=logvar_1, hidden=hidden
                    )

    def return_loss(self, X=None, X_raw=None, latent_pre=None,
                    mean_pre=None, logvar_pre=None, latent_pre_hidden=None,
                    scale_factor=1.0, cretion_loss=None, attention_loss=None, args=1):

        # output       = self.inference( X, scale_factor )
        output = self.inference(X)
        recon_x = output["recon_x"]
        disper_x = output["disper_x"]
        dropout_rate = output["dropout_rate"]

        mean_1 = output["mean_1"]
        logvar_1 = output["logvar_1"]
        latent_z1 = output["latent_z1"]

        hidden = output["hidden"]

        if self.Type == 'ZINB':
            loss = log_zinb_positive(X_raw, recon_x, disper_x, dropout_rate)

        elif self.Type == 'NB':
            loss = log_nb_positive(X_raw, recon_x, disper_x)

        elif self.Type == 'Bernoulli':  # here X and X_raw are same
            loss = binary_cross_entropy(recon_x, X, args)

        else:
            loss = mse_loss(X, recon_x)

        # # calculate KL loss for Gaussian distribution
        mean = torch.zeros_like(mean_1)
        scale = torch.ones_like(logvar_1)
        logvar_1 = torch.abs(logvar_1)
        # kl_divergence_z = kl(Normal(mean_1, logvar_1),
        #                      Normal(mean, scale)).sum(dim=1)

        atten_loss1 = torch.tensor(0.0)
        # if latent_pre is not None and latent_pre_hidden is not None:
        if latent_pre is not None:
            # if attention_loss == "KL_div":
            # atten_loss1 = cretion_loss(mean_1, logvar_1, mean_pre, logvar_pre)
            # else:
            atten_loss1 = cretion_loss(latent_z1, latent_pre)

        return loss, atten_loss1, mean_1, logvar_1, latent_z1

    def forward(self, X=None, scale_factor=1.0):

        output = self.inference(X, scale_factor)

        return output

    def fit(self, train_loader, args, model_pre, criterion, state, train=None, auxiliary=None):

        train()

        if state == 0:

            optimizer = optim.Adam(self.parameters(), lr=args.learn_rate, )

            test_like_max = 9999999
            patience_epoch = 0

            start = time.time()

            for epoch in range(1, args.max_epoch + 1):

                loss_value = 0

                for batch_idx, data in enumerate(train_loader):
                    data = data.to(args.device)
                    data = Variable(data)
                    optimizer.zero_grad()
                    loss1, _, mu, logvar, _ = self.return_loss(data)
                    loss = loss1 + regularization(mu, logvar)* args.b
                    loss.backward()
                    loss_value += loss.item()
                    optimizer.step()

                print('Epoch: {} Average loss: {:.4f}'.format(epoch, loss_value / len(train_loader.dataset)))

                if test_like_max > loss_value:
                    test_like_max = loss_value
                    patience_epoch = 0

                patience_epoch = patience_epoch + 1

                if patience_epoch >= 15:
                    print("patient with 15")
                    break

            duration = time.time() - start
            print('Finish network training, total time is: ' + str(duration) + 's')


        else:

            optimizer = optim.Adam(self.parameters(), lr=args.learn_rate, )

            test_like_max = 99999999
            patience_epoch = 0

            start = time.time()

            for epoch in range(1, args.max_epoch + 1):

                loss_value = 0

                train = train.to(args.device)
                train = Variable(train)
                optimizer.zero_grad()

                output = model_pre(auxiliary)
                latent_info = output['latent_z1']

                loss1, atten_loss, mu, logvar, _ = self.return_loss(train, latent_pre=latent_info,
                                                                       cretion_loss=criterion, args=args.alpha)
                loss = loss + regularization(mu, logvar) + 0.5 * atten_loss
                loss.backward()
                loss_value += loss.item()
                optimizer.step()

                if test_like_max > loss_value:
                    test_like_max = loss_value
                    patience_epoch = 0

                patience_epoch = patience_epoch + 1

                print('Epoch: {} Average loss: {:.4f}'.format(epoch, loss_value / len(train_loader.dataset)))

                if patience_epoch >= 30:
                    print("patient with 30")
                    break

            duration = time.time() - start
            print('Finish network Learning, total time is: ' + str(duration) + 's')


class ATDR(nn.Module):
    def __init__(self, layer_e_1, hidden1_1, Zdim_1, layer_d_1, hidden2_1,
                 layer_e_2, hidden1_2, Zdim_2, layer_d_2, hidden2_2, args,
                 ground_truth=None, ground_truth1=None, Type_1='NB', Type_2='Bernoulli',
                 attention_loss='Eucli', penality='Gaussian', droprate=0.1):

        super(ATDR, self).__init__()

        self.model1 = VAE(layer_e=layer_e_1, hidden1=hidden1_1, Zdim=Zdim_1,
                          layer_d=layer_d_1, hidden2=hidden2_1,
                          Type=Type_1, penality=penality, droprate=droprate)

        self.model2 = VAE(layer_e=layer_e_2, hidden1=hidden1_2, Zdim=Zdim_2,
                          layer_d=layer_d_2, hidden2=hidden2_2,
                          Type=Type_2, penality=penality, droprate=droprate)

        if attention_loss == 'NST':
            self.attention = NSTLoss()

        elif attention_loss == 'FT':
            self.attention = FactorTransfer()

        elif attention_loss == 'SL':
            self.attention = Similarity()

        elif attention_loss == 'CC':
            self.attention = Correlation()

        elif attention_loss == 'AT':
            self.attention = Attention()

        elif attention_loss == 'KL_div':
            self.attention = KL_diver()

        elif attention_loss == 'L1':
            self.attention = L1_dis()

        else:
            self.attention = Eucli_dis()

        self.device = args.device
        self.args = args
        self.ground_truth = ground_truth
        self.ground_truth1 = ground_truth1
        self.penality = penality
        self.attention_loss = attention_loss

    def fit_model(self, args, drugnet, infonet, DTItrain, DTItest,
                  drug_loader=None, info_loader=None, model1=None, model2=None, ):

        self.model2.fit(info_loader, args, model1, self.attention, 0, infonet, drugnet)
        self.model1.fit(drug_loader, args, model2, self.attention, 0, drugnet, infonet)

        evaluate(model1, drugnet, DTItrain, DTItest, 0)

        self.model2.fit(info_loader, args, model1, self.attention, 1, infonet, drugnet)
        self.model1.fit(drug_loader, args, model2, self.attention, 1, drugnet, infonet)

        test_auc, test_aupr = evaluate(model1, drugnet, DTItrain, DTItest, 1)

        return test_auc, test_aupr
