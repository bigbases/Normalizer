from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Autoformer, DLinear, FEDformer, iTransformer
from normalizers import NoNorm, RevIN, SAN, DDN, LightTrend
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.norm_losses import NormLosses
from layers.Autoformer_EncDec import series_decomp

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import time

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.station_type = args.station_type
        self.decomp = series_decomp(args.kernel_size)
        self.norm_setting = f"{args.use_norm}_prelr{args.station_pre_lr}_snorm{args.s_norm}_tff{args.t_ff}_kl{args.kernel_len}"

    def _build_model(self):
        station_dict = {
            'none': NoNorm,
            'revin': RevIN,
            'san': SAN,
            'ddn': DDN,
            'lt': LightTrend,
        }
        self.station = station_dict[self.args.use_norm].Model(self.args).to(self.device)
        self.station_loss = NormLosses(self.args, station=self.station).to(self.device)
        
        # [pre train, pre epoch, joint train, join epoch]
        station_setting_dict = {
            'none': [0, 0, 0, 0],
            'revin': [0, 0, 0, 0],
            'san': [1, self.args.pre_epoch, 0, 0],
            'ddn': [1, self.args.pre_epoch, 1, self.args.twice_epoch],
            'tp': [1, self.args.pre_epoch, 1, self.args.twice_epoch],
            'lt': [1, self.args.pre_epoch, 1, self.args.twice_epoch],
        }
        self.station_setting = station_setting_dict[self.args.use_norm]
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'iTransformer': iTransformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.station_optim = optim.Adam(self.station.parameters(), lr=self.args.station_pre_lr)
        return model_optim

    def _select_criterion(self):
        self.criterion = nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion, epoch):
        total_loss = []
        self.model.eval()
        self.station.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # normalize
                if self.args.use_norm == 'ddn':
                    if epoch + 1 <= self.station_setting[1]:
                        batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x, p_value=False)
                    else:
                        batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x)
                else:
                    batch_x, statistics_pred = self.station.normalize(batch_x)

                # station pretrain
                if epoch + 1 <= self.station_setting[1]:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.features == 'MS':
                        statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                    loss = self.station_loss(batch_y, statistics_pred)
                
                # model train
                else:
                    # decoder x
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    if self.args.features == 'MS':
                        statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    
                    # de-normalize
                    outputs = self.station.de_normalize(outputs, statistics_pred)
                    
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)

                total_loss.append(loss.cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        self.station.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        path_station_pre = './station_pre/' + '{}_{}_s{}_p{}'.format(self.norm_setting, self.args.data_path[:-4],
                                                                self.args.seq_len, self.args.pred_len)
        path_station = './station/' + '{}_{}_s{}_p{}'.format(setting, self.args.data_path[:-4],
                                                        self.args.seq_len, self.args.pred_len)
        if not os.path.exists(path_station_pre):
            os.makedirs(path_station_pre)
        if not os.path.exists(path_station):
            os.makedirs(path_station)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_station_model = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # If station checkpoint exists and skip_pretrain is True, load station checkpoint and skip pretraining
        skip_pretrain = self.args.load_station
        skip_pretrain_effective = False
        if skip_pretrain and self.station_setting[0] > 0:
            best_station_path = os.path.join(path_station_pre, "checkpoint.pth")
            if os.path.exists(best_station_path):
                self.station.load_state_dict(torch.load(best_station_path, map_location=self.device))
                self.station_setting[1] = 0  # pre_epoch=0 => pretrain phase skip
                skip_pretrain_effective = True
                print(f"[station_training=False] loaded station ckpt and skip pretrain: {best_station_path}")
            else:
                print(f"[station_training=False] station ckpt not found -> run pretrain as usual: {best_station_path}")

        for epoch in range(self.args.train_epochs + self.station_setting[1]):
            iter_count = 0
            train_loss = []
            
            # Load best station model after pretraining
            if (not skip_pretrain_effective) and self.station_setting[0] > 0 and epoch == self.station_setting[1]:
                best_station_path = os.path.join(path_station_pre, "checkpoint.pth")
                if os.path.exists(best_station_path):
                    self.station.load_state_dict(torch.load(best_station_path, map_location=self.device))
                    print("loading pretrained adaptive station model")
                else:
                    print(f"[warn] station ckpt not found after pretraining: {best_station_path} (continue with current weights)")

            # Add station parameters to model optim after pretraining and delay epochs for joint training
            if self.station_setting[2] > 0 and self.station_setting[3] == epoch - self.station_setting[1]:
                joint_station_ckpt = os.path.join(path_station, "checkpoint.pth")
                torch.save(self.station.state_dict(), joint_station_ckpt)
                print(f"[save] station(pretrained) before joint training: {joint_station_ckpt}")
                
                lr = model_optim.param_groups[0]['lr']
                # lr = self.args.station_joint_lr
                model_optim.add_param_group({'params': self.station.parameters(), 'lr': lr})
            
            self.model.train()
            self.station.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # normalize
                if self.args.use_norm == 'ddn':
                    if epoch + 1 <= self.station_setting[1]:
                        batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x, p_value=False)
                    else:
                        batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x)
                else:
                    batch_x, statistics_pred = self.station.normalize(batch_x)
                
                # station pretrain
                if epoch + 1 <= self.station_setting[1]:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.features == 'MS':
                        statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                    loss = self.station_loss(batch_y, statistics_pred)
                    train_loss.append(loss.item())
                
                # model train
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder x
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = self.criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        if self.args.features == 'MS':
                            statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                            
                        # de-normalize
                        outputs = self.station.de_normalize(outputs, statistics_pred)
                        
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self.criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.args.train_epochs + self.station_setting[1] - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # two-stage training schema
                    if epoch + 1 <= self.station_setting[1]:
                        self.station_optim.step()
                    else:
                        model_optim.step()
                    model_optim.zero_grad()
                    self.station_optim.zero_grad()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, self.criterion, epoch)
            test_loss = self.vali(test_data, test_loader, self.criterion, epoch)

            if epoch + 1 <= self.station_setting[1]:
                print(
                    "Station Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping_station_model(vali_loss, self.station, path_station_pre)
                adjust_learning_rate(self.station_optim, epoch + 1, self.args, self.args.station_pre_lr)
            else:
                print(
                    "Backbone Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1 - self.station_setting[1], train_steps, train_loss, vali_loss, test_loss))
                # if: joint training, else: only model training
                if self.station_setting[2] > 0 and self.station_setting[3] <= epoch - self.station_setting[1]:
                    early_stopping(vali_loss, self.model, path, self.station, path_station)
                else:
                    early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                adjust_learning_rate(model_optim, epoch + 1 - self.station_setting[1], self.args,
                                     self.args.learning_rate)
                adjust_learning_rate(self.station_optim, epoch + 1 - self.station_setting[1], self.args,
                                     self.args.station_pre_lr)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.station_setting[2] > 0:
            self.station.load_state_dict(torch.load(path_station + '/' + 'checkpoint.pth'))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            path_station = './station/' + '{}_{}_s{}_p{}'.format(setting, self.args.data_path[:-4],
                                                        self.args.seq_len, self.args.pred_len)
            self.station.load_state_dict(torch.load(os.path.join(path_station, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.station.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                input_x = batch_x

                # normalize
                if self.args.use_norm == 'ddn':
                    batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x)
                else:
                    batch_x, statistics_pred = self.station.normalize(batch_x)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder x
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_label = batch_x[:, -self.args.label_len:, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.features == 'MS':
                    statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                    
                # de-normalize
                outputs = self.station.de_normalize(outputs, statistics_pred)
                
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    x = input_x.detach().cpu().numpy()
                    gt = np.concatenate((x[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((x[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # if self.args.test_flop:
        #     test_params_flop((batch_x.shape[1], batch_x.shape[2]))
        #     exit()
        preds = np.array(preds, dtype=object)
        trues = np.array(trues, dtype=object)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # result.csv save (if file is not exist, create it)
        result_file = 'result.csv'
        if not os.path.exists(result_file):
            with open(result_file, 'w') as f:
                f.write('Setting,MSE,MAE\n')
        with open(result_file, 'a') as f:
            f.write('{},{},{}\n'.format(setting, mse, mae))
        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder x
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
