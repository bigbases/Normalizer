from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
from utils.norm_losses import NormLosses
from models import Autoformer, DLinear, FEDformer, iTransformer
from normalizers import NoNorm, RevIN, SAN, DDN, LightTrend
from layers.Autoformer_EncDec import series_decomp
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)
        self.station_type = args.station_type
        self.decomp = series_decomp(args.kernel_size)
        self.norm_setting = f"{args.use_norm}_prelr{args.station_pre_lr}_snorm{args.s_norm}_tff{args.t_ff}_kl{args.kernel_len}"

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
            
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

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Station 모델용 체크포인트 경로 설정
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
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()
        
        # Station Pretrain 스킵 로직
        skip_pretrain = self.args.load_station
        skip_pretrain_effective = False
        if skip_pretrain and self.station_setting[0] > 0:
            best_station_path = os.path.join(path_station_pre, "checkpoint.pth")
            if os.path.exists(best_station_path):
                self.station.load_state_dict(torch.load(best_station_path, map_location=self.device))
                self.station_setting[1] = 0  
                skip_pretrain_effective = True
                print(f"[station_training=False] loaded station ckpt and skip pretrain: {best_station_path}")
            else:
                print(f"[station_training=False] station ckpt not found -> run pretrain as usual")

        for epoch in range(self.args.train_epochs + self.station_setting[1]):
            iter_count = 0
            train_loss = []
            
            # Pretrain 후 최적의 Station 모델 로드
            if (not skip_pretrain_effective) and self.station_setting[0] > 0 and epoch == self.station_setting[1]:
                best_station_path = os.path.join(path_station_pre, "checkpoint.pth")
                if os.path.exists(best_station_path):
                    self.station.load_state_dict(torch.load(best_station_path, map_location=self.device))
                    print("loading pretrained adaptive station model")
            
            # Joint Training 설정 (옵션)
            if self.station_setting[2] > 0 and self.station_setting[3] == epoch - self.station_setting[1]:
                joint_station_ckpt = os.path.join(path_station, "checkpoint.pth")
                torch.save(self.station.state_dict(), joint_station_ckpt)
                lr = self.args.station_joint_lr
                model_optim.add_param_group({'params': self.station.parameters(), 'lr': lr})

            self.model.train()
            self.station.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                self.station_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # [1] Normalize 적용
                if self.args.use_norm == 'ddn':
                    if epoch + 1 <= self.station_setting[1]:
                        batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x, p_value=False)
                    else:
                        batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x)
                else:
                    batch_x, statistics_pred = self.station.normalize(batch_x)

                # [2-A] Station Pretrain 단계
                if epoch + 1 <= self.station_setting[1]:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.features == 'MS':
                        statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                    loss = self.station_loss(batch_y, statistics_pred)
                    train_loss.append(loss.item())

                # [2-B] Model Train 단계
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()

                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x, None, dec_inp, None)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    
                    # De-normalize 적용 (예측값 복원)
                    if self.args.features == 'MS':
                        statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                    outputs = self.station.de_normalize(outputs, statistics_pred)
                    
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                    loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                    loss = loss_value  # + loss_sharpness * 1e-5
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # 단계별 Optimizer Step
                if epoch + 1 <= self.station_setting[1]:
                    self.station_optim.step()
                else:
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion, epoch)
            
            # 결과 출력 및 저장 로직
            if epoch + 1 <= self.station_setting[1]:
                print("Station Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(
                    epoch + 1, train_loss, vali_loss))
                early_stopping_station_model(vali_loss, self.station, path_station_pre)
                adjust_learning_rate(self.station_optim, epoch + 1, self.args, self.args.station_pre_lr)
            else:
                print("Model Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(
                    epoch + 1 - self.station_setting[1], train_loss, vali_loss))
                
                if self.station_setting[2] > 0 and self.station_setting[3] <= epoch - self.station_setting[1]:
                    early_stopping(vali_loss, self.model, path, self.station, path_station)
                else:
                    early_stopping(vali_loss, self.model, path)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1 - self.station_setting[1], self.args, self.args.learning_rate)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.station_setting[2] > 0:
            self.station.load_state_dict(torch.load(path_station + '/' + 'checkpoint.pth'))

        return self.model

    def vali(self, train_loader, vali_loader, criterion, epoch):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        self.station.eval() ###
        with torch.no_grad():
            B, _, C = x.shape
            
            if self.args.use_norm == 'ddn':
                if epoch + 1 <= self.station_setting[1]:
                    x, statistics_pred, statistics_seq = self.station.normalize(x, p_value=False)
                else:
                    x, statistics_pred, statistics_seq = self.station.normalize(x)
            else:
                x, statistics_pred = self.station.normalize(x)
            
            ### Pre-train phase
            if epoch + 1 <= self.station_setting[1]:
                f_dim = -1 if self.args.features == 'MS' else 0
                # true = torch.from_numpy(np.array(y)).float().to(self.device)
                batch_y_list = [v[-self.args.pred_len:] for v in y]
                true_np = np.array(batch_y_list, dtype=np.float32)
                true = torch.from_numpy(true_np).to(self.device)
                if true.dim() == 2:
                    true = true.unsqueeze(-1)
                if self.args.features == 'MS':
                    statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                loss = self.station_loss(true, statistics_pred)
            
            ### Joint-train phase
            else:
                dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
                dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
                
                outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
                id_list = np.arange(0, B, 500)
                id_list = np.append(id_list, B)
                
                for i in range(len(id_list) - 1):
                    if 'Linear' in self.args.model:
                        out = self.model(x[id_list[i]:id_list[i + 1]])
                    else:
                        out = self.model(x[id_list[i]:id_list[i + 1]], None,
                                        dec_inp[id_list[i]:id_list[i + 1]], None)
                    outputs[id_list[i]:id_list[i + 1], :, :] = out.detach()

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                if self.args.features == 'MS':
                    statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                outputs = self.station.de_normalize(outputs, statistics_pred)
                
                pred = outputs.detach().cpu()
                # true = torch.from_numpy(np.array(y)).float()
                batch_y_list = [v[-self.args.pred_len:] for v in y]
                true_np = np.array(batch_y_list, dtype=np.float32)
                true = torch.from_numpy(true_np)
                batch_y_mark = torch.ones(true.shape)

                loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        self.station.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            
            # Station 모델 경로 설정 및 로드
            path_station = './station/' + '{}_{}_s{}_p{}'.format(setting, self.args.data_path[:-4],
                                                        self.args.seq_len, self.args.pred_len)
            if os.path.exists(os.path.join(path_station, 'checkpoint.pth')):
                self.station.load_state_dict(torch.load(os.path.join(path_station, 'checkpoint.pth')))
                print(f'loading station model from {path_station}')
            else:
                print("Warning: Station checkpoint not found.")

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.station.eval()
        with torch.no_grad():
            x_raw = x.clone()
            
            if self.args.use_norm == 'ddn':
                x, statistics_pred, statistics_seq = self.station.normalize(x, p_value=False)
            else:
                x, statistics_pred = self.station.normalize(x)
            
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            
            for i in range(len(id_list) - 1):
                batch_x = x[id_list[i]:id_list[i + 1]]
                batch_dec_inp = dec_inp[id_list[i]:id_list[i + 1]]
                
                if 'Linear' in self.args.model:
                    out = self.model(batch_x)
                else:
                    out = self.model(batch_x, None, batch_dec_inp, None)
                outputs[id_list[i]:id_list[i + 1], :, :] = out

                if id_list[i] % 1000 == 0:
                    print(f"Processing: {id_list[i]}")

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            
            if self.args.features == 'MS':
                statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
            outputs = self.station.de_normalize(outputs, statistics_pred)
                
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x_vis = x_raw.detach().cpu().numpy()

            for i in range(0, preds.shape[0], max(1, preds.shape[0] // 10)):
                gt = np.concatenate((x_vis[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x_vis[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return