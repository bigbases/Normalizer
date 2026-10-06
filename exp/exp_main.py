from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from importlib import import_module
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import StreamingMetrics
from utils.norm_losses import NormLosses

import csv
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    """Main training/eval runtime for the production forecasting pipeline.

    Supported ``use_norm`` values: ``none``, ``revin``, ``san``, ``ddn``,
    ``fan``, and ``lt``.

    Ablation / analysis variants (san_ms, ddn_ms, lt_joint, lt_freeze,
    naive_last, naive_linear, san_phase1, ddn_phase1) and their supporting
    plumbing (aux_trend_loss, trajectory logging, oracle-std replacement,
    load_station / explicit station_dir+station_pre_dir paths, auxiliary CSV
    sinks) live in ``exp/exp_extra.py`` (``Exp_Extra``) and are launched via
    ``run_extra_exp.py``.
    """

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.station_type = args.station_type
        self.norm_setting = (
            f"{args.use_norm}_prelr{args.station_lr}_snorm{args.s_norm}"
            f"_tff{args.t_ff}_kl{args.kernel_len}"
        )
        self.best_vali_loss = None
        self.best_epoch = -1

    def _station_paths(self, setting):
        # Main runtime uses fixed, simple subdirectories. Extra_exp overrides
        # this in Exp_Extra to honor --station_dir / --station_pre_dir.
        from pathlib import Path
        # The scheduler passes a run-scoped root so that a finished
        # search/confirm run can be cleaned up as one directory.
        station_root = Path(getattr(self.args, 'station_root', '.') or '.')
        pre_root = station_root / 'station_pre'
        joint_root = station_root / 'station'
        # Include the full setting (seed + config hash) in both paths.  The old
        # path reused a pretrained normalizer across seeds and search configs.
        pre_path = pre_root / setting
        joint_path = joint_root / (
            f"{setting}_{self.args.data_path[:-4]}"
            f"_s{self.args.seq_len}_p{self.args.pred_len}"
        )
        pre_path.mkdir(parents=True, exist_ok=True)
        joint_path.mkdir(parents=True, exist_ok=True)
        return str(pre_path), str(joint_path)

    def _result_row(self, setting, *, mse=None, mae=None, split='test'):
        return {
            'RunID': getattr(self.args, 'run_id', ''),
            'CandidateID': getattr(self.args, 'candidate_id', ''),
            'ConfigHash': getattr(self.args, 'config_hash', ''),
            'Phase': getattr(self.args, 'phase', 'final'),
            'Split': split,
            'Setting': setting,
            'Dataset': self.args.data,
            'Backbone': self.args.model,
            'Horizon': self.args.pred_len,
            'MSE': mse,
            'MAE': mae,
            'Seed': getattr(self.args, 'seed', 2021),
            'BestValMSE': self.best_vali_loss,
            'BestEpoch': self.best_epoch,
            'UseNorm': self.args.use_norm,
        }

    def _append_result_row(self, row):
        result_file = getattr(self.args, 'result_file', 'result.csv')
        result_dir = os.path.dirname(result_file)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        file_exists = os.path.exists(result_file)
        with open(result_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def record_validation_result(self, setting):
        """Persist search metadata without opening/evaluating the test split."""
        self._append_result_row(self._result_row(setting, split='validation'))

    def _build_model(self):
        station_module_name = {
            'none': 'NoNorm',
            'revin': 'RevIN',
            'san': 'SAN',
            'ddn': 'DDN',
            'fan': 'FAN',
            'lt': 'LightNorm',
        }[self.args.use_norm]
        station_mod = import_module(f'normalizers.{station_module_name}')
        self.station = station_mod.Model(self.args).to(self.device)
        self.station_loss = NormLosses(self.args, station=self.station).to(self.device)

        # [pretrain_on, pre_epochs, joint_station_on, twice_epoch]
        station_setting_dict = {
            'none':  [0, 0, 0, 0],
            # RevIN affine parameters must be optimized jointly when enabled.
            'revin': [0, 0, int(bool(self.args.affine)), 0],
            'san':   [1, self.args.pre_epoch, 0, 0],
            'ddn':   [1, self.args.pre_epoch, 1, self.args.twice_epoch],
            # FAN's frequency predictor is trained jointly with the backbone.
            'fan':   [0, 0, 1, 0],
            'lt':    [1, self.args.pre_epoch, 1, 0],
        }
        self.station_setting = station_setting_dict[self.args.use_norm]

        # Lazy imports keep optional backbones (e.g. TimeMixer++) isolated and
        # give an actionable error only when that model is selected.
        try:
            model_module = import_module(f'models.{self.args.model}')
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"Could not import backbone '{self.args.model}' from models/"
            ) from exc
        model = model_module.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        station_params = [p for p in self.station.parameters() if p.requires_grad]
        self.station_optim = (
            optim.Adam(station_params, lr=self.args.station_lr) if station_params else None
        )
        return model_optim

    def _select_criterion(self):
        self.criterion = nn.MSELoss()

    def _on_epoch_end(self, *, epoch, phase, train_loss, vali_loss, test_loss):
        """Hook called after every epoch's vali/test evaluation.

        Default implementation is a no-op. Notebook clients can either
        subclass Exp_Main and override this, or monkey-patch an instance
        (exp._on_epoch_end = my_callback) to collect training-curve data.

        Arguments (all keyword-only):
            epoch:       1-based epoch number in the combined (pretrain + joint) schedule
            phase:       'pretrain' or 'joint'
            train_loss:  average training loss across the epoch
            vali_loss:   validation loss (same definition as early_stopping uses)
            test_loss:   test loss (informational only; not used for model selection)
        """
        pass

    def _normalize_input(self, batch_x, epoch_idx):
        if self.args.use_norm == 'ddn':
            if epoch_idx is not None and epoch_idx + 1 <= self.station_setting[1]:
                batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x, p_value=False)
            else:
                batch_x, statistics_pred, statistics_seq = self.station.normalize(batch_x)
            return batch_x, statistics_pred, statistics_seq
        batch_x, statistics_pred = self.station.normalize(batch_x)
        return batch_x, statistics_pred, None

    def _prepare_decoder_input(self, batch_x, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_label = batch_x[:, -self.args.label_len:, :]
        dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
        return dec_inp

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if 'Linear' in self.args.model:
            outputs = self.model(batch_x)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def _select_f_dim(self):
        return -1 if self.args.features == 'MS' else 0

    def _trim_outputs_and_target(self, outputs, batch_y):
        f_dim = self._select_f_dim()
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return outputs, batch_y

    def _maybe_trim_statistics_for_ms(self, statistics_pred):
        if self.args.features != 'MS' or statistics_pred is None:
            return statistics_pred
        # only statistics-prediction modules expose [mean, std] pairs
        if self.args.use_norm in ('san', 'ddn'):
            return statistics_pred[:, :, [self.args.enc_in - 1, -1]]
        if self.args.use_norm in ('fan', 'lt'):
            return statistics_pred[:, :, -1:]
        return statistics_pred

    def _denormalize(self, outputs, statistics_pred, batch_y):
        statistics_pred = self._maybe_trim_statistics_for_ms(statistics_pred)
        return self.station.de_normalize(outputs, statistics_pred)

    def _training_forecast_loss(self, outputs, batch_y, statistics_pred):
        """Return the method's declared training objective.

        All methods use forecast MSE in the original value space except FAN,
        whose paper/reference code optimizes residual and main-frequency MSEs.
        Validation and final selection always use original-space forecast MSE.
        """
        statistics_pred = self._maybe_trim_statistics_for_ms(statistics_pred)
        if self.args.use_norm == 'fan':
            return self.station.training_loss(outputs, batch_y, statistics_pred)
        outputs = self.station.de_normalize(outputs, statistics_pred)
        return self.criterion(outputs, batch_y)

    def vali(self, vali_data, vali_loader, criterion, epoch):
        total_loss = []
        self.model.eval()
        self.station.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x, statistics_pred, _ = self._normalize_input(batch_x, epoch)

                if epoch + 1 <= self.station_setting[1]:
                    outputs_target = batch_y[:, -self.args.pred_len:, self._select_f_dim():].to(self.device)
                    statistics_pred = self._maybe_trim_statistics_for_ms(statistics_pred)
                    loss = self.station_loss(outputs_target, statistics_pred)
                    total_loss.append(float(loss.item()))
                    continue

                dec_inp = self._prepare_decoder_input(batch_x, batch_y)
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, batch_y_t = self._trim_outputs_and_target(outputs, batch_y)
                outputs = self._denormalize(outputs, statistics_pred, batch_y_t)
                forecast_loss = criterion(outputs.detach().cpu(), batch_y_t.detach().cpu())
                total_loss.append(float(forecast_loss.item()))

        self.model.train()
        self.station.train()
        return float(np.average(total_loss)) if total_loss else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # The test split is not even instantiated during ordinary training.
        # It can be monitored only via an explicit legacy/debug flag and is
        # never used by early stopping or hyperparameter selection.
        test_data, test_loader = None, None
        if self.args.monitor_test_during_training:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        path_station_pre, path_station = self._station_paths(setting)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_station_model = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs + self.station_setting[1]):
            if self.station_setting[0] > 0 and epoch == self.station_setting[1]:
                best_station_path = os.path.join(path_station_pre, 'checkpoint.pth')
                if os.path.exists(best_station_path):
                    self.station.load_state_dict(torch.load(best_station_path, map_location=self.device))
                    print('loading pretrained station model')

            if self.station_setting[2] > 0 and self.station_setting[3] == epoch - self.station_setting[1]:
                joint_station_ckpt = os.path.join(path_station, 'checkpoint.pth')
                torch.save(self.station.state_dict(), joint_station_ckpt)
                print(f'[save] station(pretrained) before joint training: {joint_station_ckpt}')
                # Keep the normalizer on its declared station_lr.  It is
                # stepped by station_optim below instead of being silently
                # moved into the backbone optimizer at backbone LR.

            self.model.train()
            self.station.train()
            epoch_time = time.time()
            train_loss = []
            iter_count = 0

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.station_optim is not None:
                    self.station_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x, statistics_pred, _ = self._normalize_input(batch_x, epoch)

                if epoch + 1 <= self.station_setting[1]:
                    outputs_target = batch_y[:, -self.args.pred_len:, self._select_f_dim():].to(self.device)
                    statistics_pred = self._maybe_trim_statistics_for_ms(statistics_pred)
                    loss = self.station_loss(outputs_target, statistics_pred)
                else:
                    dec_inp = self._prepare_decoder_input(batch_x, batch_y)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs, batch_y_t = self._trim_outputs_and_target(outputs, batch_y)
                            loss = self._training_forecast_loss(outputs, batch_y_t, statistics_pred)
                    else:
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, batch_y_t = self._trim_outputs_and_target(outputs, batch_y)
                        loss = self._training_forecast_loss(outputs, batch_y_t, statistics_pred)

                train_loss.append(float(loss.item()))

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * (((self.args.train_epochs + self.station_setting[1] - epoch) * train_steps) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if epoch + 1 <= self.station_setting[1]:
                        if self.station_optim is None:
                            raise RuntimeError('pretraining requested for a parameter-free normalizer')
                        scaler.step(self.station_optim)
                    else:
                        scaler.step(model_optim)
                        joint_enabled = (
                            self.station_setting[2] > 0
                            and self.station_setting[3] <= epoch - self.station_setting[1]
                        )
                        if joint_enabled:
                            if self.station_optim is None:
                                raise RuntimeError('joint training requested for a parameter-free normalizer')
                            scaler.step(self.station_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if epoch + 1 <= self.station_setting[1]:
                        if self.station_optim is None:
                            raise RuntimeError('pretraining requested for a parameter-free normalizer')
                        self.station_optim.step()
                    else:
                        model_optim.step()
                        joint_enabled = (
                            self.station_setting[2] > 0
                            and self.station_setting[3] <= epoch - self.station_setting[1]
                        )
                        if joint_enabled:
                            if self.station_optim is None:
                                raise RuntimeError('joint training requested for a parameter-free normalizer')
                            self.station_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss_avg = float(np.average(train_loss)) if train_loss else 0.0
            vali_loss = self.vali(vali_data, vali_loader, self.criterion, epoch)
            test_loss = None
            if test_loader is not None:
                test_loss = self.vali(test_data, test_loader, self.criterion, epoch)

            # Optional per-epoch hook for interactive clients (notebooks).
            # Default implementation is a no-op; see _on_epoch_end below.
            self._on_epoch_end(
                epoch=epoch + 1,
                phase=('pretrain' if epoch + 1 <= self.station_setting[1] else 'joint'),
                train_loss=train_loss_avg,
                vali_loss=vali_loss,
                test_loss=test_loss,
            )

            if epoch + 1 <= self.station_setting[1]:
                message = f"Station Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}"
                if test_loss is not None:
                    message += f" Test Loss: {test_loss:.7f}"
                print(message)
                early_stopping_station_model(vali_loss, self.station, path_station_pre)
                if self.station_optim is not None:
                    adjust_learning_rate(self.station_optim, epoch + 1, self.args, self.args.station_lr)
            else:
                current_backbone_epoch = epoch + 1 - self.station_setting[1]
                message = f"Backbone Epoch: {current_backbone_epoch}, Steps: {train_steps} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}"
                if test_loss is not None:
                    message += f" Test Loss: {test_loss:.7f}"
                print(message)
                if self.station_setting[2] > 0 and self.station_setting[3] <= epoch - self.station_setting[1]:
                    early_stopping(vali_loss, self.model, path, self.station, path_station, epoch=current_backbone_epoch)
                else:
                    early_stopping(vali_loss, self.model, path, epoch=current_backbone_epoch)
                if early_stopping.early_stop:
                    print('Early stopping')
                    break
                adjust_learning_rate(model_optim, current_backbone_epoch, self.args, self.args.learning_rate)
                if self.station_optim is not None:
                    adjust_learning_rate(self.station_optim, current_backbone_epoch, self.args, self.args.station_lr)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        if self.station_setting[2] > 0:
            station_ckpt = os.path.join(path_station, 'checkpoint.pth')
            if os.path.exists(station_ckpt):
                self.station.load_state_dict(torch.load(station_ckpt, map_location=self.device))
        self.best_vali_loss = float(early_stopping.val_loss_min) if early_stopping.val_loss_min is not None else None
        self.best_epoch = int(early_stopping.best_epoch) if early_stopping.best_epoch is not None else -1
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location=self.device))
            _, path_station = self._station_paths(setting)
            station_ckpt = os.path.join(path_station, 'checkpoint.pth')
            if os.path.exists(station_ckpt):
                self.station.load_state_dict(torch.load(station_ckpt, map_location=self.device))

        # Metrics are accumulated per batch in float64.  Holding every
        # prediction (plus the unused inputs) needed ~20 GB of host RAM for
        # Traffic/H720, and np.array(..., dtype=object) silently produced an
        # object array whenever all batches had equal shape.
        stats = StreamingMetrics()
        save_plots = bool(getattr(self.args, 'save_test_plots', False))
        folder_path = os.path.join('./test_results', setting)
        if save_plots:
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        self.station.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                input_x = batch_x
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x, statistics_pred, _ = self._normalize_input(batch_x, epoch_idx=None)
                dec_inp = self._prepare_decoder_input(batch_x, batch_y)
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, batch_y_t = self._trim_outputs_and_target(outputs, batch_y)
                outputs = self._denormalize(outputs, statistics_pred, batch_y_t)

                pred = outputs.detach().cpu().numpy()
                true = batch_y_t.detach().cpu().numpy()
                stats.update(pred, true)
                if save_plots and i % 20 == 0:
                    x = input_x.detach().cpu().numpy()
                    gt = np.concatenate((x[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((x[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))

        mse, mae, rse = stats.compute()
        print(f'mse:{mse}, mae:{mae}')

        with open('result.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write(f'mse:{mse}, mae:{mae}, rse:{rse}')
            f.write('\n\n')

        # Append compact CSV result summary. Sink is controlled by --result_file
        # (default 'result.csv' in CWD; can point to e.g. './results/NoNorm.csv'
        # to separate baselines across scripts).
        self._append_result_row(self._result_row(setting, mse=mse, mae=mae, split='test'))
        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        return
