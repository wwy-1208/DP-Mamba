from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DPMamba
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from models.DPMamba import DPMambaModel

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args.device = self.device  # 将设备信息传递到config

    def _build_model(self):
        model_dict = {
            'DP-Mamba': DPMambaModel,
        }
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        data_x = torch.from_numpy(data_set.data_x).float()
        data_y = torch.from_numpy(data_set.data_y).float()

        def calculate_stats(tensor):
            # 过滤 NaN 值
            tensor_filtered = tensor[~torch.isnan(tensor)]
            mean = torch.mean(tensor_filtered).item() if tensor_filtered.numel() > 0 else 0.0
            std = torch.std(tensor_filtered).item() if tensor_filtered.numel() > 0 else 0.0
            return mean, std

        mean_x, std_x = calculate_stats(data_x)
        assert not torch.isnan(data_x).any(), f"{flag} 输入数据包含 NaN!"
        assert not torch.isinf(data_x).any(), f"{flag} 输入数据包含 Inf!"
        assert not torch.isnan(data_y).any(), f"{flag} 标签数据包含 NaN!"
        assert not torch.isinf(data_y).any(), f"{flag} 标签数据包含 Inf!"

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if batch_y.shape[1] != self.args.pred_len:
                    batch_y_target = batch_y[:, -self.args.pred_len:, :].to(self.device)
                else:
                    batch_y_target = batch_y.to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                        loss = criterion(outputs, batch_y_target)
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark)
                    loss = criterion(outputs, batch_y_target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

                train_loss.append(loss.item())
                if scheduler is not None:
                    scheduler.step()

            train_loss_avg = np.mean(train_loss) if train_loss else float('nan')
            print("轮次: {} 耗时: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("早停")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if batch_y.shape[1] != self.args.pred_len:
                    batch_y_target = batch_y[:, -self.args.pred_len:, :].to(self.device)
                else:
                    batch_y_target = batch_y.to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    total_loss.append(float('nan'))
                    continue

                if outputs.shape != batch_y_target.shape:
                    total_loss.append(float('nan'))
                    continue

                loss = criterion(outputs.detach(), batch_y_target.detach())

                if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                    total_loss.append(loss.item())

        valid_losses = [l for l in total_loss if not np.isnan(l)]
        avg_loss = np.average(valid_losses) if valid_losses else float('nan')

        self.model.train()
        return avg_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('加载模型')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if not os.path.exists(model_path):
                print(f"错误: 在 {model_path} 未找到模型检查点")
                return
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if batch_y.shape[1] != self.args.pred_len:
                    batch_y_target = batch_y[:, -self.args.pred_len:, :].numpy()
                else:
                    batch_y_target = batch_y.numpy()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"❌ 在测试输出中检测到 NaN/Inf，位于迭代 {i}。跳过此批次的指标计算。")
                    continue

                pred = outputs.detach().cpu().numpy()
                true = batch_y_target

                if pred.shape != true.shape:
                    print(f"[测试] 形状不匹配！预测: {pred.shape}, 真实: {true.shape} at I{i}")
                    continue

                preds.append(pred)
                trues.append(true)

                if i == 0:
                    input_seq = batch_x[0, :, 0].detach().cpu().numpy()
                    true_future = true[0, :, 0]
                    pred_future = pred[0, :, 0]

                    plt.figure(figsize=(6, 5))
                    full_true_series = np.concatenate((input_seq, true_future))
                    plt.plot(np.arange(len(full_true_series)), full_true_series, label='GroundTruth', color='C0')
                    full_pred_series = np.concatenate((input_seq, pred_future))
                    plt.plot(np.arange(len(full_pred_series)), full_pred_series, label='Prediction', color='C1')
                    plt.title('ETTh1', fontsize=16)
                    plt.ylabel('Input-672-predict-96', fontsize=16)
                    plt.legend(fontsize=14)
                    plt.tick_params(axis='both', which='major', labelsize=14)
                    plt.grid(True, linestyle=':', alpha=0.6)
                    fig_save_path = os.path.join(folder_path, f'prediction_style_2_batch_{i}.png')
                    plt.savefig(fig_save_path, dpi=300)
                    print(f"新风格的预测可视化图已保存至: {fig_save_path}")
                    plt.close()

        if not preds:
            print("测试期间未收集到有效预测。无法计算指标。")
            return

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        results_folder_path = './results/' + setting + '/'
        if not os.path.exists(results_folder_path):
            os.makedirs(results_folder_path)

        np.save(results_folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse]))
        np.save(results_folder_path + 'pred.npy', preds)
        np.save(results_folder_path + 'true.npy', trues)

        return

    def _select_criterion(self):
        class SafeMSELoss(nn.MSELoss):
            def forward(self, input, target):
                input_c = torch.clamp(input, min=-1e4, max=1e4)
                target_c = torch.clamp(target, min=-1e4, max=1e4)
                return super().forward(input_c, target_c)

        return SafeMSELoss()

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # long short input
                long, short, long_mark, short_mark = batch_x, batch_y[:, :self.args.label_len, :], batch_x_mark, batch_y_mark[:, :self.args.label_len, :]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'SST':
                            outputs = self.model(long)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'SST':
                        outputs = self.model(long)
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_pred.npy', preds)
        np.save(folder_path + 'real_true.npy', trues)

        return
