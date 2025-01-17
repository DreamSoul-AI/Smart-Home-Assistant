from torch.nn import BCELoss

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        if self.args.resume:
            resume_path = os.path.join(self.args.resume, 'checkpoint.pth')
            if os.path.exists(resume_path):
                print('=======Resume model=======')
                self.model.load_state_dict(torch.load(resume_path))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _select_criterion2(self):
        criterion = BCELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion_list):
        criterion = criterion_list[0]
        criterion2 = criterion_list[1]
        total_loss = []
        self.model.eval()
        true_count = 0
        total_count = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, data_info) in enumerate(vali_loader):
                device_name, task_name, zero_trans = (data_info[k][0] for k in ('device_name', 'task_name', 'zero_transformed_value'))
                info = (device_name, task_name)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, cls_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, task_name, zero_trans)
                else:
                    outputs, cls_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, task_name, zero_trans)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                cls_outputs = cls_outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_cls = torch.where(torch.eq(batch_y, zero_trans), torch.zeros_like(batch_y),
                                          torch.ones_like(batch_y))
                batch_y_cls_mask = batch_y_cls.detach().cpu().numpy()
                cls_mask = cls_outputs.detach().cpu().numpy()
                cls_mask = np.where(cls_mask >= 0.5, 1.0, 0.0)
                # equal_elements = np.sum(batch_y_cls_mask == cls_mask)
                # total_elements = batch_y_cls_mask.size
                equal_elements = np.sum(batch_y_cls_mask[:, 0, :] == cls_mask[:, 0, :])
                total_elements = batch_y_cls_mask[:, 0, :].size
                true_count += equal_elements
                total_count += total_elements

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                pred_cls = cls_outputs.detach().cpu()
                true_cls = batch_y_cls.detach().cpu()

                loss = criterion(pred, true) + criterion2(pred_cls, true_cls)
                # loss = criterion(pred[:, 0, :], true[:, 0, :])
                total_loss.append(loss)
        accuracy = true_count / total_count * 100
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, accuracy

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
        criterion2 = self._select_criterion2()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            true_count = 0
            total_count = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, data_info) in enumerate(train_loader):
                device_name, task_name, zero_trans = (data_info[k][0] for k in ('device_name', 'task_name', 'zero_transformed_value'))
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, cls_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, device_name, task_name)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        cls_outputs = cls_outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        batch_y_cls = torch.where(torch.eq(batch_y, zero_trans), torch.zeros_like(batch_y),
                                                  torch.ones_like(batch_y))
                        loss = criterion(outputs, batch_y) + criterion2(cls_outputs, batch_y_cls)
                        # loss = criterion(outputs[:, 0, :], batch_y[:, 0, :]) + criterion2(cls_outputs[:, 0, :], batch_y_cls[:, 0, :])
                        train_loss.append(loss.item())
                else:
                    outputs, cls_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, device_name, task_name)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    cls_outputs = cls_outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y_cls = torch.where(torch.eq(batch_y, zero_trans), torch.zeros_like(batch_y),
                                              torch.ones_like(batch_y))
                    # print(batch_x.requires_grad, batch_y.requires_grad, batch_y_cls.requires_grad, outputs.requires_grad, cls_outputs.requires_grad)
                    # False False False True True

                    batch_y_cls_mask = batch_y_cls.detach().cpu().numpy()
                    cls_mask = cls_outputs.detach().cpu().numpy()
                    cls_mask = np.where(cls_mask >= 0.5, 1.0, 0.0)
                    equal_elements = np.sum(batch_y_cls_mask[:, 0, :] == cls_mask[:, 0, :])
                    total_elements = batch_y_cls_mask[:, 0, :].size
                    true_count += equal_elements
                    total_count += total_elements

                    # print(f"Accuracy: {accuracy * 100:.2f}%")

                    loss = criterion(outputs, batch_y) + criterion2(cls_outputs, batch_y_cls)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    print(f"\t\tAcc: {true_count / total_count * 100:.2f}%")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_acc = self.vali(vali_data, vali_loader, [criterion, criterion2])
            test_loss, test_acc = self.vali(test_data, test_loader, [criterion, criterion2])

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\n"
                  "Vali Cls Acc: {5:.7f} Test Cls Acc: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, vali_acc, test_acc))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        true_count = 0
        total_count = 0

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, data_info) in enumerate(test_loader):
                device_name, task_name, zero_trans = (data_info[k][0] for k in ('device_name', 'task_name', 'zero_transformed_value'))
                info = (device_name, task_name)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # print(torch.min(batch_y), torch.max(batch_y))
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, cls_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, device_name, task_name)
                else:
                    outputs, cls_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, device_name, task_name)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                cls_outputs = cls_outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                cls_outputs = cls_outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                cls_mask = np.where(cls_outputs >= 0.5, 1.0, 0.0)

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = np.where(batch_y < 0.0, 0.0, batch_y)
                    # print(np.min(batch_y), np.max(batch_y))
                    # exit()
                # print(np.max(batch_y), np.min(batch_y))
                batch_y_cls_mask = np.where(batch_y > 1.0e-07, 1.0, 0.0)
                equal_elements = np.sum(batch_y_cls_mask[:, 0, :] == cls_mask[:, 0, :])
                total_elements = batch_y_cls_mask[:, 0, :].size
                # print(equal_elements, total_elements)
                true_count += equal_elements
                total_count += total_elements
                # print(true_count, total_count)

                outputs = np.where(cls_mask == 0.0, 0.0, outputs)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # print(true[0, :, -1], input[0, :, -1])
                    # print(pred[0, :, -1], input[0, :, -1])
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        print(true_count, total_count)
        accuracy = true_count / total_count * 100
        print('Test Acc:{}'.format(accuracy))

        # compare
        compare_pit_path = os.path.join(folder_path, 'pic')
        if not os.path.exists(compare_pit_path):
            os.makedirs(compare_pit_path)

        compare_len = 100
        compare_idx = np.arange(0, len(preds), 100)[:8]
        for idx in compare_idx:
            preds_i = preds[idx:idx + compare_len, 0, :]
            trues_i = trues[idx:idx + compare_len, 0, :]
            result_compare(trues_i, preds_i, os.path.join(compare_pit_path, '{}.png'.format(idx)))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

def result_compare(data1, data2, save_path):
    # Flatten the data to make it easier to plot
    data1_flattened = data1.flatten()
    data2_flattened = data2.flatten()

    x_values = np.arange(len(data1))  # Generate x values assuming index from 0 to len-1

    fig, ax1 = plt.subplots()  # Create a new figure and an axis
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.plot(x_values, data1_flattened, color='tab:blue', label='Trues')
    ax1.plot(x_values, data2_flattened, color='tab:red', label='Preds')

    fig.tight_layout()  # Ensure that the labels do not overlap
    plt.title('Figure')
    plt.legend()
    plt.savefig(save_path)
    return
