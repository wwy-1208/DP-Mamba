# run.py
import argparse
import os
import torch
from utils.tools import count_parameters
from exp.exp_main import Exp_Main
import random
import numpy as np

torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A DUAL-PATH MAMBA WITH FIXED AND VARIABLE PATCHES FOR TIME SERIES FORECASTING')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='DP-Mamba', help='model name, should be DP-Mamba')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s, t, h, d, b, w, m]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=168,
                        help='start token length (for decoder, not used in this model but kept for compatibility)')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--fft_top_k', type=int, default=3,
                        help='Number of top frequencies to consider in the FFT expert')
    # --- Time Branch: Mamba & Patching Parameters ---
    parser.add_argument('--patch_len', type=int, default=16, help='Patch length for Time Branch')
    parser.add_argument('--stride', type=int, default=8, help='Stride for Time Branch')
    #parser.add_argument('--patch_len_ls', type=str, default='16,16', help='list of patch lengths for time branch')
    #parser.add_argument('--stride_ls', type=str, default='8,8', help='list of strides for time branch')
    parser.add_argument('--d_state', type=int, default=16, help='SSM state expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='local convolution width for Mamba')
    #parser.add_argument('--wavelet', type=str, default='db3', help='wavelet type for wavelet transform')
    #parser.add_argument('--level', type=int, default=1, help='level for multi-level wavelet decomposition')
    parser.add_argument('--ablation_mode', type=str, default='moe',
                        # --- 修改这里的choices ---
                        choices=['moe', 's_mamba_only', 'fft_only', 'sum_experts'],
                        help="MoELayerBranch ablation mode: 'moe' (mixture of experts), "
                             "'s_mamba_only', 'fft_only', or 'sum_experts'")
    # --- General Model Parameters ---
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=1, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--c_in', type=int, default=7, help='Input channel number (n_vars)')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='num of heads (used in old model, kept for compatibility)')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (for Bi-Mamba)')
    parser.add_argument('--d_ff', type=int, default=128,
                        help='dimension of fcn (not used in Mamba, kept for compatibility)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers (compatibility)')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average (compatibility)')
    parser.add_argument('--factor', type=int, default=1, help='attn factor (compatibility)')
    parser.add_argument('--distil', action='store_false', help='(compatibility)', default=True)
    parser.add_argument('--embed', type=str, default='timeF', help='(compatibility)')
    parser.add_argument('--output_attention', action='store_true', help='(compatibility)')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        # NEW setting record for testing
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_el{}_plen{}_str{}_imgS{}_imgP{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.patch_len,
            args.stride,
            args.image_size,
            args.image_periodicity,
            args.des, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()