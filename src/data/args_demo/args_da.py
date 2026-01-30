import argparse
import os
import random


def check_dirs(dirs):
    dirs = [dirs] if type(dirs) not in [list, tuple] else dirs
    for d in dirs:
        try:
            os.makedirs(d)
        except OSError:
            pass
    return


def get_args_da():
    parser = argparse.ArgumentParser()

    # Mandotory
    parser.add_argument("config_file")

    ## Common Parameters ##
    parser.add_argument('-T', '--task', default='clf', help='clf | seg')  # Classification or Segmentation
    # parser.add_argument('-D','--datasets', type=str, nargs='+', required=True, help='clf: M/MM/U (MNIST/MNIST-M/USPS) '
    #                                                                            'seg: G/C (GTA5/Cityscapes)')
    parser.add_argument("--dataset_source", required=True, type=str)
    parser.add_argument("--dataset_target", required=True, type=str)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--imsize', type=int, help='the height of the input image')
    parser.add_argument('--iter', type=int, default=10000000, help='total training iterations')
    parser.add_argument('--manualSeed', type=int, default=None)
    parser.add_argument('--ex', help='Experiment name', default="not_set")
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--tensor_freq', type=int, help='frequency of showing results on tensorboard during training.')
    parser.add_argument('--eval_freq', type=int, help='frequency of evaluation during training.')
    parser.add_argument('--CADT', type=bool, default=False)
    parser.add_argument('--load_step', type=int, help="iteration of trained networks")

    parser.add_argument('--debug_pc', default=1, type=int)  # set to 1 to activate wandb log

    ## Optimizers Parameters ##
    parser.add_argument('--lr_dra', type=float, default=0.0002)
    parser.add_argument('--lr_clf', type=float, default=4e-4)
    parser.add_argument('--lr_seg', type=float, default=6e-4)
    parser.add_argument('--lambda_g', type=float, default=0.12)
    parser.add_argument('--lambda_content', type=float, default=1.0)
    parser.add_argument('--lambda_style', type=float, default=0.5)
    parser.add_argument('--lambda_consis', type=float, default=2.0)
    parser.add_argument('--lambda_recon', type=float, default=0.2)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_step', type=int, default=20000)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--weight_decay_dra', type=float, default=1e-5)
    parser.add_argument('--weight_decay_task', type=float, default=5e-4)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument("--project_wandb", default="DRANet_READ", type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--path_log", default='checkpoint/', type=str)

    # CRNN
    parser.add_argument("--path_model", default="", type=str)
    parser.add_argument("--path_optimizer", default="", help="", type=str)
    parser.add_argument('--height_max', default=160, type=int)
    parser.add_argument('--width_max', default=1570, type=int)
    parser.add_argument('--pad_left', default=64, type=int)
    parser.add_argument('--pad_right', default=64, type=int)

    args = parser.parse_args()
    check_dirs([args.path_log + args.task + '/' + args.ex])
    args.logfile = args.path_log  + args.task + '/' + args.ex + '/' + args.ex + '.log'
    if args.task == 'seg':
        args.CADT = True
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.batch is None:
        if args.task == 'clf':
            args.batch = 8
        elif args.task == 'seg':
            args.batch = 2
    if args.imsize is None:
        if args.task == 'clf':
            # args.imsize = (160, 1698)
            args.imsize = (args.height_max, args.width_max)
        elif args.task == 'seg':
            args.imsize = 512
    if args.tensor_freq is None:
        if args.task == 'clf':
            args.tensor_freq = 1000
        elif args.task == 'seg':
            args.tensor_freq = 100
    if args.eval_freq is None:
        if args.task == 'clf':
            args.eval_freq = 5000
        elif args.task == 'seg':
            args.eval_freq = 1000

    # To refactor
    args.datasets = [args.dataset_source, args.dataset_target]

    return args
