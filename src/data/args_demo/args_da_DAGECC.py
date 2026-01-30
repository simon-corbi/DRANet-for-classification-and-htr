import argparse
import os
import random


def get_args_da():
    parser = argparse.ArgumentParser()

    # Mandotory
    parser.add_argument("config_file")

    ## Common Parameters ##
    parser.add_argument('-T', '--task', required=True, help='clf | seg')  # Classification or Segmentation
    parser.add_argument('-D','--datasets', type=str, nargs='+', required=True, help='clf: M/MM/U (MNIST/MNIST-M/USPS) '
                                                                                'seg: G/C (GTA5/Cityscapes)')

    parser.add_argument('--ex', help='Experiment name', default="not_set")
    parser.add_argument("--dataset_source", required=True, type=str)
    parser.add_argument("--dataset_target", required=True, type=str)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch', type=int, default=8)

    parser.add_argument('--iter', type=int, default=10000000, help='total training iterations')
    parser.add_argument('--manualSeed', type=int, default=None)

    parser.add_argument('--eval_freq', type=int, help='frequency of evaluation during training.',
                        default=5000)

    parser.add_argument('--load_step', type=int, help="iteration of trained networks")

    parser.add_argument('--debug_pc', default=0, type=int)
    parser.add_argument("--path_model_expert", type=str, default="")

    ## Optimizers Parameters ##
    parser.add_argument('--CADT', type=bool, default=False)
    parser.add_argument('--imsize', type=int, default=48)
    parser.add_argument('--lr_dra', type=float, default=0.0001)
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
    parser.add_argument("--path_log", default='checkpoint/', type=str)
    parser.add_argument("--logfile", default='checkpoint/', type=str)

    parser.add_argument("--project_wandb", default="DRANet_READ", type=str)
    parser.add_argument("--wandb_name", default=None, type=str)

    # # CRNN
    parser.add_argument("--path_model", default="", type=str)
    parser.add_argument('--height_max', default=48, type=int)
    parser.add_argument('--width_max', default=48, type=int)

    args = parser.parse_args()

    os.makedirs(args.path_log , exist_ok=True)

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    return args