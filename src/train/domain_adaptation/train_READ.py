from __future__ import print_function
import torch
import wandb
import sys
import os

# Get the path to the project root (3 levels up from this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.args_demo.args_da import get_args_da as get_args
from src.train.domain_adaptation.trainer_READ import Trainer


if __name__ == '__main__':
    opt = get_args()

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    opt.device = device

    if opt.debug_pc == 0:
        # Initialize wandb
        if opt.wandb_name is None:
            wandb.init(project=opt.project_wandb, config=vars(opt))
        else:
            wandb.init(project=opt.project_wandb, name=opt.wandb_name, config=vars(opt))

        print("run name wand : " + str(wandb.run.name))

    trainer = Trainer(opt)
    trainer.train()

    if opt.debug_pc == 0:
        wandb.finish()
