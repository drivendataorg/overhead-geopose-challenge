import random

import ever as er
import numpy as np
import torch
from eval_fn import geopose_evaluate_fn

er.registry.register_all()


def register_geopose_evaluate_fn(launcher):
    launcher.override_evaluate(geopose_evaluate_fn)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SEED = 2333

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    trainer = er.trainer.get_trainer('th_amp_ddp')()

    trainer.run(after_construct_launcher_callbacks=[register_geopose_evaluate_fn])
