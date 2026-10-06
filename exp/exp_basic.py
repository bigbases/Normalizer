import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # A resource-aware parent scheduler may already have restricted
            # this process to one physical GPU. Preserve that binding and use
            # logical cuda:0 inside the child.
            scheduler_bound = "CUDA_VISIBLE_DEVICES" in os.environ
            if not scheduler_bound:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            logical_gpu = 0 if scheduler_bound else self.args.gpu
            device = torch.device('cuda:{}'.format(logical_gpu))
            print('Use GPU: cuda:{} (CUDA_VISIBLE_DEVICES={})'.format(
                logical_gpu, os.environ.get("CUDA_VISIBLE_DEVICES", "")
            ))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
