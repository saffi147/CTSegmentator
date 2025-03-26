from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_and_topK_and_CE

class nnUNetTrainer_DC_CE_topk_loss(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_and_topK_and_CE()
        return loss
    
 