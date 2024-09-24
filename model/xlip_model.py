import torch
import torch.nn as nn 
import torch.nn.functional as F 
from mulcon.model.xlip import (
    Blip2Base,
    compute_sim_matrix
)
from mulcon.model.xlip_outputs import BlipOutput, BlipOutputFeatures
from mulcon.model.xlip_qformer import Blip2Qformer


class XlipModel(Blip2Qformer):
    def __init__(self, cvt_config, encoder_config, config):
        self.config = config 
        super().__init__(
            cvt_config,
            encoder_config,
            config.freeze_cvt,
            config.num_query_token,
            config.cross_attention_freq,
            config.embed_dim,
            config.max_txt_len
        )
    
    def forward(self, samples):
        return super().forward(samples)

