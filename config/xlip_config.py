from transformers import PretrainedConfig 

class XlipConfig(PretrainedConfig):
    def __init__(
        self,
        freeze_cvt=False,
        num_query_token=25, 
        cross_attention_freq=2, 
        embed_dim=256, 
        max_txt_len=256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.freeze_cvt = freeze_cvt
        self.num_query_token = num_query_token
        self.cross_attention_freq = cross_attention_freq
        self.embed_dim = embed_dim
        self.max_txt_len = max_txt_len