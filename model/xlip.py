import torch
import torch.nn as nn 
import torch.nn.functional as F 
import contextlib 
import sys 
sys.path.append("/data/lcl")
from mulcon.model.Xformer import BertLMHeadModel, BertConfig
from transformers import BertLMHeadModel as BERTLMHEADMODEL
from transformers import BertTokenizer, CvtModel, CvtConfig 
import timm

class Blip2Base(nn.Module):

    def __init__(self):
        super().__init__()
    
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/checkpoint/bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, encoder_config, cross_attention_freq=2):
        # encoder_config = BertConfig.from_pretrained("../checkpoint/bert-base-uncased")
        # encoder_config.encoder_width = vision_width
        # # encoder_config.is_decoder = True 
        # # insert cross-attention layer every other block
        # encoder_config.add_cross_attention = True
        # encoder_config.cross_attention_freq = cross_attention_freq
        # encoder_config.query_length = num_query_token
        # encoder_config.num_heads = 3
        # encoder_config.embed_dim = 768
        # encoder_config.kernel_size = 3
        # encoder_config.padding_q = 1
        # encoder_config.padding_kv = 1
        # encoder_config.stride_q = 1
        # encoder_config.stride_kv = 2
        # encoder_config.qkv_projection_method = "dw_bn"
        # encoder_config.qkv_bias = True 
        # encoder_config.attention_drop_rate = 0.0
        # encoder_config.drop_rate = 0.0
        # print(encoder_config.is_decoder)
        Qformer = BertLMHeadModel(config=encoder_config)
        Qformer.load_state_dict(BertLMHeadModel.from_pretrained("/root/autodl-tmp/checkpoint/bert-base-uncased").cpu().state_dict(), strict=False)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.embed_dim)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

#     def init_vision_encoder(
#         self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
#     ):
#         assert model_name in [
#             "eva_clip_g",
#             "eva2_clip_L",
#             "clip_L",
#         ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
#         if model_name == "eva_clip_g":
#             visual_encoder = create_eva_vit_g(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
# #         elif model_name == "eva2_clip_L":
# #             visual_encoder = create_eva2_vit_L(
# #                 img_size, drop_path_rate, use_grad_checkpoint, precision
# #             )
#         elif model_name == "clip_L":
#             visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
#         ln_vision = LayerNorm(visual_encoder.num_features)
#         self.vit_name = model_name
#         return visual_encoder, ln_vision

    def init_vision_encoder(self, cvt_config):
        pre_config = CvtConfig.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
        # cvt_config.embed_dim[2] = 768
        # vision_encoder = CvtModel(cvt_config)
        vision_encoder = CvtModel.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
        cvt_proj = nn.Linear(pre_config.embed_dim[2], cvt_config.embed_dim[2])
        self.cvt_proj = cvt_proj
        ln_vision = LayerNorm(cvt_config.embed_dim[2])
        return vision_encoder, ln_vision
    
    def init_efficientnet(self):
        model = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=True)
        efficient_net_proj = nn.Linear(1280, 768)
        return model, transforms, efficient_net_proj


    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


    # def get_optimizer_params(self, weight_decay, lr_scale=1):
    #     if self.vit_name == "eva_clip_g":
    #         vit_num_layers = self.visual_encoder.get_num_layer()
    #         lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

    #         parameter_group_names = {}
    #         parameter_group_vars = {}

    #         for name, param in self.named_parameters():
    #             if not param.requires_grad:
    #                 continue  # frozen weights
    #             if len(param.shape) == 1 or name.endswith(".bias"):
    #                 group_name = "no_decay"
    #                 this_weight_decay = 0.
    #             else:
    #                 group_name = "decay"
    #                 this_weight_decay = weight_decay
    #             if 'visual_encoder' in name:
    #                 layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
    #                 group_name = "vit_layer_%d_%s" % (layer_id, group_name)
    #             else:
    #                 layer_id = None

    #             if group_name not in parameter_group_names:
    #                 if layer_id is not None:
    #                     scale = lr_scales[layer_id]
    #                 else:
    #                     scale = 1
    #                 parameter_group_names[group_name] = {
    #                     "weight_decay": this_weight_decay,
    #                     "params": [],
    #                     "lr_scale": scale
    #                 }
    #                 parameter_group_vars[group_name] = {
    #                     "weight_decay": this_weight_decay,
    #                     "params": [],
    #                     "lr_scale": scale
    #                 }
    #             parameter_group_vars[group_name]["params"].append(param)
    #             parameter_group_names[group_name]["params"].append(name)
    #         # import json
    #         # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    #         optim_params = list(parameter_group_vars.values())
    #         return optim_params
    #     else:
    #         return super().get_optimizer_params(weight_decay,lr_scale)
    
    def get_optimizer_params(self, weight_decay, lr, lr_scale=1):
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": lr_scale,
                    "lr": lr
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": lr_scale,
                    "lr": lr
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        import json
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params


    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()