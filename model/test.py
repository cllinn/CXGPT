# import torch
# import torch.nn as nn 
# import torch.nn.functional as F 
# import sys 
# sys.path.append("/data/lcl")
# from mulcon.model.Xformer import BertLMHeadModel, BertConfig
# from transformers import BertTokenizer, CvtModel, CvtConfig
# from mulcon.model.xlip import Blip2Base
# from mulcon.model.xlip_qformer import Blip2Qformer
# from mulcon.config.xlip_config import XlipConfig
# from mulcon.model.xlip_model import XlipModel 
# from mulcon.model.xlip_lm import XlipLM
# import json 


if __name__ == "__main__":
    with open("untitled.txt", "w") as f:
        for i in range(100):
            f.write(str(i) + "\n")
    hyp = open("untitled.txt", "r")
    hyp_lines = [line[:-1] for line in hyp if line[:-1]]
    print(hyp_lines)
    # with open("../dataset/MIMIC-CXR/data_processed/data.json", "r") as f:
    #     fd = json.load(f)
    # # ls = list(fd)
    # # for i in ls[:2]:
    # #     print(i)
    # fd = {k: v for k, v in fd.items() if v.get("IMAGE", None) is not None}
    # for i, id_d in fd.items():
    #     for k, v in id_d.items():
    #         if k == "IMAGE":
    #             new_path = v.replace("/data/user/MYF", "/data/lcl/mulcon/dataset")
    #             fd[i][k] = new_path
    # with open("../dataset/MIMIC-CXR/data_processed/new_data.json", "w") as f:
    #     json.dump(fd, f, indent=4)
    # cvt_config = CvtConfig.from_json_file("../checkpoint/xlip-base/cvt_config.json")
    # encoder_config = BertConfig.from_json_file("../checkpoint/xlip-base/encoder_config.json")
    # config = XlipConfig.from_json_file("../checkpoint/xlip-base/config.json")
    # model = XlipLM(cvt_config, encoder_config, config)
    # checkpoint = torch.load("../checkpoint/xlip_lm/checkpoint/checkpoint.pt-2")
    # model.load_state_dict(checkpoint["model"])
    # model = model.to(torch.device("cuda", 0))
    # print(model.visual_encoder.device)
    # print(model)
    # encoder_config.encoder_width = cvt_config.embed_dim[2]
    # # encoder_config.is_decoder = True 
    # # insert cross-attention layer every other block
    # encoder_config.add_cross_attention = True
    # encoder_config.cross_attention_freq = 2
    # encoder_config.query_length = 25
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
    # # print(cvt_config)
    # # print(encoder_config)
    # xlip_config = XlipConfig()
    # # print(xlip_config)
    # xlip_config.to_json_file("../checkpoint/xlip-base/config.json")
    # encoder_config.to_json_file("../checkpoint/xlip-base/encoder_config.json")
    # model = Blip2Qformer(cvt_config)
    # print(model)
    # print(model.config)
    # image = torch.randn(2, 3, 224, 224)
    # text = ["a cat wearing sunglasses", "a dog wearing sunglasses"]
    # sample = {
    #     "image": image.cuda(),
    #     "text_input": text 
    # }
    # output = model.generate(sample)
    # print(output)

