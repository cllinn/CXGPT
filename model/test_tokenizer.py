import torch
import torch.nn as nn 
import torch.nn.functional as F 
import sys 
sys.path.append("/root")
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from mulcon.model.xlip import (
    Blip2Base,
    compute_sim_matrix
)
from mulcon.model.xlip_outputs import BlipOutput, BlipOutputFeatures
from mulcon.model.xlip_qformer import Blip2Qformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, AutoImageProcessor, BertLMHeadModel
from mulcon.model.xlip_model import XlipModel 
from mulcon.model.xlip_qformer import disabled_train
import logging
from PIL import Image
import numpy as np 

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/checkpoint/bert-base-uncased")
    model = BertLMHeadModel.from_pretrained("/root/autodl-tmp/checkpoint/bert-base-uncased")
    model.config.is_decoder = True
    print(model.config.is_decoder)
    inputs = tokenizer("hello ", return_tensors="pt")
    output = model.generate(**inputs, use_cache=False)
    output = tokenizer.batch_decode(output)
    print(output)
    # print(image["pixel_values"].shape)
    # print(image_2["pixel_values"] == image["pixel_values"])
    # print(llm_tokenizer.bos_token)
    # # sentence = ["hello world i am a test sentence", "hello world hello world"]
    # sentence = ["[IMAGE]"]
    # llm_tokenizer.add_special_tokens({'bos_token': "[IMAGE]"})
    # print(llm_tokenizer.bos_token)
    # a = llm_tokenizer(
    #     sentence,
    #     # [llm_tokenizer.bos_token + s for s in sentence],
    #     return_tensors="pt",
    #     padding="longest",
    #     truncation=True,
    #     max_length=20,
    #     add_special_tokens=False
    # )
    # print(a)
    # print(llm_tokenizer.convert_ids_to_tokens(a.input_ids[0]))