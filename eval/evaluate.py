import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import sys
sys.path.append("/root")
from torch.utils.data import DataLoader
from mulcon.dataloader.openi_loader import OPENI, OPENI_it2t, OPENI_t2t
from mulcon.model.xlip_model import XlipModel
from mulcon.model.xlip_lm import XlipLM
from mulcon.config.xlip_config import XlipConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from mulcon.model.Xformer import BertLMHeadModel, BertConfig
from transformers import BertTokenizer, CvtModel, CvtConfig
import argparse
from mulcon.trainer.trainer import XlipTrainer
from mulcon.trainer.LMTrainer import LMTrainer
# from mulcon.trainer.multiserial_trainer import XlipMultiGPUSerialTrainer
from mulcon.trainer.xlip_lmtrainer import XlipLMTrainer
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler


if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--dataset_path', type=str, default="../dataloader/OPENI_it2t_test.pt", help="Path to dataset")
#     parser.add_argument('--config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/config.json", help="Path to config")
#     parser.add_argument('--cvt_config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/cvt_config.json", help="Path to CVT config")
#     parser.add_argument('--encoder_config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/encoder_config.json", help="Path to encoder config")
#     parser.add_argument('--device', type=str, default="cuda:0", help="Device for training")
#     parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
#     parser.add_argument('--epochs', type=int, default=8, help="Number of epochs")
#     parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay value")
#     parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
#     parser.add_argument('--output_dir', type=str, default="../checkpoint/xlip_lm", help="Output directory")

#     args = parser.parse_args()

#     dataset_path = args.dataset_path
#     config_path = args.config_path
#     cvt_config_path = args.cvt_config_path
#     encoder_config_path = args.encoder_config_path
#     device = torch.device(args.device)
#     batch_size = args.batch_size
#     epochs = args.epochs
#     weight_decay = args.weight_decay
#     lr = args.lr
#     output_dir = args.output_dir

# #     cvt_config = CvtConfig.from_json_file(cvt_config_path)
# #     encoder_config = BertConfig.from_json_file(encoder_config_path)
# #     config = XlipConfig.from_json_file(config_path)
# #     model = XlipLM(cvt_config, encoder_config, config)
# #     # model = XlipModel(cvt_config, encoder_config, config)
#     tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/checkpoint/biogpt-large-pubmedqa", use_fast=False, truncation_side="left")
#     model = AutoModelForCausalLM.from_pretrained(
#         "/root/autodl-tmp/checkpoint/biogpt-large-pubmedqa", torch_dtype=torch.float32
#     )
#     dataset = torch.load(dataset_path)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     # params = model.get_optimizer_params(weight_decay, lr)
#     # optimizer = AdamW(params)
#     optimizer = None 
# #     # lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
#     lr_scheduler = None

# #     trainer = XlipLMTrainer(epochs, optimizer, model, dataloader, dataloader, lr_scheduler, output_dir)
#     trainer = LMTrainer(epochs, 
#         optimizer,
#         model,
#         tokenizer,
#         dataloader,
#         lr_scheduler,
#         output_dir)
#     checkpoint_path = "/root/autodl-fs/biogptl-ft/checkpoint.pt-7"
# #     checkpoint_path = "/root/autodl-fs/xlip-lm/checkpoint/checkpoint.pt-1"
#     # hyp_path = "xlip_openi_hyp.txt"
#     hyp_path = "biogpt_ihyp.txt"
#     ref_path = "biogpt_iref.txt"
# #     # orig_path = "orig_generate.txt"
# #     # hyp = open(hyp_path, "w")
# #     # with open(orig_path, "r") as f:
# #     #     for line in f:
# #     #         p = line.split("Impression:")
# #     #         hyp.write(p[1])
# #     # hyp.close()'
#     # ref_path = "xlip_openi_ref.txt"
# #     # with open(ref_path, "w") as f:
# #     #     for samples in dataloader:
# #     #         f.write(sample["text_output"] + "\n")
#     trainer.generate_eval_files(checkpoint_path, hyp_path, ref_path)
#     score = LMTrainer.evaluate(hyp_path, ref_path)
#     print(score)
    # for i in range(15):
    #     hyp_file = f"../trainer/output/train_tmp_hyp_{i}.txt"
    #     ref_file = f"../trainer/output/train_tmp_ref_{i}.txt"
    #     score = XlipLMTrainer.evaluate_by_lines(hyp_file, ref_file)
    #     print(i, score)
    #     with open("score.txt", "a") as f:
    #         f.write(f"ft{i}epoch chestxraygpt4 it2t " + str(score) + "\n")
    score = XlipLMTrainer.evaluate("rg_hyp.txt", "rg_ref.txt") 
    print(score)