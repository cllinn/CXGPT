import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import sys
sys.path.append("/root")
from torch.utils.data import DataLoader
from mulcon.dataloader.openi_loader import OPENI, OPENI_i2t, OPENI_it2t, OPENI_t2t, OPENI_image2impression
from mulcon.dataloader.mimic_loader import MIMIC_CXR
# from mulcon.dataloader.lm_loader import LMDataset
from mulcon.model.xlip_model import XlipModel
from mulcon.model.xlip_lm import XlipLM
from mulcon.model.cvt_biogpt import CvtBioGPT
from transformers import AutoTokenizer, AutoModelForCausalLM
from mulcon.config.xlip_config import XlipConfig
from mulcon.model.Xformer import BertLMHeadModel, BertConfig
from transformers import BertTokenizer, CvtModel, CvtConfig
import argparse
from mulcon.trainer.trainer import XlipTrainer
# from mulcon.trainer.multiserial_trainer import XlipMultiGPUSerialTrainer
from mulcon.trainer.xlip_lmtrainer import XlipLMTrainer
from mulcon.trainer.LMTrainer import LMTrainer
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="../dataloader/OPENI_it2t_train.pt", help="Path to dataset")
    parser.add_argument('--val_path', type=str, default="../dataloader/OPENI_it2t_test.pt", help="Path to val dataset")
    parser.add_argument('--config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/config.json", help="Path to config")
    parser.add_argument('--cvt_config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/cvt_config.json", help="Path to CVT config")
    parser.add_argument('--encoder_config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/encoder_config.json", help="Path to encoder config")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device for training")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--epochs', type=int, default=8, help="Number of epochs")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay value")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/checkpoint/cvt-biogpt-mimic", help="Output directory")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    val_path = args.val_path
    config_path = args.config_path
    cvt_config_path = args.cvt_config_path
    encoder_config_path = args.encoder_config_path
    device = torch.device(args.device)
    batch_size = args.batch_size
    epochs = args.epochs
    weight_decay = args.weight_decay
    lr = args.lr
    output_dir = args.output_dir

    cvt_config = CvtConfig.from_json_file(cvt_config_path)
    encoder_config = BertConfig.from_json_file(encoder_config_path)
    config = XlipConfig.from_json_file(config_path)
    # model = XlipModel(cvt_config, encoder_config, config)
    model = XlipLM(cvt_config, encoder_config, config)
    print(f"number query of model is {model.query_tokens.shape[1]}")
    # model = CvtBioGPT(cvt_config)
    # model = model.to(device)
    # tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/checkpoint/biogpt-large-pubmedqa", use_fast=False, truncation_side="left")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "/root/autodl-tmp/checkpoint/biogpt-large-pubmedqa", torch_dtype=torch.float32
    # )
    dataset = torch.load(dataset_path)
    valset = torch.load(val_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=2, shuffle=True)
    # params = model.get_optimizer_params(weight_decay, lr)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)

    trainer = XlipLMTrainer(epochs, optimizer, model, dataloader, val_loader, lr_scheduler, output_dir)
    # trainer = LMTrainer(epochs, optimizer, model, tokenizer, dataloader, lr_scheduler, output_dir)
    print("start training")
    trainer.train()
