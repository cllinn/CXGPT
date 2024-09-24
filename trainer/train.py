import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import sys
sys.path.append("/root")
from torch.utils.data import DataLoader
from mulcon.dataloader.openi_loader import OPENI_it2t, OPENI_i_im2t
# from mulcon.dataloader.mimic_loader import MIMIC_CXR
from mulcon.model.xlip_model import XlipModel
from mulcon.config.xlip_config import XlipConfig
from mulcon.model.Xformer import BertLMHeadModel, BertConfig
from transformers import BertTokenizer, CvtModel, CvtConfig
import argparse
from mulcon.trainer.trainer import XlipTrainer
from mulcon.trainer.multiserial_trainer import XlipMultiGPUSerialTrainer
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="../dataloader/OPENI_it2t_train.pt", help="Path to dataset")
    parser.add_argument('--config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/config.json", help="Path to config")
    parser.add_argument('--cvt_config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/cvt_config.json", help="Path to CVT config")
    parser.add_argument('--encoder_config_path', type=str, default="/root/autodl-tmp/checkpoint/xlip-base/encoder_config.json", help="Path to encoder config")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay value")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--output_dir', type=str, default="/root/autodl-fs/xlip-base", help="Output directory")

    args = parser.parse_args()

    dataset_path = args.dataset_path
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
    model = XlipModel(cvt_config, encoder_config, config)
    
    # xlip_model_path="/root/autodl-tmp/checkpoint/xlip-base/checkpoint/checkpoint.pt-19"
    # checkpoint = torch.load(xlip_model_path, map_location="cuda:0")
    # model.load_state_dict(checkpoint["model"])
    # del checkpoint
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # exit(0)
    dataset = torch.load(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    params = model.get_optimizer_params(weight_decay, lr)
    optimizer = AdamW(params)
    lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)

    trainer = XlipTrainer(epochs, optimizer, model, dataloader, lr_scheduler, output_dir)
    print("pretraining xlip model")
    trainer.train()
