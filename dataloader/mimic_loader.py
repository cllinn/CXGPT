import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
import json 
from transformers import AutoFeatureExtractor, AutoImageProcessor
from PIL import Image 
from torchvision import utils 
import numpy as np 
from tqdm import tqdm 

PROMPT_TEMPLATE = {
    "i": [
        "corresponding radiology reports is",
        "based on the chest x-ray the impression is"
    ],
    "it": [
        "radiology reports is {findings}, the corrseponding diagnostic opinions is",
        "the radiology reports {findings}, and the impression is",
        "Findings:{findings}, Impression:"
        ]
}

class MIMIC_CXR(Dataset):
    def __init__(self):
        super().__init__()
        json_path = "/root/autodl-tmp/MIMIC-CXR/data_processed/new_data.json"
        with open(json_path, "r") as f:
            fd = json.load(f)
        self.ls = list(fd)
        self.fd = fd 
        self.image_processor = AutoFeatureExtractor.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")

    def __getitem__(self, index):
        out_k = self.ls[index]  
        in_d = self.fd[out_k]
        text_input = in_d["FINDINGS"]
        text_output = in_d["IMPRESSION"]
        image_path = in_d.get("IMAGE")
        prompts = PROMPT_TEMPLATE["it"][2]
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt")
        image = image["pixel_values"].squeeze()
        sample = {
            "image": image,
            "text_input": prompts.format(findings=text_input),
            "text_output": text_output
        }
        return sample 
    
    def __len__(self):
        return len(self.ls)

if __name__ == "__main__":
    # tr = torch.load("MIMIC_it2t_train.pt")
    # for i in tr:
    #     print(i)
    #     exit(0)
    data = MIMIC_CXR()
    l = len(data)
    tr = int(0.7 * l)
    va = int(0.1 * l)
    te = l - tr - va
    train, val, test = torch.utils.data.random_split(data, [tr, va, te])
    torch.save(train, "MIMIC_it2t_train.pt")
    torch.save(val, "MIMIC_it2t_val.pt")
    torch.save(test, "MIMIC_it2t_test.pt")
    torch.save(data, "MIMIC_CXR.pt")
    # json_path = "/root/autodl-tmp/MIMIC-CXR/data_processed/new_data.json"
    # with open(json_path, "r") as f:
    #     fd = json.load(f)
    # ls = list(fd)
    # for out_k in ls:
    #     in_d = fd[out_k]
    #     image_path = in_d.get("IMAGE")
    #     if image_path is not None:
    #         image_path = image_path.replace("/data/user/MYF/", "/root/autodl-tmp/")
    #     fd[out_k]["IMAGE"] = image_path
    #     if image_path is None:
    #         # del fd[out_k]
    #         print("None")
    #         exit(0)
    # with open("/root/autodl-tmp/MIMIC-CXR/data_processed/new_data.json", "w") as f:
    #     json.dump(fd, f)
    # image_processor = AutoFeatureExtractor.from_pretrained("../checkpoint/cvt-21")
    # index = 0
    # out_k = ls[index]
    # in_d = fd[out_k]
    # image_path = in_d["IMAGE"]
    # print(image_path)
    # image = Image.open(image_path)
    # print(image)
    # image = image_processor(image, return_tensors="pt")
    # image = image["pixel_values"].clone().detach().cpu()
    # r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    # print((r-g).sum(), (r-g).sum(), (g-b).sum())
    # utils.save_image(image, "image2.png")
    # data = MIMIC_CXR()
    # R_channels = 0.0
    # G_channels = 0.0
    # B_channels = 0.0
    # loader = DataLoader(data, batch_size=32)
    # num = 0
    # for i in tqdm(loader):
    #     image = i["image"]
    #     R_image = image[:, 0, :, :]
    #     G_image = image[:, 1, :, :]
    #     B_image = image[:, 2, :, :]
    #     R_channels += R_image.sum()
    #     G_channels += G_image.sum()
    #     B_channels += B_image.sum()
    #     num += image.size(0) * image.size(2) * image.size(3)
    # R_mean = R_channels / num
    # G_mean = G_channels / num
    # B_mean = B_channels / num
    # R_channels = 0.0
    # G_channels = 0.0
    # B_channels = 0.0
    # for i in loader:
    #     image = i["image"]
    #     R_image = image[:, 0, :, :]
    #     G_image = image[:, 1, :, :]
    #     B_image = image[:, 2, :, :]
    #     R_channels += ((R_image - R_mean) ** 2).sum()
    #     G_channels += ((G_image - G_mean) ** 2).sum()
    #     B_channels += ((B_image - B_mean) ** 2).sum()
    # R_var, G_var, B_var = np.sqrt(R_channels / num), np.sqrt(G_channels / num), np.sqrt(B_channels / num)
    # print(R_mean, G_mean, B_mean, R_var, G_var, B_var)
    # print(num, len(data))

