import torch

if __name__ == "__main__":
    ft_model = "/root/autodl-tmp/checkpoint/fuxian/checkpoint/checkpoint.pt-7"
    all_model = "/root/autodl-tmp/checkpoint/xlip-lm-fuxian/checkpoint/checkpoint.pt-7"
    ft_model_state_dict = torch.load(ft_model)["model"]
    all_model_state_dict = torch.load(all_model)["model"]
    for k, v in ft_model_state_dict.items():
        name = "llm_model." + k
        all_model = all_model_state_dict[name]
        print(f"{k}, {name}", (all_model - v).sum())
    # for k, v in all_model_state_dict["model"].items():
        # print("all_model", k)