import torch
import torch.nn as nn 
import torch.nn.functional as F 


if __name__ == "__main__":
    device1 = torch.device("cuda", 1)
    device2 = torch.device("cuda", 2)
    net1 = nn.Linear(10, 5)
    net2 = nn.Linear(5, 1)
    optimizer = torch.optim.AdamW([
        {"params": net1.parameters(), "lr": 1e-3},
        {"params": net2.parameters(), "lr": 1e-4}
    ])
    input = torch.randn(32, 10)
    input = input.to(device1)
    net1 = net1.to(device1)
    net2 = net2.to(device2)
    output1 = net1(input)
    output1 = output1.to(device2)
    output = net2(output1)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()