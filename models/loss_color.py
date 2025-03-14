import torch


def color_loss(x): 
    mean_rgb = torch.mean(x,[2,3],keepdim=True)
    mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
    Drg = torch.pow(mr-mg,2)
    Drb = torch.pow(mr-mb,2)
    Dgb = torch.pow(mb-mg,2)
    k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5).mean()
    return k