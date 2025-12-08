import torch
from torch.autograd import Variable


#######################################################
#       STATISTICAL DISTANCES(LOSSES) IN PYTORCH      #
#######################################################

## Statistial Distances for 1D weight distributions
## Inspired by Scipy.Stats Statistial Distances for 1D
## Pytorch Version, supporting Autograd to make a valid Loss
## Supposing Inputs are Groups of Same-Length Weight Vectors
## Instead of (Points, Weight), full-length Weight Vectors are taken as Inputs
## Code Written by E.Bao, CASIA

def torch_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a,tensor_b,p=1))

def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))

def symmetric_earthmover(tensor_a, tensor_b, p=1, normalize=False):
    """
    Vectors are not normalized.
    Normalize here means we divide both vectors by the sum of both their weights.
    so normalize(V1) = V1 * 2/(sum(V1)+sum(V2))
    We compute the sum of the metric forward and backward, to cancel the direction bias.
    Also no pth roots, because they cause instability.
    """
    #print(tensor_a.shape, tensor_b.shape)
    norm_term = 1.0
    if normalize:
        if (torch.sum(tensor_a) + torch.sum(tensor_b)) > 0:
            norm_term = 2.0/(torch.sum(tensor_a) + torch.sum(tensor_b))

    cdf_tensor_a = torch.cumsum(norm_term * tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(norm_term * tensor_b, dim=-1)
    cdf_tensor_a_rev = torch.cumsum(torch.flip(norm_term * tensor_a,[-1]),dim=-1)
    cdf_tensor_b_rev = torch.cumsum(torch.flip(norm_term * tensor_b,[-1]),dim=-1)
    #print("-----------------------------------------------------")
    #print(tensor_a[0,:])
    #print(tensor_b[0,:])
    #print(torch.flip(tensor_a,[-1])[0,:])
    #print(torch.flip(tensor_b,[-1])[0,:])
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
        cdf_distance += torch.sum(torch.abs((cdf_tensor_a_rev-cdf_tensor_b_rev)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1)
        cdf_distance += torch.sum(torch.pow((cdf_tensor_a_rev - cdf_tensor_b_rev), 2), dim=-1)
    else:
        cdf_distance = torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1)
        cdf_distance += torch.sum(torch.pow(torch.abs(cdf_tensor_a_rev-cdf_tensor_b_rev),p),dim=-1)
    return cdf_distance/cdf_tensor_a.shape[-1]

def torch_cdf_loss_protected(tensor_a,tensor_b,p=1, normalize=True):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    if normalize:
        tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
        tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        #cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
        cdf_distance = torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1)
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss


def torch_cdf_loss(tensor_a,tensor_b,p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss

def torch_validate_distibution(tensor_a,tensor_b):
    # Zero sized dimension is not supported by pytorch, we suppose there is no empty inputs
    # Weights should be non-negetive, and with a positive and finite sum
    # We suppose all conditions will be corrected by network training
    # We only check the match of the size here
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Input weight tensors must be of the same size")
