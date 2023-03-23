'''
Only contains the scoring function for holE, based on the implementation from https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/HolE.py
It was mainly outsourced to keep the model definitions clean
'''
import torch
def conj(tensor):
    zero_shape = (list)(tensor.shape)
    one_shape = (list)(tensor.shape)
    zero_shape[-1] = 1
    one_shape[-1] -= 1
    ze = torch.zeros(size = zero_shape, device = tensor.device)
    on = torch.ones(size = one_shape, device = tensor.device)
    matrix = torch.cat([ze, on], -1)
    matrix = 2 * matrix
    return tensor - matrix * tensor
        
def real(tensor):
    dimensions = len(tensor.shape)
    return tensor.narrow(dimensions - 1, 0, 1)
    
def imag(tensor):
    dimensions = len(tensor.shape)
    return tensor.narrow(dimensions - 1, 1, 1)
    
def mul(real_1, imag_1, real_2, imag_2):
    real = real_1 * real_2 - imag_1 * imag_2
    imag = real_1 * imag_2 + imag_1 * real_2
    return torch.cat([real, imag], -1)
    
def ccorr(a, b):
    a = conj(torch.rfft(a, signal_ndim = 1, onesided = False))
    b = torch.rfft(b, signal_ndim = 1, onesided = False)
    res = mul(real(a), imag(a), real(b), imag(b))
    res = torch.ifft(res, signal_ndim = 1)
    return real(res).flatten(start_dim = -2)

def holeScore(entity1,relation,entity2):
    score = ccorr(entity1, entity2) * relation
    score = torch.sum(score, -1)
    return score

def holeScoreLoop(entity1,relation,entity2):
    score = ccorr(entity1, entity2) * relation
    score = torch.sum(score, -1).flatten()
    return score