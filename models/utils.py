import torch
from torch import nn
import torch.nn.functional as F

class MultiCropWrapper(nn.Module):
    def __init__(self, model):
        super(MultiCropWrapper, self).__init__()
        self.model = model
        self.embed_dim = model.embed_dim

    def forward(self, x):
        # x is a list of clips, each with shape [b, 1, t_i, 128, 128]
        # where t_i is the length of the i-th clip
        if not isinstance(x, list):
            x = [x]
        outputs = [self.model(x_i) for x_i in x]
        concatenated_output = torch.cat(outputs, dim=0)
        return concatenated_output


def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


token_mapping = {
    "pad": 0,
    "mask": 2,
}


def reset_parameters_to_random(state_dict, parameter_names):
    for name in parameter_names:
        if name in state_dict:
            state_dict[name] = torch.nn.init.xavier_uniform_(state_dict[name])
            print(f"[*] Reset {name} to random values.")


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, feat1, feat2):
        feats = torch.cat([feat1, feat2], dim=0)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        comb_sim = torch.cat(
        [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
        dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        return nll, sim_argsort

