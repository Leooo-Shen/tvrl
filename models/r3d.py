import torch
from torch import nn
from torchvision.transforms import v2


class R3D(nn.Module):
    def __init__(self, pretrained=True):
        super(R3D, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)
        self.model.blocks[-1] = nn.AdaptiveAvgPool3d(1)
        self.model.fc = nn.Identity()
        self.embed_dim = 2048

    @torch.no_grad()
    def forward(self, x, attn_mask=None, time_step=None, feat_op=None):
        # x is (b, c, t, h, w)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1, -1)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            # resize the input to 224x224
            x = v2.Resize((224, 224), antialias=False)(x)
        return self.model(x).squeeze(-1).squeeze(-1).squeeze(-1)
    

if __name__ == "__main__":

    x = torch.rand(2, 1, 3, 128, 128)
    attn_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0]]).bool()
    model = R3D()
    print(model)
    out = model(x, attn_mask) 
    print(out.shape)