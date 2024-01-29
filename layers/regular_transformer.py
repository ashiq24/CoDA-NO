import torch
from torch import nn
from einops import rearrange, repeat
from vit_pytorch import ViT

class vision_transformer(ViT):    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        return x[:, 1:,:]
# v = vision_transformer(
#     image_size = (100,50),
#     patch_size = (10,5),
#     num_classes = 1000,
#     dim = 500,
#     depth = 6,
#     channels = 10,
#     heads = 16,
#     mlp_dim = 500,
#     dropout = 0.0,
#     emb_dropout = 0.0
# )

# img = torch.randn(1, 10, 100, 50)

# preds = v(img)
# preds = preds.reshape(-1, 10, 100,50)
# print(preds.shape)
            