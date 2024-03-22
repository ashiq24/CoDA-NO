import torch
from einops import repeat
from vit_pytorch import ViT


class vision_transformer(ViT):
    """
    Vision Transformer model for image classification.

    Args:
        ViT (torch.nn.Module): Base Vision Transformer model.

    Attributes:
        cls_token (torch.Tensor): Token representing the class label.
        pos_embedding (torch.Tensor): Positional embeddings for the input image patches.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        transformer (torch.nn.Module): Transformer encoder for feature extraction.

    Methods:
        forward(img): Forward pass of the vision_transformer model.

    """

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        return x[:, 1:, :]
