from collections import OrderedDict
import torch
import torch.nn as nn


class UNet2d(nn.Module):
    """
    2D U-Net implementation for image segmentation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        init_features (int): Number of initial features.

    Attributes:
        encoder1 (nn.Sequential): Encoder block 1.
        pool1 (nn.MaxPool2d): Max pooling layer 1.
        encoder2 (nn.Sequential): Encoder block 2.
        pool2 (nn.MaxPool2d): Max pooling layer 2.
        bottleneck (nn.Sequential): Bottleneck block.
        upconv2 (nn.ConvTranspose2d): Transposed convolution layer 2.
        decoder2 (nn.Sequential): Decoder block 2.
        upconv1 (nn.ConvTranspose2d): Transposed convolution layer 1.
        decoder1 (nn.Sequential): Decoder block 1.
        conv (nn.Conv2d): Convolution layer.

    """

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet2d, self).__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet2d._block(features * 2, features * 4, name="bottleneck")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        """
        Helper method to create a block of convolutional layers.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels.
            name (str): Name of the block.

        Returns:
            nn.Sequential: Block of convolutional layers.

        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.ReLU()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.ReLU()),
                ]
            )
        )
