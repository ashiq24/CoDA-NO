import unittest

import torchinfo

from models.codano import CodANO, VariableEncodingArgs


class CodANOTest(unittest.TestCase):
    def test_initialization(self):
        raise NotImplementedError

    def test_forward(self):
        n_variables = 2
        n_tokens = 4  # tokens per (logical) channel
        n_channels = 3  # channels per variable
        # n_channels = n_tokens * n_channels  # physical channels
        n_modes = 16
        model = CodANO(
            # I/O per variable
            input_token_codimension=1,
            output_token_codimension=1,
            hidden_token_codimension=n_tokens,
            lifting_token_codimension=n_tokens * 2,
            n_layers=2,
            n_modes=((n_modes, n_modes), (n_modes, n_modes)),
            n_heads=(2, 2),
            n_variables=n_variables,
            variable_encoding_args=VariableEncodingArgs(
                basis="fft",
                n_channels=n_channels,
                modes_x=n_modes,
                modes_y=n_modes,
            ),
            lifting=True,
            projection=True,
        )
        """
        ====================================================================================================
        Layer (type:depth-idx)                             Output Shape              Param #
        ====================================================================================================
        CodANO                                             [4, 1, 64, 64]            --
        ├─Projection: 1-1                                  [4, 4, 64, 64]            --
        │    └─Conv2d: 2-1                                 [4, 8, 64, 64]            40
        │    └─InstanceNorm2d: 2-2                         [4, 8, 64, 64]            16
        │    └─Conv2d: 2-3                                 [4, 4, 64, 64]            36
        ├─ModuleList: 1-2                                  --                        --
        │    └─TnoBlock2d: 2-4                             [4, 4, 64, 64]            --
        │    │    └─InstanceNorm2d: 3-1                    [16, 1, 64, 64]           2
        │    │    └─FNOBlocks: 3-2                         [16, 2, 64, 64]           580
        │    │    └─FNOBlocks: 3-3                         [16, 2, 64, 64]           580
        │    │    └─FNOBlocks: 3-4                         [16, 2, 64, 64]           580
        │    │    └─FNOBlocks: 3-5                         [16, 1, 64, 64]           579
        │    │    └─InstanceNorm2d: 3-6                    [16, 1, 64, 64]           2
        │    │    └─InstanceNorm2d: 3-7                    [16, 1, 64, 64]           2
        │    │    └─FNOBlocks: 3-8                         [16, 1, 64, 64]           580
        │    │    └─InstanceNorm2d: 3-9                    [16, 1, 64, 64]           2
        │    └─TnoBlock2d: 2-5                             [4, 4, 64, 64]            --
        │    │    └─InstanceNorm2d: 3-10                   [16, 1, 64, 64]           2
        │    │    └─FNOBlocks: 3-11                        [16, 2, 64, 64]           580
        │    │    └─FNOBlocks: 3-12                        [16, 2, 64, 64]           580
        │    │    └─FNOBlocks: 3-13                        [16, 2, 64, 64]           580
        │    │    └─FNOBlocks: 3-14                        [16, 1, 64, 64]           579
        │    │    └─InstanceNorm2d: 3-15                   [16, 1, 64, 64]           2
        │    │    └─InstanceNorm2d: 3-16                   [16, 1, 64, 64]           2
        │    │    └─FNOBlocks: 3-17                        [16, 1, 64, 64]           580
        │    │    └─InstanceNorm2d: 3-18                   [16, 1, 64, 64]           2
        ├─Projection: 1-3                                  [4, 1, 64, 64]            --
        │    └─Conv2d: 2-6                                 [4, 8, 64, 64]            40
        │    └─InstanceNorm2d: 2-7                         [4, 8, 64, 64]            16
        │    └─Conv2d: 2-8                                 [4, 1, 64, 64]            9
        ====================================================================================================
        Total params: 5,971
        Trainable params: 5,971
        Non-trainable params: 0
        Total mult-adds (M): 3.23
        ====================================================================================================
        Input size (MB): 0.26
        Forward/backward pass size (MB): 17.43
        Params size (MB): 0.00
        Estimated Total Size (MB): 17.70
        ====================================================================================================
        """

        """
        ====================================================================================================
        Layer (type:depth-idx)                             Output Shape              Param #
        ====================================================================================================
        CodANO                                             [4, 1, 64, 64]            --
        ├─Projection: 1-1                                  [4, 4, 64, 64]            --
        │    └─Conv2d: 2-1                                 [4, 8, 64, 64]            40
        │    └─InstanceNorm2d: 2-2                         [4, 8, 64, 64]            16
        │    └─Conv2d: 2-3                                 [4, 4, 64, 64]            36
        ├─ModuleList: 1-2                                  --                        --
        │    └─TnoBlock2d: 2-4                             [4, 4, 64, 64]            --
        │    │    └─InstanceNorm2d: 3-1                    [16, 1, 64, 64]           2
        │    │    └─FNOBlocks: 3-2                         [16, 2, 64, 64]           --
        │    │    │    └─ModuleList: 4-1                   --                        --
        │    │    │    │    └─Conv2d: 5-1                  [16, 2, 64, 64]           2
        │    │    │    └─SpectralConvKernel2d: 4-2         [16, 2, 64, 64]           578
        │    │    └─FNOBlocks: 3-3                         [16, 2, 64, 64]           --
        │    │    │    └─ModuleList: 4-3                   --                        --
        │    │    │    │    └─Conv2d: 5-2                  [16, 2, 64, 64]           2
        │    │    │    └─SpectralConvKernel2d: 4-4         [16, 2, 64, 64]           578
        │    │    └─FNOBlocks: 3-4                         [16, 2, 64, 64]           --
        │    │    │    └─ModuleList: 4-5                   --                        --
        │    │    │    │    └─Conv2d: 5-3                  [16, 2, 64, 64]           2
        │    │    │    └─SpectralConvKernel2d: 4-6         [16, 2, 64, 64]           578
        │    │    └─FNOBlocks: 3-5                         [16, 1, 64, 64]           --
        │    │    │    └─ModuleList: 4-7                   --                        --
        │    │    │    │    └─Conv2d: 5-4                  [16, 1, 64, 64]           2
        │    │    │    └─SpectralConvKernel2d: 4-8         [16, 1, 64, 64]           577
        │    │    └─InstanceNorm2d: 3-6                    [16, 1, 64, 64]           2
        │    │    └─InstanceNorm2d: 3-7                    [16, 1, 64, 64]           2
        │    │    └─FNOBlocks: 3-8                         [16, 1, 64, 64]           --
        │    │    │    └─ModuleList: 4-9                   --                        1
        │    │    │    │    └─Conv2d: 5-5                  [16, 1, 64, 64]           1
        │    │    │    └─SpectralConvKernel2d: 4-10        [16, 1, 64, 64]           578
        │    │    │    └─ModuleList: 4-11                  --                        --
        │    │    │    │    └─InstanceNorm2d: 5-6          [16, 1, 64, 64]           --
        │    │    └─InstanceNorm2d: 3-9                    [16, 1, 64, 64]           2
        │    └─TnoBlock2d: 2-5                             [4, 4, 64, 64]            --
        │    │    └─InstanceNorm2d: 3-10                   [16, 1, 64, 64]           2
        │    │    └─FNOBlocks: 3-11                        [16, 2, 64, 64]           --
        │    │    │    └─ModuleList: 4-12                  --                        --
        │    │    │    │    └─Conv2d: 5-7                  [16, 2, 64, 64]           2
        │    │    │    └─SpectralConvKernel2d: 4-13        [16, 2, 64, 64]           578
        │    │    └─FNOBlocks: 3-12                        [16, 2, 64, 64]           --
        │    │    │    └─ModuleList: 4-14                  --                        --
        │    │    │    │    └─Conv2d: 5-8                  [16, 2, 64, 64]           2
        │    │    │    └─SpectralConvKernel2d: 4-15        [16, 2, 64, 64]           578
        │    │    └─FNOBlocks: 3-13                        [16, 2, 64, 64]           --
        │    │    │    └─ModuleList: 4-16                  --                        --
        │    │    │    │    └─Conv2d: 5-9                  [16, 2, 64, 64]           2
        │    │    │    └─SpectralConvKernel2d: 4-17        [16, 2, 64, 64]           578
        │    │    └─FNOBlocks: 3-14                        [16, 1, 64, 64]           --
        │    │    │    └─ModuleList: 4-18                  --                        --
        │    │    │    │    └─Conv2d: 5-10                 [16, 1, 64, 64]           2
        │    │    │    └─SpectralConvKernel2d: 4-19        [16, 1, 64, 64]           577
        │    │    └─InstanceNorm2d: 3-15                   [16, 1, 64, 64]           2
        │    │    └─InstanceNorm2d: 3-16                   [16, 1, 64, 64]           2
        │    │    └─FNOBlocks: 3-17                        [16, 1, 64, 64]           --
        │    │    │    └─ModuleList: 4-20                  --                        1
        │    │    │    │    └─Conv2d: 5-11                 [16, 1, 64, 64]           1
        │    │    │    └─SpectralConvKernel2d: 4-21        [16, 1, 64, 64]           578
        │    │    │    └─ModuleList: 4-22                  --                        --
        │    │    │    │    └─InstanceNorm2d: 5-12         [16, 1, 64, 64]           --
        │    │    └─InstanceNorm2d: 3-18                   [16, 1, 64, 64]           2
        ├─Projection: 1-3                                  [4, 1, 64, 64]            --
        │    └─Conv2d: 2-6                                 [4, 8, 64, 64]            40
        │    └─InstanceNorm2d: 2-7                         [4, 8, 64, 64]            16
        │    └─Conv2d: 2-8                                 [4, 1, 64, 64]            9
        ====================================================================================================
        Total params: 5,971
        Trainable params: 5,971
        Non-trainable params: 0
        Total mult-adds (M): 3.23
        ====================================================================================================
        Input size (MB): 0.26
        Forward/backward pass size (MB): 17.43
        Params size (MB): 0.00
        Estimated Total Size (MB): 17.70
        ====================================================================================================
        """

        batch_size = 4
        torchinfo.summary(model, input_size=(batch_size, 4, 64, 64), depth=5)
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
