import unittest

import torch

from fino import SpectralConvKernel2d, SpectralConvolutionKernel3D

class SpectralConvKernel2DTest(unittest.TestCase):
    def test_initialization(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="fft",
            frequency_mixer=True,
        )

        # TODO convolution.half_n_modes is immutable and should be a tuple
        self.assertEqual(tuple(convolution.half_n_modes), (8, 8))
        self.assertEqual(tuple(convolution.W1r.shape), (8, 8, 8, 8))
        self.assertEqual(tuple(convolution.W1i.shape), (8, 8, 8, 8))
        self.assertEqual(tuple(convolution.W2r.shape), (8, 8, 8, 8))
        self.assertEqual(tuple(convolution.W2i.shape), (8, 8, 8, 8))
        self.assertIsNone(convolution.forward_sht)
        self.assertIsNone(convolution.inverse_sht)

    def test_init_sht_no_scaling(self):
        sht_nlat = 18
        sht_nlon = 36
        sht_grid = "equiangular"
        isht_grid = "equiangular"
        sht_norm = "schmidt"

        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="sht",
            output_scaling_factor=None,  # default option
            sht_nlat=sht_nlat,
            sht_nlon=sht_nlon,
            sht_grid=sht_grid,
            isht_grid=isht_grid,
            sht_norm=sht_norm,
        )

        self.assertEqual(convolution.forward_sht.nlat, sht_nlat)
        self.assertEqual(convolution.forward_sht.nlon, sht_nlon)
        self.assertEqual(convolution.forward_sht.grid, sht_grid)
        self.assertEqual(convolution.forward_sht.norm, sht_norm)

        self.assertEqual(convolution.inverse_sht.nlat, sht_nlat)
        self.assertEqual(convolution.inverse_sht.nlon, sht_nlon)
        self.assertEqual(convolution.inverse_sht.grid, isht_grid)
        self.assertEqual(convolution.inverse_sht.norm, sht_norm)

    def test_init_sht_with_scaling(self):
        sht_nlat = 18
        sht_nlon = 18
        scaling = (10, 20)
        sht_grid = "equiangular"
        isht_grid = "equiangular"
        sht_norm = "schmidt"

        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="sht",
            output_scaling_factor=scaling,
            sht_nlat=sht_nlat,
            sht_nlon=sht_nlon,
            sht_grid=sht_grid,
            isht_grid=isht_grid,
            sht_norm=sht_norm,
        )

        self.assertEqual(convolution.forward_sht.nlat, sht_nlat)
        self.assertEqual(convolution.forward_sht.nlon, sht_nlon)
        self.assertEqual(convolution.forward_sht.grid, sht_grid)
        self.assertEqual(convolution.forward_sht.norm, sht_norm)

        expected_long = sht_nlon * scaling[1] // 2  # like half-modes
        self.assertEqual(convolution.inverse_sht.nlat, sht_nlat * scaling[0])
        self.assertEqual(convolution.inverse_sht.nlon, expected_long)
        self.assertEqual(convolution.inverse_sht.grid, isht_grid)
        self.assertEqual(convolution.inverse_sht.norm, sht_norm)

    def test_forward_transform_fft(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=4,
            n_modes=(8, 8),
            transform_type="fft",
        )

        # This size doesn't need to be equal to that of `modes` above:
        x = torch.randn((1, 16, 16))
        y = convolution.forward_transform(x)
        # FFT takes the real frequency modes (i.e. the upper x.size(-1)//2 + 1 modes)
        self.assertEqual(tuple(y.shape), (1, 16, 9))

    def test_forward_transform_invalid_transform(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="invalid_transform_type",
        )

        x = torch.randn((1, 64, 64))
        self.assertRaisesRegex(
            ValueError,
            # The error message should be helpful and contain the bad type:
            r"invalid_transform_type",
            convolution.forward_transform,
            x,  # positional arg to be passed above
        )

    def test_forward_transform_sht_uses_old_transform(self):
        sht_nlat = 180
        sht_nlon = 360

        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="sht",
            sht_nlat=sht_nlat,
            sht_nlon=sht_nlon,
        )
        sht1 = convolution.forward_sht

        x = torch.randn((1, sht_nlat, sht_nlon))
        y = convolution.forward_transform(x)
        sht2 = convolution.forward_sht

        self.assertEqual(id(sht1), id(sht2))
        # AssertionError: Tuples differ: (1, 180, 181) != (1, 180, 360)
        # This seems to be treating lat/long like half modes.
        # TODO(ashiq) Is this expected behavior?
        self.assertEqual(tuple(y.shape), (1, sht_nlat, sht_nlon))

    def test_forward_transform_sht_uses_new_transform(self):
        sht_nlat = 180
        sht_nlon = 360

        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="sht",
            sht_nlat=18,
            sht_nlon=36,
        )
        sht1 = convolution.forward_sht

        x = torch.randn((1, sht_nlat, sht_nlon))
        y = convolution.forward_transform(x)
        sht2 = convolution.forward_sht

        self.assertNotEqual(id(sht1), id(sht2))
        # AssertionError: Tuples differ: (1, 180, 181) != (1, 180, 360)
        # This seems to be treating lat/long like half modes.
        # TODO(ashiq) Is this expected behavior?
        self.assertEqual(tuple(y.shape), (1, sht_nlat, sht_nlon))

    def test_inverse_transform_fft(self):
        frequency_modes = 16
        physical_resolution = 64
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(frequency_modes, frequency_modes),
            transform_type="fft",
        )

        # This size DOES need to be equal to that of `modes` above:
        x = torch.randn((1, frequency_modes, frequency_modes))
        y = convolution.inverse_transform(
            x,
            target_height=physical_resolution,
            target_width=physical_resolution,
        )
        self.assertEqual(
            tuple(y.shape),
            (1, physical_resolution, physical_resolution)
        )

    def test_inverse_transform_invalid_transform(self):
        frequency_modes = 16
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(frequency_modes, frequency_modes),
            transform_type="invalid_transform_type",
        )

        x = torch.randn((1, frequency_modes, frequency_modes))
        self.assertRaisesRegex(
            ValueError,
            # The error message should be helpful and contain the bad type:
            r"invalid_transform_type",
            convolution.inverse_transform,
            x,  # positional arg to be passed above
            180,  # target_height
            360,  # target_width
        )

    def test_inverse_transform_sht_uses_old_transform(self):
        sht_nlat = 180
        sht_nlon = 360

        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="sht",
            sht_nlat=sht_nlat,
            sht_nlon=sht_nlon,
        )
        isht1 = convolution.inverse_sht

        x = torch.randn((1, sht_nlat, sht_nlon // 2 + 1))
        y = convolution.inverse_transform(x, sht_nlat, sht_nlon)
        isht2 = convolution.inverse_sht

        self.assertEqual(id(isht1), id(isht2))
        self.assertEqual(tuple(y.shape), (1, sht_nlat, sht_nlon))

    def test_inverse_transform_sht_uses_new_transform_mismatched_source(self):
        sht_nlat = 180
        sht_nlon = 360

        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="sht",
            sht_nlat=sht_nlat,
            sht_nlon=sht_nlon,
        )
        isht1 = convolution.inverse_sht

        # Mismatch on input tensor shape will instantiate a new SHT transform:
        x = torch.randn((1, 18, 36))
        y = convolution.inverse_transform(x, sht_nlat, sht_nlon)
        isht2 = convolution.inverse_sht

        self.assertNotEqual(id(isht1), id(isht2))
        self.assertEqual(tuple(y.shape), (1, sht_nlat, sht_nlon))

    def test_inverse_transform_sht_uses_new_transform_mismatched_target(self):
        sht_nlat = 180
        sht_nlon = 360

        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="sht",
            sht_nlat=sht_nlat,
            sht_nlon=sht_nlon,
        )
        isht1 = convolution.inverse_sht

        x = torch.randn((1, sht_nlat, sht_nlon))
        # Mismatch on requested output tensor shape
        # will instantiate a new SHT transform:
        target_height = 18
        target_width = 36
        y = convolution.inverse_transform(x, target_height, target_width)
        isht2 = convolution.inverse_sht

        self.assertNotEqual(id(isht1), id(isht2))
        self.assertEqual(tuple(y.shape), (1, target_height, target_width))

    def test_forward_propagation(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="fft",
        )

        x = torch.randn((1, 4, 64, 64))
        y1 = convolution(x)
        self.assertEqual(tuple(y1.shape), (1, 8, 64, 64))

        y2 = convolution(x, output_shape=(128, 128))
        self.assertEqual(tuple(y2.shape), (1, 8, 128, 128))

    def test_forward_propagation_with_output_scaling(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="fft",
            output_scaling_factor=(4, 4),
            frequency_mixer=True,
        )

        x = torch.randn((1, 4, 64, 64))
        y1 = convolution(x)
        self.assertEqual(tuple(y1.shape), (1, 8, 256, 256))

        # Explicitly passing in ``output_shape`` overrides ``scaling_factor`` state:
        y2 = convolution(x, output_shape=(128, 128))
        self.assertEqual(tuple(y2.shape), (1, 8, 128, 128))


class SpectralConvKernel3DTest(unittest.TestCase):
    """
    TODO:
    * half_n_modes: init and usage [ in fwd() ]
    * upper/lower modes /W[ri]/ parameters and shape.
    ---
    * forward_transform
    ** does "fft"
    ** throws otherwise
    ---
    * forward_transform
    ** does "fft"
    ** throws otherwise
    ---
    * forward
    ** check shape with scaling, output shape, and by default
    ** does it fill all the output channels?
    ** bias term biases (!)
    """
    def test_initialization(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16, 16),
            transform_type="fft",
            frequency_mixer=True,
        )

        expected_modes = (8, 8, 8)
        expected_shape = expected_modes * 2
        # TODO convolution.half_n_modes is immutable and should be a tuple
        self.assertEqual(tuple(convolution.half_n_modes), expected_modes)
        self.assertEqual(
            [tuple(w.shape) for w in convolution.weights_re],
            [expected_shape for _ in convolution.weights_re],
        )
        self.assertEqual(
            [tuple(w.shape) for w in convolution.weights_im],
            [expected_shape for _ in convolution.weights_im],
        )

    def test_forward_transform_fft(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=4,
            n_modes=(8, 8, 8),
            transform_type="fft",
        )

        # This size doesn't need to be equal to that of `modes` above:
        x = torch.randn((1, 16, 16, 16))
        y = convolution.forward_transform(x)
        # FFT takes the real frequency modes (i.e. the upper x.size(-1)//2 + 1 modes)
        self.assertEqual(tuple(y.shape), (1, 16, 16, 9))

    def test_forward_transform_invalid_transform(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16, 16),
            transform_type="invalid_transform_type",
        )

        x = torch.randn((1, 32, 32, 32))
        self.assertRaisesRegex(
            ValueError,
            # The error message should be helpful and contain the bad type:
            r"invalid_transform_type",
            convolution.forward_transform,
            x,  # positional arg to be passed above
        )

    def test_inverse_transform_fft(self):
        frequency_modes = 8
        physical_resolution = 32
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=6,
            n_modes=(frequency_modes, frequency_modes, frequency_modes),
            transform_type="fft",
        )

        # This size DOES need to be equal to that of `modes` above:
        x = torch.randn((1, frequency_modes, frequency_modes, frequency_modes))
        y = convolution.inverse_transform(
            x,
            duration=physical_resolution,
            height=physical_resolution,
            width=physical_resolution,
        )
        self.assertEqual(
            tuple(y.shape),
            (1, physical_resolution, physical_resolution, physical_resolution)
        )

    def test_inverse_transform_invalid_transform(self):
        frequency_modes = 16
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=8,
            n_modes=(frequency_modes, frequency_modes, frequency_modes),
            transform_type="invalid_transform_type",
        )

        x = torch.randn((1, frequency_modes, frequency_modes, frequency_modes))
        self.assertRaisesRegex(
            ValueError,
            # The error message should be helpful and contain the bad type:
            r"invalid_transform_type",
            convolution.inverse_transform,
            x,  # positional arg to be passed above
            frequency_modes,  # duration
            frequency_modes,  # height
            frequency_modes,  # width
        )

    def test_forward_propagation(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16, 16),
            transform_type="fft",
        )

        x = torch.randn((1, 4, 32, 32, 32))
        y1 = convolution(x)
        self.assertEqual(tuple(y1.shape), (1, 8, 32, 32, 32))

        y2 = convolution(x, output_shape=(48, 48, 48))
        self.assertEqual(tuple(y2.shape), (1, 8, 48, 48, 48))

    def test_forward_propagation_with_output_scaling(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=8,
            n_modes=(8, 8, 8),
            transform_type="fft",
            output_scaling_factor=(2, 2),
            frequency_mixer=True,
        )

        x = torch.randn((1, 4, 16, 16, 16))
        y1 = convolution(x)
        self.assertEqual(tuple(y1.shape), (1, 8, 32, 32, 32))

        # Explicitly passing in ``output_shape`` overrides ``scaling_factor`` state:
        y2 = convolution(x, output_shape=(24, 24, 24))
        self.assertEqual(tuple(y2.shape), (1, 8, 24, 24, 24))


if __name__ == '__main__':
    unittest.main()
