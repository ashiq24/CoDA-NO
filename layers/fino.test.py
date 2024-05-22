import unittest
from typing import Optional

import numpy as np
import torch

from fino import SpectralConvKernel2d, SpectralConvolutionKernel3D

DEVICE: torch.device = "cuda" if torch.cuda.is_available() else "cpu"


class SpectralConvKernel2DWrapper(torch.nn.Module):
    """A simple wrapper to access Parameters during testing."""

    def __init__(self, convolution: SpectralConvKernel2d):
        super().__init__()
        self.convolution = convolution

    def forward(
        self,
        x: torch.Tensor,
        indices: int = 0,
        output_shape: Optional[torch.Size] = None,
    ):
        return self.convolution.forward(
            x,
            indices=indices,
            output_shape=output_shape
        )


class SpectralConvKernel3DWrapper(torch.nn.Module):
    """A simple wrapper to access Parameters during testing."""

    def __init__(self, convolution: SpectralConvolutionKernel3D):
        super().__init__()
        self.convolution = convolution

    def forward(
        self,
        x: torch.Tensor,
        indices: int = 0,
        output_shape: Optional[torch.Size] = None,
    ):
        return self.convolution.forward(
            x,
            indices=indices,
            output_shape=output_shape
        )


class SpectralConvKernel2DTest(unittest.TestCase):
    def test_initialization(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="fft",
            frequency_mixer=True,
        )

        self.assertEqual(tuple(convolution.W1.shape), (8, 9, 8, 9))
        self.assertEqual(tuple(convolution.W2.shape), (8, 9, 8, 9))
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
        scaling = [10, 20]
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

    @torch.no_grad()
    def test_forward_transform_fft(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=4,
            n_modes=(8, 8),
            transform_type="fft",
        )

        # This size doesn't need to be equal to that of `modes` above:
        x = torch.randn((1, 16, 16), dtype=torch.float64)
        y = convolution.forward_transform(x)
        # FFT takes the real frequency modes (i.e. the upper x.size(-1)//2 + 1
        # modes)
        self.assertEqual(tuple(y.shape), (1, 16, 9))
        self.assertEqual(y.dtype, torch.complex64)

    @torch.no_grad()
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

    @torch.no_grad()
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

        x = torch.randn((1, sht_nlat, sht_nlon), dtype=torch.float32)
        y = convolution.forward_transform(x)
        sht2 = convolution.forward_sht

        self.assertEqual(id(sht1), id(sht2))
        self.assertEqual(tuple(y.shape), (1, sht_nlat, sht_nlon))
        self.assertEqual(y.dtype, torch.complex64)

    @torch.no_grad()
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

        x = torch.randn((1, sht_nlat, sht_nlon), dtype=torch.float32)
        y = convolution.forward_transform(x)
        sht2 = convolution.forward_sht

        self.assertNotEqual(id(sht1), id(sht2))
        self.assertEqual(tuple(y.shape), (1, sht_nlat, sht_nlon))
        self.assertEqual(y.dtype, torch.complex64)

    @torch.no_grad()
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
        x = torch.randn((1, frequency_modes, frequency_modes),
                        dtype=torch.complex64)
        y = convolution.inverse_transform(
            x,
            target_height=physical_resolution,
            target_width=physical_resolution,
        )
        self.assertEqual(
            tuple(y.shape),
            (1, physical_resolution, physical_resolution)
        )
        self.assertEqual(y.dtype, torch.float32)

    @torch.no_grad()
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

    @torch.no_grad()
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

        x = torch.randn((1, sht_nlat, sht_nlon // 2 + 1),
                        dtype=torch.complex64)
        y = convolution.inverse_transform(x, sht_nlat, sht_nlon)
        isht2 = convolution.inverse_sht

        self.assertEqual(id(isht1), id(isht2))
        self.assertEqual(tuple(y.shape), (1, sht_nlat, sht_nlon))
        self.assertEqual(y.dtype, torch.float32)

    @torch.no_grad()
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
        x = torch.randn((1, 18, 36), dtype=torch.complex64)
        y = convolution.inverse_transform(x, sht_nlat, sht_nlon)
        isht2 = convolution.inverse_sht

        self.assertNotEqual(id(isht1), id(isht2))
        self.assertEqual(tuple(y.shape), (1, sht_nlat, sht_nlon))
        self.assertEqual(y.dtype, torch.float32)

    @torch.no_grad()
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

        x = torch.randn((1, sht_nlat, sht_nlon), dtype=torch.complex64)
        # Mismatch on requested output tensor shape
        # will instantiate a new SHT transform:
        target_height = 18
        target_width = 36
        y = convolution.inverse_transform(x, target_height, target_width)
        isht2 = convolution.inverse_sht

        self.assertNotEqual(id(isht1), id(isht2))
        self.assertEqual(tuple(y.shape), (1, target_height, target_width))
        self.assertEqual(y.dtype, torch.float32)

    @torch.no_grad()
    def test_forward_propagation(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="fft",
        )

        x = torch.randn((1, 4, 64, 64), dtype=torch.float32)
        y1 = convolution(x)
        self.assertEqual(tuple(y1.shape), (1, 8, 64, 64))
        self.assertEqual(y1.dtype, torch.float32)

        y2 = convolution(x, output_shape=(128, 128))
        self.assertEqual(tuple(y2.shape), (1, 8, 128, 128))
        self.assertEqual(y2.dtype, torch.float32)

    @torch.no_grad()
    def test_forward_propagation_with_output_scaling(self):
        convolution = SpectralConvKernel2d(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16),
            transform_type="fft",
            output_scaling_factor=[4, 4],
            frequency_mixer=True,
        )

        x = torch.randn((1, 4, 64, 64), dtype=torch.float32)
        y1 = convolution(x)
        self.assertEqual(tuple(y1.shape), (1, 8, 256, 256))
        self.assertEqual(y1.dtype, torch.float32)

        # Explicitly passing in ``output_shape`` overrides ``scaling_factor``
        # state:
        y2 = convolution(x, output_shape=(128, 128))
        self.assertEqual(tuple(y2.shape), (1, 8, 128, 128))
        self.assertEqual(y2.dtype, torch.float32)

    def test_backwards_propagation(self):
        convolution = SpectralConvKernel2d(
            in_channels=2,
            out_channels=2,
            n_modes=(16, 16),
            transform_type="fft",
            n_layers=4,
            frequency_mixer=True,
        )
        convolution_w = SpectralConvKernel2DWrapper(convolution)

        resolution = 0.01
        x = np.arange(-5, 5, resolution)
        y = np.arange(-5, 5, resolution)
        xx, yy = np.meshgrid(x, y, sparse=True)

        # Learn to advance the plane wave one quarter phase forward:
        phase = np.pi / 4.0
        displacement_in = torch.tensor(
            np.sin(xx * np.sqrt(2.0)) + np.sin(yy * np.sqrt(3.0)),
            device=DEVICE,
        )
        velocity_in = torch.tensor(
            np.sqrt(2.0) * np.cos(xx * np.sqrt(2.0)) +
            np.sqrt(3.0) * np.cos(yy * np.sqrt(3.0)),
            device=DEVICE,
        )
        displacement_out = torch.tensor(
            np.sin(xx * np.sqrt(2.0) + phase) +
            np.sin(yy * np.sqrt(3.0) + phase),
            device=DEVICE,
        )
        velocity_out = torch.tensor(
            np.sqrt(2.0) * np.cos(xx * np.sqrt(2.0) + phase) +
            np.sqrt(3.0) * np.cos(yy * np.sqrt(3.0) + phase),
            device=DEVICE,
        )

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            convolution_w.parameters(),
            # XXX: At 0.05, loss below is no longer monotonic(-ally
            # decreasing).
            lr=0.04,
            # XXX: At 0.005, loss below is no longer monotonic(-ally
            # decreasing).
            weight_decay=0.004,
        )

        losses = []
        for _ in range(10):
            optimizer.zero_grad()

            [[position_pred, velocity_pred]] = convolution_w.forward(
                torch.concatenate([
                    displacement_in.unsqueeze(0),
                    velocity_in.unsqueeze(0),
                ]).unsqueeze(0)
            )

            loss = loss_fn(
                displacement_out.float().view(1, -1),
                position_pred.float().view(1, -1)
            ) + loss_fn(
                velocity_out.float().view(1, -1),
                velocity_pred.float().view(1, -1)
            )
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        self.assertTrue(
            all(_next < _prev for _next, _prev in zip(
                losses[1:], losses[:-1])),
            "Expected losses to be monotonically decreasing:\n"
            f"[{', '.join('{:.3f}'.format(loss) for loss in losses)}]"
        )


class SpectralConvKernel3DTest(unittest.TestCase):

    @torch.no_grad()
    def test_initialization(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16, 16),
            transform_type="fft",
            frequency_mixer=True,
        )

        expected_modes = (8, 8, 9)
        expected_shape = expected_modes * 2
        self.assertEqual(
            [tuple(w.shape) for w in convolution.mixing_weights],
            [expected_shape for _ in convolution.mixing_weights],
        )

    @torch.no_grad()
    def test_forward_transform_fft(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=4,
            n_modes=(8, 8, 8),
            transform_type="fft",
        )

        # This size doesn't need to be equal to that of `modes` above:
        x = torch.randn((1, 16, 16, 16), dtype=torch.float64)
        y = convolution.forward_transform(x)
        # FFT takes the real frequency modes (i.e. the upper x.size(-1)//2 + 1
        # modes)
        self.assertEqual(tuple(y.shape), (1, 16, 16, 9))
        self.assertEqual(y.dtype, torch.complex64)

    @torch.no_grad()
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

    @torch.no_grad()
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
        x = torch.randn(
            (1, frequency_modes, frequency_modes, frequency_modes),
            dtype=torch.complex64,
        )
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
        self.assertEqual(y.dtype, torch.float32)

    @torch.no_grad()
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

    @torch.no_grad()
    def test_forward_propagation(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=8,
            n_modes=(16, 16, 16),
            transform_type="fft",
        )

        x = torch.randn((1, 4, 32, 32, 32), dtype=torch.float32)
        y1 = convolution(x)
        self.assertEqual(tuple(y1.shape), (1, 8, 32, 32, 32))
        self.assertEqual(y1.dtype, torch.float32)

        y2 = convolution(x, output_shape=(48, 48, 48))
        self.assertEqual(tuple(y2.shape), (1, 8, 48, 48, 48))
        self.assertEqual(y2.dtype, torch.float32)

    @torch.no_grad()
    def test_forward_propagation_with_output_scaling(self):
        # TODO(mogab) ``SpectralConv`` doesn't accept ``tuple``s
        # for ``scaling_factor`` - FIXME
        convolution = SpectralConvolutionKernel3D(
            in_channels=4,
            out_channels=8,
            n_modes=(8, 8, 8),
            transform_type="fft",
            output_scaling_factor=[2, 2, 2],
            frequency_mixer=True,
        )

        x = torch.randn((1, 4, 16, 16, 16), dtype=torch.float32)
        y1 = convolution.forward(x)
        self.assertEqual(tuple(y1.shape), (1, 8, 32, 32, 32))
        self.assertEqual(y1.dtype, torch.float32)

        # Explicitly passing in ``output_shape`` overrides ``scaling_factor``
        # state:
        y2 = convolution.forward(x, output_shape=(24, 24, 24))
        self.assertEqual(tuple(y2.shape), (1, 8, 24, 24, 24))
        self.assertEqual(y2.dtype, torch.float32)

    def test_backwards_propagation(self):
        convolution = SpectralConvolutionKernel3D(
            in_channels=2,
            out_channels=2,
            n_modes=(16, 16, 16),
            transform_type="fft",
            n_layers=4,
            frequency_mixer=True,
        )
        convolution_w = SpectralConvKernel3DWrapper(convolution)

        resolution = 0.02
        x = np.arange(-5.0, 5.0, resolution)
        y = np.arange(-5.0, 5.0, resolution)
        t = np.arange(0.0, 5.0, resolution)
        tt, xx, yy = np.meshgrid(t, x, y, indexing='ij', sparse=True)

        displacement = torch.tensor(
            np.sin(xx * np.sqrt(2.0) + tt * np.sqrt(5.0)) +
            np.sin(yy * np.sqrt(3.0) + tt * np.sqrt(7.0)),
            device=DEVICE,
        )
        velocity = torch.tensor(
            np.sqrt(5.0) * np.cos(xx * np.sqrt(2.0) * tt * np.sqrt(5.0)) +
            np.sqrt(7.0) * np.cos(yy * np.sqrt(3.0) * tt * np.sqrt(7.0)),
            device=DEVICE,
        )

        loss_fn = torch.nn.MSELoss()
        # XXX: These training parameters nominally work, but the
        # loss stays high (>7.0). I'd like to be able to test in a
        # scenario when it gets low, so as a
        # TODO(mogab) Tune LR/weight decay below, or find a better assertion
        # than "Is the loss monotonically decreasing?"
        optimizer = torch.optim.Adam(
            convolution_w.parameters(),
            # XXX: At 1.0e-3, loss below is no longer monotonic(-ally
            # decreasing).
            lr=1.0e-4,
            # XXX: At 1.0e-4, loss below is no longer monotonic(-ally
            # decreasing).
            weight_decay=1.0e-5,
        )

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            displacement_in = displacement[:100].unsqueeze(0)
            velocity_in = velocity[:100].unsqueeze(0)

            # Take the first 2.0 seconds as input ...
            [[position_pred, velocity_pred]] = convolution_w.forward(
                torch.concatenate([displacement_in, velocity_in]).unsqueeze(0)
            )

            # ... and learn the next 2.0 seconds as output:
            # (i.e. 2.0 <= t < 4.0)
            loss = loss_fn(
                displacement[100:200].float().view(1, -1),
                position_pred.float().view(1, -1)
            ) + loss_fn(
                velocity[100:200].float().view(1, -1),
                velocity_pred.float().view(1, -1)
            )
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        self.assertTrue(
            all(_next < _prev for _next, _prev in zip(
                losses[1:], losses[:-1])),
            "Expected losses to be monotonically decreasing:\n"
            f"[{', '.join('{:.3f}'.format(loss) for loss in losses)}]"
        )


if __name__ == '__main__':
    unittest.main()
