from dataclasses import dataclass
import itertools
import unittest
from typing import Callable, Optional, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neuralop.layers.fno_block import FNOBlocks

from attention import TNOBlock, TnoBlock2d, TNOBlock3D, NO_OP

DEVICE: torch.device = "cuda" if torch.cuda.is_available() else "cpu"


class TNOBlock2DWrapper(nn.Module):
    def __init__(self, tno_block: TnoBlock2d):
        super().__init__()
        self.tno_block = tno_block

    def forward(
        self,
        x: torch.Tensor,
        output_shape: Optional[torch.Size] = None,
    ):
        return self.tno_block.forward(x, output_shape=output_shape)


class TNOBlock3DWrapper(nn.Module):
    def __init__(self, tno_block: TNOBlock3D):
        super().__init__()
        self.tno_block = tno_block

    def forward(
        self,
        x: torch.Tensor,
        output_shape: Optional[torch.Size] = None,
    ):
        return self.tno_block.forward(x, output_shape=output_shape)


@dataclass
class ConvolutionParameters:
    in_channels: int
    out_channels: int
    n_layers: int
    n_modes: tuple[int]
    non_linearity: Callable


class ConvolutionTestCase(unittest.TestCase):
    def assertConvolutionIsLike(
        self,
        expected: ConvolutionParameters,
        fno_block: FNOBlocks,
    ):
        self.assertEqual(expected.in_channels, fno_block.in_channels)
        self.assertEqual(expected.out_channels, fno_block.out_channels)
        self.assertEqual(expected.n_layers, fno_block.n_layers)
        # TODO n_modes (later max_n_modes) _should_ be an immutable tuple
        self.assertEqual(expected.n_modes, tuple(fno_block.n_modes))
        self.assertEqual(expected.non_linearity, fno_block.non_linearity)


class AbstractTNOBlockTest(ConvolutionTestCase):
    """
    TODO :test_init: transformation shapes, parameter shapes
    TODO :test_init: what about when equivariant=False
    """
    tno_block: Optional[TNOBlock] = None
    token_codimension: Optional[int] = None
    n_modes: Optional[tuple[int]] = None
    n_dim: Optional[int] = None
    norm: Optional[Any] = None

    def setUp(self):
        raise NotImplementedError

    @torch.no_grad()
    def _check_initialization(self, n_heads: int, kqv_non_linear: bool):
        self.assertEqual(self.tno_block.variable_codimension, self.token_codimension)
        self.assertEqual(self.tno_block.token_codimension, self.token_codimension)
        expected_head_codimension = max(self.token_codimension // n_heads, 1)
        self.assertEqual(self.tno_block.head_codimension, expected_head_codimension)
        self.assertEqual(self.tno_block.mixer_token_codimension, self.token_codimension)

        # The alternative to importing NO_OP below is to take each testable
        # convolution and check whether it does pointwise activation on its
        # output. Do things quick and dirty now:

        expected_out_channels = n_heads * expected_head_codimension
        expected_activation = F.gelu  # default
        kqv_activation = expected_activation if kqv_non_linear else NO_OP
        assert self.n_modes is not None  # satisfy type-checker

        kqv_args = ConvolutionParameters(
            in_channels=self.token_codimension,
            out_channels=expected_out_channels,
            n_layers=1,  # hard-coded
            n_modes=self.n_modes,
            non_linearity=kqv_activation,
        )
        self.assertConvolutionIsLike(kqv_args, self.tno_block.K)
        self.assertConvolutionIsLike(kqv_args, self.tno_block.Q)
        self.assertConvolutionIsLike(kqv_args, self.tno_block.V)
        if expected_out_channels == self.token_codimension:
            self.assertIsNone(self.tno_block.proj)
        else:
            self.assertConvolutionIsLike(
                ConvolutionParameters(
                    in_channels=expected_out_channels,
                    out_channels=self.token_codimension,
                    n_layers=1,  # hard-coded
                    n_modes=self.n_modes,
                    non_linearity=kqv_activation,
                ),
                self.tno_block.proj,
            )
        self.assertConvolutionIsLike(
            ConvolutionParameters(
                in_channels=self.token_codimension,
                out_channels=self.token_codimension,
                n_layers=2,  # hard-coded
                n_modes=self.n_modes,
                non_linearity=expected_activation,
            ),
            self.tno_block.mixer,
        )

        # Default ``nn.InsanceNorm_nD`` with attributes ``norm.n_features``
        # and ``norm.affine`` below:
        for norm in (
            self.tno_block.attention_normalizer,
            self.tno_block.norm1,
            self.tno_block.norm2,
            self.tno_block.mixer_out_normalizer,
        ):
            self.assertTrue(isinstance(norm, self.norm)),
            self.assertEqual(self.token_codimension, norm.num_features)
            self.assertTrue(norm.affine)


class TNOBlock2DTest(AbstractTNOBlockTest):
    """
    TODO
        - tests w/out permutation equivariance
        - test projection for if `n_heads % token_codim != 0`

    Optionally, to test intermediate shapes [ex. in ``forward()``]:
      *  add a ``verbose`` arg,
      *  gate intermediate shape logging behind ``verbose``
      *  capture logs to peek intermediate shape states
    """
    n_dim: int = 2
    norm = nn.InstanceNorm2d
    token_codimension: int
    n_modes: tuple[int, ...]
    tno_block: TnoBlock2d

    def setUp(self) -> None:
        self.token_codimension = 4
        self.n_modes = (8,) * TNOBlock2DTest.n_dim

        self.tno_block = TnoBlock2d(
            n_modes=self.n_modes,
            token_codimension=self.token_codimension,
            permutation_eq=True,
            per_channel_attention=False,
            n_head=2,
        )

    @torch.no_grad()
    def test_initialization(self):
        self.token_codimension = 8
        for n_heads, non_linear in itertools.product(
            (1, 2, 3, 4, 6, 8, 12, 16),
            (True, False),
        ):
            with self.subTest(split=f"{n_heads=};{non_linear=}"):
                self.tno_block = TnoBlock2d(
                    n_modes=self.n_modes,
                    token_codimension=self.token_codimension,
                    permutation_eq=True,
                    per_channel_attention=False,
                    n_head=n_heads,
                    kqv_non_linear=non_linear,
                )
                self._check_initialization(n_heads, non_linear)

    # OP * add a ``verbose`` arg to ``compute_attention()``,
    #    * gate intermediate shape logging behind ``verbose``
    #    * capture logs to peek intermediate shape states
    # This would also exercise that logging is working as expected.
    @torch.no_grad()
    def test_compute_attention(self):
        batch_size = 2
        length = 32
        xa = torch.randn(torch.Size([
            batch_size * 4,  # batch-size x variable count
            self.token_codimension,
            length,  # height
            length,  # width
        ]))

        ya = self.tno_block.compute_attention(xa, batch_size)
        self.assertEqual(
            (batch_size * 4, self.token_codimension, length, length),
            tuple(ya.shape),
        )

    # OP * add a ``verbose`` arg to ``forward_propagation()``,
    #    * gate intermediate shape logging behind ``verbose``
    #    * capture logs to peek intermediate shape states
    # This would also exercise that logging is working as expected.
    @torch.no_grad()
    def test_forward_propagation(self):
        batch_size = 2
        length = 32
        xa = torch.randn(torch.Size([
            batch_size,
            self.token_codimension * 4,  # tokens x variable count
            length,  # height
            length,  # width
        ]))

        ya = self.tno_block(xa, batch_size)  # forward pass
        self.assertEqual(
            (batch_size, self.token_codimension * 4, length, length),
            tuple(ya.shape),
        )

    def test_backwards_propagation(self):
        self.token_codimension = 3
        self.tno_block = TnoBlock2d(
            n_modes=self.n_modes,
            token_codimension=self.token_codimension,
            permutation_eq=True,
            per_channel_attention=False,
        )
        tno_block_w = TNOBlock2DWrapper(self.tno_block)

        resolution = 0.02
        x = np.arange(-5, 5, resolution, dtype=np.float32)
        y = np.arange(-5, 5, resolution, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, sparse=True)

        displacement_in = torch.tensor(
            np.sin(xx * np.sqrt(2.0)) + np.sin(yy * np.sqrt(3.0)),
            device=DEVICE,
        )
        # These tensors need to be expanded because of the sparse ``numpy``
        # representation from ``meshgrid``
        displacement_code1 = torch\
            .tensor(np.sin(xx), device=DEVICE)\
            .expand(displacement_in.shape)
        displacement_code2 = torch\
            .tensor(np.sin(yy), device=DEVICE)\
            .expand(displacement_in.shape)

        velocity_in = torch.tensor(
            np.sqrt(2.0) * np.cos(xx * np.sqrt(2.0)) +
            np.sqrt(3.0) * np.cos(yy * np.sqrt(3.0)),
            device=DEVICE,
        )
        # These tensors need to be expanded because of the sparse ``numpy``
        # representation from ``meshgrid``
        velocity_code1 = torch\
            .tensor(np.sin(xx + yy), device=DEVICE)\
            .expand(velocity_in.shape)
        velocity_code2 = torch\
            .tensor(np.sin(yy - xx), device=DEVICE)\
            .expand(velocity_in.shape)

        # Learn to advance the plane wave one quarter phase forward:
        phase = np.pi / 4.0
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
        learning_rate = 0.01
        optimizer = torch.optim.Adam(
            tno_block_w.parameters(),
            lr=learning_rate,
            weight_decay=learning_rate / 10.0,
        )

        losses = []
        for _ in range(10):
            optimizer.zero_grad()

            [pred] = tno_block_w.forward(
                torch.concatenate([
                    displacement_in.unsqueeze(0),
                    displacement_code1.unsqueeze(0),
                    displacement_code2.unsqueeze(0),
                    velocity_in.unsqueeze(0),
                    velocity_code1.unsqueeze(0),
                    velocity_code2.unsqueeze(0),
                ]).unsqueeze(0)
            )
            displacement_pred = pred[0:3].sum(axis=0)
            velocity_pred = pred[3:6].sum(axis=0)

            loss = loss_fn(
                displacement_out.float().view(1, -1),
                displacement_pred.float().view(1, -1)
            ) + loss_fn(
                velocity_out.float().view(1, -1),
                velocity_pred.float().view(1, -1)
            )
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        self.assertTrue(
            all(_next < _prev for _next, _prev in zip(losses[1:], losses[:-1])),
            "Expected losses to be monotonically decreasing:\n"
            f"[{', '.join('{:.3f}'.format(loss) for loss in losses)}]"
        )


class TNOBlock3DTest(AbstractTNOBlockTest):
    """
    TODO tests w/out permutation equivariance
    TODO test projection layer when `n_heads % token_codim != 0`

    Option for testing intermediate shapes (ex. in ``forward()``):
      *  add a ``verbose`` arg,
      *  gate intermediate shape logging behind ``verbose``
      *  capture logs to peek intermediate shape states
    """
    n_dim = 3
    norm = nn.InstanceNorm3d
    token_codimension: int
    n_modes: tuple[int, ...]
    tno_block: TNOBlock3D

    def setUp(self) -> None:
        self.token_codimension = 4
        self.n_modes = (8,) * TNOBlock3DTest.n_dim

        self.tno_block = TNOBlock3D(
            n_modes=self.n_modes,
            token_codimension=self.token_codimension,
            permutation_eq=True,
            per_channel_attention=False,
        )

    @torch.no_grad()
    def test_initialization(self):
        self.token_codimension = 4
        for n_heads, non_linear in itertools.product(
            (1, 2, 3, 4, 6, 8),
            (True, False),
        ):
            with self.subTest(split=f"{n_heads=};{non_linear=}"):
                self.tno_block = TNOBlock3D(
                    n_modes=self.n_modes,
                    token_codimension=self.token_codimension,
                    permutation_eq=True,
                    per_channel_attention=False,
                    n_head=n_heads,
                    kqv_non_linear=non_linear,
                )
                self._check_initialization(n_heads, non_linear)

    # OP * add a ``verbose`` arg to ``compute_attention()``,
    #    * gate intermediate shape logging behind ``verbose``
    #    * capture logs to peek intermediate shape states
    # This would also exercise that logging is working as expected.
    @torch.no_grad()
    def test_compute_attention(self):
        batch_size = 2
        length = 32
        xa = torch.randn(torch.Size([
            batch_size * 4,  # batch-size x variable count
            self.token_codimension,
            length // 4,  # duration
            length,  # height
            length,  # width
        ]))

        ya = self.tno_block.compute_attention(xa, batch_size)
        self.assertEqual(
            (
                batch_size * 4,
                self.token_codimension,
                length // 4,
                length,
                length,
            ),
            tuple(ya.shape),
        )

    # OP * add a ``verbose`` arg to ``forward_propagation()``,
    #    * gate intermediate shape logging behind ``verbose``
    #    * capture logs to peek intermediate shape states
    # This would also exercise that logging is working as expected.
    @torch.no_grad()
    def test_forward_propagation(self):
        batch_size = 2
        length = 32
        xa = torch.randn(torch.Size([
            batch_size,
            self.token_codimension * 4,  # tokens x variable count
            length // 4,  # duration
            length,  # height
            length,  # width
        ]))

        ya = self.tno_block(xa, batch_size)  # forward pass
        self.assertEqual(
            (
                batch_size,
                self.token_codimension * 4,
                length // 4,
                length,
                length,
            ),
            tuple(ya.shape),
        )

    def test_backwards_propagation(self):
        self.token_codimension = 4
        self.tno_block = TNOBlock3D(
            n_modes=self.n_modes,
            token_codimension=self.token_codimension,
            permutation_eq=True,
            per_channel_attention=False,
            n_head=2,
        )
        tno_block_w = TNOBlock3DWrapper(self.tno_block)

        resolution = 0.05
        x = np.arange(-5.0, 5.0, resolution, dtype=np.float32)
        y = np.arange(-5.0, 5.0, resolution, dtype=np.float32)
        t = np.arange(0.0, 5.0, resolution, dtype=np.float32)
        tt, xx, yy = np.meshgrid(t, x, y, indexing='ij', sparse=True)

        displacement = torch.tensor(
            np.sin(xx * np.sqrt(2.0) + tt * np.sqrt(5.0)) +
            np.sin(yy * np.sqrt(3.0) + tt * np.sqrt(7.0)),
            device=DEVICE,
        )
        # displacement_in = torch.tensor(
        #     np.sin(xx * np.sqrt(2.0)) + np.sin(yy * np.sqrt(3.0)),
        #     device=DEVICE,
        # )
        # These tensors need to be expanded because of the sparse ``numpy``
        # representation from ``meshgrid``
        displacement_code1 = torch\
            .tensor(np.sin(xx), device=DEVICE)\
            .expand(displacement.shape)
        displacement_code2 = torch\
            .tensor(np.sin(yy), device=DEVICE)\
            .expand(displacement.shape)
        displacement_code3 = torch\
            .tensor(np.sin(tt), device=DEVICE)\
            .expand(displacement.shape)

        # velocity_in = torch.tensor(
        #     np.sqrt(2.0) * np.cos(xx * np.sqrt(2.0)) +
        #     np.sqrt(3.0) * np.cos(yy * np.sqrt(3.0)),
        #     device=DEVICE,
        # )
        velocity = torch.tensor(
            np.sqrt(5.0) * np.cos(xx * np.sqrt(2.0) * tt * np.sqrt(5.0)) +
            np.sqrt(7.0) * np.cos(yy * np.sqrt(3.0) * tt * np.sqrt(7.0)),
            device=DEVICE,
        )
        # These tensors need to be expanded because of the sparse ``numpy``
        # representation from ``meshgrid``
        velocity_code1 = torch\
            .tensor(np.sin(xx + yy), device=DEVICE)\
            .expand(velocity.shape)
        velocity_code2 = torch\
            .tensor(np.sin(yy + tt), device=DEVICE)\
            .expand(velocity.shape)
        velocity_code3 = torch\
            .tensor(np.sin(tt + xx), device=DEVICE)\
            .expand(velocity.shape)

        loss_fn = torch.nn.MSELoss()
        learning_rate = 0.01
        optimizer = torch.optim.Adam(
            tno_block_w.parameters(),
            lr=learning_rate,
            weight_decay=learning_rate / 10.0,
        )

        losses = []
        for _ in range(10):
            optimizer.zero_grad()

            t0 = 0
            # Take the first 2.0 seconds as input ...
            t1 = int(2.0 / resolution)
            # ... and learn the next 2.0 seconds as output:
            # (i.e. 2.0 <= t < 4.0)
            t2 = int(4.0 / resolution)

            [pred] = tno_block_w.forward(
                torch.concatenate([
                    displacement[t0:t1].unsqueeze(0),
                    displacement_code1[t0:t1].unsqueeze(0),
                    displacement_code2[t0:t1].unsqueeze(0),
                    displacement_code3[t0:t1].unsqueeze(0),
                    velocity[t0:t1].unsqueeze(0),
                    velocity_code1[t0:t1].unsqueeze(0),
                    velocity_code2[t0:t1].unsqueeze(0),
                    velocity_code3[t0:t1].unsqueeze(0),
                ]).unsqueeze(0)
            )
            s1 = self.token_codimension
            s2 = self.token_codimension * 2
            displacement_pred = pred[0:s1].sum(axis=0)
            velocity_pred = pred[s1:s2].sum(axis=0)

            loss = loss_fn(
                displacement[t1:t2].float().view(1, -1),
                displacement_pred.float().view(1, -1)
            ) + loss_fn(
                velocity[t1:t2].float().view(1, -1),
                velocity_pred.float().view(1, -1)
            )
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        self.assertTrue(
            all(_next < _prev for _next, _prev in zip(losses[1:], losses[:-1])),
            "Expected losses to be monotonically decreasing:\n"
            f"[{', '.join('{:.3f}'.format(loss) for loss in losses)}]"
        )


if __name__ == '__main__':
    unittest.main()
