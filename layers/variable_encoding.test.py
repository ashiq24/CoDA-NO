import unittest

import torch
import torch_harmonics as th

from variable_encoding import VariableEncoding2d

class VariableEncoding2DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_channels = 4
        self.modes = (16, 16)

    @torch.no_grad()
    def test_forward_sht(self):
        """Uses the old, stored transformation or a new one."""
        self.variable_encoding = VariableEncoding2d(
            self.n_channels,
            self.modes,
            basis='sht',
        )

        # Uses the old transform:
        u = torch.randn(self.modes, dtype=torch.float32)
        v = self.variable_encoding.forward(u)
        self.assertEqual(torch.Size([self.n_channels, *self.modes]), v.shape)

        # Uses the new transform with the new size:
        big_modes = tuple([2 * m for m in self.modes])
        u = torch.randn(big_modes, dtype=torch.float32)
        v = self.variable_encoding.forward(u)
        self.assertEqual(torch.Size([self.n_channels, *big_modes]), v.shape)

    @torch.no_grad()
    def test_forward_fft(self):
        self.variable_encoding = VariableEncoding2d(
            self.n_channels,
            self.modes,
            basis='fft',
        )

        u = torch.randn(self.modes, dtype=torch.float32)
        v = self.variable_encoding.forward(u)
        self.assertEqual(torch.Size([self.n_channels, *self.modes]), v.shape)

        big_modes = tuple([2 * m for m in self.modes])
        u = torch.randn(big_modes, dtype=torch.float32)
        v = self.variable_encoding.forward(u)
        self.assertEqual(torch.Size([self.n_channels, *big_modes]), v.shape)


if __name__ == '__main__':
    unittest.main()
