import unittest

import torch

from models.codano import Projection, ProjectionT


def permute(a: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Permutes the first and last (of 2) tokens.

    Note that this returns a new Tensor and leaves the argument unchanged.
    This function is also its own inverse. Ideally,
    ```
    x = permute(permute(x))
    ```

    Each token must consist of ``n_channels`` channels.
    """
    b = torch.empty_like(a)
    # Set the last token of `a` as the first token of `b`,
    b[:n_channels] = a[n_channels:]
    # and set the first token of `a` as the last token of `b`.
    b[n_channels:] = a[:n_channels]
    return b


class ProjectionTest(unittest.TestCase):
    def test_initialization(self):
        # Tests default when `hidden_channels=None`
        in_channels = 4
        out_channels = 8
        projection1 = Projection(
            in_channels,
            out_channels,
            permutation_invariant=False,
        )
        self.assertEqual(in_channels, projection1.fc1.in_channels)
        self.assertEqual(in_channels, projection1.fc1.out_channels)  # hidden_channels
        self.assertEqual(in_channels, projection1.norm.num_features)  # hidden_channels
        self.assertEqual(in_channels, projection1.fc2.in_channels)  # hidden_channels
        self.assertEqual(out_channels, projection1.fc2.out_channels)
        self.assertFalse(projection1.permutation_invariant)

        hidden_channels = 12
        projection2 = Projection(
            in_channels,
            out_channels,
            hidden_channels=hidden_channels,
            permutation_invariant=True,
        )
        self.assertEqual(in_channels, projection2.fc1.in_channels)
        self.assertEqual(hidden_channels, projection2.fc1.out_channels)
        self.assertEqual(hidden_channels, projection2.norm.num_features)
        self.assertEqual(hidden_channels, projection2.fc2.in_channels)
        self.assertEqual(out_channels, projection2.fc2.out_channels)
        self.assertTrue(projection2.permutation_invariant)

    @torch.no_grad()
    def test_forward(self):
        in_channels = 4
        out_channels = 8
        projection = Projection(
            in_channels,
            out_channels,
            hidden_channels=out_channels,
            permutation_invariant=False,
        )

        length = 16
        x = torch.randn([in_channels, length, length])
        y = projection(x.unsqueeze(0))

        self.assertEqual((1, out_channels, length, length), tuple(y.shape))

    @torch.no_grad()
    def test_forward_permutation_equivariant(self):
        in_channels = 4
        out_channels = 8
        projection = Projection(
            in_channels,
            out_channels,
            hidden_channels=out_channels,
            permutation_invariant=True,
        )

        length = 16
        x = torch.randn([2 * in_channels, length, length])
        y = projection(x.unsqueeze(0))

        permuted_y = projection(permute(x, in_channels).unsqueeze(0))

        torch.testing.assert_close(
            y.squeeze(0), permute(permuted_y.squeeze(0), out_channels),
            atol=1.0e-9,
            rtol=1.0e-9,
        )

class ProjectionTTest(unittest.TestCase):
    def test_initialization(self):
        # Tests default when `hidden_channels=None`
        in_channels = 4
        out_channels = 8
        projection1 = ProjectionT(
            in_channels,
            out_channels,
            permutation_invariant=False,
        )
        self.assertEqual(in_channels, projection1.fc1.in_channels)
        self.assertEqual(in_channels, projection1.fc1.out_channels)  # hidden_channels
        self.assertEqual(in_channels, projection1.norm.num_features)  # hidden_channels
        self.assertEqual(in_channels, projection1.fc2.in_channels)  # hidden_channels
        self.assertEqual(out_channels, projection1.fc2.out_channels)
        self.assertFalse(projection1.permutation_invariant)

        hidden_channels = 12
        projection2 = ProjectionT(
            in_channels,
            out_channels,
            hidden_channels=hidden_channels,
            permutation_invariant=True,
        )
        self.assertEqual(in_channels, projection2.fc1.in_channels)
        self.assertEqual(hidden_channels, projection2.fc1.out_channels)
        self.assertEqual(hidden_channels, projection2.norm.num_features)
        self.assertEqual(hidden_channels, projection2.fc2.in_channels)
        self.assertEqual(out_channels, projection2.fc2.out_channels)
        self.assertTrue(projection2.permutation_invariant)

    @torch.no_grad()
    def test_forward(self):
        in_channels = 2
        out_channels = 4
        projection = ProjectionT(
            in_channels,
            out_channels,
            hidden_channels=out_channels,
            permutation_invariant=False,
        )

        length = 8
        x = torch.randn([in_channels, length, length, length])
        y = projection(x.unsqueeze(0))

        self.assertEqual(
            (out_channels, length, length, length),
            tuple(y.squeeze(0).shape),
        )

    @torch.no_grad()
    def test_forward_permutation_equivariant(self):
        in_channels = 2
        out_channels = 4
        projection = ProjectionT(
            in_channels,
            out_channels,
            hidden_channels=out_channels,
            permutation_invariant=True,
        )

        length = 8
        x = torch.randn([2 * in_channels, length, length, length])
        y = projection(x.unsqueeze(0))
        permuted_y = projection(permute(x, in_channels).unsqueeze(0))

        torch.testing.assert_close(
            y.squeeze(0), permute(permuted_y.squeeze(0), out_channels),
            atol=1.0e-9,
            rtol=1.0e-9,
        )

if __name__ == '__main__':
    unittest.main()
