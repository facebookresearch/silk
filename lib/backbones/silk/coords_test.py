# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from silk.backbones.silk.coords import Conv2dCoordinateMapping


class _UnitTests(unittest.TestCase):
    def test_conv2d_mapping(self):
        # "same" padding
        mapping = Conv2dCoordinateMapping(kernel_size=3, stride=1, padding=1)

        self.assertEqual(mapping.reverse(0.5), 0.5)
        self.assertEqual(mapping.reverse(1.5), 1.5)

        mapping = Conv2dCoordinateMapping(kernel_size=7, stride=1, padding=3)

        self.assertEqual(mapping.reverse(0.5), 0.5)
        self.assertEqual(mapping.reverse(1.5), 1.5)

        # "valid" padding
        mapping = Conv2dCoordinateMapping(kernel_size=3, stride=1, padding=0)

        self.assertEqual(mapping.reverse(0.5), 1.5)
        self.assertEqual(mapping.reverse(1.5), 2.5)

        mapping = Conv2dCoordinateMapping(kernel_size=7, stride=1, padding=0)

        self.assertEqual(mapping.reverse(0.5), 3.5)
        self.assertEqual(mapping.reverse(1.5), 4.5)

        # stride 2, "same" padding
        mapping = Conv2dCoordinateMapping(kernel_size=3, stride=2, padding=1)

        self.assertEqual(mapping.reverse(0.5), 0.5)
        self.assertEqual(mapping.reverse(1.5), 2.5)

        mapping = Conv2dCoordinateMapping(kernel_size=7, stride=2, padding=3)

        self.assertEqual(mapping.reverse(0.5), 0.5)
        self.assertEqual(mapping.reverse(1.5), 2.5)

        # stride 2, "valid" padding
        mapping = Conv2dCoordinateMapping(kernel_size=3, stride=2, padding=0)

        self.assertEqual(mapping.reverse(0.5), 1.5)
        self.assertEqual(mapping.reverse(1.5), 3.5)

        mapping = Conv2dCoordinateMapping(kernel_size=7, stride=2, padding=0)

        self.assertEqual(mapping.reverse(0.5), 3.5)
        self.assertEqual(mapping.reverse(1.5), 5.5)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
