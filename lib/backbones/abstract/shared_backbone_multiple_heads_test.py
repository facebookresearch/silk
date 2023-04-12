# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from silk.backbones.abstract.shared_backbone_multiple_heads import (
    SharedBackboneMultipleHeads,
)
from silk.backbones.silk.coords import function_coordinate_mapping_provider


class _UnitTests(unittest.TestCase):
    def test_shared_backbone_multiple_heads(self):
        @function_coordinate_mapping_provider()
        def backbone(x):
            return 2 * x

        @function_coordinate_mapping_provider()
        def head_a(x):
            return x + 0

        @function_coordinate_mapping_provider()
        def head_b(x):
            return x + 1

        @function_coordinate_mapping_provider()
        def head_c(x):
            return x + 2

        model = SharedBackboneMultipleHeads(
            backbone=backbone, input_name="number", backbone_output_name="double_number"
        )
        model.add_heads(a=head_a)

        # check properties
        self.assertEqual(model.input_name, "number")
        self.assertEqual(model.backbone_output_name, "double_number")
        self.assertEqual(model.head_names, ("a",))

        # check first head computation
        self.assertEqual(model.forward_flow("a", 20), 40)

        # add additional heads
        model.add_heads(b=head_b)
        model.add_heads(c=head_c)

        # check exposed heads
        self.assertEqual(model.head_names, ("a", "b", "c"))

        # check head computation
        self.assertEqual(
            model.forward_flow(("a", "b", "c", "double_number", "number"), 10),
            (20, 21, 22, 20, 10),
        )


if __name__ == "__main__":
    unittest.main()
