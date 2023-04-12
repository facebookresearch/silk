# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import asdict, dataclass
from typing import Dict, List, Union

from silk.config.formatter import get_formatter


@dataclass
class _MockClass:
    a: str
    b: Dict[str, Union[int, str]]
    c: List[Union[int, str]]


class _UnitTests(unittest.TestCase):
    def _check_formatter(self, obj, name, expected_output):
        formatter = get_formatter(name)

        # should return the same result
        output_0 = formatter(obj)
        output_1 = formatter(obj)
        self.assertEqual(output_0, output_1)

        # should be equal to the expected output
        self.assertEqual(output_0, expected_output)

    def test_get_formatter(self):
        obj = _MockClass(
            a=0,
            b={"a": 2, "b": "b", "c": 3},
            c=1,
        )

        self._check_formatter(obj, "none", None)
        self._check_formatter(
            asdict(obj), "json", '{"a": 0, "b": {"a": 2, "b": "b", "c": 3}, "c": 1}'
        )
        self._check_formatter(
            asdict(obj),
            "yaml",
            """a: 0
b:
  a: 2
  b: b
  c: 3
c: 1
""",
        )
        self._check_formatter(obj, "python", str(obj))


if __name__ == "__main__":
    unittest.main()
