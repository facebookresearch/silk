# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

from silk.flow import ConditionalReturn, Flow


class _MockFlow:
    def __init__(self) -> None:
        self.flow = Flow("x", "y")

        self.flow.define_transition("b", self.i_to_b, "x")
        self.flow.define_transition("a", self.b_to_a, "b", w="y")
        self.flow.define_transition(("ta", "tb"), self.tuple_a_and_b, "a", "b")
        self.flow.define_transition("z", operator.add, "ta", Flow.Constant(-120))

    def b_to_a(self, b, w):
        return w * b + 1

    def i_to_b(self, x, t=3):
        return x * 2 + t

    def tuple_a_and_b(self, a, b):
        return (a, b)

    def __call__(self, outputs, *args, **kwargs):
        return self.flow(outputs, *args, **kwargs)


class _UnitTests(unittest.TestCase):
    @staticmethod
    def _make_mock_flow():
        return _MockFlow()

    def test_flow(self):
        flow = _UnitTests._make_mock_flow()

        print(flow("a", 20, 1))
        print(flow("a", 20, 2))

        print(flow("b", x=20))
        print(flow("b", x=20, y=2))

        print(flow(("b", "a"), x=20, y=1))
        print(flow(("b", "a"), x=20, y=2))
        print(flow("ta", x=20, y=2))
        print(flow("tb", x=20, y=2))

        print(flow(("x", "y"), 20, 1))
        print(flow(("x", "y"), x=20, y=2))

        fn = flow.flow.with_outputs(("tb", "ta"))
        print(fn(x=20, y=1))
        print(fn(20, y=2))

        fn = flow.flow.with_outputs("z")
        print(fn(x=20, y=1))
        print(fn(20, y=2))

        print(flow.flow.tape_as_pseudocode(fn._tape))

    def test_conditional_return(self):
        def fn(a, b, c, d, vars):
            cr = ConditionalReturn(
                vars,
                {"a0", "b0", "c0", "d0", "e0"},
                from_locals=True,
            )

            a0 = a

            cr.gather()
            if cr.should_return():
                return cr.return_value()

            a0 = None
            b0 = b

            if cr.gathered():
                return cr.return_value()

            del a0
            del b0

            if cr.gathered(e0=(a, b, c, d)):
                return cr.return_value()

            c0 = c  # noqa: F841
            d0 = d  # noqa: F841

            return cr.return_value()

        self.assertEqual(fn(a=1, b=2, c=3, d=4, vars="a0"), 1)
        self.assertEqual(fn(a=1, b=2, c=3, d=4, vars=["a0"]), (1,))
        self.assertEqual(fn(a=1, b=2, c=3, d=4, vars=["a0", "b0"]), (1, 2))
        self.assertEqual(fn(a=1, b=2, c=3, d=4, vars=["a0", "b0", "c0"]), (1, 2, 3))
        self.assertEqual(
            fn(a=1, b=2, c=3, d=4, vars=["a0", "b0", "c0", "d0"]), (1, 2, 3, 4)
        )
        self.assertEqual(
            fn(a=1, b=2, c=3, d=4, vars=["a0", "b0", "c0", "d0", "e0"]),
            (1, 2, 3, 4, (1, 2, 3, 4)),
        )

        self.assertEqual(
            fn(a=1, b=2, c=3, d=4, vars=["c0", "b0", "a0", "d0"]), (3, 2, 1, 4)
        )


if __name__ == "__main__":
    unittest.main()
