# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from silk.transforms.abstract import (
    Compose,
    Lambda,
    MethodCall,
    NamedContext,
    ToNamedContext,
)


class _UnitTests(unittest.TestCase):
    def test_named_context_init(self):
        ctx = NamedContext()

        self.assertEqual(len(ctx), 0)

        ctx = NamedContext(a=0, b=1, c=2)

        self.assertEqual(len(ctx), 3)
        ctx.ensure_exists("a", "b", "c")

        ctx = NamedContext({"a": 0, "b": 1, "c": 2})

        self.assertEqual(len(ctx), 3)
        ctx.ensure_exists("a", "b", "c")

    def test_named_context_add(self):
        ctx = NamedContext()
        ctx.add("add.incorrect", 0)
        ctx = ctx.add("add.correct", 1)

        with self.assertRaises(RuntimeError):
            ctx = ctx.add("add.correct", 2)

        ctx = ctx.add("add.correct", 3, allow_exist=True)

        ctx.ensure_exists("add.correct")
        ctx.ensure_not_exists("add.incorrect")

        self.assertEqual(len(ctx), 1)
        self.assertEqual(ctx["add.correct"], 3)

    def test_named_context_remove(self):
        ctx = NamedContext()
        ctx = ctx.add("remove.incorrect", 0)
        ctx = ctx.add("remove.correct", 1)

        with self.assertRaises(RuntimeError):
            ctx = ctx.remove("remove.not.exist")

        ctx.remove("remove.incorrect")
        ctx = ctx.remove("remove.correct")

        ctx.ensure_exists("remove.incorrect")
        ctx.ensure_not_exists("remove.correct")

        self.assertEqual(len(ctx), 1)
        self.assertEqual(ctx["remove.incorrect"], 0)

    def test_named_context_rename(self):
        ctx = NamedContext()
        ctx = ctx.add("name.old", 0)
        ctx = ctx.rename("name.old", "name.new")
        ctx.rename("name.new", "name.new.incorrect")

        ctx.ensure_exists("name.new")
        ctx.ensure_not_exists("name.old", "name.new.incorrect")

        self.assertEqual(len(ctx), 1)
        self.assertEqual(ctx["name.new"], 0)

    def test_named_context_map(self):
        ctx = NamedContext()
        ctx = ctx.add("a", 0)
        ctx = ctx.add("b", 1)
        ctx = ctx.add("c", 2)

        def operator(val, inc=0, mul=1):
            return val * mul + inc

        # incorrect use of map
        ctx.map(operator, +1)

        self.assertEqual(ctx["a"], 0)
        self.assertEqual(ctx["b"], 1)
        self.assertEqual(ctx["c"], 2)

        ctx = ctx.map(operator, +1)

        self.assertEqual(ctx["a"], 1)
        self.assertEqual(ctx["b"], 2)
        self.assertEqual(ctx["c"], 3)

        ctx = ctx.map(operator, inc=-1, mul=+2)

        self.assertEqual(ctx["a"], 1)
        self.assertEqual(ctx["b"], 3)
        self.assertEqual(ctx["c"], 5)

    def test_named_context_map_only(self):
        ctx = NamedContext()
        ctx = ctx.add("a", 0)
        ctx = ctx.add("b", 1)
        ctx = ctx.add("c", 2)

        def operator(val, inc=0, mul=1):
            return val * mul + inc

        # incorrect use of map_only
        ctx.map_only(["a", "b"], operator, +1)

        self.assertEqual(ctx["a"], 0)
        self.assertEqual(ctx["b"], 1)
        self.assertEqual(ctx["c"], 2)

        ctx = ctx.map_only(["a", "b"], operator, +1)

        self.assertEqual(ctx["a"], 1)
        self.assertEqual(ctx["b"], 2)
        self.assertEqual(ctx["c"], 2)

        ctx = ctx.map_only(["b", "c"], operator, inc=-1, mul=+2)

        self.assertEqual(ctx["a"], 1)
        self.assertEqual(ctx["b"], 3)
        self.assertEqual(ctx["c"], 3)

    def test_to_named_context(self):
        transf = ToNamedContext("a", "b", None, "c")
        ctx = transf((0, 1, 2, 3))

        self.assertEqual(len(ctx), 3)
        self.assertEqual(ctx["a"], 0)
        self.assertEqual(ctx["b"], 1)
        self.assertEqual(ctx["c"], 3)

        with self.assertRaises(RuntimeError):
            ctx = transf((0, 1, 2))

        with self.assertRaises(RuntimeError):
            ctx = transf((0, 1, 2, 3, 4))

    def test_lambda_and_compose(self):
        ctx = NamedContext(a=0)

        def operator(val, inc=0, mul=1):
            return val * mul + inc

        transf = Compose(
            Lambda("b", operator, "@a", 1, mul=1),
            Lambda("c", operator, "@b", "@b", mul=1),
            Lambda("d", operator, "@c", inc="@b", mul="@c"),
        )

        ctx = transf(ctx)

        self.assertEqual(len(ctx), 4)
        self.assertEqual(ctx["a"], 0)
        self.assertEqual(ctx["b"], 1)
        self.assertEqual(ctx["c"], 2)
        self.assertEqual(ctx["d"], 5)

    def test_lambda_with_no_name(self):
        ctx = NamedContext(a=13, b=3, c=5)

        def operator(val, inc=0, mul=1):
            return val * mul + inc

        transf = Lambda(None, operator, "@a", "@b", "@c")

        result = transf(ctx)

        self.assertFalse(isinstance(result, NamedContext))
        self.assertEqual(result, 13 * 5 + 3)

    def test_lambda_with_whole_context_as_input(self):
        ctx = NamedContext(a=13, b=3, c=5)

        def operator(ctx, inc=0, mul=1):
            return ctx["a"] * mul + inc

        transf = Lambda("d", operator, "@", "@b", "@c")

        result = transf(ctx)

        self.assertIsInstance(result, NamedContext)
        self.assertEqual(result["d"], 13 * 5 + 3)

    def test_method_call(self):
        class A:
            def method(self, a, b=1):
                return a + b

        ctx = NamedContext(a=13, b=3, obj=A())

        transf_by_func = MethodCall("c", "@obj", A.method, "@a", "@b")
        transf_by_name = MethodCall("c", "@obj", "method", "@a", "@b")

        result = transf_by_func(ctx)

        self.assertIsInstance(result, NamedContext)
        self.assertEqual(result["c"], 13 + 3)

        result = transf_by_name(ctx)

        self.assertIsInstance(result, NamedContext)
        self.assertEqual(result["c"], 13 + 3)


def main():
    unittest.main()


if __name__ == "__main__":
    unittest.main()
