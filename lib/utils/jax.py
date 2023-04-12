# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# source : https://github.com/lucidrains/jax2torch/blob/main/jax2torch/jax2torch.py
# modified `vjp` use for 2x speed-up on large inputs

import os
from functools import wraps
from inspect import signature

import jax
import jax.numpy as jnp
import torch
from jax import dlpack as jax_dlpack, tree_flatten, tree_unflatten
from jax.tree_util import tree_map
from torch.utils import dlpack as torch_dlpack

# To avoid having jax taking all the VRAM
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def j2t(x_jax, device=None):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    if device:
        x_torch = x_torch.to(device)
    return x_torch


def t2j(x_torch, device=None):
    x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
    if device:
        x_torch = x_torch.to(device)
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax


def tree_t2j(x_torch, device=None):
    return tree_map(
        lambda t: t2j(t, device=device) if isinstance(t, torch.Tensor) else t,
        x_torch,
    )


def tree_j2t(x_jax, device=None):
    return tree_map(
        lambda t: j2t(t, device=device) if isinstance(t, jnp.ndarray) else t,
        x_jax,
    )


def tree_get_devices(x_torch):
    return tree_map(
        lambda t: t.device if isinstance(t, torch.Tensor) else None, x_torch
    )


def find_unique_device(devices):
    device = None
    flat_d, _ = tree_flatten(devices)
    for d in flat_d:
        if d:
            if device is None:
                device = d
            else:
                if d != device:
                    raise RuntimeError(
                        f"there should be a unique device in the pytree, found {device} and {d}"
                    )
    return device


def tree_set_devices(x_torch, devices):
    flat_x, tree_x = tree_flatten(x_torch)
    flat_d, tree_d = tree_flatten(devices)

    assert tree_d == tree_x

    flat_r = [
        x.to(d) if isinstance(x, torch.Tensor) else x for x, d in zip(flat_x, flat_d)
    ]

    return tree_unflatten(tree_x, flat_r)


def jax2torch(fn, backward_pass=True):
    class JaxFun(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            args, jax_device, torch_device = args[:-2], args[-2], args[-1]

            if torch_device is None:
                torch_device = find_unique_device(tree_get_devices(args))

            args = tree_t2j(args, device=jax_device)

            if backward_pass:
                y_, ctx.fun_vjp = jax.vjp(fn, *args)
                ctx.jax_device = jax_device
            else:
                y_ = fn(*args)

            return tree_j2t(y_, device=torch_device)

        @staticmethod
        def backward(ctx, *grad_args):
            if not backward_pass:
                return (None,) * (len(grad_args) + 2)

            device = find_unique_device(tree_get_devices(grad_args))
            grad_args = (
                tree_t2j(grad_args, device=ctx.jax_device)
                if len(grad_args) > 1
                else t2j(grad_args[0])
            )

            grads = ctx.fun_vjp(grad_args)

            grads = tuple(t if isinstance(t, jnp.ndarray) else None for t in grads) + (
                None,
                None,
            )

            return tree_j2t(grads, device=device)

    @wraps(fn)
    def inner(*args, jax_device=None, torch_device=None, **kwargs):
        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        return JaxFun.apply(
            *(tuple(bound.arguments.values()) + (jax_device, torch_device))
        )

    return inner


def delayed_vjp(fun):
    @jax.custom_vjp
    def new_fun(*args):
        return fun(*args)

    def fun_fwd(*args):
        return new_fun(*args), args

    def fun_bwd(args, g):
        _, vjp = jax.vjp(fun, *args)
        return vjp(g)

    new_fun.defvjp(fun_fwd, fun_bwd)

    return new_fun
