"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Static class for the negative operation. Used to group helper static methods for forward and backward passes of the negative operation."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs forward pass of the negative operation.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1 : input tensor

        Returns:
        -------
            Tensor: Result of the negative operation.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs backward pass of the negative operation.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Static class for the inverse operation. Used to group helper static methods for forward and backward passes of the inverse operation."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs forward pass of the inverse operation.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1 : input tensor

        Returns:
        -------
            Tensor: Result of the inverse operation.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs backward pass of the inverse operation.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Static class for the addition operation. Used to group helper static methods for forward and backward passes of the addition operation."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs forward pass of the addition operation.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1 : first input tensor
            t2 : second input tensor

        Returns:
        -------
            Tensor: Result of the addition operation.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs backward pass of the addition operation.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to inputs t1 and t2.

        """
        return grad_output, grad_output


class All(Function):
    """Static class for the all operation. Used to group helper static methods for forward and backward passes of the all operation."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# Task 2.3.
class Mul(Function):
    """Static class for element-wise multiplication of tensors. Used to group helper static methods for forward and backward passes of element-wise multiplication operation."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs forward pass of element-wise multiplication.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Result of element-wise multiplication.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs backward pass of element-wise multiplication.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to inputs t1 and t2.

        """
        (t1, t2) = ctx.saved_values
        return t2.f.mul_zip(t2, grad_output), t1.f.mul_zip(t1, grad_output)


class Sigmoid(Function):
    """Static class for sigmoid activation function. Used to group helper static methods for forward and backward passes of the sigmoid activation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs forward pass of sigmoid activation function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Result of sigmoid activation function.

        """
        sigmoid_t1 = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sigmoid_t1)
        return sigmoid_t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs backward pass of sigmoid activation function.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (sigmoid_t1,) = ctx.saved_values
        return grad_output.f.mul_zip(
            grad_output,
            sigmoid_t1.f.mul_zip(
                sigmoid_t1,
                sigmoid_t1.f.add_zip(tensor([1.0]), sigmoid_t1.f.neg_map(sigmoid_t1)),
            ),
        )


class ReLU(Function):
    """Static class for ReLU activation function. Used to group helper static methods for forward and backward passes of the ReLU activation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs forward pass of ReLU activation function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Result of ReLU activation function.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs backward pass of ReLU activation function.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (t1,) = ctx.saved_values
        return t1.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """Static class for natural logarithm function. Used to group helper static methods for forward and backward passes of the natural logarithm function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs forward pass of natural logarithm function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Result of natural logarithm function.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs backward pass of natural logarithm function.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (t1,) = ctx.saved_values
        return t1.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """Static class for exponential function. Used to group helper static methods for forward and backward passes of the exponential function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs forward pass of exponential function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: Input tensor.

        Returns:
        -------
            Tensor: Result of exponential function.

        """
        exp_t1 = t1.f.exp_map(t1)
        ctx.save_for_backward(exp_t1)
        return exp_t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs backward pass of exponential function.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (exp_t1,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, exp_t1)


class Sum(Function):
    """Static class for sum reduction function. Used to group helper static methods for forward and backward passes of the sum reduction function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Performs forward pass of sum reduction function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: Input tensor.
            dim: Dimension along which to reduce.

        Returns:
        -------
            Tensor: Result of sum reduction function.

        """
        ctx.save_for_backward(t1.shape, dim)
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Performs backward pass of sum reduction function.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, float]: Gradients with respect to inputs t1 and dim.

        """
        shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    """Static class for less than comparison function. Used to group helper static methods for forward and backward passes of the less than comparison function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs forward pass of less than comparison function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Result of less than comparison function.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs backward pass of less than comparison function.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to inputs t1 and t2.

        """
        (t1_shape, t2_shape) = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class EQ(Function):
    """Static class for equality comparison function. Used to group helper static methods for forward and backward passes of the equality comparison function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs forward pass of equality comparison function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Result of equality comparison function.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs backward pass of equality comparison function.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to inputs t1 and t2.

        """
        (t1_shape, t2_shape) = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class IsClose(Function):
    """Static class for close comparison function. Used to group helper static methods for forward and backward passes of the close comparison function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs forward pass of close comparison function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Tensor: Result of close comparison function.

        """
        return t1.f.is_close_zip(t1, t2)

    # No backward function needed for IsClose


class Permute(Function):
    """Static class for permutation function. Used to group helper static methods for forward and backward passes of the permutation function."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Performs forward pass of permutation function.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            t1: Input tensor.
            order: Tensor containing the permutation order.

        Returns:
        -------
            Tensor: Result of permutation function.

        """
        order_list = order.to_numpy().astype(int).tolist()
        ctx.save_for_backward(order_list)
        return t1._new(t1._tensor.permute(*order_list))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Performs backward pass of permutation function.

        Args:
        ----
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, float]: Gradients with respect to inputs t1 and order.

        """
        (order_list,) = ctx.saved_values
        # Create a new list to store the inverse permutation order
        undo_permute_order = [0] * len(order_list)
        # Populate the inverse permutation order by mapping each new position to its original position
        for original_axis_position, new_axis_position in enumerate(order_list):
            undo_permute_order[new_axis_position] = original_axis_position
        # Apply the inverse permutation to the gradient output
        return grad_output._new(grad_output._tensor.permute(*undo_permute_order)), 0.0


class View(Function):
    """Static class for the view operation. Used to group helper static methods for forward and backward passes of the view operation."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Performs forward pass of the view operation.

        Args:
        ----
            ctx: Context object to save information for backward pass.
            a: Input tensor.
            shape: Tensor containing the new shape.

        Returns:
        -------
            Tensor: Result of the view operation.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the central difference gradient for a given function and tensor. Central difference is a numerical method used to approximate the derivative of a function.

    Args:
    ----
        f: The function to differentiate.
        *vals: The input tensors.
        arg: The index of the tensor to differentiate with respect to.
        epsilon: The small perturbation value.
        ind: The index of the element to perturb.

    Returns:
    -------
        float: The computed gradient.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
