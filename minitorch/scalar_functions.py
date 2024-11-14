from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the scalar function to the given inputs. Under the hood, this creates a new scalar variable from the result with a the history of the computation. This history is later used for backpropagation.

        Args:
        ----
            vals: The inputs to the function.

        Returns:
        -------
            The result of the scalar function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the addition function.

        Args:
        ----
            ctx: The context for the function.
            a: The first input to the function.
            b: The second input to the function.

        Returns:
        -------
            The result of the addition function.

        """
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the addition function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the log function.

        Args:
        ----
            ctx: The context for the function.
            a: The input to the function.

        Returns:
        -------
            The result of the log function.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the log function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the multiplication function.

        Args:
        ----
            ctx: The context for the function.
            a: The first input to the function.
            b: The second input to the function.

        Returns:
        -------
            The result of the multiplication function.

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the multiplication function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the inputs.

        """
        (a, b) = ctx.saved_values
        return operators.mul(d_output, b), operators.mul(d_output, a)


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the inverse function.

        Args:
        ----
            ctx: The context for the function.
            a: The input to the function.

        Returns:
        -------
            The result of the inverse function.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the inverse function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the negation function.

        Args:
        ----
            ctx: The context for the function.
            a: The input to the function.

        Returns:
        -------
            The result of the negation function.

        """
        # Doesn't need to save any values for the backward pass because the negation function is linear
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the negation function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        return operators.neg(d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the exponential function.

        Args:
        ----
            ctx: The context for the function.
            a: The input to the function.

        Returns:
        -------
            The result of the exponential function.

        """
        # We save the exponential of a for the backward pass because the derivative of the exponential function is dependent on the value of the exponential function, helps avoid recomputing it
        exp_a = operators.exp(a)
        ctx.save_for_backward(exp_a)
        return exp_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the exponential function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (exp_a,) = ctx.saved_values
        return operators.mul(d_output, exp_a)


class LT(ScalarFunction):
    """Less-than function $f(x, y) = x < y$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the less-than function.

        Args:
        ----
            ctx: The context for the function.
            a: The first input to the function.
            b: The second input to the function.

        Returns:
        -------
            The result of the less-than function.

        """
        # Doesn't need to save any values for the backward pass because the less-than function isn't differentiable
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the less-than function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the inputs.

        """
        # Returns 0 for both derivatives because the less-than function isn't differentiable and can be treated like a constant function here
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = x == y$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the equal function.

        Args:
        ----
            ctx: The context for the function.
            a: The first input to the function.
            b: The second input to the function.

        Returns:
        -------
            The result of the equal function.

        """
        # Doesn't need to save any values for the backward pass because the equal function isn't differentiable
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the equal function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the inputs.

        """
        # Returns 0 for both derivatives because the equal function isn't differentiable and can be treated like a constant function here
        return 0.0, 0.0


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the ReLU function.

        Args:
        ----
            ctx: The context for the function.
            a: The input to the function.

        Returns:
        -------
            The result of the ReLU function.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the ReLU function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    r"""Sigmoid function $f(x) = \frac{1}{1 + e^{-x}}$", with helper static functions grouped into the ScalarFunction class"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the sigmoid function.

        Args:
        ----
            ctx: The context for the function.
            a: The input to the function.

        Returns:
        -------
            The result of the sigmoid function.

        """
        # We save the sigmoid of a for the backward pass because the derivative of the sigmoid function is dependent on the value of the sigmoid function, helps avoid recomputing it
        sigmoid_a = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_a)
        return sigmoid_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the sigmoid function.

        Args:
        ----
            ctx: The context for the function.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (sigmoid_a,) = ctx.saved_values
        #  Derivative of sigmoid function is sigmoid(a) * (1 - sigmoid(a))
        return operators.mul(
            d_output,
            operators.mul(sigmoid_a, operators.add(1, operators.neg(sigmoid_a))),
        )
