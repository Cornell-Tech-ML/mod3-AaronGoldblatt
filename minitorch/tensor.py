"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the requires_grad attribute of the tensor to the specified value. If x is True, it sets the attribute to True, indicating that the tensor requires gradient computation. If x is False, it sets the attribute to False, indicating that the tensor does not require gradient computation.

        Args:
        ----
            x (bool): The value to set the requires_grad attribute to.

        Returns:
        -------
            None

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks if the tensor requires gradient computation. This is true if the tensor has been used in a function that requires gradient computation, i.e. it has a history.

        Args:
        ----
            None

        Returns:
        -------
            bool: True if the tensor requires gradient computation, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns a numpy array with the same data as the tensor.

        Args:
        ----
            None

        Returns:
        -------
            npt.NDArray[np.float64]: A numpy array with the same data as the tensor.

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor of zeros with the given shape. If no shape is provided, it creates a tensor of zeros with the same shape as the current tensor.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the tensor to create.

        Returns:
        -------
            Tensor: A tensor of zeros with the given shape.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is a constant, i.e. it has no history.

        Args:
        ----
            None

        Returns:
        -------
            bool: True if the tensor is a constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parents of the tensor.

        Args:
        ----
            None

        Returns:
        -------
            Iterable[Variable]: The parents of the tensor.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule of calculus in tensor form to compute the derivatives of the parents of the tensor. The chain rule is a fundamental concept in calculus that allows us to find the derivative of a composite function by breaking it down into simpler parts. In the context of tensors, this rule is used to propagate the derivative of the output tensor back through the computational graph to the input tensors.

        Args:
        ----
            d_output : derivative of the output tensor

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: The derivatives of the parents of the tensor.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation to compute the gradients of the tensor with respect to its inputs. This method is used to compute the gradients of the tensor during the backward pass of the automatic differentiation process.

        Args:
        ----
            grad_output : derivative of the output tensor

        Returns:
        -------
            None

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    # Task 2.3.
    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor.

        Args:
        ----
            None

        Returns:
        -------
            int: The total number of elements in the tensor.

        """
        return int(operators.prod(self.shape))

    @property
    def dims(self) -> int:
        """Returns the number of dimensions in the tensor.

        Args:
        ----
            None

        Returns:
        -------
            int: The number of dimensions in the tensor.

        """
        return len(self.shape)

    def __add__(self, b: TensorLike) -> Tensor:
        """Element-wise addition.

        Args:
        ----
            b (TensorLike): The tensor to add element-wise.

        Returns:
        -------
            Tensor: The result of element-wise addition.

        """
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        """Element-wise subtraction.

        Args:
        ----
            b (TensorLike): The tensor to subtract element-wise.

        Returns:
        -------
            Tensor: The result of element-wise subtraction.

        """
        return Add.apply(self, Neg.apply(self._ensure_tensor(b)))

    def __mul__(self, b: TensorLike) -> Tensor:
        """Element-wise multiplication.

        Args:
        ----
            b (TensorLike): The tensor to multiply element-wise.

        Returns:
        -------
            Tensor: The result of element-wise multiplication.

        """
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: TensorLike) -> Tensor:
        """Element-wise less than comparison.

        Args:
        ----
            b (TensorLike): The tensor to compare element-wise.

        Returns:
        -------
            Tensor: The result of element-wise less than comparison.

        """
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        """Element-wise equality comparison.

        Args:
        ----
            b (TensorLike): The tensor to compare element-wise.

        Returns:
        -------
            Tensor: The result of element-wise equality comparison.

        """
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        """Element-wise greater than comparison.

        Args:
        ----
            b (TensorLike): The tensor to compare element-wise.

        Returns:
        -------
            Tensor: The result of element-wise greater than comparison.

        """
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        """Element-wise negation.

        Args:
        ----
            None

        Returns:
        -------
            Tensor: The negated tensor.

        """
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        """Element-wise addition with reversed operands.

        Args:
        ----
            b (TensorLike): The tensor to add element-wise with reversed operands.

        Returns:
        -------
            Tensor: The result of element-wise addition with reversed operands.

        """
        return Add.apply(self._ensure_tensor(b), self)

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Element-wise multiplication with reversed operands.

        Args:
        ----
            b (TensorLike): The tensor to multiply element-wise with reversed operands.

        Returns:
        -------
            Tensor: The result of element-wise multiplication with reversed operands.

        """
        return Mul.apply(self._ensure_tensor(b), self)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Returns True if all elements are true.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce. If None, reduces all dimensions.

        Returns:
        -------
            Tensor: A tensor with the result of the reduction.

        """
        if dim is None:
            return All.apply(
                # Flatten the tensor into 1D, doesn't use contiguous() this operation isn't made more efficient with contiguous array
                self.view(self.size),
                # Set to take all over first dimension, which is the only dimension so result will be true if all elements in the tensor are true
                self._ensure_tensor(0),
            )
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, b: TensorLike) -> Tensor:
        """Element-wise close comparison.

        Args:
        ----
            b (TensorLike): The tensor to compare element-wise.
            rtol (float): The relative tolerance.
            atol (float): The absolute tolerance.

        Returns:
        -------
            Tensor: The result of element-wise close comparison.

        """
        return IsClose.apply(self, self._ensure_tensor(b))

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise.

        Args:
        ----
            None

        Returns:
        -------
            Tensor: The result of applying the sigmoid function element-wise.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU function element-wise.

        Args:
        ----
            None

        Returns:
        -------
            Tensor: The result of applying the ReLU function element-wise.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Applies the natural logarithm element-wise.

        Args:
        ----
            None

        Returns:
        -------
            Tensor: The result of applying the natural logarithm element-wise.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function element-wise.

        Args:
        ----
            None

        Returns:
        -------
            Tensor: The result of applying the exponential function element-wise.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sum of elements over a given dimension.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce. If None, reduces all dimensions.

        Returns:
        -------
            Tensor: A tensor with the result of the reduction.

        """
        if dim is None:
            return Sum.apply(
                # Flatten the tensor into 1D, uses contiguous() because summing is more efficient with contiguous array
                self.contiguous().view(self.size),
                # Set to take sum over first dimension, which is the only dimension so result will sum all elements in the tensor
                self._ensure_tensor(0),
            )
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Mean of elements over a given dimension.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce. If None, reduces all dimensions.

        Returns:
        -------
            Tensor: A tensor with the result of the reduction.

        """
        if dim is None:
            return self.sum() / self.size
        else:
            return self.sum(dim) / self.shape[dim]

    def permute(self, *order: int, dim: Optional[int] = None) -> Tensor:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order (int): The permutation order.
            dim (Optional[int]): The dimension to permute. If None, permutes all dimensions.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        return Permute.apply(
            self,
            # Convert the order tuple to a tensor
            tensor(list(order)),
        )

    def view(self, *shape: int) -> Tensor:
        """View the tensor as a different shape, without changing the underlying data.

        Args:
        ----
            *shape (int): The new shape.

        Returns:
        -------
            Tensor: The tensor viewed as the new shape.

        """
        return View.apply(
            self,
            # Convert the shape tuple to a tensor
            tensor(list(shape)),
        )

    def zero_grad_(self) -> None:
        """Sets the gradient of the tensor to None.

        Args:
        ----
            None

        Returns:
        -------
            None.

        """
        self.grad = None
