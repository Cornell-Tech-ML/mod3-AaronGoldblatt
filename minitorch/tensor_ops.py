from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

import numpy as np


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce
            start (float): optional, start value for the reduction

        Returns:
        -------
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        """Applies a function element-wise to input tensor, storing results in output tensor. Handles broadcasting if input and output shapes differ.

        Args:
        ----
            out: Storage for the output tensor.
            out_shape: Shape of the output tensor.
            out_strides: Strides of the output tensor.
            in_storage: Storage for the input tensor.
            in_shape: Shape of the input tensor.
            in_strides: Strides of the input tensor.

        Returns:
        -------
            None. Results are stored in-place in the `out` storage.

        """
        # Task 2.3.
        # Initialize the indices for the output and input tensors as lists of zeros the same length as the shape
        out_index = np.array([0] * len(out_shape), dtype=np.int32)
        in_index = np.array([0] * len(in_shape), dtype=np.int32)
        for out_position in range(len(out)):
            # Convert the output position to an index in the output tensor
            to_index(out_position, out_shape, out_index)
            # Broadcast the output index to the input shape to obtain the corresponding input tensor's index to the current output index
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # Calculate the position in the 1D lower-level storage of the input tensor
            in_storage_position = index_to_position(in_index, in_strides)
            # Calculate the position in the 1D lower-level storage of the output tensor
            out_storage_position = index_to_position(out_index, out_strides)
            # Apply the function to the data at the input and output positions and store the result in the output tensor
            out[out_storage_position] = fn(in_storage[in_storage_position])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        """Applies a binary function element-wise to two input tensors, storing results in a single output tensor. Handles broadcasting if input shapes differ from the output shape.

        Args:
        ----
            out: Storage for the output tensor.
            out_shape: Shape of the output tensor.
            out_strides: Strides of the output tensor.
            a_storage: Storage for the first input tensor.
            a_shape: Shape of the first input tensor.
            a_strides: Strides of the first input tensor.
            b_storage: Storage for the second input tensor.
            b_shape: Shape of the second input tensor.
            b_strides: Strides of the second input tensor.

        Returns:
        -------
            None. Results are stored in-place in the `out` storage.

        """
        # Task 2.3.
        # Initialize the indices for the output and both input tensors as lists of zeros the same length as the shape
        a_index = np.array([0] * len(a_shape), dtype=np.int32)
        b_index = np.array([0] * len(b_shape), dtype=np.int32)
        out_index = np.array([0] * len(out_shape), dtype=np.int32)
        for out_position in range(len(out)):
            # Convert the output position to an index in the output tensor
            to_index(out_position, out_shape, out_index)
            # Broadcast the output index to both of the input tensors' shapes to obtain each input tensor's index to the current output index
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # Calculate the position in the 1D lower-level storage of each input tensor
            a_storage_position = index_to_position(a_index, a_strides)
            b_storage_position = index_to_position(b_index, b_strides)
            # Calculate the position in the 1D lower-level storage of the output tensor
            out_storage_position = index_to_position(out_index, out_strides)
            # Apply the function to the data of both input tensors and store the result in the output tensor
            out[out_storage_position] = fn(
                a_storage[a_storage_position], b_storage[b_storage_position]
            )

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        """Applies a reduction function along a specified dimension of the input tensor.The output tensor will have the same shape as the input tensor, except the dimension being reduced will have size 1.

        Args:
        ----
            out: Storage for the output tensor.
            out_shape: Shape of the output tensor.
            out_strides: Strides of the output tensor.
            a_storage: Storage for the input tensor.
            a_shape: Shape of the input tensor.
            a_strides: Strides of the input tensor.
            reduce_dim: The dimension along which to reduce.

        Returns:
        -------
            None. Results are stored in-place in the `out` storage.

        """
        # Task 2.3.
        # Initialize an index for the output tensor, with the same length as its shape
        out_index = np.array([0] * len(out_shape), dtype=np.int32)
        # Get the size of the reduction dimension
        dim_size = a_shape[reduce_dim]
        for out_position in range(len(out)):
            # Convert the output position to an index in the output tensor
            to_index(out_position, out_shape, out_index)
            # Compute the corresponding storage position in the 1D representation of the output tensor
            out_storage_position = index_to_position(out_index, out_strides)
            # Initialize the reduction result with the first value in the reduction dimension
            result = None
            for reduce_position in range(dim_size):
                # Create a copy of out_index to adjust for the input index
                a_index = out_index.copy()
                # Set the reduction dimension index
                a_index[reduce_dim] = reduce_position
                # Compute the corresponding storage position in the 1D representation of the input tensor
                a_storage_position = index_to_position(a_index, a_strides)
                # Apply the reduction function to accumulate values
                if result is None:
                    result = a_storage[
                        a_storage_position
                    ]  # Initialize with the first value
                else:
                    result = fn(result, a_storage[a_storage_position])
                # Store the reduction result in the output tensor
                out[out_storage_position] = result

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
