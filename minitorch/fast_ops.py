from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to compile a function with Numba's JIT compiler with 'inline' always enabled, which ensures that all JIT compiled functions are inlined for better performance.

    Args:
    ----
        fn (Fn): Function to be JIT compiled
        **kwargs: Additional keyword arguments to pass to Numba's JIT compiler

    Returns:
    -------
        Fn: JIT compiled version of the input function with inlining enabled

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

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
        # Implemented for Task 3.1.
        # Check if input and output tensors are stride-aligned and have the same shape
        if list(in_shape) == list(out_shape) and list(in_strides) == list(out_strides):
            # Directly apply the function without index calculations
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            # Handle non-aligned tensors with index calculations
            for out_flat_index in prange(len(out)):
                # Initialize arrays to hold multi-dimensional indices for output and input tensors
                out_multi_index = np.zeros(MAX_DIMS, np.int32)
                in_multi_index = np.zeros(MAX_DIMS, np.int32)
                # Convert the flat index to a multi-dimensional index for the output tensor
                to_index(out_flat_index, out_shape, out_multi_index)
                # Broadcast the output index to the input index, aligning dimensions
                broadcast_index(out_multi_index, out_shape, in_shape, in_multi_index)
                # Calculate the position in the input storage using the input multi-dimensional index
                in_storage_position = index_to_position(in_multi_index, in_strides)
                # Calculate the position in the output storage using the output multi-dimensional index
                out_storage_position = index_to_position(out_multi_index, out_strides)
                # Apply the function to the input value and store the result in the output storage
                out[out_storage_position] = fn(in_storage[in_storage_position])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

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
        # Implemented for Task 3.1.
        # Check if `out`, `a`, and `b` are stride-aligned and have the same shape
        if list(a_strides) == list(b_strides) == list(out_strides) and list(
            a_shape
        ) == list(b_shape) == list(out_shape):
            # Directly apply the function without index calculations
            for flat_index in prange(len(out)):
                out[flat_index] = fn(a_storage[flat_index], b_storage[flat_index])
        else:
            # Handle non-aligned tensors with index calculations
            for flat_index in prange(len(out)):
                # Initialize arrays to hold multi-dimensional indices for output and input tensors
                out_multi_index: Index = np.empty(MAX_DIMS, np.int32)
                a_multi_index: Index = np.empty(MAX_DIMS, np.int32)
                b_multi_index: Index = np.empty(MAX_DIMS, np.int32)
                # Convert the flat index to a multi-dimensional index for the output tensor
                to_index(flat_index, out_shape, out_multi_index)
                out_storage_position = index_to_position(out_multi_index, out_strides)
                # Broadcast the output index to the input indices, aligning dimensions
                broadcast_index(out_multi_index, out_shape, a_shape, a_multi_index)
                a_storage_position = index_to_position(a_multi_index, a_strides)
                broadcast_index(out_multi_index, out_shape, b_shape, b_multi_index)
                b_storage_position = index_to_position(b_multi_index, b_strides)
                # Apply the function to the input values and store the result in the output storage
                out[out_storage_position] = fn(
                    a_storage[a_storage_position], b_storage[b_storage_position]
                )

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

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
        # Implemented for Task 3.1.
        reduction_size = a_shape[reduce_dim]
        reduction_stride = a_strides[reduce_dim]
        # Iterate over the output tensor in parallel
        for output_flat_index in prange(len(out)):
            output_multi_dim_index: Index = np.empty(MAX_DIMS, np.int32)
            # Convert the flat index to a multi-dimensional index for the output tensor
            to_index(output_flat_index, out_shape, output_multi_dim_index)
            # Calculate the position in the output storage
            output_storage_position = index_to_position(
                output_multi_dim_index, out_strides
            )
            # Calculate the starting position in the input storage
            input_storage_position = index_to_position(
                output_multi_dim_index, a_strides
            )
            # Initialize the temporary result with the current output value
            temp_result = out[output_storage_position]
            # Perform the reduction operation along the specified dimension, not in parallel because of dependencies in reduction operation
            for _ in range(reduction_size):
                temp_result = fn(temp_result, a_storage[input_storage_position])
                input_storage_position += reduction_stride
            # Store the result back in the output storage
            out[output_storage_position] = temp_result

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
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
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # Ensure the dimensions are compatible for matrix multiplication
    assert (
        a_shape[-1] == b_shape[-2]
    ), "Incompatible dimensions for matrix multiplication"

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Implemented for Task 3.2.
    # Loop over the batch dimension, which corresponds to out_shape[0]
    for batch_index in prange(out_shape[0]):
        # Loop over the first dimension of tensor 'a'
        for row_index in prange(out_shape[1]):
            # Loop over the second dimension of tensor 'b'
            for col_index in prange(out_shape[2]):
                # Calculate the starting positions in a_storage and b_storage using batch and row/column indices
                a_position = batch_index * a_batch_stride + row_index * a_strides[1]
                b_position = batch_index * b_batch_stride + col_index * b_strides[2]
                # Initialize accumulator for the dot product
                dot_product_accumulator = 0.0
                # Compute the dot product over the shared dimension (2nd of 'a' and 1st of 'b')
                for shared_dim_index in range(a_shape[2]):
                    dot_product_accumulator += (
                        a_storage[a_position] * b_storage[b_position]
                    )
                    # Update positions in the shared dimension using strides
                    a_position += a_strides[2]
                    b_position += b_strides[1]
                # Calculate the position in the output tensor and store the accumulated result
                out_position = (
                    batch_index * out_strides[0]
                    + row_index * out_strides[1]
                    + col_index * out_strides[2]
                )
                out[out_position] = dot_product_accumulator


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
