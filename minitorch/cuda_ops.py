# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to compile a function just-in-time (JIT) for CUDA with the argument device set to True. This creates device functions that can only be called from other CUDA functions, and not from the CPU/host code.

    Args:
    ----
        fn: Function to compile.
        **kwargs: Additional keyword arguments to pass to the `jit` decorator.

    Returns:
    -------
        JIT compiled device function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator to compile a function just-in-time (JIT) for CUDA. This creates functions that can be called from the CPU/host code and that serve as the entry points for GPU computation. They cannot be called from other CUDA functions.

    Args:
    ----
        fn: Function to compile.
        **kwargs: Additional keyword arguments to pass to the `jit` decorator.

    Returns:
    -------
        JIT compiled function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates a CUDA function that applies a binary operation element-wise between two tensors. This method takes a binary function (operating on two floats) and returns a new function that applies this operation to two tensors element-wise, with broadcasting support. The computation is performed on the GPU using CUDA.

        Args:
        ----
            fn (Callable[[float, float], float]): Binary function to apply element-wise.
                Should take two floats as input and return a float.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that takes two tensors and returns a new tensor containing the element-wise application of fn. The output shape is determined by broadcasting rules.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a CUDA function that reduces a tensor along a specified dimension. This method takes a reduction function (operating on two floats) and returns a new function that applies this operation to a tensor along a specified dimension, with broadcasting support. The computation is performed on the GPU using CUDA.

        Args:
        ----
            fn (Callable[[float, float], float]): Reduction function to apply.
                Should take two floats as input and return a float.
            start (float): Starting value for the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that takes a tensor and an integer dimension and returns a new tensor containing the reduction along the specified dimension.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Creates a CUDA function that performs matrix multiplication between two tensors. This method takes two tensors and returns a new tensor containing the result of the matrix multiplication. The computation is performed on the GPU using CUDA.

        Args:
        ----
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the result of the matrix multiplication.

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # Implemented for Task 3.3.
        # Check if the current thread is within the bounds of the output tensor.
        if i < out_size:
            # Convert the linear index 'i' to a multi-dimensional index 'out_index' based on 'out_shape'.
            to_index(i, out_shape, out_index)
            # Broadcast 'out_index' to 'in_index' based on 'in_shape' to handle broadcasting rules.
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # Calculate the position in the input storage using the 'in_index' and 'in_strides'.
            in_position = index_to_position(in_index, in_strides)
            # Calculate the position in the output storage using the 'out_index' and 'out_strides'.
            out_position = index_to_position(out_index, out_strides)
            # Apply the function 'fn' to the input value and store the result in the output storage.
            out[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Implemented for Task 3.3.
        # Check if the current thread is within the bounds of the output tensor.
        if i < out_size:
            # Convert the linear index 'i' to a multi-dimensional index 'out_index' based on 'out_shape'.
            to_index(i, out_shape, out_index)
            # Broadcast 'out_index' to 'a_index' based on 'a_shape' to handle broadcasting rules.
            broadcast_index(out_index, out_shape, a_shape, a_index)
            # Broadcast 'out_index' to 'b_index' based on 'b_shape' to handle broadcasting rules.
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # Calculate the position in the 'a' storage using the 'a_index' and 'a_strides'.
            a_position = index_to_position(a_index, a_strides)
            # Calculate the position in the 'b' storage using the 'b_index' and 'b_strides'.
            b_position = index_to_position(b_index, b_strides)
            # Apply the function 'fn' to the input values and store the result in the output storage.
            out[i] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Implemented for Task 3.3.
    # Initialize shared memory with the input values
    if i < size:
        # Load the input value into shared memory at the current thread's position
        cache[pos] = a[i]
    else:
        # If the thread index is out of bounds, initialize shared memory to zero
        cache[pos] = 0.0
    # Synchronize threads within the block to ensure all threads have loaded their data
    cuda.syncthreads()
    # Perform reduction within the block using a parallel reduction algorithm
    stride = 1
    while stride < BLOCK_DIM:
        # Check if the current thread should perform an addition at this stride level
        if pos % (2 * stride) == 0:
            # Add the value from the neighboring thread at the current stride distance
            cache[pos] += cache[pos + stride]
        # Double the stride for the next iteration
        stride *= 2
        # Synchronize threads to ensure all additions at this stride level are complete
        cuda.syncthreads()
    # Write the result of the block's sum to the output array
    if pos == 0:
        # Only the first thread in the block writes the result to the global memory
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice CUDA reduction by summing blocks of a 1D tensor. This function demonstrates parallel reduction on GPU by dividing the input tensor into blocks of size THREADS_PER_BLOCK (32) and summing each block. The result will contain partial sums, with each element representing the sum of up to 32 consecutive elements from the input tensor. Note that this is not a complete reduction of the tensor, but rather a practice function intended for learning CUDA reduction patterns.

    Args:
    ----
        a (Tensor): Input 1D tensor to be reduced.

    Returns:
    -------
        TensorData: A TensorData object containing partial sums.
            The output size is fixed at 2 elements, where each element
            contains the sum of a block of input values.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Implemented for Task 3.3.
        # Convert the block index to a multi-dimensional index for output
        to_index(out_pos, out_shape, out_index)
        # Initialize reduction parameters
        reduction_stride = 1
        # Size of dimension being reduced
        dimension_size = a_shape[reduce_dim]
        # Load data into shared memory cache if the thread index is within the dimension size
        if pos < dimension_size:
            # Set the index for the reduction dimension
            out_index[reduce_dim] = pos
            # Calculate position in input storage
            input_position = index_to_position(out_index, a_strides)
            # Load value into shared memory
            cache[pos] = a_storage[input_position]
        else:
            # For threads beyond dimension size, initialize with reduction identity value
            cache[pos] = reduce_value
        # Parallel reduction in shared memory
        while reduction_stride < BLOCK_DIM:
            # Synchronize all threads before next reduction step
            numba.cuda.syncthreads()
            # Only threads at even positions (relative to current stride) perform reduction, the number of which is halved each iteration
            if pos % (reduction_stride * 2) == 0:
                # Combine values with strided pair using reduction function and store the result in cache[pos]
                cache[pos] = fn(cache[pos], cache[pos + reduction_stride])
            # Double the stride for next iteration
            reduction_stride *= 2
        # Ensure all reductions are complete
        numba.cuda.syncthreads()
        # Thread 0 writes final result to global memory
        if pos == 0:
            # Calculate position in output storage
            output_position = index_to_position(out_index, out_strides)
            # Write the result, which is at cache[0], to the output storage
            out[output_position] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # Implemented for Task 3.4.
    # Allocate shared memory for matrices A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # Get the thread indices within the block
    row = cuda.threadIdx.x
    col = cuda.threadIdx.y
    # Only process if thread indices are within matrix bounds, otherwise thread does nothing
    if row < size and col < size:
        # Copy data from global to shared memory, converting 2D indices to 1D index using row-major ordering
        row_offset = row * size
        # Calculate the position in the input matrices using the row offset and column index, which is the same in both matrices since they are both size N x N
        position = row_offset + col
        a_shared[row, col] = a[position]
        b_shared[row, col] = b[position]
        # Ensure all threads have finished copying to shared memory
        cuda.syncthreads()
        # Initialize accumulator for dot product
        dot_product_accumulator = 0.0
        # Compute dot product for this element
        for inner_dimension in range(size):
            dot_product_accumulator += (
                a_shared[row, inner_dimension] * b_shared[inner_dimension, col]
            )
        # Write result to global memory, converting 2D indices to 1D index for output
        out[position] = dot_product_accumulator


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice CUDA matrix multiplication on small square matrices. This function demonstrates basic matrix multiplication on GPU using shared memory. It is designed for learning purposes and only works with same-size, square matrices where the size is less than THREADS_PER_BLOCK (32).

    Args:
    ----
        a (Tensor): First input tensor of shape (size, size)
        b (Tensor): Second input tensor of shape (size, size)

    Returns:
    -------
        TensorData: Result of matrix multiplication with shape (size, size)

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    assert (
        a_shape[-1] == b_shape[-2]
    ), "Incompatible dimensions for matrix multiplication"

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # Implemented for Task 3.4.
    # Initialize accumulator for the dot product result, which allows us to accumulate the result across multiple threads and only write to global memory once we have the final result
    dot_product_accumulator = 0.0
    # Iterate over the shared dimension in blocks of size BLOCK_DIM (32), which allows us to handle the case where the matrix is larger than the block size
    for block_start in range(0, a_shape[2], BLOCK_DIM):
        # Load a block of matrix A into shared memory
        # Calculate current position in the k dimension, or inner dimension that is shared between the two matrices and is being iterated over
        k = block_start + pj
        # Only copy if the thread indices are within the bounds of the input matrix a
        if i < a_shape[1] and k < a_shape[2]:
            # Calculate the batch offset using the batch stride and the batch index
            batch_offset = a_batch_stride * batch
            # Calculate the row offset using the row stride and the row index
            row_offset = a_strides[1] * i
            # Calculate the column offset using the column stride and k, which represents the column of the input matrix a since the inner dimension in a is the column dimension, i.e. a is size M x K
            col_offset = a_strides[2] * k
            # Calculate global memory position using all the offsets
            a_position = batch_offset + row_offset + col_offset
            # Copy the value from the input matrix a to its shared memory, only one time per kernel
            a_shared[pi, pj] = a_storage[a_position]
        # Load a block of matrix B into shared memory
        # Calculate current position in the k dimension, or inner dimension that is shared between the two matrices and is being iterated over
        k = block_start + pi
        # Only copy if the thread indices are within the bounds of the input matrix b
        if j < b_shape[2] and k < b_shape[1]:
            # Calculate the batch offset using the batch stride and the batch index
            batch_offset = b_batch_stride * batch
            # Calculate the row offset using the row stride and k, which represents the row of the input matrix b since the inner dimension in b is the row dimension, i.e. b is size K x N
            row_offset = b_strides[1] * k
            # Calculate the column offset using the column stride and the column index
            col_offset = b_strides[2] * j
            # Calculate global memory position using all the offsets
            b_position = batch_offset + col_offset + row_offset
            # Copy the value from the input matrix b to its shared memory, only one time per kernel
            b_shared[pi, pj] = b_storage[b_position]
        # Ensure all threads have finished loading data from input matrices into their respective shared memory for the current block
        cuda.syncthreads()
        # Compute partial dot product for this block of the output matrix
        for inner_dim in range(BLOCK_DIM):
            # Only compute if we haven't exceeded the actual matrix dimensions
            if block_start + inner_dim < a_shape[2]:
                # Multiply and accumulate corresponding elements
                dot_product_accumulator += (
                    a_shared[pi, inner_dim] * b_shared[inner_dim, pj]
                )
    # Write final result to global memory if indices are within bounds
    if i < out_shape[1] and j < out_shape[2]:
        # Calculate the batch offset using the batch stride and the batch index
        batch_offset = out_strides[0] * batch
        # Calculate the row offset using the row stride and the row index
        row_offset = out_strides[1] * i
        # Calculate the column offset using the column stride and the column index
        col_offset = out_strides[2] * j
        # Calculate the output position using the batch offset, row offset, and column offset
        out_position = batch_offset + row_offset + col_offset
        # Write the result to the output storage at the calculated position, only one time per kernel
        out[out_position] = dot_product_accumulator


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
