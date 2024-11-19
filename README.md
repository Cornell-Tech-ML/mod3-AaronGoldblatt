# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1 and 3.2
### Task 3.1/2.1: Diagnostic Output of Script `project/parallel_check.py`
```
$ python project/parallel_check.py
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\User\Repositories\Cornell-Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (163)
-----------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                              |
        out: Storage,                                                                                      |
        out_shape: Shape,                                                                                  |
        out_strides: Strides,                                                                              |
        in_storage: Storage,                                                                               |
        in_shape: Shape,                                                                                   |
        in_strides: Strides,                                                                               |
    ) -> None:                                                                                             |
        # Implemented for Task 3.1.                                                                    |
        # Check if input and output tensors are stride-aligned and have the same shape                     |
        if list(in_shape) == list(out_shape) and list(in_strides) == list(out_strides):                    |
            # Directly apply the function without index calculations                                       |
            for i in prange(len(out)):---------------------------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                                 |
        else:                                                                                              |
            # Handle non-aligned tensors with index calculations                                           |
            for out_flat_index in prange(len(out)):--------------------------------------------------------| #3
                # Initialize arrays to hold multi-dimensional indices for output and input tensors         |
                out_multi_index = np.zeros(MAX_DIMS, np.int32)---------------------------------------------| #0
                in_multi_index = np.zeros(MAX_DIMS, np.int32)----------------------------------------------| #1
                # Convert the flat index to a multi-dimensional index for the output tensor                |
                to_index(out_flat_index, out_shape, out_multi_index)                                       |
                # Broadcast the output index to the input index, aligning dimensions                       |
                broadcast_index(out_multi_index, out_shape, in_shape, in_multi_index)                      |
                # Calculate the position in the input storage using the input multi-dimensional index      |
                in_storage_position = index_to_position(in_multi_index, in_strides)                        |
                # Calculate the position in the output storage using the output multi-dimensional index    |
                out_storage_position = index_to_position(out_multi_index, out_strides)                     |
                # Apply the function to the input value and store the result in the output storage         |
                out[out_storage_position] = fn(in_storage[in_storage_position])                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (181) is hoisted out of
the parallel loop labelled #3 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_multi_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (182) is hoisted out of
the parallel loop labelled #3 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: in_multi_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (220)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\User\Repositories\Cornell-Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (220)
-----------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                |
        out: Storage,                                                                                                        |
        out_shape: Shape,                                                                                                    |
        out_strides: Strides,                                                                                                |
        a_storage: Storage,                                                                                                  |
        a_shape: Shape,                                                                                                      |
        a_strides: Strides,                                                                                                  |
        b_storage: Storage,                                                                                                  |
        b_shape: Shape,                                                                                                      |
        b_strides: Strides,                                                                                                  |
    ) -> None:                                                                                                               |
        # Implemented for Task 3.1.                                                                                      |
        # Check if `out`, `a`, and `b` are stride-aligned and have the same shape                                            |
        if list(a_strides) == list(b_strides) == list(out_strides) and list(a_shape) == list(b_shape) == list(out_shape):    |
            # Directly apply the function without index calculations                                                         |
            for flat_index in prange(len(out)):------------------------------------------------------------------------------| #4
                out[flat_index] = fn(a_storage[flat_index], b_storage[flat_index])                                           |
        else:                                                                                                                |
            # Handle non-aligned tensors with index calculations                                                             |
            for flat_index in prange(len(out)):------------------------------------------------------------------------------| #5
                # Initialize arrays to hold multi-dimensional indices for output and input tensors                           |
                out_multi_index: Index = np.empty(MAX_DIMS, np.int32)                                                        |
                a_multi_index: Index = np.empty(MAX_DIMS, np.int32)                                                          |
                b_multi_index: Index = np.empty(MAX_DIMS, np.int32)                                                          |
                # Convert the flat index to a multi-dimensional index for the output tensor                                  |
                to_index(flat_index, out_shape, out_multi_index)                                                             |
                out_storage_position = index_to_position(out_multi_index, out_strides)                                       |
                # Broadcast the output index to the input indices, aligning dimensions                                       |
                broadcast_index(out_multi_index, out_shape, a_shape, a_multi_index)                                          |
                a_storage_position = index_to_position(a_multi_index, a_strides)                                             |
                broadcast_index(out_multi_index, out_shape, b_shape, b_multi_index)                                          |
                b_storage_position = index_to_position(b_multi_index, b_strides)                                             |
                # Apply the function to the input values and store the result in the output storage                          |
                out[out_storage_position] = fn(a_storage[a_storage_position], b_storage[b_storage_position])                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #4, #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (241) is hoisted out of
the parallel loop labelled #5 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_multi_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (242) is hoisted out of
the parallel loop labelled #5 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: a_multi_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (243) is hoisted out of
the parallel loop labelled #5 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: b_multi_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (279)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\User\Repositories\Cornell-Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (279)
-----------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(
   |
        out: Storage,
   |
        out_shape: Shape,
   |
        out_strides: Strides,
   |
        a_storage: Storage,
   |
        a_shape: Shape,
   |
        a_strides: Strides,
   |
        reduce_dim: int,
   |
    ) -> None:
   |
        # Implemented for Task 3.1.
   |
        reduction_size = a_shape[reduce_dim]
   |
        reduction_stride = a_strides[reduce_dim]
   |
        # Iterate over the output tensor in parallel
   |
        for output_flat_index in prange(len(out)):---------------------------------------------------------------------------------------------| #6
            output_multi_dim_index: Index = np.empty(MAX_DIMS, np.int32)
   |
            # Convert the flat index to a multi-dimensional index for the output tensor
   |
            to_index(output_flat_index, out_shape, output_multi_dim_index)
   |
            # Calculate the position in the output storage
   |
            output_storage_position = index_to_position(output_multi_dim_index, out_strides)
   |
            # Calculate the starting position in the input storage
   |
            input_storage_position = index_to_position(output_multi_dim_index, a_strides)
   |
            # Initialize the temporary result with the current output value
   |
            temp_result = out[output_storage_position]
   |
            # Perform the reduction operation along the specified dimension, not in parallel because of dependencies in reduction operation    |
            for _ in range(reduction_size):
   |
                temp_result = fn(temp_result, a_storage[input_storage_position])
   |
                input_storage_position += reduction_stride
   |
            # Store the result back in the output storage
   |
            out[output_storage_position] = temp_result
   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #6).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (293) is hoisted out of
the parallel loop labelled #6 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: output_multi_dim_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\User\Repositories\Cornell-
Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (312)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\User\Repositories\Cornell-Tech\CS-5781\mod3-AaronGoldblatt\minitorch\fast_ops.py (312)
-------------------------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                                             |
    out: Storage,                                                                                                        |
    out_shape: Shape,                                                                                                    |
    out_strides: Strides,                                                                                                |
    a_storage: Storage,                                                                                                  |
    a_shape: Shape,                                                                                                      |
    a_strides: Strides,                                                                                                  |
    b_storage: Storage,                                                                                                  |
    b_shape: Shape,                                                                                                      |
    b_strides: Strides,                                                                                                  |
) -> None:                                                                                                               |
    """NUMBA tensor matrix multiply function.                                                                            |
                                                                                                                         |
    Should work for any tensor shapes that broadcast as long as                                                          |
                                                                                                                         |
    ```                                                                                                                  |
    assert a_shape[-1] == b_shape[-2]                                                                                    |
    ```                                                                                                                  |
                                                                                                                         |
    Optimizations:                                                                                                       |
                                                                                                                         |
    * Outer loop in parallel                                                                                             |
    * No index buffers or function calls                                                                                 |
    * Inner loop should have no global writes, 1 multiply.                                                               |
                                                                                                                         |
                                                                                                                         |
    Args:                                                                                                                |
    ----                                                                                                                 |
        out (Storage): storage for `out` tensor                                                                          |
        out_shape (Shape): shape for `out` tensor                                                                        |
        out_strides (Strides): strides for `out` tensor                                                                  |
        a_storage (Storage): storage for `a` tensor                                                                      |
        a_shape (Shape): shape for `a` tensor                                                                            |
        a_strides (Strides): strides for `a` tensor                                                                      |
        b_storage (Storage): storage for `b` tensor                                                                      |
        b_shape (Shape): shape for `b` tensor                                                                            |
        b_strides (Strides): strides for `b` tensor                                                                      |
                                                                                                                         |
    Returns:                                                                                                             |
    -------                                                                                                              |
        None : Fills in `out`                                                                                            |
                                                                                                                         |
    """                                                                                                                  |
    # Ensure the dimensions are compatible for matrix multiplication                                                     |
    assert a_shape[-1] == b_shape[-2], "Incompatible dimensions for matrix multiplication"                               |
                                                                                                                         |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                               |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                               |
                                                                                                                         |
    # TODO: Implement for Task 3.2.                                                                                      |
    # Loop over the batch dimension, which corresponds to out_shape[0]                                                   |
    for batch_index in prange(out_shape[0]):-----------------------------------------------------------------------------| #9
        # Loop over the first dimension of tensor 'a'                                                                    |
        for row_index in prange(out_shape[1]):---------------------------------------------------------------------------| #8
            # Loop over the second dimension of tensor 'b'                                                               |
            for col_index in prange(out_shape[2]):-----------------------------------------------------------------------| #7
                # Calculate the starting positions in a_storage and b_storage using batch and row/column indices         |
                a_position = batch_index * a_batch_stride + row_index * a_strides[1]                                     |
                b_position = batch_index * b_batch_stride + col_index * b_strides[2]                                     |
                # Initialize accumulator for the dot product                                                             |
                dot_product_accumulator = 0.0                                                                            |
                # Compute the dot product over the shared dimension (2nd of 'a' and 1st of 'b')                          |
                for shared_dim_index in range(a_shape[2]):                                                               |
                    dot_product_accumulator += a_storage[a_position] * b_storage[b_position]                             |
                    # Update positions in the shared dimension using strides                                             |
                    a_position += a_strides[2]                                                                           |
                    b_position += b_strides[1]                                                                           |
                # Calculate the position in the output tensor and store the accumulated result                           |
                out_position = batch_index * out_strides[0] + row_index * out_strides[1] + col_index * out_strides[2]    |
                out[out_position] = dot_product_accumulator                                                              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #9, #8).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--9 is a parallel loop
   +--8 --> rewritten as a serial loop
      +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (parallel)
      +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (serial)
      +--7 (serial)



Parallel region 0 (loop #9) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#9).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Task 3.4
### Task 3.4.1: Comparison Graph for Matrix Multiplication Using Fast (CPU) and GPU
**Regular Graph**:
<img src="images\task3_4\Performance Comparison Regular Graph.png" width="50%">

**Log-Log Graph**:
<img src="images\task3_4\Performance Comparison Log-Log Graph.png" width="50%">

### Task 3.4.2: Comparison Graph Generation Script Log
```
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
Running size 64
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.003069718678792318), 'gpu': np.float64(0.0057163238525390625)}
Running size 128
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.015105644861857096), 'gpu': np.float64(0.013212283452351889)}
Running size 256
{'fast': np.float64(0.09365320205688477), 'gpu': np.float64(0.045696099599202476)}
Running size 512
{'fast': np.float64(0.9755202134450277), 'gpu': np.float64(0.18802698453267416)}
Running size 1024
{'fast': np.float64(8.23149315516154), 'gpu': np.float64(0.9942088921864828)}

Timing summary
Size: 64
    fast: 0.00307
    gpu: 0.00572
Size: 128
    fast: 0.01511
    gpu: 0.01321
Size: 256
    fast: 0.09365
    gpu: 0.04570
Size: 512
    fast: 0.97552
    gpu: 0.18803
Size: 1024
    fast: 8.23149
    gpu: 0.99421
```

## Task 3.5
### Task 3.5.1: Simple Dataset
#### Task 3.5.1.1: CPU - Simple Dataset
50 data points
Time Per Epoch: 0.119s

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 500
- hidden layers: 100

<img src="images\task3_5\1. simple\cpu\1. Dataset.png" width="50%">
<img src="images\task3_5\1. simple\cpu\2. Hidden Layer and Initial Setting.png" width="50%">
<img src="images\task3_5\1. simple\cpu\3. Hyperparameters and Results.png" width="50%">
<img src="images\task3_5\1. simple\cpu\4. Loss Graph and Table.png" width="50%">

**Simple CPU Training Log**:
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 4.805775014532324, correct: 41
Epoch: 10/500, loss: 2.740380589045123, correct: 48
Epoch: 20/500, loss: 1.08828740725263, correct: 48
Epoch: 30/500, loss: 1.7753023224835316, correct: 50
Epoch: 40/500, loss: 0.4682037288420724, correct: 50
Epoch: 50/500, loss: 1.4262402418916114, correct: 50
Epoch: 60/500, loss: 0.5436520641654073, correct: 50
Epoch: 70/500, loss: 0.4555430536305608, correct: 50
Epoch: 80/500, loss: 0.39999040409596637, correct: 50
Epoch: 90/500, loss: 0.6873504616378866, correct: 50
Epoch: 100/500, loss: 0.8995280664624152, correct: 50
Epoch: 110/500, loss: 0.4710381460244473, correct: 50
Epoch: 120/500, loss: 0.2902585023946693, correct: 50
Epoch: 130/500, loss: 0.18828286751641674, correct: 50
Epoch: 140/500, loss: 0.6287993423051236, correct: 50
Epoch: 150/500, loss: 0.3874868071618293, correct: 50
Epoch: 160/500, loss: 0.5772169389198476, correct: 50
Epoch: 170/500, loss: 0.6002718480311565, correct: 50
Epoch: 180/500, loss: 0.5707255452743282, correct: 50
Epoch: 190/500, loss: 0.4618634291238531, correct: 50
Epoch: 200/500, loss: 0.21012690006681178, correct: 50
Epoch: 210/500, loss: 0.052632776192168045, correct: 50
Epoch: 220/500, loss: 0.009424580182087416, correct: 50
Epoch: 230/500, loss: 0.34160723289150996, correct: 50
Epoch: 240/500, loss: 0.02035004436328414, correct: 50
Epoch: 250/500, loss: 0.18825006185083387, correct: 50
Epoch: 260/500, loss: 0.3716514756053981, correct: 50
Epoch: 270/500, loss: 0.5831821989732409, correct: 50
Epoch: 280/500, loss: 0.09030381876148255, correct: 50
Epoch: 290/500, loss: 0.1401395376317558, correct: 50
Epoch: 300/500, loss: 0.17561787995990652, correct: 50
Epoch: 310/500, loss: 0.012883388055918356, correct: 50
Epoch: 320/500, loss: 0.09221706710222721, correct: 50
Epoch: 330/500, loss: 0.18883892387134474, correct: 50
Epoch: 340/500, loss: 0.17557202639843564, correct: 50
Epoch: 350/500, loss: 0.13033913458665114, correct: 50
Epoch: 360/500, loss: 0.13087921495246285, correct: 50
Epoch: 370/500, loss: 0.321010281112334, correct: 50
Epoch: 380/500, loss: 0.18367194090018593, correct: 50
Epoch: 390/500, loss: 0.015505298777980946, correct: 50
Epoch: 400/500, loss: 0.15579651925649116, correct: 50
Epoch: 410/500, loss: 0.09627793122304669, correct: 50
Epoch: 420/500, loss: 0.4450169025882106, correct: 50
Epoch: 430/500, loss: 0.06161252679437731, correct: 50
Epoch: 440/500, loss: 0.01952400116969191, correct: 50
Epoch: 450/500, loss: 0.08922999468879295, correct: 50
Epoch: 460/500, loss: 0.08282170918195776, correct: 50
Epoch: 470/500, loss: 0.03445276561871205, correct: 50
Epoch: 480/500, loss: 0.23121485120305033, correct: 50
Epoch: 490/500, loss: 0.0534413205857732, correct: 50
```

#### Task 3.5.1.2: GPU - Simple Dataset
50 data points
Time Per Epoch: Xs

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 500
- hidden layers: 100

**Simple GPU Training Log**:
```
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  4.6654938871948355 correct 36
Epoch  10  loss  1.674902513787757 correct 47
Epoch  20  loss  0.1810884002220403 correct 50
Epoch  30  loss  1.4058473947962085 correct 50
Epoch  40  loss  0.38120674415148204 correct 47
Epoch  50  loss  0.23493432987547308 correct 50
Epoch  60  loss  0.12105216818362133 correct 50
Epoch  70  loss  0.3592741880051916 correct 49
Epoch  80  loss  0.5577781959289381 correct 50
Epoch  90  loss  1.050204253100938 correct 50
Epoch  100  loss  0.4085038386234219 correct 50
Epoch  110  loss  0.43276581206097664 correct 50
Epoch  120  loss  0.030628321218879893 correct 50
Epoch  130  loss  0.0206182327136636 correct 50
Epoch  140  loss  0.11066356136791064 correct 50
Epoch  150  loss  0.21893727776157373 correct 50
Epoch  160  loss  0.13058431891640648 correct 50
Epoch  170  loss  0.09045138427690663 correct 50
Epoch  180  loss  0.09873688959104797 correct 50
Epoch  190  loss  0.22712419663202585 correct 50
Epoch  200  loss  0.1788859338671556 correct 50
Epoch  210  loss  0.1292295384537704 correct 50
Epoch  220  loss  0.003604703857580745 correct 50
Epoch  230  loss  0.16878754654668088 correct 50
Epoch  240  loss  0.21182021197921194 correct 50
Epoch  250  loss  0.3290085972522911 correct 50
Epoch  260  loss  0.11484384964662744 correct 50
Epoch  270  loss  0.0003600811349889055 correct 50
Epoch  280  loss  0.04294582987299248 correct 50
Epoch  290  loss  0.03531995719082791 correct 50
Epoch  300  loss  0.04556083316897192 correct 50
Epoch  310  loss  0.12401938115924777 correct 50
Epoch  320  loss  0.0003475691559147917 correct 50
Epoch  330  loss  0.074572511064606 correct 50
Epoch  340  loss  0.006151793735319139 correct 50
Epoch  350  loss  0.01939479285074038 correct 50
Epoch  360  loss  9.107840732339743e-05 correct 50
Epoch  370  loss  0.02125473249380904 correct 50
Epoch  380  loss  0.1376800090267279 correct 50
Epoch  390  loss  0.03499769323721483 correct 50
Epoch  400  loss  0.01179178676048381 correct 50
Epoch  410  loss  0.01973186685792328 correct 50
Epoch  420  loss  0.018091544642520208 correct 50
Epoch  430  loss  0.0062634492661401715 correct 50
Epoch  440  loss  0.09963730566927269 correct 50
Epoch  450  loss  0.008258677886920406 correct 50
Epoch  460  loss  0.0002887364755850184 correct 50
Epoch  470  loss  9.310347213468984e-05 correct 50
Epoch  480  loss  0.07774002141897965 correct 50
Epoch  490  loss  0.014184310672249905 correct 50

real	14m6.195s
user	13m55.857s
sys	0m5.275s
```

### Task 3.5.2: Split Dataset
#### Task 3.5.2.1: CPU - Split Dataset
50 data points
Time Per Epoch: 0.105s

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 500
- hidden layers: 100

<img src="images\task3_5\2. split\cpu\1. Dataset.png" width="50%">
<img src="images\task3_5\2. split\cpu\2. Hidden Layer and Initial Setting.png" width="50%">
<img src="images\task3_5\2. split\cpu\3. Hyperparameters and Results.png" width="50%">
<img src="images\task3_5\2. split\cpu\4. Loss Graph and Table.png" width="50%">

**Split CPU Training Log**:
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 7.989069141689213, correct: 25
Epoch: 10/500, loss: 6.347317547308546, correct: 37
Epoch: 20/500, loss: 5.467303616931577, correct: 43
Epoch: 30/500, loss: 5.825555631413826, correct: 44
Epoch: 40/500, loss: 3.6674669235930497, correct: 36
Epoch: 50/500, loss: 3.616152209075461, correct: 45
Epoch: 60/500, loss: 3.506027654270239, correct: 48
Epoch: 70/500, loss: 2.418524602729474, correct: 41
Epoch: 80/500, loss: 2.309150275783908, correct: 50
Epoch: 90/500, loss: 1.629376447479323, correct: 40
Epoch: 100/500, loss: 1.6683380502290135, correct: 45
Epoch: 110/500, loss: 2.1067929935104184, correct: 50
Epoch: 120/500, loss: 1.7176511090556055, correct: 50
Epoch: 130/500, loss: 1.7811709541178282, correct: 50
Epoch: 140/500, loss: 1.9177802631241259, correct: 49
Epoch: 150/500, loss: 0.8998794109727732, correct: 50
Epoch: 160/500, loss: 2.6606444650402867, correct: 49
Epoch: 170/500, loss: 1.1599114124366696, correct: 50
Epoch: 180/500, loss: 1.0448853412903072, correct: 50
Epoch: 190/500, loss: 0.9002598884884636, correct: 50
Epoch: 200/500, loss: 0.8317498650509674, correct: 50
Epoch: 210/500, loss: 2.1951870486249567, correct: 44
Epoch: 220/500, loss: 0.5277485747187025, correct: 50
Epoch: 230/500, loss: 1.2750675313360251, correct: 50
Epoch: 240/500, loss: 0.8713298171940196, correct: 50
Epoch: 250/500, loss: 0.7517760128954043, correct: 50
Epoch: 260/500, loss: 0.63557938894386, correct: 50
Epoch: 270/500, loss: 0.46651005884639507, correct: 50
Epoch: 280/500, loss: 0.3000842604017046, correct: 50
Epoch: 290/500, loss: 0.9539312294385657, correct: 50
Epoch: 300/500, loss: 0.3293366087053279, correct: 50
Epoch: 310/500, loss: 0.329204797451658, correct: 50
Epoch: 320/500, loss: 0.19266406040110456, correct: 50
Epoch: 330/500, loss: 0.5229811330747368, correct: 50
Epoch: 340/500, loss: 0.7875289089684194, correct: 50
Epoch: 350/500, loss: 0.19168567847548898, correct: 50
Epoch: 360/500, loss: 0.36246322501520123, correct: 50
Epoch: 370/500, loss: 0.47209855287146585, correct: 50
Epoch: 380/500, loss: 0.40933514946177435, correct: 50
Epoch: 390/500, loss: 0.24093726134491503, correct: 50
Epoch: 400/500, loss: 0.14252686126354722, correct: 50
Epoch: 410/500, loss: 0.3593436502031117, correct: 50
Epoch: 420/500, loss: 0.3025745609149297, correct: 50
Epoch: 430/500, loss: 0.08738830419208904, correct: 50
Epoch: 440/500, loss: 0.41248805330159993, correct: 50
Epoch: 450/500, loss: 0.13531257790206275, correct: 50
Epoch: 460/500, loss: 0.23803636207118423, correct: 50
Epoch: 470/500, loss: 0.1014722642725165, correct: 50
Epoch: 480/500, loss: 0.22905008093060839, correct: 50
Epoch: 490/500, loss: 0.1802670566387022, correct: 50
```

#### Task 3.5.2.2: GPU - Split Dataset
50 data points
Time Per Epoch: Xs

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 500
- hidden layers: 100

**Split GPU Training Log**:
```
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  7.174588981115167 correct 24
Epoch  10  loss  5.566592020587255 correct 36
Epoch  20  loss  6.713155594622647 correct 41
Epoch  30  loss  5.332420258789707 correct 43
Epoch  40  loss  4.59138173333331 correct 46
Epoch  50  loss  2.768046146388307 correct 47
Epoch  60  loss  3.0595139652943875 correct 48
Epoch  70  loss  2.9778377019516973 correct 47
Epoch  80  loss  2.583398926125989 correct 47
Epoch  90  loss  1.63391423595501 correct 49
Epoch  100  loss  2.916437841464219 correct 46
Epoch  110  loss  1.87748384854277 correct 50
Epoch  120  loss  3.3763308338905786 correct 49
Epoch  130  loss  0.9795868661243698 correct 46
Epoch  140  loss  0.49161417872956925 correct 50
Epoch  150  loss  1.407628483262446 correct 49
Epoch  160  loss  0.9514994345577021 correct 50
Epoch  170  loss  0.7472536324009353 correct 49
Epoch  180  loss  1.8115774013091381 correct 50
Epoch  190  loss  0.8481100961492758 correct 50
Epoch  200  loss  0.6427490016884864 correct 50
Epoch  210  loss  0.8063370947477364 correct 49
Epoch  220  loss  0.6318561947682341 correct 50
Epoch  230  loss  1.2023276683090114 correct 50
Epoch  240  loss  0.23398489653336282 correct 50
Epoch  250  loss  0.22139919105851527 correct 50
Epoch  260  loss  0.649550097414852 correct 50
Epoch  270  loss  1.1818544574924335 correct 50
Epoch  280  loss  0.4437170156803806 correct 50
Epoch  290  loss  0.4133712803217894 correct 50
Epoch  300  loss  1.2740417037139824 correct 47
Epoch  310  loss  0.14001415280262505 correct 47
Epoch  320  loss  1.1425119457252364 correct 50
Epoch  330  loss  0.11376943855978065 correct 50
Epoch  340  loss  1.4674585533303417 correct 48
Epoch  350  loss  0.41823131843615674 correct 50
Epoch  360  loss  0.49291171500067227 correct 50
Epoch  370  loss  0.10271416795997695 correct 47
Epoch  380  loss  0.1970925406832982 correct 50
Epoch  390  loss  0.061366879704291645 correct 50
Epoch  400  loss  0.1470820180998377 correct 48
Epoch  410  loss  0.16102868407656937 correct 50
Epoch  420  loss  0.7910155383264164 correct 50
Epoch  430  loss  0.36713482461248664 correct 50
Epoch  440  loss  0.2354876531830829 correct 49
Epoch  450  loss  0.0504006778910423 correct 50
Epoch  460  loss  0.2695600372359634 correct 50
Epoch  470  loss  0.056674187671676056 correct 50
Epoch  480  loss  0.08729000634094516 correct 49
Epoch  490  loss  0.8290379446356428 correct 50

real	13m58.093s
user	13m48.856s
sys	0m5.239s
```

### Task 3.5.3: XOR Dataset
#### Task 3.5.3.1: CPU - XOR Dataset
50 data points
Time Per Epoch: 0.075s

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 500
- hidden layers: 100

<img src="images\task3_5\3. xor\cpu\1. Dataset.png" width="50%">
<img src="images\task3_5\3. xor\cpu\2. Hidden Layer and Initial Setting.png" width="50%">
<img src="images\task3_5\3. xor\cpu\3. Hyperparameters and Results.png" width="50%">
<img src="images\task3_5\3. xor\cpu\4. Loss Graph and Table.png" width="50%">

**XOR CPU Training Log**:
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 6.542948087247501, correct: 34
Epoch: 10/500, loss: 4.630086478939592, correct: 39
Epoch: 20/500, loss: 3.9344131251842507, correct: 38
Epoch: 30/500, loss: 2.0663080971314263, correct: 44
Epoch: 40/500, loss: 4.6671104705732915, correct: 45
Epoch: 50/500, loss: 2.9447047580406647, correct: 46
Epoch: 60/500, loss: 3.1108997970072325, correct: 45
Epoch: 70/500, loss: 0.874716160152569, correct: 46
Epoch: 80/500, loss: 1.0381219014727905, correct: 45
Epoch: 90/500, loss: 3.5171121525670364, correct: 45
Epoch: 100/500, loss: 3.1149919280153404, correct: 47
Epoch: 110/500, loss: 1.3902817100686817, correct: 45
Epoch: 120/500, loss: 2.2475425922226235, correct: 45
Epoch: 130/500, loss: 5.056761454128022, correct: 44
Epoch: 140/500, loss: 2.712723572527033, correct: 46
Epoch: 150/500, loss: 1.0651764591063235, correct: 46
Epoch: 160/500, loss: 1.4815504599924199, correct: 48
Epoch: 170/500, loss: 1.1804330653533477, correct: 47
Epoch: 180/500, loss: 1.9896937587326768, correct: 46
Epoch: 190/500, loss: 3.6755995324166757, correct: 46
Epoch: 200/500, loss: 0.9999967102367949, correct: 47
Epoch: 210/500, loss: 1.7333363652695448, correct: 48
Epoch: 220/500, loss: 0.5566897719609589, correct: 47
Epoch: 230/500, loss: 3.0951629220381216, correct: 47
Epoch: 240/500, loss: 0.27253170687332634, correct: 48
Epoch: 250/500, loss: 2.469868463861854, correct: 48
Epoch: 260/500, loss: 1.8876338805197028, correct: 46
Epoch: 270/500, loss: 2.0094040823894503, correct: 47
Epoch: 280/500, loss: 1.0898059030851632, correct: 47
Epoch: 290/500, loss: 1.8635859838902056, correct: 48
Epoch: 300/500, loss: 0.9842607849666166, correct: 47
Epoch: 310/500, loss: 1.1773673682379786, correct: 48
Epoch: 320/500, loss: 0.5295861301393505, correct: 48
Epoch: 330/500, loss: 0.5593698881727903, correct: 48
Epoch: 340/500, loss: 0.8609865140515897, correct: 48
Epoch: 350/500, loss: 1.9432724557165368, correct: 48
Epoch: 360/500, loss: 1.1831242977284055, correct: 49
Epoch: 370/500, loss: 2.7607178496647085, correct: 48
Epoch: 380/500, loss: 0.3144552916398004, correct: 50
Epoch: 390/500, loss: 1.4999953721903834, correct: 49
Epoch: 400/500, loss: 1.2591031938791004, correct: 50
Epoch: 410/500, loss: 1.5322196603899092, correct: 48
Epoch: 420/500, loss: 1.0375728850641375, correct: 50
Epoch: 430/500, loss: 0.9071983991973679, correct: 48
Epoch: 440/500, loss: 1.5159769297231676, correct: 49
Epoch: 450/500, loss: 1.07934877012558, correct: 48
Epoch: 460/500, loss: 0.8410080798172681, correct: 49
Epoch: 470/500, loss: 1.420589213568013, correct: 49
Epoch: 480/500, loss: 1.3869616115339907, correct: 50
Epoch: 490/500, loss: 1.2997807902602128, correct: 50
```

#### Task 3.5.3.2: GPU - XOR Dataset
50 data points
Time Per Epoch: Xs

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 500
- hidden layers: 200

<img src="images\task3_5\4. bigger model - simple\cpu\1. Dataset.png" width="50%">
<img src="images\task3_5\4. bigger model - simple\cpu\2. Hidden Layer and Initial Setting.png" width="50%">
<img src="images\task3_5\4. bigger model - simple\cpu\3. Hyperparameters and Results.png" width="50%">
<img src="images\task3_5\4. bigger model - simple\cpu\4. Loss Graph and Table.png" width="50%">

**XOR GPU Training Log**:
```
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  6.752775366755557 correct 40
Epoch  10  loss  4.165446399049184 correct 43
Epoch  20  loss  3.6279098166990456 correct 42
Epoch  30  loss  4.380567873904038 correct 44
Epoch  40  loss  3.6747068203986353 correct 44
Epoch  50  loss  5.829683138372165 correct 44
Epoch  60  loss  1.919534169836239 correct 44
Epoch  70  loss  3.8756328612148465 correct 42
Epoch  80  loss  4.730898451200794 correct 47
Epoch  90  loss  2.557283476508084 correct 45
Epoch  100  loss  1.8236390011625052 correct 46
Epoch  110  loss  2.6372157670648746 correct 48
Epoch  120  loss  1.0675620488141417 correct 46
Epoch  130  loss  2.1490234340303624 correct 47
Epoch  140  loss  0.4545892901023372 correct 48
Epoch  150  loss  4.991099035357574 correct 47
Epoch  160  loss  3.2702272663180407 correct 48
Epoch  170  loss  2.7209867424705183 correct 46
Epoch  180  loss  0.9106612422338731 correct 48
Epoch  190  loss  0.5329384386261294 correct 48
Epoch  200  loss  1.3247924224660659 correct 48
Epoch  210  loss  1.8743297917823638 correct 48
Epoch  220  loss  0.9931501413636481 correct 48
Epoch  230  loss  1.5286117809439763 correct 48
Epoch  240  loss  1.1292557128576197 correct 48
Epoch  250  loss  1.6062809948143113 correct 48
Epoch  260  loss  0.5303214202898513 correct 49
Epoch  270  loss  1.6708807907081824 correct 49
Epoch  280  loss  0.4425402207182916 correct 48
Epoch  290  loss  1.9128639130664797 correct 49
Epoch  300  loss  0.2367383058207615 correct 48
Epoch  310  loss  2.327740986807623 correct 50
Epoch  320  loss  0.6383725738778308 correct 48
Epoch  330  loss  1.2753898935577705 correct 48
Epoch  340  loss  1.6682102067248814 correct 50
Epoch  350  loss  1.0107512632988551 correct 48
Epoch  360  loss  1.2133190717686297 correct 49
Epoch  370  loss  0.5628491745528185 correct 49
Epoch  380  loss  1.681710445030033 correct 50
Epoch  390  loss  0.11730566228568844 correct 48
Epoch  400  loss  0.24530056761978716 correct 49
Epoch  410  loss  0.321600565107234 correct 49
Epoch  420  loss  1.2462207558964096 correct 48
Epoch  430  loss  0.6711088340583627 correct 49
Epoch  440  loss  0.2212673255520656 correct 50
Epoch  450  loss  0.05791504534229715 correct 49
Epoch  460  loss  0.5233588007090495 correct 49
Epoch  470  loss  0.6178621256509126 correct 49
Epoch  480  loss  0.10745922467548087 correct 50
Epoch  490  loss  0.16298087598394023 correct 50

real	14m4.900s
user	13m55.615s
sys	0m5.147s
```

### Task 3.5.4: Bigger Model - Simple Dataset
#### Task 3.5.4.1: CPU - Bigger Model Simple Dataset
50 data points
Time Per Epoch: 0.167s

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 500
- hidden layers: 200

**Bigger Model Simple CPU Training Log**:
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 5.498057636435462, correct: 34
Epoch: 10/500, loss: 0.9512376201227832, correct: 49
Epoch: 20/500, loss: 1.2504020101736737, correct: 50
Epoch: 30/500, loss: 0.6095335332164868, correct: 50
Epoch: 40/500, loss: 0.4850106267377643, correct: 50
Epoch: 50/500, loss: 0.623814684545131, correct: 50
Epoch: 60/500, loss: 0.6966482911014259, correct: 50
Epoch: 70/500, loss: 0.6421993397624752, correct: 50
Epoch: 80/500, loss: 0.31230237779601894, correct: 50
Epoch: 90/500, loss: 0.24459157479439317, correct: 50
Epoch: 100/500, loss: 0.1528128677212402, correct: 50
Epoch: 110/500, loss: 0.28573384764394, correct: 50
Epoch: 120/500, loss: 0.06715525702282849, correct: 50
Epoch: 130/500, loss: 0.24030423617910532, correct: 50
Epoch: 140/500, loss: 0.06586992932636349, correct: 50
Epoch: 150/500, loss: 0.06437502224196236, correct: 50
Epoch: 160/500, loss: 0.281073474800087, correct: 50
Epoch: 170/500, loss: 0.39328396068044746, correct: 50
Epoch: 180/500, loss: 0.3260698794596112, correct: 50
Epoch: 190/500, loss: 0.3362148445642298, correct: 50
Epoch: 200/500, loss: 0.08201237034858734, correct: 50
Epoch: 210/500, loss: 0.19059585657276645, correct: 50
Epoch: 220/500, loss: 0.31950237733168324, correct: 50
Epoch: 230/500, loss: 0.06399987110663836, correct: 50
Epoch: 240/500, loss: 0.043989173551623954, correct: 50
Epoch: 250/500, loss: 0.28112521774997973, correct: 50
Epoch: 260/500, loss: 0.03307354229056451, correct: 50
Epoch: 270/500, loss: 0.12187381511144381, correct: 50
Epoch: 280/500, loss: 0.002248706031400742, correct: 50
Epoch: 290/500, loss: 0.212758348170272, correct: 50
Epoch: 300/500, loss: 0.07785902524924304, correct: 50
Epoch: 310/500, loss: 0.18814639792810806, correct: 50
Epoch: 320/500, loss: 0.05465355259292452, correct: 50
Epoch: 330/500, loss: 0.16591564914835405, correct: 50
Epoch: 340/500, loss: 0.18596290714154684, correct: 50
Epoch: 350/500, loss: 0.015727380873340295, correct: 50
Epoch: 360/500, loss: 0.0813479296703818, correct: 50
Epoch: 370/500, loss: 0.025532152479034457, correct: 50
Epoch: 380/500, loss: 0.14937419166547533, correct: 50
Epoch: 390/500, loss: 0.1637608488859727, correct: 50
Epoch: 400/500, loss: 0.03614023363199804, correct: 50
Epoch: 410/500, loss: 0.028518198121696134, correct: 50
Epoch: 420/500, loss: 0.014836399011646064, correct: 50
Epoch: 430/500, loss: 0.07448987677969282, correct: 50
Epoch: 440/500, loss: 0.05799701299206973, correct: 50
Epoch: 450/500, loss: 0.1402032070879377, correct: 50
Epoch: 460/500, loss: 0.04755704022501959, correct: 50
Epoch: 470/500, loss: 0.028451905552267696, correct: 50
Epoch: 480/500, loss: 0.14429004233328446, correct: 50
Epoch: 490/500, loss: 0.020382489791282758, correct: 50
```

#### Task 3.5.4.2: GPU - Bigger Model Simple Dataset
50 data points
Time Per Epoch: Xs

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 500
- hidden layers: 200

**Bigger Model Simple GPU Training Log**:
```
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 13 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 13 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 63 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 63 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 63 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 49 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 13 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 14 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  11.738022631502265 correct 46
Epoch  10  loss  1.221322714613364 correct 49
Epoch  20  loss  1.1230809823206425 correct 49
Epoch  30  loss  1.343979259176759 correct 50
Epoch  40  loss  0.2567169311037842 correct 50
Epoch  50  loss  0.009938769414168255 correct 50
Epoch  60  loss  0.2771472200298923 correct 50
Epoch  70  loss  0.06129137876221251 correct 50
Epoch  80  loss  0.49803254383841145 correct 49
Epoch  90  loss  0.27925800298258474 correct 50
Epoch  100  loss  0.83525811472838 correct 50
Epoch  110  loss  0.4021695329413916 correct 50
Epoch  120  loss  0.3941231119688402 correct 50
Epoch  130  loss  0.015105117312317493 correct 50
Epoch  140  loss  0.1531359506901961 correct 50
Epoch  150  loss  0.03490245610278743 correct 50
Epoch  160  loss  0.09663591985019154 correct 50
Epoch  170  loss  0.00025322624345060424 correct 50
Epoch  180  loss  0.08087783919759745 correct 50
Epoch  190  loss  0.36231811585012713 correct 50
Epoch  200  loss  0.1667651570131819 correct 50
Epoch  210  loss  0.025436842944655123 correct 50
Epoch  220  loss  0.13975961753964375 correct 50
Epoch  230  loss  0.0017827848858911722 correct 50
Epoch  240  loss  0.5965020828773185 correct 50
Epoch  250  loss  0.0012049577501712615 correct 50
Epoch  260  loss  0.3060157830132207 correct 50
Epoch  270  loss  0.00202885644051991 correct 50
Epoch  280  loss  0.0024722003553995058 correct 50
Epoch  290  loss  0.12718586512242355 correct 50
Epoch  300  loss  0.31687749093092676 correct 50
Epoch  310  loss  0.4151249893787584 correct 50
Epoch  320  loss  0.08618998578385574 correct 50
Epoch  330  loss  0.05218502512636108 correct 50
Epoch  340  loss  0.015957401571514707 correct 50
Epoch  350  loss  0.10579776042790467 correct 50
Epoch  360  loss  0.26859146100165454 correct 50
Epoch  370  loss  0.253605260682983 correct 50
Epoch  380  loss  0.07055298325840992 correct 50
Epoch  390  loss  0.059034306661555944 correct 50
Epoch  400  loss  0.08476612863149573 correct 50
Epoch  410  loss  0.00628877936620914 correct 50
Epoch  420  loss  0.1226779839070192 correct 50
Epoch  430  loss  0.04172086790609712 correct 50
Epoch  440  loss  0.0003322442772927407 correct 50
Epoch  450  loss  0.04172434326226778 correct 50
Epoch  460  loss  0.09742442783232033 correct 50
Epoch  470  loss  0.10814270733796325 correct 50
Epoch  480  loss  0.1354871854162931 correct 50
Epoch  490  loss  0.1420711866130135 correct 50

real	14m54.037s
user	14m41.803s
sys	0m5.748s
```
