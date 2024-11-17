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
### Diagnostic Output of Script `project/parallel_check.py`
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
