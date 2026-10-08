# Speeding up operations research algorithms on parallel hardware

## Introduction

### Why GPUs?

- Single cores are no longer getting much faster, computing power increases from parallelism
- Initially designed for computer graphics, processing pixels in parallel
- CUDA (2007) made the switch to a general programming interface
- Alexnet (2012) showed they can be used in deep learning
- Most supercomputing capabilities are now GPU-driven, but they are in every laptop too

### CPU versus GPU

- CPU: few cores optimized for sequential, complex operations, low latency, large memory
- GPU: many cores optimized for parallel, simple operations, high throughput, small memory
- GPUs perform many floating point operations per second (100 TFLOPS on Float32, 100x that of CPUs)[^1]
- GPUs have high memory bandwidth (1000 GB/s, 10x that of CPUs)

### Challenges of parallelization

- Additional complexity: performing more work
- Memory-bound applications
- Heterogeneous, unpredictable input data
- Synchronization operations

## How a GPU works

### Communication

- CPU is the host, GPUs are the devices
- Device code is composed of kernels, executed in a data-parallel manner
- Kernels are interlaced with host instructions, launched asynchronously
- Transfer of data or instructions between CPU and GPU is expensive

### Theoretical structure

- Grid of blocks (1D, 2D or 3D), each with up to 1024 threads
- Within a block, threads are executed in lockstep
- Single Instruction, Multiple Thread paradigm

### Hardware structure

- Several Streaming Multiprocessors (142 on the L40s), each with many CUDA cores (128 on the L40s)
- All threads in a block are assigned to the same SM, but one SM will get several blocks
- Threads within a block are physically in the same place and can interact with one another (synchronization, memory)
- Limited number of SMs + limited number of blocks executed at the same time by an SM = no guarantee on the simultaneity or sequence between two blocks

### Warps and divergence

- Blocks are split into warps of 32 threads which are physically executed in lockstep
- In the presence of control flow, all paths are taken sequentially: one pass for `if` then one pass for `else`
- Typical example is boundary conditions (only affects one warp)

### Example kernel

- Each thread knows its block and thread ID, which replaces a loop

``` julia
using KernelAbstractions
@kernel function mul2_kernel(A)
  I = @index(Global)
  A[I] = 2 * A[I]
end
```

- Grid and block shapes are specified by host at kernel launch

``` julia
import CUDA
dev = CUDABackend()
A = ones(1024, 1024)
ev = mul2_kernel(dev, 64)(A, ndrange=size(A))
synchronize(dev)
all(A .== 2.0)
```

### Matmul kernel

Grid maps to output matrix, one thread per dot product:

``` julia
using KernelAbstractions

@kernel function matmul_kernel!(output, a, b)
    i, j = @index(Global, NTuple)
    tmp_sum = zero(eltype(output))
    for k in axes(a, 2)
        tmp_sum += a[i, k] * b[k, j]
    end
    output[i, j] = tmp_sum
end

function matmul!(output, a, b)
    backend = get_backend(a)
    kernel! = matmul_kernel!(backend)
    kernel!(output, a, b, ndrange = size(output))
    return
end
```

## Performance considerations

### Roofline model

- Key quantity: compute to global memory access ratio (or computational intensity), expressed in FLOP/B
- Example: matrix multiplication
- Roofline diagram: computational intensity (FLOP/B) as X axis, computational throughput (GFLOP/s) as Y axis
- Two regimes: memory-bound (\\\alpha\\ = peak bandwidth in B/s times computational throughput) then compute-bound (\\\beta\\ = peak throughput in GB/s)

### Memory hierarchy

- Host memory + several kinds of on-device memory
- Global memory (or DRAM), per grid
  - Constant memory, part of global but read-only
  - Local memory, part of global but owned by thread
- Shared (or scratchpad) memory, per block
- Registers, per thread

### Fighting memory limits

- Improve data reuse to reach peak FLOPS
- Use shared memory as scratchpad to store data from global memory
- Coalesce global memory accesses (make sure adjacent threads access adjacent memory locations)
- Example: tiling for matmul
- Warps are ordered linearly (column-major vs row-major, depending on the language)

### Resource allocation

- Hardware-dependent resource allocation enables transparent scalability
- (Too) many threads/warps per SM is good for latency tolerance
- But each thread induces a certain occupancy (amount of memory, registers, etc)

### Launching kernels

- Kernel launch has overhead
- Kernel fusion at program scale improves performance

## Sparse matrices and graphs

### Sparse matrix formats

- COO (coordinate-wise)
- CSR (row-wise)
- ELL (CSR + data padding + transposition)
- SELL (ELL by chunk)
- JDS (CSR + sort by row length)
- Hybrids
- Tradeoffs:
  - space efficiency
  - accessibility (coefficient, row, column)
  - flexibility (modifying, appending)

### SpMV kernels

- COO (one thread per input coefficient): atomic operations
- CSR (one thread per input row): load balancing, divergence and coalescing issues
- ELL (one thread per input row): low space efficiency, divergence issues

### Are fancy sparse formats worth it?

- For one-off computations, no
- For repeated computations, yes
- Either way, SpMV much less efficient than dense equivalent: no data reuse

### Graph algorithms with linear algebra

- GraphBLAS specification
- Custom semirings
- Example: BFS

## Shortest paths and preprocessing

### Forget about Dijkstra

- Real graphs are huge but don’t move
- Preprocessing the graph can lead to huge speedups (6 orders of magnitude)
- Done in practice for continental-size applications like Google Maps
- Needs adaptation if the weights change

### Other applications of preprocessing

- Decomposition methods (multicommodity flow)
- Fixed point algorithms (traffic assignment)
- Decision-focused learning

### Contraction hierarchies

- Iteratively contract nodes according to a priority order
- Add shortcuts to preserve shortest path distances while reducing diameter
- At query time, search in shallow acyclic graphs
- Better ordering means longer preprocessing but less fill-in

### Running it on the GPU (PHAST)

- Get rid of expensive data structures (priority queues)
- Improve memory locality by reordering vertices
- In the topological order, process levels in parallel
- Process several shortest path trees in parallel

## Linear and integer programming

### Algorithms for LP

- Simplex: variable-wise updates, linear solve
- Interior point: linear solve
- Primal-dual methods: no linear solve
- Visualization on <https://lpviz.net/>

### What’s wrong with linear solves

- Memory overhead of the factorization
- Sparse factorization and updates
- Doesn’t scale well on GPU
- Not necessarily the bottleneck

### PDLP

- First-order algorithm with primal and dual variables
- Just matrix-vector multiplications (SpMV)
- Low precision, high scalability
- Original CPU and GPU implementations in Julia (Google, then MIT)

### What makes it go brrrr

- Preconditioning
- Restarting
- Adaptive step size
- Averaging / reflection
- Feasibility polishing

### Areas of improvement

- Presolve
- Crossover
- Hyperparameter tuning

### Algorithms for MILP

- Branch & bound: very sequential
- Cutting planes: very sequential
- Primal heuristics: sometimes parallelizable

### GPU-friendly MILP heuristics

- PDLP as the relaxation solver
- Feasibility pump
- Fix-and-propagate
- Large neighborhood search
- Collaboration between CPU and GPU

## Julia tooling

### CUDA-specific code

- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- Math libraries: cuBLAS.jl, cuSPARSE.jl, etc.
- [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl)

### Hardware-agnostic code

- [GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl)
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
- [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl)
- [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)

### Deep learning

- [Enzyme.jl](https://github.com/EnzymeAd/Enzyme.jl)
- [Lux.jl](https://github.com/LuxDL/Lux.jl)

### Mathematical programming

Modeling (including batched):

- [JuMP.jl](https://github.com/jump-dev/JuMP.jl)[^2]
- [Convex.jl](https://github.com/jump-dev/Convex.jl)
- [ExaModels.jl](https://github.com/exanauts/ExaModels.jl)
- [BatchNLPKernels.jl](https://github.com/LearningToOptimize/BatchNLPKernels.jl)
- Experimental: [BatchOptInterface.jl](https://github.com/klamike/BatchOptInterface.jl), [BatchQuadraticModels.jl](https://github.com/klamike/BatchQuadraticModels.jl)

Solvers:

- [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel)
- [CoolPDLP.jl](https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl)
- [cuOpt.jl](https://github.com/jump-dev/cuOpt.jl)
- [cuPDLPx.jl](https://github.com/MIT-Lu-Lab/CuPDLPx.jl)
- [MadIPM.jl](https://github.com/MadNLP/MadIPM.jl)
- [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl)
- [SCS.jl](https://github.com/jump-dev/SCS.jl)

### Graph algorithms

Nothing… yet

## Python tooling

### CUDA-specific code

- [cupy](https://github.com/cupy/cupy)
- [nvmath-python](https://github.com/NVIDIA/nvmath-python)
- [cutile-python](https://github.com/nvidia/cutile-python)

### Hardware-agnostic code

- [Array API](https://github.com/data-apis/array-api/)
- [DLPack](https://github.com/dmlc/dlpack)
- [numba](https://github.com/numba/numba)
- [triton](https://github.com/triton-lang/triton)
- [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)

### Deep learning

- [PyTorch](https://github.com/pytorch/pytorch)
- [JAX](https://github.com/jax-ml/jax)

### Mathematical programming

Modeling:

- [cvxpy](https://github.com/cvxpy/cvxpy/)
- [pyomo](https://github.com/pyomo/pyomo)

Solvers:

- [cuopt](https://github.com/NVIDIA/cuopt)
- [MPAX](https://github.com/MIT-Lu-Lab/MPAX)

### Graph algorithms

- [networkx](https://github.com/networkx/networkx)
- [cugraph](https://github.com/rapidsai/cugraph)

## References

Applegate, David, Mateo Diaz Diaz, Oliver Hinder, et al. 2021. “Practical Large-Scale Linear Programming Using Primal-Dual Hybrid Gradient.” Paper presented Thirty-Fifth Conference on Neural Information Processing Systems. May 21. <https://openreview.net/forum?id=_eXwwWOyqT_>.

Besard, Tim, Christophe Foket, and Bjorn De Sutter. 2019. “Effective Extensible Programming: Unleashing Julia on GPUs.” *IEEE Transactions on Parallel and Distributed Systems* 30 (4): 827–41. <https://doi.org/10.1109/TPDS.2018.2872064>.

Blin, Nicolas, Stefano Gualandi, Christopher Maes, Andrea Lodi, and Bartolomeo Stellato. 2026. “Batched First-Order Methods for Parallel LP Solving in MIP.” Version 1. Pre-published. <https://doi.org/10.48550/ARXIV.2601.21990>.

Çördük, Akif, Piotr Sielski, Alice Boucher, and Kumar Aatish. 2025. “GPU-Accelerated Primal Heuristics for Mixed Integer Programming.” Pre-published October 23. <https://doi.org/10.48550/arXiv.2510.20499>.

Hijma, Pieter, Stijn Heldens, Alessio Sclocco, Ben van Werkhoven, and Henri E. Bal. 2023. “Optimization Techniques for GPU Programming.” *ACM Comput. Surv.* 55 (11): 239:1–81. <https://doi.org/10.1145/3570638>.

Hwu, Wen-mei W., David B. Kirk, and Izzat El Hajj. 2022. *Programming Massively Parallel Processors: A Hands-on Approach*. Morgan Kaufmann. <https://books.google.com?id=7H9dEAAAQBAJ>.

Kepner, Jeremy, Peter Aaltonen, David Bader, et al. 2016. “Mathematical Foundations of the GraphBLAS.” *2016 IEEE High Performance Extreme Computing Conference (HPEC)*, September, 1–9. <https://doi.org/10.1109/HPEC.2016.7761646>.

Lu, Haihao, Zedong Peng, and Jinwen Yang. 2024. “MPAX: Mathematical Programming in JAX.” Pre-published December 12. <https://doi.org/10.48550/arXiv.2412.09734>.

Lu, Haihao, and Jinwen Yang. 2025a. “An Overview of GPU-based First-Order Methods for Linear Programming and Extensions.” Pre-published June 2. <https://doi.org/10.48550/arXiv.2506.02174>.

Lu, Haihao, and Jinwen Yang. 2025b. “cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid Gradient for Linear Programming in Julia.” *Operations Research* 73 (6): 3440–52. <https://doi.org/10.1287/opre.2024.1069>.

Merckx, Jules. 2025. “Building Bridges: Julia as an MLIR Frontend.” Pre-published February 14. <https://doi.org/10.48550/arXiv.2503.04771>.

Nicusan, Andrei-Leonard, Dominik Werner, Simon Branford, Simon Hartley, Andrew J. Morris, and Kit Windows-Yule. 2025. “AcceleratedKernels.jl: Cross-Architecture Parallel Algorithms from a Unified, Transpiled Codebase.” Pre-published July 22. <https://doi.org/10.48550/arXiv.2507.16710>.

Perumalla, Kalyan, and Maksudul Alam. 2021. “Design Considerations for GPU-based Mixed Integer Programming on Parallel Computing Platforms.” (New York, NY, USA), ICPP Workshops ’21, September 23, 1–7. <https://doi.org/10.1145/3458744.3473366>.

Shin, Sungho, François Pacaud, and Mihai Anitescu. 2024. “Accelerating Optimal Power Flow with GPUs: SIMD Abstraction of Nonlinear Programs and Condensed-Space Interior-Point Methods.” Pre-published February 26. <http://arxiv.org/abs/2307.16830>.

Sountsov, Pavel, Colin Carroll, and Matthew D. Hoffman. 2024. “Running Markov Chain Monte Carlo on Modern Hardware and Software.” Pre-published November 6. <https://doi.org/10.48550/arXiv.2411.04260>.

Tjusila, Gennesaret Kharistio, Alexander Hoen, Nils-Christian Kempke, et al. 2026. “CHAP: A Hybrid GPU-CPU Heuristic for MIP.” Version 1. Pre-published. <https://doi.org/10.48550/ARXIV.2605.05086>.

## Footnotes

[^1]: stats given for the NVIDIA L40s cards installed on ENPC’s cluster

[^2]: see the [batching issue](https://github.com/jump-dev/MathOptInterface.jl/issues/2904)
