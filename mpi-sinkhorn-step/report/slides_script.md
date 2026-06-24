# Presentation Script: MPI Sinkhorn Optimal Transport Solver

## Slide 1 - Title

**Say:**  
Good morning/afternoon. We are presenting our Parallel and Distributed Programming project: an MPI Sinkhorn optimal transport solver. The focus is not only implementing the algorithm, but also setting up a real physical-machine VM cluster and evaluating how the program scales.

**Transition:**  
I will first summarize the project requirements and what our group delivered.

## Slide 2 - Project Goal and Deliverables

**Say:**  
The assignment requires a C or C++ MPI program, running on at least three physical machines, with a clear explanation of parallelism, decomposition, communication, correctness, and performance. Our final implementation is a C++17 Sinkhorn solver with sequential and MPI modes. We ran it on router/Ethernet and Wi-Fi clusters, logged results as JSON and CSV, and also implemented a dynamic scheduling extension.

**Transition:**  
Now I will explain the numerical problem behind the project.

## Slide 3 - Problem: Entropy-Regularized Optimal Transport

**Say:**  
Optimal transport tries to move a source distribution to a target distribution with minimum cost. We solve the entropy-regularized version using Sinkhorn-Knopp iterations. The key computation is repeated multiplication with a dense kernel matrix K. Since K has size N by N, the computation grows quadratically with the input size.

**Transition:**  
This dense matrix structure is exactly why the problem can be parallelized by rows.

## Slide 4 - Parallel Algorithm

**Say:**  
We use data-level parallelism. Each MPI rank owns a contiguous block of rows of the kernel matrix. Updating local u values can be done independently. The difficult part is updating v, because it needs the global column contribution from every rank. We solve that with MPI Allreduce. This means each iteration has a local compute phase followed by one global synchronization.

**Transition:**  
Here is the same algorithm in pseudocode form.

## Slide 5 - MPI Sinkhorn Pseudocode

**Say:**  
The pseudocode shows the core loop. Each rank builds its local rows, computes local u, accumulates local column sums, and then calls Allreduce to get the global column denominator. After that, every rank updates the replicated v vector. This is simple and reliable, but the Allreduce over a length-N vector becomes the main communication bottleneck.

**Transition:**  
Next, I will describe the cluster hardware used for running this program.

## Slide 6 - Cluster and Hardware

**Say:**  
Our cluster is heterogeneous. The VMs expose four virtual CPU cores, but the physical CPUs and memory limits differ. The table includes Geekbench 6 single-core and multi-core scores as a rough reference. The master and worker 1 use i5-9300H machines. Worker 2 uses a Ryzen 5 5600H machine, and worker 3 uses an i3-1115G4 machine. We also upgraded VM memory during the project to reduce memory pressure. The router setup uses a TP-Link Archer C20; it is stable for demo, but its LAN ports are only 100 Mbps, so network communication can still limit MPI performance.

**Transition:**  
Now I will outline the benchmark plan.

## Slide 7 - Benchmark Plan

**Say:**  
The experiments follow the course requirements. First, we run an input-size sweep from 2000 to 12000. Second, we calibrate a longer workload by increasing iterations until runtime reaches two to three minutes. Third, we sweep process count. Fourth, we run the 2N speedup experiment at N equals 16000. We also compare network setups and evaluate dynamic scheduling.

**Transition:**  
Before the main results, we compare the initial Python prototype with the final C++ implementation.

## Slide 8 - Python Prototype vs C++ Implementation

**Say:**  
The first MVP was implemented in Python with mpi4py because it was fast to validate correctness. The final course submission uses C++ MPI. This chart compares Python and C++ on a two-node benchmark. The result is mixed for very small sizes because overhead and implementation details dominate, but C++ is the path we use for all final cluster benchmarks because it satisfies the requirement and gives us direct control over timing, memory, and MPI behavior.

**Transition:**  
Next we move to the main router and Wi-Fi cluster results.

## Slide 9 - Input-Size Sweep: Router vs Wi-Fi

**Say:**  
The left plot shows runtime versus input size. The right plot shows speedup for three nodes. MPI becomes useful when the input is large enough. At N equals 12000, the latest Wi-Fi three-node run reaches about 2.38 times speedup. The best observed three-node router point reaches 2.8859 times speedup, which is about 96.2 percent efficiency. We treat that as the best observed point, while the later report-suite results are more conservative.

**Transition:**  
The next slide shows what happens when we increase process count.

## Slide 10 - Process Sweep and 2N Speedup

**Say:**  
At N equals 8000, increasing process count helps up to around four processes. After that, the benefit stops because communication becomes more expensive relative to local compute. In the 2N experiment at N equals 16000, np equals 4 gives the best speedup, about 3.585 times. np equals 8 is slower than np equals 4, which confirms that Allreduce communication is the main bottleneck at higher process counts.

**Transition:**  
Now we show the workload calibration and load-balance analysis required by the assignment.

## Slide 11 - Calibration and Load Balance

**Say:**  
To satisfy the two- to three-minute workload requirement, we keep N equals 8000 and increase the iteration count. At 2000 iterations, runtime is around 164 seconds, which fits the target. The load-balance chart shows per-rank compute and communication or waiting time. Total times are almost equal because ranks synchronize, but the compute/communication split shows machine heterogeneity.

**Transition:**  
Because communication is important, we also tested network and communication variants.

## Slide 12 - Network and Communication Experiments

**Say:**  
The left chart compares two-node wireless topologies. The result shows that wireless network behavior can affect speedup significantly. The right chart compares communication variants at N equals 12000. Lazy synchronization every five iterations gives the best runtime because it reduces the number of column synchronizations. Float32 communication is also useful because it halves the payload while keeping numerical error small.

**Transition:**  
The next slide discusses the dynamic scheduling extension.

## Slide 13 - Dynamic Scheduling

**Say:**  
 Dynamic scheduling was implemented as chunk-based partitioning. The baseline is still a distributed MPI run: static contiguous means each rank on each machine owns a fixed continuous row interval. We compare that baseline with weighted equal chunks, adaptive weighted chunks, manual weights, and runtime queue. The result is not universally better than static partitioning. Adaptive weighted chunks improve the large N equals 12000 case by about 3.7 percent, but manual weights can hurt badly if the assumed machine speeds are wrong. Runtime queue is correct as a demonstration of dynamic assignment, but it is too slow for the performance path.

**Transition:**  
I will close with the main conclusions.

## Slide 14 - Conclusions

**Say:**  
The project satisfies the core requirements: a real C++ MPI solver, real multi-machine VM cluster, sequential baseline, correctness validation, and performance analysis. Row-wise data decomposition works well for large dense workloads. The main limitation is communication: every iteration needs an Allreduce over a length-N vector. For demo, the best setup is the router/Ethernet three-machine cluster with N equals 8000 and 2000 iterations. Dynamic scheduling is a promising extension, but the stable contiguous partition remains the reliable baseline.

**Transition:**  
The final slide lists where the raw results are stored.

## Slide 15 - Backup: Key Raw Result Locations

**Say:**  
These are the result folders used to build the report and slides. They contain the raw JSON logs, CSV summaries, and benchmark outputs. This makes the numbers in the report reproducible and easy to verify.

**Transition:**  
Thank you. We are ready for questions.
