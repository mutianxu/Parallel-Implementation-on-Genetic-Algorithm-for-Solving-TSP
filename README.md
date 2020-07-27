# Parallel Implementation on Genetic Algorithm for Solving TSP
## Final project of UCI EECS224-High Performance Computing
 Traveling salesman problem (TSP) is quite known in the field of combinatorial optimization. There had been many flexible attempts to address this problem using genetic algorithms (GA).With the significantly increasing of the dataset in real world, the sequential method constrains the performance of algorithms. This paper proposes three parallel methods based on using sequential GA to solve TSP problem, aiming at improving the efficiency of sequential solutions. In this paper, we parallelized the initial population and the evaluation parts through dividing them into several sub parts and used different cores to process different sub parts. Besides, we parallelized another greedy algorithm when initializing the population to set an upper bound, and we also provided non-migration and migration methods, which will be introduced in the introduction section. Using these parallel methods, we ultimately got a better path with shorter distance than that of sequential algorithm and reached four times speedup.
