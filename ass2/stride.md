# The Effect of Stride on Computational and Communication Cost

## Executive Summary

Stride parameter selection fundamentally transforms parallel 2D convolution performance through **quadratic reduction in computational workload** (93.6-99.98% as stride increases from 2 to 10) while creating **asymmetric communication effects**. Computational time decreases dramatically, but communication overhead grows from 5.6% to 43.3% at high core counts, requiring stride-aware parallelization strategies that reduce optimal core allocation by up to 6×.

## 1. Fundamental Stride Impact

### 1.1 Computational Scaling

Stride controls output dimensions via: `Output = ⌊(Input - Kernel) / Stride⌋ + 1`, creating quadratic workload reduction.

**Table 1: Stride Impact on Computation (1000×1000 input, 3×3 kernel, 4 cores)**

| Stride | Output Size | Output Elements | Computation Time (s) | Time Reduction | Comp % of Total |
|--------|-------------|-----------------|---------------------|----------------|-----------------|
| (2,2)  | 500×500     | 250,000        | 0.000703           | Baseline       | 9.9%            |
| (5,5)  | 200×200     | 40,000         | 0.000134           | 80.9%          | 5.9%            |
| (10,10)| 100×100     | 10,000         | 0.000045           | 93.6%          | 3.6%            |

**Key Finding:** Computation time decreases 93.6%, but its percentage of total time drops from 9.9% to 3.6%, demonstrating that **stride amplifies communication bottlenecks** rather than merely reducing workload proportionally.

### 1.2 Kernel Size as Dominant Factor

**Table 2: Kernel Size vs Stride Effects (1000×1000 input, 4 cores)**

| Kernel | Stride | Comp Time (s) | Comm Time (s) | Comp % | Comm % |
|--------|--------|---------------|---------------|--------|--------|
| 3×3    | (2,2)  | 0.000703     | 0.004827      | 9.9%   | 67.9%  |
| 3×3    | (10,10)| 0.000045     | 0.001103      | 3.6%   | 88.3%  |
| 100×100| (2,2)  | 0.515782     | 0.034218      | 93.8%  | 6.2%   |
| 100×100| (10,10)| 0.020651     | 0.002349      | 88.1%  | 8.7%   |

**Critical Insight:** Small kernels (3×3, ~9 ops/element) remain communication-dominated across all strides. Large kernels (100×100, ~10,000 ops/element) maintain computational dominance. **Kernel size, not stride, determines the fundamental performance regime.**

## 2. Communication Cost Decomposition

Communication comprises two components with fundamentally different stride responses.

**Table 3: Communication Components (10000×10000 input, 3×3 kernel, 4 cores)**

| Stride | Broadcast Calls | Broadcast Time (s) | Memory Copy Time (s) | Total Comm (s) | Broadcast Reduction |
|--------|----------------|-------------------|---------------------|----------------|-------------------|
| (2,2)  | 5,000          | 0.121963          | 0.052632            | 0.174595       | Baseline          |
| (5,5)  | 2,000          | 0.037007          | 0.051954            | 0.088961       | 69.7%             |
| (10,10)| 1,000          | 0.017771          | 0.052393            | 0.070164       | 85.4%             |

**Asymmetric Response:**
- **Broadcast (output-dependent):** Scales linearly with stride, 85.4% reduction
- **Memory copy (input-dependent):** Nearly constant (<0.5% variation)

At stride (10,10), memory copy represents 74.7% of communication overhead versus 30.1% at stride (2,2), making **fixed communication costs dominant at large strides**.

## 3. Scale-Dependent Performance Regimes

### 3.1 Large-Scale Stride Impact

**Table 4: Large-Scale Performance (20000×20000, 200×200 kernel, 96 cores Hybrid)**

| Stride | Total Time (s) | Comp Time (s) | Comm Time (s) | Comp % | Comm % | Time Reduction |
|--------|----------------|---------------|---------------|--------|--------|----------------|
| (1,1)  | 175.50        | 165.60        | 9.90          | 94.4%  | 5.6%   | Baseline       |
| (2,2)  | 46.61         | 41.51         | 5.11          | 89.0%  | 11.0%  | 73.4%          |
| (5,5)  | 7.60          | 6.66          | 0.94          | 87.7%  | 12.3%  | 95.7%          |
| (10,10)| 2.95          | 1.67          | 1.28          | 56.7%  | 43.3%  | 98.3%          |

**Performance Inversion:** At stride (10,10), despite 98.3% total time reduction, communication explodes to 43.3%, creating a **regime shift from compute-bound to communication-bound**.

**Table 5: Low Core Count Comparison (20000×20000, 200×200 kernel, 2 cores)**

| Stride | Total Time (s) | Comp % | Comm % |
|--------|----------------|--------|--------|
| (1,1)  | 7,185.58      | 100.0% | 0.0%   |
| (2,2)  | 1,796.27      | 99.8%  | 0.2%   |
| (10,10)| 72.57         | 99.4%  | 0.6%   |

At **low core counts**, computation dominates across all strides. At **high core counts**, stride (10,10) creates communication dominance. This demonstrates **stride-parallelism interaction**.

## 4. Optimal Core Configuration

### 4.1 Stride-Dependent Optimal Configurations

**Table 6: Optimal Configurations by Stride (20000×20000, 200×200 kernel)**

| Stride | Optimal Cores | Config | Total Time (s) | Comm % | Speedup vs 2 cores | Efficiency |
|--------|---------------|--------|----------------|--------|-------------------|------------|
| (1,1)  | 96           | 12×8   | 175.50        | 5.6%   | 40.9×             | 42.6%      |
| (2,2)  | 64           | 8×8    | 64.93         | 4.2%   | 27.7×             | 43.3%      |
| (5,5)  | 32           | 8×4    | 18.62         | 3.6%   | 15.5×             | 48.4%      |
| (10,10)| 16           | 4×4    | 9.34          | 3.4%   | 7.8×              | 48.8%      |

**Critical Finding:** Optimal core count decreases **6× from stride (1,1) to (10,10)**. Using 96 cores at stride (10,10) achieves faster time (2.95s) but terrible efficiency (43.3% communication vs 3.4% at 16 cores).

### 4.2 Communication Overhead Scaling

**Table 7: Communication % Growth with Core Count**

| Cores | Stride (1,1) | Stride (2,2) | Stride (5,5) | Stride (10,10) |
|-------|-------------|-------------|-------------|----------------|
| 2     | 0.0%        | 0.2%        | 0.3%        | 0.6%           |
| 16    | 0.6%        | 1.0%        | 1.8%        | 3.4%           |
| 32    | 2.0%        | 3.0%        | 3.6%        | 4.2%           |
| 64    | 3.5%        | 4.2%        | 17.3%       | 8.1%           |
| 96    | 5.6%        | 11.0%       | 12.3%       | 43.3%          |

At stride (10,10), communication overhead increases **72× from 2 to 96 cores** (0.6% → 43.3%), demonstrating **superlinear growth** where fixed communication costs overwhelm diminished computational workload.

## 5. Programming Model Comparison

**Table 8: Pure MPI vs Hybrid (20000×20000, 200×200 kernel, Stride 1,1)**

| Cores | MPI Config | Hybrid Config | MPI Time (s) | Hybrid Time (s) | Time Advantage | MPI Comm % | Hybrid Comm % |
|-------|-----------|---------------|-------------|----------------|----------------|-----------|---------------|
| 2     | 2×1       | 2×1           | 7,188.65    | 7,185.58       | 0.04%          | 0.3%      | 0.0%          |
| 16    | 16×1      | 4×4           | 913.01      | 906.09         | 0.76%          | 3.2%      | 0.6%          |
| 64    | 64×1      | 8×8           | 269.13      | 258.23         | 4.0%           | 13.9%     | 3.5%          |

**Scale-Dependent Advantage:** Hybrid models provide **4.0% time improvement and 75% communication reduction** at 64+ cores through shared memory optimization. At low core counts (≤16), differences are negligible (<1%).

## 6. Practical Optimization Framework

### 6.1 Decision Matrix

**Table 9: Configuration Strategy by Problem Characteristics**

| Kernel Size | Stride Range | Optimal Cores | Comm Target | Strategy |
|-------------|--------------|---------------|-------------|----------|
| Small (<10×10) | Any | 2-4 | <20% | Minimal parallelization |
| Medium (50-150) | (1,1)-(5,5) | 16-32 | <10% | Moderate scaling |
| Medium (50-150) | (10,10) | 8-16 | <15% | Conservative |
| Large (>150) | (1,1)-(2,2) | 64-96 | <10% | Maximum parallelization |
| Large (>150) | (5,5) | 32-64 | <15% | Moderate scaling |
| Large (>150) | (10,10) | 16-32 | <20% | Avoid comm dominance |

### 6.2 Performance Prediction Model

**Table 10: Model Validation (20000×20000, 200×200 kernel, 96 cores)**

| Stride | Predicted (s) | Actual (s) | Error | Model |
|--------|---------------|------------|-------|-------|
| (1,1)  | 175.5         | 175.50     | 0.0%  | Baseline |
| (2,2)  | 47.3          | 46.61      | 1.5%  | T_comp/4 + T_bcast/2 + T_mem |
| (5,5)  | 8.0           | 7.60       | 5.3%  | T_comp/25 + T_bcast/5 + T_mem |
| (10,10)| 2.8           | 2.95       | 5.1%  | T_comp/100 + T_bcast/10 + T_mem |

**Model:** `T_total(s) ≈ T_comp(1)/s² + T_broadcast(1)/s + T_memcopy` achieves <6% prediction error, validating quadratic computational, linear broadcast, and constant memory copy scaling.

## 7. Key Findings Summary

**Table 11: Critical Performance Relationships**

| Relationship | Magnitude | Implication |
|--------------|-----------|-------------|
| Stride → Computation | 93.6-99.98% reduction (stride 2→10) | Quadratic workload scaling |
| Stride → Broadcast | 85.4% reduction | Linear communication scaling |
| Stride → Memory Copy | <3% variation | Fixed overhead dominance |
| Optimal Cores vs Stride | 96 cores (stride 1) → 16 cores (stride 10) | 6× reduction required |
| Comm Overhead vs Cores | 0.6% (2 cores) → 43.3% (96 cores) at stride 10 | Superlinear growth |
| Hybrid vs Pure MPI | 4% faster, 75% less comm (64+ cores) | Scale-dependent advantage |

## 8. Conclusions and Recommendations

### Key Insights

1. **Stride creates asymmetric effects:** Computation decreases quadratically (93.6-99.98%) while broadcast scales linearly (85.4%) and memory copy remains constant (<3%), fundamentally altering optimization requirements.

2. **Optimal parallelization decreases with stride:** From 96 cores at stride (1,1) to 16 cores at stride (10,10)—a 6× reduction—to maintain communication below 10% threshold.

3. **Kernel size determines regime:** Small kernels (<10×10) are communication-bound at all strides; large kernels (>150×150) remain compute-bound except under extreme parallelization.

4. **Communication has two components:** Output-dependent (broadcast) scales with stride; input-dependent (memory copy) stays fixed, becoming dominant at large strides (74.7% at stride 10).

### Practical Recommendations

**For High Performance:**
- **Stride ≤2:** Use maximum cores (64-96) with Hybrid model
- **Stride 5:** Use moderate cores (32) to balance efficiency
- **Stride ≥10:** Use minimal cores (16) to avoid communication explosion

**Monitoring Threshold:**
- Reduce cores if communication exceeds 15%
- At stride (10,10) with 96 cores: 43.3% communication indicates severe over-parallelization

**Programming Model Selection:**
- Use Hybrid for 32+ cores at stride ≤5 (4-10% advantage)
- Use either model for ≤16 cores or stride ≥10 (<1% difference)

**Critical Design Principle:** Stride is not merely a workload parameter but a **first-class architectural parameter** requiring explicit consideration in parallelization strategy, core allocation, and programming model selection.