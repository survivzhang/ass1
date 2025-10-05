# CITS3402/CITS5507 Assignment 2 - 实验设计方案

**小组成员**: Jiazheng Guo (24070858), Zichen Zhang (24064091)

---

## 1. 实验环境配置

### 1.1 硬件平台
- **HPC系统**: Setonix (主要) + Kaya (备用)
- **节点配置**:
  - Setonix: 每个 node 有 128 cores (2 × AMD EPYC 7763 64-core)
  - 使用 4 nodes (满足 Assignment 要求的最大节点数)
- **总计算资源**: 96 cores (4 nodes × 4 processes/node × 6 threads/process)

### 1.2 测试模式
| 模式 | 说明 | MPI Processes | OpenMP Threads | 总 Cores |
|------|------|---------------|----------------|----------|
| **Serial** | 单线程基准测试 | 1 | 1 | 1 |
| **OpenMP** | 共享内存并行 | 1 | 96 | 96 |
| **MPI** | 分布式内存并行 | 96 | 1 | 96 |
| **Hybrid** | MPI + OpenMP 混合 | 16 | 6 | 96 |

### 1.3 测试参数范围
- **矩阵大小 (H×W)**: 2000×2000, 4000×4000, 6000×6000, 8000×8000, 10000×10000
- **Kernel 大小**: 主要使用 5×5 (与 Assignment 1 保持一致)
- **Stride 参数**: 1×1, 2×2, 3×3, 4×4
- **时间限制**: 每个测试 < 15 分钟

---

## 2. 实验设计

### 实验 1: 四种模式性能对比

**目的**: 对比 Serial, OpenMP, MPI, Hybrid 四种模式的性能差异

**测试配置**:
| 测试用例 | 矩阵大小 | Kernel | Stride | Serial | OpenMP | MPI | Hybrid |
|---------|---------|--------|--------|--------|--------|-----|--------|
| Test 1.1 | 4000×4000 | 5×5 | 1×1 | 1×1 | 1×96 | 96×1 | 16×6 |
| Test 1.2 | 6000×6000 | 5×5 | 1×1 | 1×1 | 1×96 | 96×1 | 16×6 |
| Test 1.3 | 8000×8000 | 5×5 | 1×1 | 1×1 | 1×96 | 96×1 | 16×6 |

**SLURM 配置**:
```bash
# Serial
srun -n 1 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 1 -sW 1 -m serial

# OpenMP (1 process × 96 threads)
export OMP_NUM_THREADS=96
srun -n 1 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 1 -sW 1 -m omp

# MPI (96 processes × 1 thread)
export OMP_NUM_THREADS=1
srun -n 96 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 1 -sW 1 -m mpi

# Hybrid (16 processes × 6 threads)
export OMP_NUM_THREADS=6
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

**性能指标**:
- Execution Time (秒)
- Speedup = T_serial / T_parallel
- Efficiency = Speedup / Cores × 100%
- 对比分析: 哪种模式最快？为什么？

**可复用 Assignment 1 数据**:
- ✅ Serial baseline (已测试过相同矩阵大小)
- ✅ OpenMP 性能数据 (可能需要调整 thread 数量到 96)

---

### 实验 2: Hybrid 配置优化

**目的**: 找到最优的 MPI processes × OpenMP threads 配置

**测试配置** (固定 96 cores, 矩阵 4000×4000):
| 配置 | Processes | Threads | Nodes | 说明 |
|------|-----------|---------|-------|------|
| Config 2.1 | 4 | 24 | 1 | 少进程，多线程 (单节点内) |
| Config 2.2 | 8 | 12 | 2 | 平衡配置 |
| Config 2.3 | 12 | 8 | 3 | - |
| Config 2.4 | 16 | 6 | 4 | **默认配置** |
| Config 2.5 | 24 | 4 | 4 | - |
| Config 2.6 | 32 | 3 | 4 | 多进程，少线程 |

**示例命令**:
```bash
# Config 2.4: 16 processes × 6 threads
export OMP_NUM_THREADS=6
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

**分析重点**:
- 通信开销 vs 共享内存效率
- NUMA 影响
- 最优配置是什么？

---

### 实验 3: Stride 影响分析 (重要!)

**目的**: 分析 stride 参数对计算和通信成本的影响 (Assignment 2 必须要求)

**测试配置** (固定 4000×4000, Hybrid 16×6):
| 测试用例 | Stride | 输出大小 | 计算量 | 通信量 |
|---------|--------|----------|--------|--------|
| Test 3.1 | 1×1 | 4000×4000 | 100% | 100% |
| Test 3.2 | 2×2 | 2000×2000 | ~25% | ~25% |
| Test 3.3 | 3×3 | 1334×1334 | ~11% | ~11% |
| Test 3.4 | 4×4 | 1000×1000 | ~6% | ~6% |

**示例命令**:
```bash
# Stride 1×1
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

# Stride 2×2
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 2 -sW 2 -m hybrid

# Stride 3×3
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 3 -sW 3 -m hybrid

# Stride 4×4
srun -n 16 ./conv_stride_test -H 4000 -W 4000 -kH 5 -kW 5 -sH 4 -sW 4 -m hybrid
```

**分析重点**:
- Stride 如何影响输出大小？
- 计算时间如何变化？
- MPI 通信开销如何变化？
- Speedup 是否与理论预期一致？

---

### 实验 4: Strong Scaling (强扩展性)

**目的**: 固定问题规模，增加计算资源，测试并行效率

**测试配置** (固定 8000×8000, Kernel 5×5, Stride 1×1):
| 测试用例 | Cores | Processes | Threads | 理论 Speedup |
|---------|-------|-----------|---------|-------------|
| Test 4.1 | 6 | 1 | 6 | 1× (基准) |
| Test 4.2 | 12 | 2 | 6 | 2× |
| Test 4.3 | 24 | 4 | 6 | 4× |
| Test 4.4 | 48 | 8 | 6 | 8× |
| Test 4.5 | 96 | 16 | 6 | 16× |

**示例命令**:
```bash
export OMP_NUM_THREADS=6

# 6 cores
srun -n 1 ./conv_stride_test -H 8000 -W 8000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

# 96 cores
srun -n 16 ./conv_stride_test -H 8000 -W 8000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

**性能指标**:
- 实际 Speedup vs 理论 Speedup
- Parallel Efficiency = Speedup / Cores
- Scalability 曲线
- 通信开销随 cores 增加的变化

---

### 实验 5: 不同矩阵大小测试

**目的**: 测试不同问题规模下的性能表现

**测试配置** (Hybrid 16×6, Stride 1×1):
| 测试用例 | 矩阵大小 | 数据量 (MB) | 预计时间 |
|---------|---------|------------|----------|
| Test 5.1 | 2000×2000 | ~16 MB | < 1 分钟 |
| Test 5.2 | 4000×4000 | ~64 MB | ~2 分钟 |
| Test 5.3 | 6000×6000 | ~144 MB | ~5 分钟 |
| Test 5.4 | 8000×8000 | ~256 MB | ~10 分钟 |
| Test 5.5 | 10000×10000 | ~400 MB | ~15 分钟 |

**示例命令**:
```bash
export OMP_NUM_THREADS=6

# Small
srun -n 16 ./conv_stride_test -H 2000 -W 2000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid

# XX-Large
srun -n 16 ./conv_stride_test -H 10000 -W 10000 -kH 5 -kW 5 -sH 1 -sW 1 -m hybrid
```

**分析重点**:
- 问题规模对性能的影响
- Cache 效率
- 通信开销占比

---

### 实验 6: (可选) Weak Scaling

**目的**: 保持每个 core 的工作量恒定，增加总计算资源

**测试配置** (每 core 处理约 167×167 的数据):
| 测试用例 | Cores | 矩阵大小 | 计算量/core |
|---------|-------|----------|------------|
| Test 6.1 | 16 | 2000×2000 | 恒定 |
| Test 6.2 | 32 | 2828×2828 | 恒定 |
| Test 6.3 | 64 | 4000×4000 | 恒定 |
| Test 6.4 | 96 | 4899×4899 | 恒定 |

**理想情况**: 执行时间应保持恒定

---

## 3. 性能指标收集清单

### 3.1 必须收集的数据
- [x] **Execution Time** (秒)
- [x] **Speedup** = T_serial / T_parallel
- [x] **Parallel Efficiency** = Speedup / Cores × 100%
- [x] **Communication Overhead** (MPI 模式)
- [x] **Memory Usage** (峰值)

### 3.2 输出数据格式
每个测试应记录:
```
Mode: Hybrid
Nodes: 4
Processes: 16
Threads/Process: 6
Matrix: 4000×4000
Kernel: 5×5
Stride: 1×1
Execution Time: X.XX seconds
Speedup: X.XX
Efficiency: XX.X%
```

---

## 4. Assignment 1 数据复用

### 4.1 可直接使用的数据
从 `Assignment1_report_24070858_24064091.pdf`:
- ✅ **Serial baseline**: 100×100, 1000×1000, 10000×10000 矩阵
- ✅ **OpenMP 性能**: 相同矩阵大小的多线程数据
- ✅ **Kernel 大小影响**: 10×10, 100×100, 999×999 (可作为对比参考)

### 4.2 需要重新测试的数据
- ❌ **MPI only** 模式 (Assignment 1 没有)
- ❌ **Hybrid** 模式 (Assignment 1 没有)
- ❌ **Stride != 1** 的所有测试 (Assignment 1 没有 stride)
- ❌ **96 cores** 的配置 (Assignment 1 可能用的 core 数不同)

---

## 5. Report 写作要求对照

根据 Marking Rubric (总分 30):

| 评分项 | 分数 | 实验对应 | 说明 |
|--------|------|---------|------|
| 实现 Serial + Stride | 1 | - | 已完成代码 |
| 矩阵生成和 I/O | 1 | - | 已完成代码 |
| MPI 并行实现 | 2 | 实验1, 4 | 需要详细描述算法 |
| Hybrid 并行实现 | 3 | 实验1, 2 | 需要详细描述 MPI+OpenMP 配合 |
| **并行化描述** | 5 | 所有实验 | 分布式+共享内存策略 |
| **数据分解和分布** | 2 | 实验1, 4 | Row-based decomposition |
| **通信策略和同步** | 3 | 实验1, 3 | MPI_Bcast 分析 |
| **性能指标和分析** | 10 | 所有实验 | **最重要!** Speedup, Efficiency, Stride 影响 |
| 格式和呈现 | 3 | - | 图表、表格、清晰度 |

---

## 6. 时间安排

### 6.1 测试执行顺序
1. **Day 1**: 实验 1 (模式对比) - 验证所有模式正常工作
2. **Day 2**: 实验 3 (Stride 影响) - 收集 stride 数据
3. **Day 3**: 实验 2 (Hybrid 配置) + 实验 4 (Strong Scaling)
4. **Day 4**: 实验 5 (矩阵大小) + 数据整理
5. **Day 5-7**: 撰写 Report

### 6.2 预计总 CPU Hours (Setonix)
- 实验 1-5: ~30 tests × ~5 min × 96 cores ≈ **240 core-hours**
- 预留调试和重测: 100 core-hours
- **总计**: ~350 core-hours

---

## 7. 数据可视化计划

### 7.1 必须的图表
1. **Bar Chart**: 四种模式性能对比 (实验1)
2. **Line Chart**: Hybrid 配置优化曲线 (实验2)
3. **Line Chart**: Stride 对执行时间的影响 (实验3)
4. **Scalability Curve**: Strong Scaling (实验4)
5. **Table**: 详细性能数据总结

### 7.2 工具
- Python + Matplotlib / Seaborn
- Excel / Google Sheets (备用)

---

## 8. 注意事项

### 8.1 Setonix HPC 限制
- ⚠️ **CPU hours 有限** - 避免重复测试
- ⚠️ **队列等待时间** - 提前规划任务提交
- ⚠️ **15分钟时间限制** - 避免过大的矩阵

### 8.2 数据验证
- ✅ 每个测试运行 3 次取平均值 (减少误差)
- ✅ 检查输出正确性 (与 serial 对比)
- ✅ 记录异常情况和错误

### 8.3 Report 重点
- ✅ **Stride 影响必须详细分析** (Assignment 2 特有要求)
- ✅ **通信开销分析** (MPI_Bcast, 数据传输)
- ✅ **Cache 和内存布局** (每个 MPI process 内的优化)
- ✅ **Speedup 和 Efficiency 曲线** (最高分项)

---

## 9. 脚本清单

- [x] `performance_test_96cores.sh` - 主性能测试脚本
- [x] `test_conv_stride.sh` - 正确性测试脚本
- [ ] `plot_results.py` - 数据可视化脚本 (待创建)
- [ ] `collect_data.sh` - 结果收集脚本 (待创建)

---

**最后修改时间**: 2025-10-05
**状态**: 初稿 - 待修改
