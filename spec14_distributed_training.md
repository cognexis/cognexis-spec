# 14. Distributed Training and Memory Efficiency

## 14.1 Purpose

Cognexis is designed to scale from small research models to very large checkpoints. Distributed training must support high-throughput data parallelism, tensor/model parallelism, pipeline staging, activation memory control, and reliable checkpointing. Recurrent depth introduces additional concerns because the same block is unrolled multiple times and gradients from all loops accumulate into shared parameters.

## 14.2 Parallelism Modes

Supported parallelism modes:

- Data parallelism: replicate the model and split batches across workers.
- Tensor parallelism: shard large matrix operations across devices.
- Pipeline parallelism: place different model stages on different devices.
- Fully sharded data parallelism: shard parameters, gradients, and optimizer states.
- Sequence/context parallelism: partition long sequence dimensions for attention.

The implementation may combine modes. A typical large configuration uses tensor parallelism inside blocks, data parallelism across batches, and sharded optimizer state.

## 14.3 Cognexis Stage Partitioning

The Prelude, Recurrent Core, and Coda form natural pipeline boundaries:

```text
stage 0: embeddings + Prelude
stage 1: Recurrent Core
stage 2: Coda + final norm + LM head
```

The recurrent stage may dominate wall-clock time when loop counts are high. Pipeline schedules must account for variable loop counts; otherwise adaptive scheduling can create load imbalance. For early implementations, fixed loop counts are easier to pipeline efficiently.

## 14.4 Tensor Parallel Attention and FFN

Attention projection weights may be sharded by head:

```text
W_q shards -> query heads per tensor-parallel rank
W_k/W_v shards -> kv heads per tensor-parallel rank
W_o -> reduce-scatter or all-reduce output
```

FFN weights may be column-parallel for `W_gate`/`W_up` and row-parallel for `W_down`. The recurrent core uses the same sharded weights every loop. Gradients should be accumulated locally across loops before collective synchronization when possible.

## 14.5 Gradient Accumulation Through Loops

Backpropagation through recurrent unrolls accumulates gradients from every loop application into the same parameters:

```text
grad_R = sum_t grad_R_from_loop_t
```

Distributed implementations should avoid all-reducing recurrent gradients once per loop. Preferred behavior is local accumulation across loops followed by one synchronization per optimizer step or per gradient accumulation boundary.

## 14.6 Activation Memory

Recurrent unrolling can store many intermediate states. Memory controls:

- Gradient checkpointing/rematerialization.
- Selective retention of loop states.
- Activation offloading to CPU/NVMe for extreme models.
- Mixed precision activations.
- Sequence parallelism for long contexts.

Checkpointing strategy should be configurable, for example:

```yaml
activation_checkpointing:
  prelude: every_block
  recurrent: every_2_loops
  coda: every_block
```

The strategy must preserve exact gradients or document approximation.

## 14.7 Mixed Precision

Recommended precision policy:

- BF16 compute for matrix operations where supported.
- FP32 optimizer master weights.
- FP32 or backend-recommended accumulation for reductions.
- FP32 loss computation for numerical stability when needed.
- BF16 KV cache by default.

FP16 requires loss scaling and is more sensitive to recurrent instability. FP8 and quantized training are experimental and require additional validation.

## 14.8 Communication Scheduling

Communication should overlap with computation:

- All-reduce data-parallel gradients while later gradients are still computing.
- Use reduce-scatter/all-gather for sharded optimizer states.
- Overlap data prefetch with forward/backward.
- Delay recurrent gradient synchronization until loop accumulation is complete.

Adaptive loop counts may reduce synchronization efficiency if ranks finish at different times. Sequence-level loop counts should be broadcast per global step during early distributed training.

## 14.9 Checkpointing

A distributed checkpoint must include:

- Model parameters.
- Optimizer state.
- Learning rate scheduler state.
- Loop curriculum state.
- Tokenizer manifest.
- Resolved config.
- Data loader state.
- RNG states for all ranks.
- Scheduler/value-head state.

Checkpoint writes should be atomic from the perspective of resume. Use a temporary directory and a final manifest file that is written only after all shards are complete.

## 14.10 Fault Tolerance

Large jobs should tolerate worker interruption. Requirements:

- Periodic checkpoints.
- Rank-local progress tracking.
- Resume from last complete checkpoint.
- Validation that all parameter shards belong to the same step.
- Clear failure if checkpoint shards are missing or inconsistent.

Elastic world-size changes are optional. If supported, checkpoint partitioning must be independent of world size or include a resharding tool.

## 14.11 Memory-Efficient Inference

Serving also needs memory controls:

- Paged KV cache.
- Batch compaction when sequences finish.
- KV cache dtype selection.
- Weight quantization.
- Maximum concurrent tokens per device.
- Request-level loop budgets.

Token-wise scheduling can reduce compute but may complicate dense GPU kernels. Benchmark both masked dense execution and sparse compaction before enabling token-wise production serving.

## 14.12 Rust Implementation Notes

Rust code should isolate distributed primitives behind traits:

```rust
pub trait Collective {
    fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()>;
    fn all_gather(&self, tensor: &Tensor) -> Result<Tensor>;
    fn reduce_scatter(&self, tensor: &Tensor) -> Result<Tensor>;
}
```

The training loop should not depend directly on one communication backend. Provide adapters for the selected runtime. Ensure tensor lifetimes and device ownership are explicit to avoid use-after-free around asynchronous collectives.

## 14.13 Validation

Distributed validation must include:

- Single-rank and multi-rank loss equivalence for small models.
- Tensor-parallel output equivalence to unsharded reference.
- Recurrent gradient accumulation equivalence.
- Checkpoint save/load across ranks.
- Resume after interrupted write.
- Deterministic loop sampling across ranks.
- Memory usage tests for high loop counts.
- Throughput benchmarks with fixed and adaptive loops.
