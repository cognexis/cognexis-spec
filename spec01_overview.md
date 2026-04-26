# 01. Overview

## 01.1 Purpose

Cognexis is a decoder-only, compute-adaptive language model architecture. Its central design choice is to separate stored parameter count from effective reasoning depth. Instead of scaling depth exclusively by adding more unique transformer layers, Cognexis reuses a shared recurrent transformer block for a configurable number of iterations. The same checkpoint can therefore serve requests at different quality, latency, and compute budgets.

This specification defines the full system contract for implementing, training, evaluating, and operating Cognexis. It is written as an engineering specification rather than a research summary: each document states expected behavior, component boundaries, runtime interfaces, validation requirements, and implementation risks. Where the architecture admits multiple implementation strategies, the specification names the preferred default and the acceptable alternatives.

The documents are intended to support a Rust implementation, but most requirements are framework independent. A conforming implementation may use any tensor backend, GPU runtime, or distributed training layer as long as the public behavior, numerical semantics, configuration fields, and evaluation outputs match the contracts described here.

## 01.2 System Goals

Cognexis has five primary goals:

- Decouple parameter count from reasoning depth by reusing a recurrent core.
- Expose inference-time loop count as a controllable compute knob.
- Support fixed-depth, sequence-adaptive, and token-wise adaptive loop execution.
- Preserve standard decoder-only language model behavior for tokenization, causal masking, training loss, and autoregressive generation.
- Provide enough observability to evaluate the quality, latency, safety, and stability impact of recurrent depth.

The architecture is not intended to make computation free. Additional recurrent loops still increase FLOPs and wall-clock latency. The intended benefit is that a single model can allocate that computation selectively rather than training and serving multiple models for different depth budgets.

## 01.3 Architectural Summary

Cognexis is organized into five major runtime stages:

1. Tokenization converts raw text into token IDs and special-token markers.
2. Embedding lookup maps token IDs into hidden vectors and prepares positional information.
3. The Prelude applies a stack of conventional transformer blocks with unique parameters.
4. The Recurrent Core applies one shared transformer block repeatedly.
5. The Coda and language modeling head transform the final hidden state into vocabulary logits.

The model is autoregressive. During both training and inference, position `i` must not attend to future positions `j > i`. Causal masking remains required even when the recurrent core is applied multiple times. Each recurrent loop refines the same sequence representation under the same causal constraint.

The effective block depth for a fixed loop count is:

```text
effective_depth = num_prelude_blocks + loop_count * num_recurrent_blocks + num_coda_blocks
```

The default architecture uses exactly one recurrent block whose weights are shared across loops. The configuration may represent this as `num_recurrent_blocks = 1`; values greater than one are experimental and must be treated as an extension, not the baseline.

## 01.4 Core Operating Modes

Cognexis supports three loop execution modes:

- `fixed`: every sequence and token runs the same configured loop count.
- `adaptive_sequence`: a scheduler decides when to stop for the whole sequence.
- `adaptive_token`: each token position may halt independently while other positions continue.

Fixed mode is the baseline for training, debugging, benchmark comparison, and deterministic production operation. Adaptive sequence mode is the preferred first production scheduler because it has a simpler execution model. Token-wise scheduling is more efficient for heterogeneous sequences but requires more complex masking, cache management, and observability.

Every implementation must enforce a hard maximum loop count. Adaptive decisions may stop early, but they must never exceed `max_loops` or a request-level compute budget.

## 01.5 Data Flow

The high-level forward pass is:

```text
input_text
  -> tokenizer.encode
  -> token_ids
  -> embedding(token_ids)
  -> prelude(hidden, causal_mask)
  -> recurrent_core(hidden, scheduler, loop_budget)
  -> coda(hidden, causal_mask)
  -> lm_head(hidden)
  -> logits
```

During training, logits are compared against the next-token targets with cross-entropy loss. During inference, logits are consumed by a sampler or decoder policy that emits the next token. The loop scheduler observes hidden-state statistics, confidence signals, value-head predictions, and budget state. It must not alter token IDs or bypass safety policies.

## 01.6 Document Map

The specification is split into component-focused documents:

- `spec02_tokenizer.md` through `spec10_lm_head.md` define core model components.
- `spec11_config.md` through `spec16_prefill_decode.md` define configuration, data, training, stability, and inference execution.
- `spec17_scheduler_design.md` through `spec19_value_head.md` define adaptive scheduling.
- `spec20_evaluation_metrics.md` through `spec24_safety_monitoring.md` define evaluation, experiments, safety, and operations.
- `spec25_glossary.md` defines shared terminology.
- `spec26_implementation_outline.md` through `spec28_conclusion.md` define implementation structure, known limitations, and project closure.

Each document is normative for its own component and informational for surrounding components. Cross-document conflicts should be resolved by prioritizing the more specific component document unless the configuration document defines an explicit global invariant.

## 01.7 Required Invariants

A conforming Cognexis implementation must satisfy these invariants:

- Token IDs produced by the tokenizer must be identical during training and inference for the same tokenizer model and normalization settings.
- The embedding matrix and language modeling head must agree on `vocab_size` and `hidden_dim`; weight tying is the default.
- Attention must be causal in all decoder-only paths.
- Recurrent core parameters must be shared across loop iterations.
- The scheduler must respect `min_loops`, `max_loops`, and request-level compute budgets.
- Training metadata must record sampled loop counts, scheduler mode, tokenizer version, and architecture configuration.
- Evaluation results must report both task quality and compute usage.
- Safety filters and budget enforcement must remain active regardless of loop mode.

## 01.8 Non-Goals

This specification does not prescribe a specific pretrained checkpoint, dataset license, cluster topology, GPU kernel implementation, or serving framework. It also does not require a single universal scheduler. Different deployments may choose a conservative fixed loop count, a learned value-head scheduler, or a rule-based scheduler depending on latency requirements and validation maturity.

The specification also does not claim that recurrent depth universally improves all tasks. Some tasks will saturate early or degrade at excessive loop counts. The evaluation and monitoring documents exist to measure those effects rather than assume them away.

## 01.9 Compatibility Expectations

The implementation should interoperate with common LLM tooling where practical:

- Tokenizer artifacts should be loadable from a stable file format.
- Checkpoints should include complete configuration metadata.
- Evaluation outputs should be exported as JSONL or CSV.
- Inference APIs should expose loop budget, scheduler mode, sampling parameters, and safety options explicitly.
- Logs should include enough structured fields to reconstruct loop decisions for a request without exposing sensitive user content by default.

Compatibility must not override correctness. If a third-party runtime cannot represent recurrent loop sharing, token-wise halting, or budget enforcement safely, an adapter layer must enforce those semantics before claiming Cognexis compatibility.

## 01.10 Acceptance Criteria

The complete Cognexis system is acceptable when:

- Unit tests cover tokenizer round trips, causal masks, attention shapes, recurrent loop reuse, scheduler bounds, and LM-head tying.
- Integration tests run a small model through prefill, decode, fixed loops, adaptive loops, save/load, and evaluation.
- Benchmark scripts produce quality-versus-depth and latency-versus-depth reports.
- Safety and budget policies are enforced in generation APIs.
- Documentation describes every public configuration field and every exported metric.
- The white paper can remain a high-level narrative while the `spec*.md` files contain the detailed implementation contract.
