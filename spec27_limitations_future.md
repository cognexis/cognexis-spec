# 27. Limitations and Future Work

## 27.1 Purpose

Cognexis introduces flexible recurrent depth, but the design creates technical and research risks. This document records known limitations so implementations can validate them directly and future work can prioritize the most important gaps.

## 27.2 Stability at High Loop Counts

The largest risk is unstable recurrent dynamics at depths beyond the training distribution. Even with normalization and residual scaling, repeated transformations can:

- Explode hidden norms.
- Collapse to uninformative fixed points.
- Oscillate between states.
- Become overconfident in wrong predictions.

Future work should explore contractive recurrent blocks, better spectral constraints, implicit fixed-point solvers, loop-specific normalization-free stabilizers, and training objectives that reward stable refinement.

## 27.3 Latency Variability

Adaptive scheduling produces variable response times. This is useful for efficiency but challenging for interactive products with strict latency service-level objectives. Token-wise scheduling can increase variability further because different positions halt at different depths.

Future work should develop schedulers with explicit latency guarantees, deadline-aware policies, and hardware-aware batching strategies.

## 27.4 Scheduler Generalization

A learned scheduler may work well on validation tasks and fail on new domains. It may halt too early on unfamiliar reasoning tasks or waste loops on prompts where confidence is miscalibrated.

Future work should study:

- Domain-adaptive scheduler calibration.
- Uncertainty-aware value heads.
- Meta-learned halting policies.
- Online telemetry feedback without unsafe self-training.
- Conservative fallback policies for out-of-distribution prompts.

## 27.5 Training Complexity

Recurrent-depth training is more complex than standard decoder-only training. It requires loop curricula, recurrent gradient handling, stability monitoring, optional intermediate losses, and scheduler training data.

Future work should simplify training recipes and identify minimal settings that preserve most benefits. A useful milestone would be a small set of model-size-specific default configs that are stable without extensive tuning.

## 27.6 Cache Semantics

KV caching for recurrent hidden states is more subtle than standard transformer caching because hidden states change by loop. Incorrect cache reuse can silently produce wrong outputs.

Future work should define optimized recurrent cache schemes with formal equivalence or bounded approximation error. Until then, correctness-first recomputation may be necessary in some paths.

## 27.7 Token-Wise Execution Efficiency

Token-wise scheduling is conceptually attractive but may not save wall-clock time on dense accelerators unless kernels can skip inactive tokens efficiently. Masked dense execution provides correctness but little compute savings.

Future work should explore block-sparse kernels, active-token compaction, scheduler-aware batching, and hardware support for dynamic token execution.

## 27.8 Overthinking

More loops can degrade quality. This may be caused by recurrent amplification of noise, excessive confidence, or distribution shift at depths underrepresented in training.

Future work should develop anti-overthinking objectives, loop dropout, adaptive residual scaling, and schedulers that detect harmful refinement before output quality falls.

## 27.9 Evaluation Burden

Cognexis requires evaluating a two-dimensional surface: task quality across compute depth. This is more expensive than single-depth evaluation and can slow iteration.

Future work should create cheaper proxy metrics that predict depth scaling behavior, plus standard benchmark subsets for rapid loop-scaling checks.

## 27.10 Safety Uncertainty

The relationship between recurrent depth and safety is not guaranteed. Deeper loops might improve refusal reasoning in some cases and amplify harmful content in others.

Future work should evaluate safety across depths, train safety-aware value heads, and design policies that cap or redirect computation for risky requests.

## 27.11 Multimodal Extensions

Cognexis currently specifies a text-first decoder-only model. Extending recurrent depth to images, audio, or video introduces new questions:

- Which tokens or patches receive extra loops?
- How are modality-specific encoders integrated?
- Does recurrence refine shared multimodal latent states or only language states?
- How are compute budgets allocated across modalities?

Future work may combine recurrent depth with vision encoders, audio tokenizers, or unified multimodal token streams.

## 27.12 Tool Use and Retrieval

External tools and retrieval systems can reduce the need for internal recurrent computation or interact with it. The model may need to decide whether to think longer, retrieve, call a calculator, or generate.

Future work should study joint scheduling of loops and tool calls, including cost models that compare internal compute with external tool latency and reliability.

## 27.13 Hardware-Aware Design

The best loop policy depends on hardware. A policy efficient on one GPU may be inefficient on another due to kernel launch overhead, memory bandwidth, or batching behavior.

Future work should integrate runtime performance models into the scheduler and develop kernels specialized for recurrent shared-weight execution.

## 27.14 Checkpoint Interoperability

Standard LLM runtimes may not directly support shared recurrent blocks or adaptive loop policies. Exporting Cognexis checkpoints to common formats may require custom graph definitions or runtime plugins.

Future work should define an interchange format for recurrent-depth transformers and adapters for widely used inference runtimes.

## 27.15 Research Roadmap

High-priority future work:

- Establish stable baseline training recipes.
- Validate loop scaling on diverse tasks.
- Calibrate value-head schedulers.
- Build correct recurrent cache optimizations.
- Develop token-wise sparse kernels.
- Study safety behavior across depth.
- Publish reproducible ablation suites.

The architecture should evolve based on these measurements rather than assuming recurrent depth is universally beneficial.
