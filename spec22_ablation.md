# 22. Ablation and Sensitivity Analysis

## 22.1 Purpose

Ablation studies determine which Cognexis components are necessary and how sensitive the system is to hyperparameters. They protect the project from architectural assumptions that are plausible but unverified. Ablations should be designed as controlled experiments with reproducible configs.

## 22.2 Required Component Ablations

Run or define experiments for:

- No recurrent core: replace recurrence with one standard block or bypass recurrence.
- Fixed loops only: disable adaptive scheduling.
- No Prelude: feed embeddings directly into recurrence.
- No Coda: feed recurrence directly into final norm and LM head.
- No recurrent gating.
- No input injection.
- No residual scaling.
- No value head: use rule-based scheduling only.
- No token-wise scheduling: use sequence-level scheduling.

Each ablation should report quality, compute, stability metrics, and parameter count.

## 22.3 Hyperparameter Sensitivity

Analyze sensitivity to:

- `max_loops`.
- `min_loops`.
- Recurrent residual scale.
- Gate initialization.
- Number of Prelude blocks.
- Number of Coda blocks.
- Hidden dimension.
- Number of attention heads and KV heads.
- FFN intermediate dimension.
- Normalization epsilon.
- RoPE scaling.
- Loop curriculum distribution.
- Value-head threshold.

Sensitivity experiments may use smaller models before confirming findings at larger scale.

## 22.4 Scheduler Sensitivity

Scheduler-specific sweeps:

- `min_delta`.
- Confidence threshold.
- Predicted gain threshold.
- Minimum loops.
- Maximum loops.
- Cost model.
- Safety risk weight.
- Token-wise versus sequence-level halting.

Report false-halt and false-continue rates when oracle traces are available. Thresholds should be selected from validation data and then tested on held-out tasks.

## 22.5 Stability Ablations

Stability controls should be ablated carefully:

- Remove spectral monitoring.
- Remove gradient clipping.
- Change residual scale.
- Disable gating.
- Switch RMSNorm to LayerNorm.
- Increase loop count beyond training range.

Some ablations can destabilize training. Run them with non-finite detection and smaller budgets before scaling.

## 22.6 Experimental Design

Each ablation must specify:

- Baseline config.
- Single changed factor.
- Training budget or checkpoint source.
- Evaluation suite.
- Random seeds.
- Hardware.
- Expected hypothesis.
- Stopping criteria for unstable runs.

Avoid comparing an ablated small model to a baseline trained with much larger compute unless the experiment is explicitly about compute tradeoffs.

## 22.7 Result Interpretation

For each ablation, answer:

- Did quality improve, degrade, or remain unchanged?
- Did compute or memory improve?
- Did stability change?
- Did scheduler behavior change?
- Is the component worth its complexity?

A component can be justified by stability or latency even if aggregate quality changes little.

## 22.8 Sensitivity Plots

Recommended plots:

- Metric versus hyperparameter value.
- Latency versus loop threshold.
- Hidden norm versus residual scale.
- Average loops versus value-head threshold.
- Quality-compute Pareto curves.

Use the same y-axis scales across related plots when possible to avoid misleading comparisons.

## 22.9 Configuration Management

Ablations should be generated from base configs using structured overrides:

```text
base.yaml + overrides/no_gate.yaml
base.yaml + overrides/no_coda.yaml
```

The resolved config for every run must be saved with results. Feature flags should be explicit and should not depend on environment variables hidden from the result manifest.

## 22.10 Rust Implementation Notes

The implementation should support ablations through config, not code edits. Example:

```yaml
model:
  recurrent:
    gating: false
    input_injection: none
```

For components that truly require compilation choices, use Cargo features with clear names and record enabled features in result metadata.

## 22.11 Validation

Ablation tooling tests must cover:

- Override application.
- Resolved config serialization.
- Parameter count reporting.
- Disabled component paths.
- Scheduler threshold sweeps.
- Result aggregation across runs.
- Failure marking for unstable experiments.
