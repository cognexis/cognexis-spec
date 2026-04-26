# 24. Safety, Monitoring and Observability

## 24.1 Purpose

Cognexis must be safe and observable in training, evaluation, and production. Compute-adaptive depth adds new operational questions: how many loops were used, why did the scheduler halt, did deeper loops improve or harm safety, and were compute budgets enforced? This document defines required controls and telemetry.

## 24.2 Safety Boundaries

Safety controls must exist outside the model weights. The model may learn safer behavior, but serving must still enforce:

- Input policy checks.
- Output policy checks.
- Special-token injection prevention.
- Compute budget limits.
- Code execution sandboxing.
- Abuse monitoring.
- Request cancellation and timeouts.

Adaptive scheduling must not bypass these controls.

## 24.3 Input Filtering

Input filtering should detect:

- Disallowed harmful requests.
- Prompt injection attempts targeting system or scheduler controls.
- Requests to reveal hidden prompts or internal chain data.
- Excessive context length or resource abuse.
- PII or sensitive data handling requirements.

Filtering decisions should happen before expensive prefill when possible. Some requests may be refused without invoking the model.

## 24.4 Output Filtering

Generated output should be checked for:

- Harmful instructions.
- Toxic or abusive content.
- Private data leakage.
- Unsafe code.
- Policy-violating tool calls.
- Formatting violations for constrained outputs.

Output filtering may happen token-by-token for streaming or after full generation. Token-level filtering has lower latency for early stops but can be harder to implement correctly.

## 24.5 Scheduler Safety

Scheduler-specific risks:

- Prompt injection asks for maximum loops to consume resources.
- Deeper loops amplify unsafe content.
- Scheduler halts too early on safety-critical classification.
- Value head underestimates risk for unfamiliar prompts.
- Token-wise halting ignores important context tokens.

Mitigations:

- Cap user-selectable loop budgets.
- Include safety risk signals in halt decisions.
- Enforce minimum loops for safety classifiers if used.
- Log halt reasons and loop distributions.
- Evaluate safety metrics across loop counts.

## 24.6 Compute Budget Enforcement

Every request must enforce:

- Maximum prompt tokens.
- Maximum generated tokens.
- Maximum loops per token or sequence.
- Total loop budget.
- Wall-clock timeout.
- Memory/cache allocation limit.

Budget checks must be hard runtime guards, not just scheduler preferences. On budget exhaustion, generation should stop with `stop_reason = budget_exhausted` or return a controlled error depending on product policy.

## 24.7 Observability Fields

Structured logs should include:

- Request ID.
- Model/checkpoint ID.
- Tokenizer checksum.
- Prompt token count.
- Generated token count.
- Loop mode.
- Loop counts summary.
- Halt reasons.
- Prefill latency.
- Decode latency.
- Safety filter actions.
- Error or stop reason.
- Hardware/backend identifier.

Logs should not include raw prompt or output text by default. If content logging is enabled for debugging, it must follow privacy and retention policy.

## 24.8 Metrics

Expose metrics such as:

- Requests per second.
- Tokens per second.
- Average loops per request.
- Loop count histogram.
- Scheduler halt reason counts.
- Safety refusal rate.
- Budget exhaustion rate.
- Cache memory usage.
- Non-finite activation/logit count.
- Error rate by category.
- Latency percentiles.

Metrics should be suitable for dashboards and alerts.

## 24.9 Tracing

Use tracing spans for:

- Tokenization.
- Prefill.
- Each recurrent loop or loop group.
- Scheduler observation.
- Coda.
- LM head.
- Sampling.
- Safety filtering.
- Streaming emission.

High-cardinality fields should be limited. Loop-level tracing can be sampled in production to control overhead.

## 24.10 Red Teaming

Red-team suites should cover:

- Prompt injection against scheduler controls.
- Requests that try to force budget exhaustion.
- Harmful content at shallow and deep loop counts.
- Long-context distraction attacks.
- Code-generation malware prompts.
- Data exfiltration attempts through tool calls.
- Overconfidence or hallucination under high loops.

Results should feed into safety filters, scheduler caps, and evaluation benchmarks.

## 24.11 Incident Response

Production systems should support:

- Disabling adaptive scheduling quickly.
- Lowering maximum loops globally.
- Blocking unsafe prompt patterns.
- Rolling back checkpoints.
- Increasing logging sample rates.
- Draining or canceling abusive requests.

Operational controls should be configuration-driven and auditable.

## 24.12 Rust Implementation Notes

Use structured logging and metrics crates:

```rust
tracing::info!(
    request_id = %request_id,
    loop_mode = ?loop_mode,
    loops_mean = loops_mean,
    halt_reason = ?halt_reason,
    "generation_complete"
);
```

Define a `SafetyContext` passed through generation:

```rust
pub struct SafetyContext {
    pub input_flags: SafetyFlags,
    pub output_flags: SafetyFlags,
    pub budget: ComputeBudget,
    pub policy_mode: PolicyMode,
}
```

Safety modules should wrap generation APIs rather than being embedded inside low-level tensor modules.

## 24.13 Validation

Safety and observability tests must cover:

- Input filter refusal path.
- Output filter stop or rewrite path.
- Special-token injection rejection.
- Loop budget enforcement.
- Timeout handling.
- Structured log fields.
- Metrics emission.
- Red-team fixture execution.
- Adaptive scheduler safety override.
