# 23. Instruction Tuning and Code Generation

## 23.1 Purpose

Instruction tuning adapts a pretrained Cognexis model to follow user instructions, maintain dialogue format, and produce helpful task-oriented responses. Code generation fine-tuning adapts the model to produce syntactically valid and semantically correct code. Both settings benefit from compute-adaptive depth because task difficulty varies widely.

## 23.2 Data Format

Instruction examples should use structured messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are Cognexis."},
    {"role": "user", "content": "Summarize this text."},
    {"role": "assistant", "content": "Summary..."}
  ],
  "metadata": {
    "source": "dataset_name",
    "task": "summarization",
    "difficulty": "medium"
  }
}
```

The loader renders this structure with the configured chat template. The template version must be stored with the fine-tuned checkpoint.

## 23.3 Loss Masking

Instruction tuning should normally compute loss only on assistant response tokens:

```text
loss_mask = 1 for assistant content
loss_mask = 0 for system/user/template tokens
```

Exceptions, such as training the model to reproduce tool-call syntax, must be explicit. Incorrect loss masking can train the model to imitate user prompts or system messages.

## 23.4 Loop Budget During Fine-Tuning

Fine-tuning should preserve multi-depth competence. Recommended approach:

- Start with fixed loop counts sampled from the pretrained curriculum range.
- Include shallow and deep depths in every epoch.
- Add task-aware depth sampling after baseline stability.
- Train or calibrate the adaptive scheduler on held-out instruction examples.

Do not fine-tune only at the maximum loop count unless the deployment will never use shallow loops.

## 23.5 Prompt-Level Depth Controls

Some deployments may expose user or system controls for quality tiers:

```text
fast: lower max loops
balanced: default adaptive budget
thorough: higher max loops within policy limits
```

These controls should be structured request options, not arbitrary user text, whenever possible. If prompt tokens are used to indicate depth, they must be reserved special tokens and protected from untrusted injection.

## 23.6 Code Generation Data

Code generation datasets should include:

- Problem statement.
- Function signature or file context.
- Reference solution.
- Unit tests when available.
- Language tag.
- Difficulty and source metadata.

The tokenizer must preserve whitespace, indentation, punctuation, and rare identifiers. Data cleaning should remove duplicate benchmark solutions from training splits when evaluating code tasks.

## 23.7 Code Evaluation

Code generation should be evaluated by functional correctness:

- Run generated code against unit tests.
- Report pass@1 and pass@k.
- Capture compile errors, runtime errors, timeouts, and wrong answers.
- Use sandboxed execution.
- Fix random seeds for stochastic generation.

Text overlap metrics are not sufficient for code correctness.

## 23.8 Adaptive Depth for Code

Code tasks may use additional signals:

- Syntax parser success.
- Compiler diagnostics.
- Unit test feedback in offline reranking.
- Model confidence over structural tokens.
- Problem difficulty metadata.

Online generation should not run arbitrary untrusted code unless a sandbox is configured. Offline evaluation can use test execution to analyze whether deeper loops improve pass rates.

## 23.9 Tool and Function Calling

If Cognexis supports tool calls, the chat template must define:

- Tool declaration format.
- Tool-call token format.
- Tool-result message format.
- Loss mask behavior for tool calls.
- Safety policy for tool arguments.

Loop depth may be adjusted around tool calls. For example, the scheduler may allocate more loops before deciding whether to call a tool, then fewer loops when summarizing a deterministic tool result.

## 23.10 Safety Filtering

Instruction and code datasets must be filtered for:

- Harmful instructions.
- Private or copyrighted sensitive content where prohibited.
- Malware or credential-stealing code.
- Prompt injections in training examples.
- Toxic or abusive dialogue.

Filtering should record counts and reasons. Safety filtering is not a substitute for runtime safety policies.

## 23.11 Fine-Tuning Hyperparameters

Instruction tuning usually uses:

- Lower learning rate than pretraining.
- Shorter training duration.
- Smaller or domain-balanced datasets.
- Optional parameter-efficient fine-tuning adapters.
- Regular evaluation to prevent overfitting.

If using LoRA or adapters, specify whether adapters apply to Prelude, recurrent core, Coda, value head, or all modules. Recurrent-core adapters are shared across loops by default.

## 23.12 Rust Implementation Notes

Recommended components:

```text
src/training/sft/
  dataset.rs
  chat_collator.rs
  loss_mask.rs
  code_eval.rs
  tool_format.rs
```

Code execution should run outside the main training process or inside a restricted sandbox. The evaluation harness should treat timeouts and sandbox violations as structured outcomes.

## 23.13 Validation

Instruction and code tuning tests must cover:

- Chat template rendering.
- Assistant-only loss masks.
- Special-token injection prevention.
- Loop sampling during fine-tuning.
- Code indentation preservation.
- Sandbox timeout handling.
- pass@k calculation.
- Tool-call formatting if enabled.
