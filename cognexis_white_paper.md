# Cognexis: A Compute‑Adaptive Recurrent‑Depth Large Language Model

## Abstract

Cognexis is a family of decoder‑only large language models (LLMs) that separate reasoning depth from parameter count through the use of recurrent computation.  Instead of scaling performance by stacking ever‑deeper transformer layers, Cognexis reuses a shared recurrent reasoning core multiple times at inference.  This design allows the model to allocate more computation to difficult inputs while keeping the number of stored parameters constant.  Cognexis provides a complete stack that includes a tokenizer, embedding layers, a **Prelude–Recurrent–Coda** architecture, an adaptive loop scheduler, and a training and evaluation framework.  The models support both fixed and adaptive iteration counts, exposing reasoning depth as a runtime parameter.  We outline the architecture, implementation details, training strategy, evaluation metrics, and runtime scheduling for Cognexis and discuss the implications of compute‑adaptive reasoning for future language models.

## 1 Introduction

### 1.1 Motivation

Modern LLMs scale capabilities by increasing model size (number of parameters) and depth (number of distinct transformer layers).  Each layer applies a unique set of weights to the hidden representation, so deeper models require more parameters.  While this approach yields performance gains, it couples reasoning depth tightly to storage footprint and inference latency: deeper models require more memory and longer wall‑clock time even when the input does not require complex reasoning.  Furthermore, training very deep networks is increasingly expensive and challenging.

**Recurrent‑Depth Transformers (RDTs)** propose a different scaling route.  Instead of stacking many distinct transformer layers, an RDT reuses a single recurrent block multiple times.  Each iteration refines the hidden state, emulating deeper networks without increasing the number of parameters.  This reuse suggests that reasoning depth can be treated as a function of compute rather than model size: more iterations allow for deeper reasoning, while fewer iterations offer fast responses.  Cognexis builds on this idea and turns it into a full LLM with a compute‑adaptive interface.

### 1.2 Contributions

This work introduces Cognexis and makes the following contributions:

1. **Compute‑Adaptive Architecture:**  We design a decoder‑only architecture where reasoning depth is determined by the number of recurrent iterations at inference.  The model uses a Prelude of conventional transformer blocks, a shared recurrent block, and a Coda for final integration.
2. **Complete LLM Stack:**  The project includes a tokenizer, embedding layers with rotary positional encoding, weight‑tied language modeling head, training scripts, and inference pipeline.  It supports parameter scales of 8B, 64B, 256B, and 1.28 T.
3. **Adaptive Loop Scheduler:**  An inference scheduler decides how many iterations to perform based on expected marginal gains in accuracy per unit of computation (the **Depth Efficiency Index** defined in §7).  This yields compute‑adaptive reasoning: the model can stop early on easy inputs or loop further when deeper reasoning pays off.
4. **Evaluation Metrics:**  We introduce the **Depth Efficiency Index (DEI)**, **Loop Saturation Point (LSP)**, and **Overthinking Threshold (OT)** to evaluate how efficiently additional iterations improve performance.
5. **Specification:**  We provide a build‑ready specification with configuration parameters for different model sizes, pseudocode for forward and backward passes, memory layout details, and guidelines for distributed training and inference.

## 2 Background and Related Work

### 2.1 Transformer Scaling

Transformers are the foundation of most modern LLMs.  Their self‑attention mechanism allows long‑range context modeling.  Scaling these models typically involves increasing the number of layers and hidden dimensions.  However, memory and compute costs grow super‑linearly with depth and width.  Deeper architectures also face optimization difficulties such as vanishing gradients and training instabilities.

### 2.2 Recurrent Architectures

Recurrent neural networks (RNNs) use parameter sharing across time steps to model sequences.  Deep equilibrium models (DEQs) and iterative refinement networks apply the same block repeatedly until convergence.  These approaches reduce parameter count but require careful design to ensure stability.  Adaptive computation time (ACT) networks allow variable numbers of iterations on different inputs, trading off between accuracy and efficiency.

### 2.3 Depth‑Adaptive LLMs

Recent research explores latent reasoning through iterative refinement rather than static depth.  Some proposed architectures use a recurrent core where each iteration deepens the representation.  Open‑source interpretations of this idea implement basic recurrent‑depth transformers but often lack a full training pipeline and evaluation framework.  Cognexis fills this gap by providing a complete system for research and deployment.

## 3 Architecture

### 3.1 Model Overview

Cognexis is a decoder‑only LLM that processes an input sequence and autoregressively predicts the next token.  The model consists of three stages:

1. **Prelude:**  Several standard transformer blocks build initial token‑level representations.  These layers operate exactly like those in GPT‑style models, using multi‑head causal self‑attention and feed‑forward networks.  The Prelude stabilizes the hidden state before recurrence.
2. **Recurrent Core:**  A shared block applies to the hidden state multiple times.  Each iteration comprises a self‑attention module and feed‑forward layers with parameter sharing.  Residual connections and layer normalization ensure stability.  The hidden state evolves over iterations, enabling deeper reasoning without extra parameters.
3. **Coda:**  Additional transformer blocks integrate information after recurrence and prepare the hidden state for output.  These layers mix token information across positions to produce high‑quality logits.

Finally, a linear projection tied to the embedding matrix (weight tying) produces vocabulary logits.  Softmax yields the next‑token distribution.

### 3.2 Tokenization and Embeddings

The model uses a SentencePiece tokenizer trained on a large corpus to build a subword vocabulary (e.g., 50k tokens).  Input text is converted into token IDs with special tokens for the beginning of sequence (BOS), end of sequence (EOS), padding, and potential chat turns (system/user/assistant tokens).  Token embeddings map each ID to a learned vector.  Rotary position encodings (RoPE) are applied to the query and key vectors in attention, enabling flexible extrapolation to longer contexts.

### 3.3 Prelude

The Prelude comprises \(L_p\) blocks (typically 6–12).  Each block contains:

* **LayerNorm** to normalize hidden states.
* **Multi‑Head Self‑Attention (MSA)** with query, key, and value projections.  We use **grouped query attention (GQA)** by default, where queries are divided into \(g\) groups with shared keys and values to reduce memory.
* **Feed‑Forward Network (FFN)** with a gated activation (e.g., GELU or SwiGLU), followed by a linear projection.
* **Residual Connections** after each sub‑layer.

### 3.4 Recurrent Core

The recurrent core is the hallmark of Cognexis.  It contains a single transformer‑style block \(R\) with shared parameters.  At iteration \(t\), the hidden state \(h_t\) is updated as:

```
h_{t+1} = R(h_t)
```

where \(R(\cdot)\) includes layer normalization, MSA, FFN, and residual connections.  The core reuses the same weights for every iteration.  To maintain stability, we enforce a spectral normalization constraint on the weight matrices or track the spectral radius during training.  An optional gating mechanism allows the network to decide how much of the previous state to retain at each iteration:

```
z_t = σ(W_z h_t + b_z)
h_{t+1} = z_t ⊙ R(h_t) + (1 – z_t) ⊙ h_t
```

### 3.5 Coda

The Coda mirrors the Prelude with \(L_c\) transformer blocks.  It integrates information across tokens after recurrence.  This stage allows the hidden representations to capture global dependencies and prepares the features for output.

### 3.6 Language Modeling Head

The final hidden state passes through an RMS normalization and then a linear projection tied to the token embeddings.  The projection yields logits over the vocabulary.  During training, the model minimizes the cross‑entropy loss between the predicted distribution and the ground‑truth next token.

### 3.7 Effective Depth

The **effective depth** of Cognexis is defined as the total number of sequential transformations applied to the hidden state.  If there are \(L_p\) Prelude blocks, a recurrent core with \(L_r\) unique blocks shared for \(N\) iterations, and \(L_c\) Coda blocks, then the effective depth \(D_{eff}\) is:

\[D_{eff} = L_p + (N \cdot L_r) + L_c\]

This formula implies that increasing the loop count \(N\) increases the effective depth without increasing the number of unique parameters.  Effective depth is purely conceptual and should not be conflated with parameter count (see §7 for more discussion on compute vs. parameters).

## 4 Training

### 4.1 Objective

Cognexis is trained to minimize the negative log‑likelihood of the next token given the previous context.  Formally, for a sequence \(x_1, x_2, \dots, x_T\), the model maximizes

\[\sum_{t=1}^{T-1} \log P_θ(x_{t+1} | x_1, x_2, \dots, x_t).\]

### 4.2 Datasets

Training requires large and diverse corpora of text.  In practice, one might use a mixture of public web data, books, Wikipedia, and curated datasets to create a **Common Crawl‑style** dataset.  The training set should be sufficiently large to support models up to trillions of parameters.  Pretraining and fine‑tuning data filtering follow industry standard procedures: de‑duplication, removal of harmful content, tokenization quality checks, and balancing across domains.

### 4.3 Loop Curriculum

Training must account for the recurrent depth.  Using a fixed loop count can lead to poor generalization at other depths.  We propose a **loop curriculum**:

1. **Warm‑up:**  Train for a number of epochs with a low loop count (e.g., \(N=2\)).  This stabilizes the model and prevents divergence.
2. **Randomized loops:**  Sample the loop count for each batch from a distribution (e.g., uniform from 1 to the target maximum).  This encourages the model to perform well at variable depths.
3. **Depth expansion:**  Gradually increase the maximum loop count as training progresses.  The model learns to generalize beyond its initial depth.

This curriculum helps the model adapt to different loop counts at inference and reduces the chance of catastrophic forgetting at low depths.

### 4.4 Optimization

Cognexis uses AdamW or any modern variant for optimization.  For large models, we employ memory‑efficient techniques like gradient checkpointing, fused kernels, and mixed precision (FP16 or bfloat16).  Distributed training frameworks such as Fully‑Sharded Data Parallel (FSDP) or ZeRO partition the model across GPUs.  We also use **tensor parallelism** in the attention and FFN modules to distribute computation across multiple devices.

### 4.5 Stability and Regularization

Recurrent models can be prone to divergence if the spectral radius of the linear transformations exceeds 1.  To ensure stability, we monitor the spectral norm of the weight matrices and optionally apply spectral normalization.  Layer normalization and residual scaling also help maintain stability.  Regularization techniques include dropout, weight decay, and stochastic depth in the recurrent iterations.

## 5 Inference and Generation

### 5.1 Prefill and Decode

Autoregressive generation comprises two phases: **prefill** and **decode**.  In the prefill phase, the model processes the entire input context in parallel.  Recurrent iterations are applied at each token position.  In the decode phase, the model generates one token at a time using cached key–value (KV) states.  Because the recurrent core reuses parameters, we also cache the hidden state at each iteration to avoid recomputation.  The inference algorithm must handle KV cache for both the Prelude/Coda and the recurrent core.

### 5.2 Loop Scheduling

A key feature of Cognexis is the ability to adjust the number of recurrent iterations at inference time.  The inference API accepts a **loop budget** (maximum iterations) or a **compute budget** (FLOP limit).  The scheduler then decides how many loops to run based on the expected performance gain.

We implement an **adaptive scheduler** based on the **Depth Efficiency Index (DEI)** defined in §7.  The scheduler monitors the marginal improvement of the model’s confidence or predicted quality after each iteration.  When the improvement per unit compute falls below a threshold, the scheduler stops.  A simpler fixed loop mode runs a constant number of iterations for all tokens.

### 5.3 Token‑Adaptive Depth

Instead of using the same loop count for all positions, Cognexis can allocate more iterations to “hard” tokens and fewer to “easy” tokens.  At each step, the scheduler examines the hidden state’s change or the predicted probability of the target token.  If the probability is already high and changing slowly, the loop count can be reduced.  Conversely, ambiguous tokens are assigned more loops.  This token‑adaptive strategy reduces average inference latency without sacrificing accuracy.

### 5.4 API Interface

An example generation API call:

```
model.generate(
    input_ids,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    loop_mode="adaptive",
    loop_budget=16,
)
```

The `loop_mode` parameter accepts values `fixed`, `adaptive`, or `token_adaptive`.  `loop_budget` sets the maximum number of iterations.  Additional arguments control sampling temperature, nucleus sampling (`top_p`), and stopping criteria.

## 6 Model Configurations

Cognexis supports multiple parameter scales.  Each configuration defines hidden dimension size, number of heads, number of Prelude/Coda blocks, number of recurrent blocks, and the maximum loop count during training.  Table 1 summarizes typical settings for the 8B, 64B, 256B, and 1.28 T models.

| Model            | Hidden Size | Attention Heads | Prelude Blocks \(L_p\) | Recurrent Blocks \(L_r\) | Coda Blocks \(L_c\) | Maximum Loops (training) | Parameters |
|-----------------|------------|----------------|-----------------------|---------------------------|---------------------|---------------------------|-----------|
| **Cognexis‑8B**   | 4096       | 32             | 8                     | 1                         | 8                   | 12                        | ~8 B      |
| **Cognexis‑64B**  | 8192       | 64             | 10                    | 1                         | 10                  | 16                        | ~64 B     |
| **Cognexis‑256B** | 12288      | 96             | 12                    | 1                         | 12                  | 20                        | ~256 B    |
| **Cognexis‑1.28T**| 16384      | 128            | 16                    | 1                         | 16                  | 24                        | ~1.28 T   |

The hidden size scales with the model size, and the number of attention heads increases accordingly.  The recurrent core always uses a single block, but training uses more iterations for larger models.  Effective depth can be adjusted at inference using the loop scheduler, independently of parameter count.

## 7 Evaluation Metrics

Standard LLM evaluation metrics such as perplexity, multiple‑choice accuracy, and code generation tests remain relevant.  However, because Cognexis operates on a new axis—recurrent depth—we require additional metrics.

### 7.1 Depth Efficiency Index (DEI)

Let \(M(N)\) be a performance metric (e.g., accuracy on a dataset) with \(N\) loops and \(C(N)\) be the compute cost in FLOPs.  For a baseline loop count \(N_0\), the **Depth Efficiency Index** at \(N\) is:

\[\mathrm{DEI}(N) = \frac{M(N) - M(N_0)}{C(N) - C(N_0)}.\]

DEI measures the performance gain per additional unit of compute.  Higher DEI indicates that extra loops are providing good returns on compute.  Plotting DEI across loop counts reveals diminishing returns and helps choose a suitable loop budget.

### 7.2 Loop Saturation Point (LSP)

Define the **Loop Saturation Point** as

\[\mathrm{LSP} = \underset{N}{\arg\max} \; \mathrm{DEI}(N).\]

The LSP identifies the most efficient reasoning depth: adding more loops beyond LSP yields lower marginal gains.  Inference schedulers can use LSP as a guideline for default loop budgets.

### 7.3 Overthinking Threshold (OT)

The **Overthinking Threshold** is the smallest loop count \(N\) such that performance begins to decline with further iterations:

\[\mathrm{OT} = \min \{ N : M(N+1) < M(N) \}.\]

Overthinking occurs when extra iterations introduce noise or destabilize the hidden state.  The scheduler should avoid exceeding OT.

### 7.4 Depth Gain Ratio (DGR)

The **Depth Gain Ratio** compares the total gain from recurrence against the baseline model (with no recurrence):

\[\mathrm{DGR} = \frac{M(N_{\max}) - M(N_0)}{M(N_0)}.\]

Higher DGR indicates that recurrence significantly improves performance relative to the baseline.

## 8 Runtime Scheduling

### 8.1 Scheduler Design

The adaptive scheduler aims to maximize performance under a compute budget.  After each iteration, the scheduler estimates the potential improvement from the next loop.  The estimate can be based on the change in hidden state norm, the difference in log probabilities of top tokens, or a small **value head** trained to predict the marginal gain (see §8.3).  If the expected improvement per compute is below a threshold (derived from DEI curves), the scheduler halts.

### 8.2 Token‑Wise Scheduling

In token‑adaptive mode, the scheduler decides loop counts for each token independently.  This can be implemented by storing separate “loop counters” for each position and updating them until the halting criterion is satisfied.  Care must be taken to synchronize positions that depend on each other via attention.  A simple strategy is to allow tokens to finish early but keep their hidden state unchanged in subsequent iterations.

### 8.3 Value Head (Optional)

To predict future gains more accurately, we can train a lightweight **value head** on top of the recurrent hidden state.  The value head outputs a scalar representing the expected improvement in loss after an additional loop.  During training, we record the improvement at each iteration and regress the value head to these improvements.  At inference, the scheduler uses the value head to decide whether to continue.

## 9 Security and Safety Considerations

Cognexis inherits the safety concerns of other LLMs: potential to generate harmful content, biases in training data, and privacy risks.  Additional considerations for recurrent models include:

* **Loop Stability:**  Excessive iterations may lead to divergent representations and unpredictable outputs.  Hard limits and monitoring of hidden state norms are necessary.
* **Resource Exhaustion:**  Adaptive loops consume variable compute.  Systems must enforce budgets to prevent denial‑of‑service via maliciously long prompts.
* **Timing Side Channels:**  Observers might infer input difficulty from inference latency.  In high‑risk settings, introducing randomization into loop counts can mitigate this risk.

## 10 Limitations and Future Work

While Cognexis demonstrates the promise of compute‑adaptive reasoning, several limitations remain:

* **Training Complexity:**  Designing loop curricula and ensuring stability across depths is more challenging than training static models.
* **Convergence Guarantees:**  The recurrent core may not always converge to a fixed point.  Further research is needed on controlling dynamics.
* **Tool Integration:**  Many advanced LLM systems integrate external tools (search, calculators).  Extending the recurrent mechanism to handle tool calls and streaming inputs is an open problem.
* **Scaling to Longer Contexts:**  The recurrence operates on all tokens simultaneously, which may limit context length compared to transformer architecture optimized for long sequences.

Future work includes developing task‑adaptive value heads, exploring other recurrent architectures (e.g., hybrid state‑space models), and integrating planning or tool‑use capabilities into the recurrence.

## 11 Conclusion

Cognexis reimagines scaling laws for language models by decoupling reasoning depth from parameter count.  Through a reusable recurrent block, the model achieves deeper effective computation without increasing memory footprint.  The compute‑adaptive interface allows dynamic allocation of loops at inference, trading off between latency and performance.  We described the complete architecture, training procedures, evaluation metrics, runtime scheduling, and configurations for models ranging from 8B to 1.28 T parameters.  By providing a full specification and adaptive scheduler, Cognexis offers a new research direction and practical framework for building efficient and flexible LLMs.