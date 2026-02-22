
# ffnc — FFN Warmup Compression Research

This repository documents the RA1 Labs FFN (Feed-Forward Network) width-compression research protocol. The goal is to empirically determine safe compression boundaries for FFN capacity annealing during training (not sparsity pruning). This README contains the experimental protocol, metrics, failure conditions, and deliverables required for the study.

**Contents**
- Research objective
- Hypothesis
- Model setup (fixed)
- Experimental variable
- Compression strategy
- Metrics to track
- Failure conditions
- Deliverables and reporting
- Practical notes and suggested tooling

## 1. Research Objective
Determine the maximum safe compression speed for FFN width reduction during training such that:
- Knowledge acquisition is preserved
- Recall and factual memory remain stable
- Training dynamics remain smooth (no irreversible loss spikes)

The final compressed model should match the functional behavior of the larger over-parameterized model while having a target size of approximately 75M parameters.

## 2. Hypothesis
Early training benefits from over-parameterized FFNs for feature discovery and memorization. As representations stabilize, FFN capacity becomes redundant and may be reduced if compression is guided by stability signals. There exists a maximum compression rate beyond which knowledge degradation becomes irreversible.

This experiment is designed to empirically identify that boundary.

## 3. Model Setup (Fixed)
- Architecture: RA1 / Rabbit (decoder-only)
- Attention, embeddings, and depth: frozen across experiments
- FFN: variable-width (the only component allowed to change)
- Optimizer, learning rate schedule, dataset: identical across runs
- Target final model size: ~75M parameters

Only FFN width is allowed to change; everything else must remain constant to isolate the effect of capacity annealing.

## 4. Experimental Variable
- Independent variable: FFN compression rate (percentage reduction per compression step)
- Controlled variables: training data order, random seeds, and handling of optimizer state during and after compression

Maintain consistent data shuffling and seed logging to ensure comparability.

## 5. Compression Strategy

5.1 Starting Point
- Initial FFN ratio: 4× (or the largest stable over-parameterized configuration available)
- Train until early learning stabilizes and stability indicators are observed:
	- Loss slope flattens
	- FFN gradient-norm variance decreases
	- Activation patterns show redundancy

5.2 Compression Schedule
- FFN width reductions must be incremental. No single step may exceed 15% width reduction.
- Suggested allowed step size: 10–15% width reduction per step.
- Example schedule: 4.0× → 3.6× → 3.2× → 2.8× → 2.4× → 2.0× → 1.5× → 1.0×

5.3 Recovery Requirement (Critical)
- After each compression step, continue training until loss recovers to within ≤2–3% of the pre-compression baseline.
- If loss does not recover within a reasonable token budget for that stage, the step was too aggressive: rollback and reduce step size.

Do not compress on a fixed clock; let the dynamics decide when the model has recovered sufficiently.

## 6. Metrics to Track (Mandatory)

6.1 Training Stability
- Training loss curve (tokens vs loss)
- Loss recovery time after compression (tokens to recover)
- Gradient norm (FFN layers only)

6.2 Knowledge Retention
- Run identical evaluation prompts before and after each compression step:
	- Factual recall
	- Long-tail entities
	- Memorization-sensitive samples
- Track accuracy / exact-match / relevant metrics and compare before vs after each step.

6.3 Functional Drift (Advanced)
- If feasible, compute model-output comparisons:
	- Output logits KL divergence on a held evaluation set
	- Monitor sharp divergence spikes after compression

## 7. Failure Conditions (Stop Criteria)
Immediately halt or rollback compression if:
- Loss spike does not recover within the allocated budget
- Recall accuracy drops consistently after a compression step
- Model behavior qualitatively degrades (hallucination, repetition, shallow responses)

These indicate compression exceeded safe limits and require a rollback and more conservative schedule.

## 8. Deliverables
For each experimental run, produce:
- Compression timeline: tokens vs FFN size (and cumulative compression rate)
- Maximum safe compression rate identified (per-stage and overall)
- Loss and stability plots (pre/post each compression): loss curves, gradient norms, recovery times
- Evaluation report: recall metrics before/after each step, KL divergence traces (if available)
- Summary of failure points and recommended safety margin
- Recommended production training schedule (step sizes, recovery budgets)

Deliver these artifacts as tracked experiment logs, a short PDF/markdown report, and visual plots.

## 9. Recommended Logging & Experiment Tracking
- Log these per-step items in a structured format (CSV/JSON):
	- Step index, wallclock, tokenizer tokens processed, FFN width (absolute & relative), pre-step loss, post-step loss, tokens-to-recover, gradient-norm statistics, seed, config hash
- Save model checkpoints before each compression step (for rollback)
- Use deterministic seeds where possible and record RNG states
- Version-control the exact configuration and training script used for the run

Suggested columns for a CSV experimental log:
step, tokens_at_step, ffn_ratio, ffn_width, loss_before, loss_after, tokens_to_recover, gradnorm_mean, gradnorm_var, recall_metric_before, recall_metric_after, note

## 10. Practical Notes and Suggested Tooling
- Checkpointing: keep a copy of the checkpoint immediately before each compression step for safe rollback.
- Visualization: produce an aggregated plot of tokens vs cost (loss) and tokens vs FFN size.
- Evaluation harness: create a deterministic evaluation script that runs identical prompts and records logits and metrics.
- Automated rollback: implement a small wrapper that applies compression, resumes training, monitors recovery, and restores pre-step checkpoint if the recovery condition is not met within the allocated budget.

## 11. Research Mindset
- This task is boundary-finding, not tuning for best final perplexity. Push compression until it breaks, identify where and why, and define a safety margin.
- Prefer conservatism and careful measurement when uncertain.

## 12. Next Steps (Suggested)
1. Implement an automated compression hook that: saves checkpoint → reduces FFN width → resumes training → monitors recovery → commits or rolls back.
2. Add structured logging and visualization scripts.
3. Run a small pilot (1–2 compression steps) to validate tooling and recovery thresholds.