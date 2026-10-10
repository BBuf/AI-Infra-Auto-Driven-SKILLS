---
name: optimization-analysis
description: "Analyze a proposed AI-infrastructure performance change by tracing saved costs, eligibility conditions, added costs, and controlled validation. Use to assess an optimization hypothesis, identify workload factors, or establish its applicability and regression boundaries."
---

# Optimization Analysis

Turn a proposed change into a causal hypothesis and a bounded validation plan.
Work from the user's code, trace, PR, or conceptual example. Match the depth to
the question; a conceptual discussion does not require launching a benchmark.
Respond in the user's language.

## Establish the change

Name the target metric and show the relevant before/after execution path.
Follow the request or data through its producer, queue, execution, consumer,
and release points as needed. Identify where the proposed change acts.

Separate source observations, measured results, and hypotheses. If no source
or measurement is available, state the assumptions and analyze the mechanism;
do not turn an illustrative path into a claim about a current implementation.

## Four-question analysis

Use this table as the default compact output. Fill it with concrete operations
and decision criteria, not generic advantages and disadvantages.

| Question | What to establish |
| --- | --- |
| What cost is removed? | Which computation, copy, wait, allocation, or call disappears? Explain how it could affect the target metric. Less internal work alone does not establish an end-to-end improvement. |
| Under what conditions does it hold? | Separate correctness prerequisites from factors affecting benefit. Check relevant workload properties, implementation capabilities, hardware, and deployment. Distinguish external workload, runtime state, and tunable policy parameters. |
| What cost is added? | Look for work shifted to a consumer, resource, or other request; include lifetime and management changes. Do not invent a tradeoff when the change may remove pure waste. State remaining uncertainty. |
| What comparison would validate it? | Specify baseline/candidate, controlled inputs, correctness checks, mechanism observations, and target metrics. Say what result would support or contradict the proposed explanation. |

## Explain the decisive causal relationship

After the table, explain the one or two relationships that determine the
decision. Derive candidate factors from the changed costs rather than listing
every imaginable variable. Ask whether a cost is removed, deferred, or moved.
Distinguish resource contention from actual preemption, and execution overlap
from request selection policy when these mechanisms matter.

Use simple cost models only as hypotheses. First vary a plausible major factor
in a controlled comparison, then check important interactions. Do not assume
GPU time is linear in bytes, operations, or batch size.

For each A/B pair, hold the external workload and unrelated configuration
comparable. Do not freeze internal batches or schedules that the optimization
is supposed to change. A controlled replay can isolate a mechanism; a separate
serving comparison must include effects on other stages and requests.

Use repeated or counterbalanced measurements when timing is involved. Check
that the intended path actually ran, and distinguish correctness, mechanism,
and unprofiled end-to-end evidence. For detailed experiment controls, consult
[paired validation](../llm-serving-auto-benchmark/references/paired-validation.md)
when needed; its kernel-specific checks apply only to relevant changes.

## State applicability and the next decision

Finish with:

- **Applicability:** separate demonstrated benefit, demonstrated regressions,
  and untested conditions. Without experiments, use candidate conditions and
  possible failure modes instead of claiming a proven range.
- **Next step:** choose the smallest useful source check or experiment that
  could change the decision. With sufficient evidence, give an enablement or
  fallback recommendation and identify how its conditions can be recognized.

A benefit in a meaningful subset of workloads can be valid. Check workloads
not used for tuning before generalizing. An observed runtime signal is not
automatically a reliable or cheap policy input; any proposed adaptive rule
needs its own validation.

Do not require exhaustive factor searches. Stop the current investigation when
additional plausible tests are unlikely to change the scoped enablement
decision and remaining risks are bounded. Unresolved correctness or lifetime
requirements still block recommending the affected path. Revisit the boundary
when deployment scope expands or counterexamples appear.

## Reference use

Read [the chunk-queue example](references/chunk-queue-example.md) when analyzing
incremental concatenation, or when a worked example of the output is useful.
It is an unmeasured illustration, not performance evidence.

References supply task-specific reasoning, examples, procedures, or schemas;
they need not all follow the four-question format. Keep the shared analysis
method here, and load supporting material only when it changes a judgment.
