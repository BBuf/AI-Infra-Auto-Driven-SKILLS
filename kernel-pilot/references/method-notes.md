# Method Notes

Source read:

- https://arxiv.org/abs/2603.24517
- https://arxiv.org/pdf/2603.24517

## Core Thesis

The useful idea is to replace a fixed mutate-and-test stage with a
self-directed coding agent. The agent owns one full iteration: inspect previous
solutions, consult a kernel knowledge base, edit code, run correctness,
benchmark, diagnose failures, and either continue revising or record a
candidate.

The abstract loop is:

```text
NextCandidate = Agent(previous_lineage, knowledge_base, scoring_function)
```

where:

- `previous_lineage` is the sequence of prior candidates and scores;
- `knowledge_base` is the curated set of references and implementation rules;
- `scoring_function` gates on correctness before considering speed.

## Iteration Step

One KernelPilot iteration should:

1. inspect lineage, failed attempts, and prior profiles;
2. consult only the approved references for the implementation lane;
3. propose one optimization hypothesis;
4. edit the candidate implementation;
5. run correctness;
6. benchmark every configured case;
7. diagnose failures or regressions;
8. revise until the candidate is worth recording.

KernelPilot records failed directions in lineage notes because those failures
are useful for later human review and for changing direction after stagnation.

## Continuous Optimization

This framework uses a single-lineage loop by default. It is not a broad archive
or population search. The practical goal is to keep the agent focused on one
operator until the requested speedup and no-regression gate are both proven.

## Kernel Knowledge Base

For GPU-kernel work, the knowledge base should include:

- local `gpu-kernel-ako4all/references`;
- architecture-specific material for the target GPU;
- CUDA, Triton, CUTLASS, or CuTe references matching the implementation lane;
- high-quality external kernel repositories selected as references only;
- task-specific tests, benchmark shapes, and forbidden source files.

The current KernelPilot catalog includes FlashInfer, FlashAttention,
Tencent hpc-ops, and DeepGEMM. The target project remains available for
integration and validation, but not as an external kernel reference.

## Evidence Style

Every performance claim should preserve:

- exact hardware and software environment;
- exact shapes, dtypes, and distributions;
- correctness tolerance and reference oracle;
- baseline and candidate latency for every case;
- geomean and minimum per-case speedup;
- lineage notes explaining which hypotheses worked and failed.
