# Pull Request DAG

## Dependencies

| Node | Scope | Depends on | Merge state |
| --- | --- | --- | --- |
| P0 | Public infrastructure: remote target/draft loader and recurrent-state transport contract | none | planned |
| P1 | Model spine: config, weights, hybrid attention, MoE, state, eager fallback | P0 | planned |
| P2 | Protocol/VLM: streaming parser, image processor, vision encoder/projector | P1 | planned |
| P3 | Platform/packaging: B200 dispatch and public image dependencies | P1 | planned |
| P4 | Validation/docs: risk-pair CI, cookbook, release lock, follow-up ownership | P1, P2, P3 | planned |

The integration PR visualizes the five nodes and cannot merge before P0
through P4 close their gates.

## Merge Gates

| Node | Gate to merge |
| --- | --- |
| P0 | Target and draft remote-load parity; PD recurrent-state round trip |
| P1 | Complete weight audit; eager/reference parity; TP8 liveness |
| P2 | Fragmented-marker streaming tests; image preprocessing parity |
| P3 | Reproducible image build; eager fallback and dispatcher tests |
| P4 | All seven gates, public recipes, sanitizer, and named follow-up owner |
