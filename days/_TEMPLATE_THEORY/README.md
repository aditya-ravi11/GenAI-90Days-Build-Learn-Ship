# Day {{DAY}} — {{TITLE}}

## Objectives
- [ ] Primary: {{PRIMARY_OBJECTIVE}}
- [ ] Secondary: {{SECONDARY_OBJECTIVE}}

## TL;DR
- {{POINT_1}}
- {{POINT_2}}
- {{POINT_3}}

## Concepts & Intuition
Briefly explain the intuition. Include 1 diagram and 1 key formula.

```mermaid
graph TD
Q[Query] --> DotProd[Dot product with Keys]
DotProd --> Scale[Scale by sqrt(d_k)]
Scale --> Softmax[Softmax]
Softmax --> Weights[Attention Weights]
Weights --> Sum[Weighted sum of Values]
```

### Key Formula
\[ \text{{Attention}}(Q,K,V) = \text{{softmax}}\left( \frac{{QK^\top}}{{\sqrt{{d_k}}}} \right) V \]

## Mini-Experiment
- Run: `python tiny_demo.py`
- What this shows: {{WHAT_IT_SHOWS}}

## Notes & Claims
- Claim 1: {{CLAIM_1}}  
- Evidence: {{EVIDENCE_1}}

## Open Questions
- {{QUESTION_1}}

## References
- {{REF_1}}
