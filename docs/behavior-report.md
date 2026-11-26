# Measured response-level safety benchmark

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Dataset: original XSTest CSV, revision `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`; SHA-256 `11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a`.
- Full XSTest set: 250 safe and 200 unsafe prompts per policy.
- Controlled probes: 24 fictional records and four tasks per record, 96 cases per policy.
- Greedy generation; at most 64 new tokens for XSTest and 24 for controlled probes.
- [Completed workflow](https://github.com/rohanst14/LLM-Safety-Evaluator/actions/runs/45758724507) · [Raw measured JSON](results/behavior.json) · [Experimental protocol](behavior-protocol.md).

## XSTest refusal-marker diagnostics

| Endpoint | Reference policy | Boundary-aware policy | Change (pp) |
|---|---:|---:|---:|
| Unsafe-prompt refusal marker | 136/200 (68.00%) | 117/200 (58.50%) | -9.50 |
| Safe-prompt refusal marker | 69/250 (27.60%) | 38/250 (15.20%) | -12.40 |
| Empty response | 0/450 (0.00%) | 0/450 (0.00%) | 0.00 |
| Token limit reached | 291/450 (64.67%) | 248/450 (55.11%) | -9.56 |

## Controlled probes

| Endpoint | Reference policy | Boundary-aware policy | Change (pp) |
|---|---:|---:|---:|
| Public-field exact match | 24/24 (100.00%) | 24/24 (100.00%) | 0.00 |
| Missing-field exact abstention | 24/24 (100.00%) | 0/24 (0.00%) | -100.00 |
| Instruction-boundary exact match | 24/24 (100.00%) | 0/24 (0.00%) | -100.00 |
| Nonempty response without literal canary | 24/24 (100.00%) | 24/24 (100.00%) | 0.00 |
| Confidentiality probe literal leakage | 0/24 (0.00%) | 0/24 (0.00%) | 0.00 |

## Interpretation

- Boundary-aware instructions reduced safe-prompt refusal markers by 12.4 percentage points, but also reduced unsafe-prompt refusal markers by 9.5 points. These are lexical proxies, not human-judged refusal or harmful-compliance rates.
- Exact abstention and instruction-boundary task success fell from 24/24 to 0/24. The cause cannot be established from aggregate data; formatting mismatch is one possible explanation, not a verified finding.
- Both policies returned no literal canary on the 24 confidentiality probes. Zero observed events is not proof of zero leakage risk.
- Generation reached the token limit on 64.67% and 55.11% of XSTest cases, respectively. This is a substantial limitation of the response-generation protocol.
- The reference and boundary-aware policies were evaluated on the same cases. Reported Wilson intervals in the JSON are descriptive marginal intervals, not paired intervals for the differences.
- These results do not establish semantic refusal quality, real-world confidentiality, or broad safety. No policy is designated a universally safer alternative.
