# Personality and disposition benchmark coverage

## Question

Which established benchmarks measure model personality or disposition, and how many of the 30 bilateral adversary models already have public scores?

## Short answer

- No public benchmark found has exact scores for 10 of the 30 models.
- The best exact coverage is 3 models for *How Different AI Chatbots Behave?*, a battery of behavioral-economics games.
- The best psychometric coverage is 1 model for *Personality Traits in Large Language Models*.
- A family-name match is not an exact score.
  - For example, a score for `GPT-4o` or Claude Opus cannot be assigned to a dated checkpoint without evidence that it is the same endpoint.
- Existing scores therefore cannot support the proposed 30-model regression.
- The practical option is to run one fixed disposition battery on the exact 30 endpoints.

## Coverage rule

- An exact match requires the same dated API snapshot or the same open-model release, size, and instruction variant.
- An undated product name does not count.
- A nearby checkpoint, a different reasoning setting, or another generation in the same family does not count.
- The count refers to public numeric results reported by the benchmark authors, not models on which the code could in principle run.

## Main benchmark audit

| Benchmark or battery | Main construct | Public artifact | Exact matches in 30 | Exact matched models | Use in this paper |
|---|---|---:|---:|---|---|
| How Different AI Chatbots Behave? | Altruism, trust, cooperation, risk, fairness, strategic choice | Paper tables and figures | **3** | `gpt-4o-2024-05-13`; `gpt-4o-mini-2024-07-18`; `claude-3-haiku-20240307` | Best existing revealed-preference source, but too sparse for regression |
| Personality Traits in Large Language Models | Big Five through IPIP-NEO and BFI, plus validity tests | Paper and supplements | **1** | `gpt-4o-mini-2024-07-18` | Best psychometric foundation; rerun it on the roster |
| TRAIT | Big Five and Short Dark Triad, with self-report and scenarios | Dataset, code, paper results | **0** | None | Strong candidate for a new exact-roster run |
| Evaluating Large Language Models with Psychometrics | Personality, emotion, theory of mind, and related psychological tests | Benchmark and paper tables | **0** | None | Broad, but mixes disposition with social capability |
| AI Psychometrics | Psychological profiles from established human inventories | Paper results | **0** | None | Useful methodological reference; existing model coverage is obsolete |
| Personality testing of LLMs: limited temporal stability | Big Five and prosociality across repeated measurements | Paper results | **0** | None | Important reliability warning; not a roster score source |
| Social desirability biases in Big Five personality surveys | Big Five response bias under questionnaire framing | Paper results | **0** | None | Important validity warning; not a clean disposition covariate |
| PersonaLLM | Controlled expression of assigned Big Five personas | Dataset, code, paper results | **0** | None | Measures persona following, not the model's default disposition |
| ValueBench | Values from 44 psychometric inventories and 453 dimensions | Dataset, code, paper results | **0** | None | Broad values battery; scoring depends partly on evaluator-model judgments |
| Value Compass Benchmarks | Multi-framework value evaluation | Benchmark, demo paper, and interface | **0 exact published matches found** | None | Promising framework, but no usable exact-roster score table was found |
| MoralBench | Moral foundations and moral-value understanding | Dataset, code, paper tables | **0** | None | Relevant to moral disposition, but older model set and some tasks test understanding |
| LLM-GLOBE | Nine GLOBE cultural-value dimensions | Paper tables | **0** | None | Useful for cultural orientation; model names are too coarse or old |
| Model-written evaluations | Sycophancy, power seeking, risk aversion, corrigibility, and related behaviors | Public datasets, code, paper results | **0** | None | Strong behavioral constructs; rerun selected datasets on exact endpoints |
| LLM economicus | Inequity aversion, risk and loss aversion, and time preference | Paper results | **0** | None | Good economic-preference battery; exact published coverage is zero |
| Machine Psychology of Cooperation | Cooperation in repeated social dilemmas | Code and paper results | **0** | None | Directly relevant, but published experiments use older GPT-3.5 snapshots |

## What should count as a disposition benchmark?

- Include measures of a stable response tendency under a fixed context.
  - Examples are agreeableness, altruism, trust, risk tolerance, inequity aversion, cooperation, sycophancy, and power seeking.
- Keep capability tests separate.
  - Theory of mind, emotion recognition, social knowledge, and moral-reasoning accuracy can measure what a model understands rather than what it tends to choose.
- Keep persona-control tests separate.
  - PersonaLLM asks whether a model can portray a requested personality.
  - It does not directly measure the model's default behavior.
- Do not treat questionnaire answers as direct evidence of behavior.
  - Several studies find framing effects, social-desirability bias, and weak temporal stability.

## Recommendation

- Run a short exact-roster battery instead of merging published scores.
  - Use TRAIT or the IPIP-NEO/BFI protocol for Big Five traits.
  - Add revealed-preference tasks for altruism, trust, cooperation, risk tolerance, loss aversion, time preference, and inequity aversion.
  - Add a small set of model-written evaluations for sycophancy and power seeking.
- Repeat each prompt with option-order permutations and several seeds.
- Estimate a small number of preregistered disposition factors.
- Regress bilateral payoff on capability and disposition together.
  - Report partial R-squared or cross-validated incremental prediction from disposition after capability.
  - Cluster uncertainty by model and correct for the small 30-model sample.
- Treat this as new measurement work, not a quick join against an existing leaderboard.

## Primary sources

- [How Different AI Chatbots Behave? Benchmarking Large Language Models in Behavioral Economics Games](https://arxiv.org/abs/2412.12362)
- [Personality Traits in Large Language Models](https://arxiv.org/abs/2307.00184)
- [TRAIT: Personality Testset designed for LLMs with Psychometrics](https://aclanthology.org/2025.findings-naacl.180/)
- [TRAIT repository](https://github.com/pull-ups/TRAIT)
- [Evaluating Large Language Models with Psychometrics](https://arxiv.org/abs/2406.17675)
- [AI Psychometrics](https://doi.org/10.1177/17456916231214460)
- [Personality testing of LLMs: limited temporal stability, but highlighted prosociality](https://doi.org/10.1098/rsos.240180)
- [Large language models display human-like social desirability biases in Big Five personality surveys](https://doi.org/10.1093/pnasnexus/pgae533)
- [PersonaLLM](https://aclanthology.org/2024.findings-naacl.229/)
- [ValueBench](https://aclanthology.org/2024.acl-long.111/)
- [ValueBench repository](https://github.com/Value4AI/ValueBench)
- [Value Compass Benchmarks](https://aclanthology.org/2025.acl-demo.64/)
- [MoralBench](https://arxiv.org/abs/2406.04428)
- [LLM-GLOBE](https://arxiv.org/abs/2411.06032)
- [Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)
- [LLM economicus? Mapping the Behavioral Biases of LLMs via Utility Theory](https://arxiv.org/abs/2408.02784)
- [Machine Psychology of Cooperation](https://arxiv.org/abs/2305.07970)

## Limits of this search

- There is no official registry that defines every top disposition benchmark.
- This audit includes the main reusable and frequently cited benchmark families found through title, citation, repository, and model-table searches through 2026-08-16.
- A paper can publish aggregate figures without releasing a machine-readable score table.
- The zero counts mean zero verified exact public scores, not that the benchmark cannot be run on the model.
