# Credentials, provider routing, and queue transport review

**Question**

What must a new user configure before a public experiment command can run safely and preserve the requested model roster?

**Short answer**

Use one shared environment loader and one explicit provider plan, correct the Google key name, stop undeclared provider changes, and remove credentials from persistent queue records before publishing the launcher.

- This review inspected source files and public OpenAI authentication documentation on 2026-09-07.
- No experiment, API credential test, queue request, monitor check, job submission, or test suite was run.
- No credential values, live environment values, or queue contents were inspected.
- Queue permissions, account access, endpoint availability, and current funds remain unverified.
- The only written artifact is this report.

## What credentials does the current code require?

| Selected route | Credential variable | Current requirement and source |
| --- | --- | --- |
| Native OpenAI | `OPENAI_API_KEY` | The runner requires this provider's key pool for every selected `api_type=openai` model in its [credential preflight](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398). |
| Native Anthropic | `ANTHROPIC_API_KEY` | The same [preflight](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:404) checks `api_type=anthropic`. |
| Native Google Gemini | `GOOGLE_API_KEY` | The [factory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:95), [key discovery](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:27), and [Google client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:3175) all use this name. |
| OpenRouter | `OPENROUTER_API_KEY` | Every OpenRouter model needs this provider's key pool, including requests sent through the [file queue](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:224). |
| Local cluster models | No provider API key in this path | The factory requires an existing [local model path](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:535). |
| Native xAI, if selected | `XAI_API_KEY` | This separate route uses the [runner preflight](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:402) and a [different file proxy](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:3054). |

- Require the union of providers in the resolved roster, including all per-run model overrides.
  - A provider key is shared by that user's agents and is not required separately for each seat.
  - `gpt-5-nano` currently means native OpenAI in the [model catalog](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265).
  - `gpt-5-nano-high` currently means OpenRouter in the [same catalog](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:275).
  - `llama-3.3-70b-instruct` also means OpenRouter in the [catalog](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:778).
  - A GPT-5-nano baseline with an OpenRouter Llama adversary therefore requires both `OPENAI_API_KEY` and `OPENROUTER_API_KEY` through the current runner.
  - Model availability was not tested, so a catalog entry is not evidence that an account can call it today.
- The credential template does not match the native Google implementation.
  - The [template](/scratch/gpfs/DANQIC/jz4391/bargain/.env.example:5) lists `GEMINI_API_KEY`, which the inspected execution paths do not read.
  - Replace that entry with `GOOGLE_API_KEY` and document which Gemini aliases use OpenRouter instead.
  - The template also lists `PERPLEXITY_API_KEY`, `HUGGING_FACE_TOKEN`, and `WANDB_*`, which the inspected main runner, factory, and provider clients do not use.
  - Label those entries optional only if an identified public workflow actually needs them.
- Key pools support grouped variables through [key discovery](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:73).
  - Grouped names follow `<GROUP>_<PROVIDER_VARIABLE>` with optional suffixes such as `_1` and `_2`.
  - `LLM_KEY_GROUP_ORDER` must name a group before discovery will consider its variables.
  - Discovery uppercases group names, orders numeric suffixes, removes duplicate values, and appends the ordinary provider variable last.
  - Ordinary onboarding should use one key per required provider and leave grouped rotation as an explicit advanced option.
- The current checks establish presence, not usable credentials.
  - [Discovery](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:105) accepts any nonempty value without checking whitespace-only text or known template markers.
  - [OpenRouter construction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:224) only warns about an unexpected key prefix.
  - These checks do not establish authentication, funds, model access, or successful output.

## How are environment files and transports selected?

- The inspected entry points do not share environment-file behavior.
  - The [main runner](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398) reads the process environment and has no dotenv loader.
  - The [TTC wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:156) copies the process environment without reading an environment file.
  - The [multi-agent batch wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1521) reads `/scratch/gpfs/DANQIC/jz4391/bargain/.env` into a copy of the process environment.
  - Its `setdefault` behavior gives existing environment variables priority over file entries.
  - Its handwritten parser splits at `=`, strips outer quotes, and does not implement shell `export`, interpolation, multiline values, or inline-comment semantics.
  - A copied template alone therefore does not configure the main runner or the TTC wrapper.
- Native OpenAI uses `OPENAI_TRANSPORT`, with `auto`, `direct`, and `proxy` accepted in [construction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2686).
  - `auto` selects the file queue when `SLURM_JOB_ID` is present and direct SDK calls otherwise in [_use_file_proxy](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2742).
  - The queue directory is `OPENAI_PROXY_POLL_DIR`, with `/home/jz4391/openrouter_proxy` as the current default.
  - `OPENAI_PROXY_CLIENT_POLL_INTERVAL` defaults to 0.1 seconds and `OPENAI_PROXY_CLIENT_TIMEOUT` defaults to 6000 seconds.
  - Invalid numeric settings silently use those defaults.
- OpenRouter uses a different meaning of `auto` in its [transport selection](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:393).
  - It tries direct HTTPS first, including under Slurm, then uses the file queue on a classified connectivity failure.
  - Once this change happens, later calls by that agent continue through the queue and a warning records the change.
  - `OPENROUTER_PROXY_POLL_DIR` defaults to `/home/jz4391/openrouter_proxy`.
  - `OPENROUTER_PROXY_CLIENT_POLL_INTERVAL`, `OPENROUTER_PROXY_CLIENT_TIMEOUT`, and `OPENROUTER_API_TIMEOUT` default to 0.1, 6000, and 300 seconds in [construction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:245).
  - `OPENROUTER_PROXY_PROBE_TIMEOUT` is parsed, but the inspected request path uses the client timeout without a separate probe.
  - Direct OpenRouter calls use standard HTTP proxy environment settings because [the session enables `trust_env`](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:330).
- The batch wrapper changes the default OpenRouter transport to `proxy` and its client/API timeouts to 9000/1800 seconds in [the subprocess environment](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717).
  - External users can set `OPENROUTER_TRANSPORT=direct` explicitly before that wrapper runs.
  - A portable launcher must not make a new user's direct-network run depend on `/home/jz4391/openrouter_proxy`.
- Anthropic and Google have native direct clients in the inspected implementation.
  - Anthropic creates an [SDK client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2417).
  - Google calls [the Google SDK](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:3228).
  - Neither has the equivalent explicit native file-queue transport in this code.
  - Moving either model to OpenRouter is a provider change and must be declared in the run specification.
- Native OpenAI direct and queue execution differ beyond transport.
  - The installed SDK reads `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, and `OPENAI_PROJECT_ID` in its [async constructor](/scratch/gpfs/DANQIC/jz4391/bargain/.venv/lib/python3.14/site-packages/openai/_client.py:499).
  - The [queue envelope](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2765) hardcodes the native endpoint and includes neither organization nor project headers.
  - The installed Anthropic SDK also reads `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` in its [async constructor](/scratch/gpfs/DANQIC/jz4391/bargain/.venv/lib/python3.14/site-packages/anthropic/_client.py:333).
  - A doctor check must identify these settings by name and validate declared endpoint overrides without printing their raw values.

## Where can provider selection or the roster change?

- Runtime native-to-OpenRouter recovery is enabled by default.
  - `OPENROUTER_PROVIDER_FALLBACK` is defined in [the agent module](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:234), and [the environment parser](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:290) returns true when it is absent.
  - `OPENROUTER_PROVIDER_FALLBACK=0` disables that runtime recovery path.
  - [Route inference](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:315) can strip dates from model identifiers and translate reasoning controls.
  - A dated native model and an undated routed alias must not be presented as an identical historical rerun.
- The factory does not apply that runtime switch to its missing-key recovery.
  - Its [missing-native-key branch](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:188) can create an OpenRouter agent directly.
  - The public runner currently stops missing native credentials in [its preflight](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:419), so direct factory callers and runner callers can behave differently.
  - The factory can also [skip unknown or unavailable models](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:137) and only fails when no agents remain.
  - The experiment then constructs its model map by [zipping created agents with requested names](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:289), which can mislabel later seats after a skipped model.
- Runtime route order is not simply the order requested in `LLM_KEY_GROUP_ORDER`.
  - The [route builder](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:712) explicitly prioritizes `PRIMARY` and `SECONDARY` native keys, then those OpenRouter groups, before remaining keys.
  - Google also explicitly prioritizes `GROUP_A`.
  - An ordinary native key can therefore appear after grouped OpenRouter keys.
- Existing recovery records are incomplete for saved experiment provenance.
  - The [fallback response](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:675) records source provider, destination provider, model identifiers, and the triggering error.
  - Normal [phase saving](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1351) keeps content, selected token fields, and the original agent's model information.
  - The [token extractor](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:898) preserves `openrouter_transport` but does not copy `provider_fallback`, `model_used`, or native OpenAI's generic `transport` field.
  - The [interaction writer](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1103) therefore lacks a complete record of the actual provider used for each call.

## Are credentials kept out of persistent files and logs?

- The queue stores credentials in request files.
  - [OpenRouter headers](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:609) include bearer authorization, and [request serialization](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:546) writes those headers to disk.
  - The [native OpenAI queue path](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2765) does the same.
  - The [monitor archives processed requests](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:236) without removing those headers.
  - These writers do not specify private directory/file modes, check ownership, or reject symbolic links.
  - Actual exposure depends on the filesystem mode and access-control rules, which this review did not inspect.
- The queue monitor trusts each request's URL and headers.
  - [Request processing](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:194) forwards the envelope directly to the [HTTP sender](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:150).
  - No endpoint allowlist or credential-to-provider binding appears in that path.
  - A public queue service needs an explicit trust boundary because a user who can write a request can direct the monitor's network request.
- The name `_safe_message` overstates the current protection.
  - The [function](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:490) only escapes newlines and truncates text.
  - [Failure reports](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:610), [retry warnings](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:780), [fallback metadata](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:683), and [monitor error logs](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:225) can retain provider exception text without credential redaction.
  - Key-label fields avoid directly writing key values, but provider messages can still contain sensitive text.
  - This is an unsafe logging path, not evidence that a live secret was disclosed.
- An older configuration API can also serialize credentials.
  - [The model registry serializer](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:802) serializes provider dataclasses including their `api_key` fields.
  - It also writes [the `default_api_keys` map](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:825).
  - The separate [agent configuration serializer](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/agent_factory.py:118) already uses environment-variable references instead.
  - A new public wrapper should reuse credential references and must not use the registry serializer with populated key fields.
- Git ignores only the standard environment filename in [the current ignore rule](/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:2).
  - Read-only `git check-ignore` confirmed that `/scratch/gpfs/DANQIC/jz4391/bargain/.env.local` and `/scratch/gpfs/DANQIC/jz4391/bargain/.env.production` do not match that rule.
  - Add a pattern for private environment variants and explicitly retain `/scratch/gpfs/DANQIC/jz4391/bargain/.env.example` as the public template.

## What is the smallest useful implementation?

- Add a shared environment loader and use it before model resolution in all public entry points.
  - Let the process environment override one explicitly selected environment file, or the repository environment file when none is selected.
  - Use one declared parser implementation and reject malformed entries without printing their values.
  - Return safe source metadata containing each variable name, its source type, and its source path.
  - Treat credentials as runtime-only objects and exclude them from commands, resolved experiment JSON, reports, and public exports.
- Reuse `StrongModelAgentFactory.resolve_model_config` to build a pure `resolve_required_routes(models, run_config)` API.
  - Return each seat's requested alias, exact provider model identifier, provider, transport, required credential labels, and declared recovery policy.
  - Use this result in both the runner's preflight and agent construction.
  - Require one created agent for each requested seat and fail before a game starts if any seat cannot be constructed.
  - Default provider recovery to disabled in the public interface and apply the same rule in the factory and runtime agent.
  - Preserve any explicitly permitted historical recovery policy in a versioned manifest.
- Add a shared provider runtime configuration with explicit transport and timeout values.
  - Require an absolute queue path when queue transport is selected outside an explicitly selected Della profile.
  - Reject invalid enum/numeric values instead of replacing them with another mode or timeout.
  - Materialize the final values and any endpoint override in a sanitized run record.
  - Native OpenAI queue execution must preserve approved project/organization routing or fail when unsupported settings are present.
- Fix queue credential retention before making queue transport public.
  - As an immediate change, create private files/directories, publish requests atomically, validate ownership, and archive a redacted request record.
  - OpenRouter currently writes [directly to its visible request filename](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:553), unlike native OpenAI's atomic temporary-file publication.
  - For a shared service, use a versioned request envelope with a provider identifier and credential label, and let the monitor resolve the secret locally.
  - Bind each provider identifier to an approved endpoint and reject arbitrary URLs, authentication headers, and symbolic-link request files.
  - Retain request IDs, model IDs, payload hashes, timestamps, usage, errors, and bounded recovery events without retaining authorization values.
  - Preserve existing externally managed monitor operations during the interface migration and reject unsupported envelope versions clearly.
- Add one redaction utility for structured records and exception text.
  - Remove authentication fields and known in-memory credential values before logs or files are written.
  - Apply it to provider reports, retry logs, monitor errors, fallback metadata, and all configuration serializers.
  - Persist actual provider, actual model identifier, transport, credential label, attempt ID, and recovery event with each interaction.

## What should onboarding and doctor do?

- Onboarding should show only the providers needed by the selected preset.
  - Ask users to place their own keys in a private environment file or inject them from a key manager.
  - Do not ask users to paste keys into the CLI, issue reports, experiment JSON, or chat.
  - Use private file permissions and avoid shell tracing when credentials enter the process environment.
  - OpenAI explicitly recommends server-side environment variables or a key-management service and says keys must not be exposed in client code in its [authentication documentation](https://developers.openai.com/api/reference/overview#authentication).
  - Explain that the existing queue path still requires a provider key on the submitting process until a monitor-side credential design is implemented.
- The proposed `bargain doctor <preset>` command should be offline by default.
  - Resolve the exact preset, seeds, roster, seat order, provider overrides, and transport without creating agents or results.
  - Check only required credential names and report `present`, `missing`, `empty`, `template value`, or `ignored grouped key` without showing secret text.
  - Detect `GEMINI_API_KEY` without `GOOGLE_API_KEY` as a setup error for native Google routes.
  - Check selected provider dependencies without importing modules that create output files or start clients.
  - Native Google needs `google-generativeai`, which is currently only a comment in [the requirements file](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:37).
  - Report the installed SDK versions because [the requirements file](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1) gives lower bounds rather than a tested dependency lock.
  - Validate the output parent and queue path, their permissions, available space, and whether client and monitor configuration identify the same shared directory.
  - Inspect directory metadata only and never read another request's credential-bearing contents.
  - Assume the Della monitor is externally managed and do not start it or run a monitor health probe.
  - Check local model assets before suggesting any download and fail if the selected model path is missing.
  - Detect undeclared endpoint overrides and native-provider routes that cannot use the selected restricted-network profile.
  - Distinguish `configuration valid` from `integration tested` in the result.
- Any network validation should be a separate, explicitly selected smoke-test operation.
  - Use the exact selected provider, model, transport, and credentials with a small declared output budget.
  - Test each distinct route in the selected roster and retain sanitized request/response provenance.
  - Do not substitute a cheaper model or alternate provider after failure.
  - Keep smoke-test output separate from research results and leave a failed test marked failed.

## Which tests and compatibility decisions remain?

- Existing [key-rotation tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_provider_key_rotation.py:30), [provider-recovery tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_openrouter_provider_fallback.py:87), and [transport tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_openrouter_transport.py:38) provide useful starting points but were not run in this review.
- Add focused tests for environment precedence, quoted values, malformed files, whitespace-only keys, template values, unused groups, and missing required providers.
- Add tests that the runner and factory resolve identical routes, disabled recovery stays disabled, and one missing local model cannot reduce or relabel the roster.
- Add tests for redaction in nested headers, exception messages, fallback records, serializers, and processed queue archives using test-only credentials.
- Add filesystem tests for private permissions, atomic publication, symbolic-link rejection, unsupported envelope versions, and bounded waiting.
- Add interaction tests that prove actual provider/model/transport metadata survives saving, including a permitted provider change.
- Preserve seeds, sampled rosters, seat assignments, game settings, and TTC reasoning controls when adding credentials or a launcher.
  - A transport change should retain the same provider endpoint and payload contract.
  - An endpoint, provider, model alias, or reasoning-control change must create a distinct resolved run identity.
  - Historical provider changes cannot be reconstructed reliably from the ordinary model alias alone when the saved interaction dropped recovery metadata.
- Decide whether the public release initially supports only direct network execution or also includes a versioned queue service.
- Decide which historical manifests explicitly permit provider recovery and how those runs are labeled in new analysis.
- Real integration validation remains incomplete until an authorized smoke test succeeds on every supported route.
