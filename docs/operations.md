# Experiment Operations

## Scope

This guide explains credentials, provider transport, key rotation, and logs. Do not put secret values in the repository.

## Set One Key For Each Provider

Use the standard environment variables for simple runs:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export XAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

Many Slurm scripts read a key file. Set `BARGAIN_API_KEYS_ENV` when the file is not at the default path.

```bash
export BARGAIN_API_KEYS_ENV=/path/to/api_keys.env
```

Set file permissions to `600`. Never commit the key file.

## Set Key Groups

The provider layer can use ordered key groups. Set the group order first.

```bash
export LLM_KEY_GROUP_ORDER=PRIMARY,SECONDARY
```

Then, set one or more provider keys in each group:

```bash
export PRIMARY_OPENAI_API_KEY_1="..."
export PRIMARY_OPENROUTER_API_KEY_1="..."
export SECONDARY_OPENAI_API_KEY_1="..."
export SECONDARY_OPENROUTER_API_KEY_1="..."
```

The provider layer uses the groups in the specified order. It uses the standard provider variable as the last alternative.

The code rotates keys after a key-specific failure. It retries a transient provider failure with the same key.

Failure reports contain environment-variable names. They do not contain secret values.

## Select OpenRouter Transport

`OPENROUTER_TRANSPORT` accepts these values:

- `auto` uses the proxy in a Slurm job. It uses direct HTTPS outside Slurm.
- `proxy` always uses the file queue.
- `direct` always uses direct HTTPS.

The default value is `auto`.

Use the proxy on a compute node that has no internet access:

```bash
export OPENROUTER_TRANSPORT=proxy
export OPENROUTER_PROXY_POLL_DIR=/home/jz4391/openrouter_proxy
```

The compute job writes a request file to the queue. `negotiation/openrouter_proxy_monitor.py` reads the request on a networked node.

The monitor writes the response to the same queue. The compute job then reads the response.

The cluster workflow assumes that the monitor is already active. Do not start another monitor for each Slurm job.

## Diagnose OpenRouter Errors

For a `401` error, do these steps:

1. Confirm that `OPENROUTER_API_KEY` has a value.
2. Confirm that the value starts with `sk-or-v1-`.
3. Run a direct API test on a node that has internet access.
4. Replace the key when the direct test also gives a `401` error.

For a proxy timeout, do these steps:

1. Confirm that `OPENROUTER_PROXY_POLL_DIR` is the shared queue path.
2. Confirm that the compute job can write to the queue.
3. Confirm that the monitor uses the same queue path.
4. Examine the proxy response and processed directories.

Do not use direct HTTPS as a compute-node fix. Compute nodes usually do not have internet access.

## Limit Provider Load

Use Slurm array limits to control concurrent API calls.

```bash
./experiments/results/scaling_experiment/configs/slurm/submit_all.sh all \
  --max-concurrent 10
```

Start with a small limit. Increase the limit only after a successful test.

The provider layer retries transient failures with exponential delays. It also writes a report when the retry budget ends.

## Examine Logs

Cluster logs usually exist below `logs/cluster/` or in a result root. Use the log utility from the repository root.

```bash
source scripts/log_utils.sh
latest_log
latest_log err
recent_logs 10
tail_latest err 100
follow_latest out
```

You can also run the utility as a command:

```bash
scripts/log_utils.sh --latest err
scripts/log_utils.sh --recent 20
scripts/log_utils.sh --tail err 100
```

Use `squeue -u "$USER"` to examine Slurm state. Search error logs for `429`, `RateLimit`, `401`, and `Traceback`.

## Examine Provider Reports

The default provider report is:

```text
experiments/results/provider_failures.md
```

A batch can set a report path in its environment. Batch reports usually exist in the batch `monitoring/` directory.

The report gives the provider, model, failure class, key label, and event count.
