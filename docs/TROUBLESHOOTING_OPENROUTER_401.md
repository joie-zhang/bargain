# Troubleshooting OpenRouter 401 "User not found" Errors

## Problem
You're getting non-stop OpenRouter 401 errors with the message "User not found."

## Root Cause
This error typically means:
1. **API key is invalid or expired** - The most common cause
2. **API key doesn't have required permissions** - Less common
3. **OpenRouter account is suspended** - Rare but possible
4. **API key format is incorrect** - Should start with `sk-or-v1-`

## Solution Steps

### Step 1: Test Your API Key
Run the diagnostic script to test your API key:

```bash
python scripts/test_openrouter_key.py
```

Or test with a specific key:
```bash
python scripts/test_openrouter_key.py "your-api-key-here"
```

### Step 2: Verify Your API Key
1. Go to https://openrouter.ai/keys
2. Check if your API key is:
   - ✅ Active (not deleted/revoked)
   - ✅ Has credits/balance
   - ✅ Has the correct permissions

### Step 3: Check Environment Variable
Make sure `OPENROUTER_API_KEY` is set correctly:

```bash
echo $OPENROUTER_API_KEY
```

If it's not set or empty, set it:
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-actual-key-here"
```

### Step 4: Generate a New API Key (if needed)
If your current key is invalid:
1. Go to https://openrouter.ai/keys
2. Delete the old key (if it exists)
3. Generate a new API key
4. Update your environment variable:
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-your-new-key-here"
   ```

### Step 5: Restart Or Check The Proxy Monitor
On Della-PLI, this repo now defaults to the shared proxy monitor first and only
falls back to direct `openrouter.ai` access if the proxy path is unavailable.

If you manage the monitor manually on `della-vis1.princeton.edu`, restart it there:

```bash
tmux ls | grep openrouter_proxy_monitor
tmux attach -t openrouter_proxy_monitor
# or start a fresh session:
tmux new -s openrouter_proxy_monitor
cd /scratch/gpfs/DANQIC/jz4391/bargain
source .venv/bin/activate
python negotiation/openrouter_proxy_monitor.py
```

## Code Changes Made

I've added validation and better error messages:

1. **API Key Validation** - The `OpenRouterAgent` now validates the API key on initialization
2. **Better Error Messages** - More helpful error messages that explain what went wrong
3. **Diagnostic Script** - `scripts/test_openrouter_key.py` to test your API key

## Prevention

To avoid this issue in the future:
- ✅ Always validate API keys before using them
- ✅ Check OpenRouter account status regularly
- ✅ Use the diagnostic script when setting up new environments
- ✅ Keep API keys secure and don't commit them to git

## Still Having Issues?

If the diagnostic script passes but you still get 401 errors:
1. Check if multiple processes are using different API keys
2. Verify the proxy monitor is running on the vis node and watching `/home/jz4391/openrouter_proxy`
3. Check OpenRouter status page for service outages
4. Contact OpenRouter support if account issues persist
