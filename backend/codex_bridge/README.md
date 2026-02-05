# ZeroRL Codex Bridge

This directory contains a tiny Node wrapper around the official `@openai/codex-sdk`.

- Entry point: `run-agent.mjs`
- Called from Python: `/Users/tarun/zero_rl/zerorl/backend/codex_client.py`

## Install
```bash
cd /Users/tarun/zero_rl/zerorl/backend/codex_bridge
npm install
```

## Required env
Set one of:
- `CODEX_API_KEY`
- `OPENAI_API_KEY`
