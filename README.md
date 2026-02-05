# ZeroRL

ZeroRL is an AI-powered RL environment studio for the OpenAI Codex Hackathon.

## Stack
- Frontend: Next.js 14 + TypeScript + Tailwind
- Backend: FastAPI + WebSockets
- RL: Gymnasium + Pygame + Stable-Baselines3
- Agent runtime: Codex SDK bridge (`@openai/codex-sdk`)

## Important Runtime Rules
- Backend is **Python 3.11 only**.
- Codex generation is **mandatory** now (no template fallback).
- Unsaved environments live in backend memory until you press **Save To Gallery**.

## Environment Setup
1. Backend env file
```bash
cp /Users/tarun/zero_rl/zerorl/backend/.env.example /Users/tarun/zero_rl/zerorl/backend/.env
```
Set at least one key in `/Users/tarun/zero_rl/zerorl/backend/.env`:
- `OPENAI_API_KEY=...` or `CODEX_API_KEY=...`

2. Frontend env file
```bash
cp /Users/tarun/zero_rl/zerorl/frontend/.env.local.example /Users/tarun/zero_rl/zerorl/frontend/.env.local
```

## How Codex SDK Is Wired
- Orchestrator: `/Users/tarun/zero_rl/zerorl/backend/orchestrator.py`
- Python bridge client: `/Users/tarun/zero_rl/zerorl/backend/codex_client.py`
- Official SDK runner: `/Users/tarun/zero_rl/zerorl/backend/codex_bridge/run-agent.mjs`

Flow:
1. FastAPI `/api/chat` triggers orchestrator.
2. Orchestrator runs agents in parallel (Architect/Rewards/Spaces), then validation fix loop, then Docs/Trainer.
3. Each agent request goes through official Codex SDK with structured output schema.
4. Thread IDs are reused per environment so follow-up prompts modify previous env context.

## Run Locally
```bash
conda create -n zerorl python=3.11 -y
conda activate zerorl

cd /Users/tarun/zero_rl/zerorl/backend
pip install -r requirements.txt

cd /Users/tarun/zero_rl/zerorl/backend/codex_bridge
npm install

cd /Users/tarun/zero_rl/zerorl/backend
python -m uvicorn main:app --reload --port 8000
```

```bash
cd /Users/tarun/zero_rl/zerorl/frontend
npm install
npm run dev
```

## prompt-kit Components
`@prompt-kit/core` is not published as an npm package. Use shadcn registry installs from prompt-kit:

```bash
cd /Users/tarun/zero_rl/zerorl/frontend
npx shadcn@latest add "https://prompt-kit.com/c/prompt-input.json"
npx shadcn@latest add "https://prompt-kit.com/c/message.json"
npx shadcn@latest add "https://prompt-kit.com/c/markdown.json"
npx shadcn@latest add "https://prompt-kit.com/c/loader.json"
```

## Core UX Features
- **Env control tab**: step/reset actions update frame, reward, and action history live.
- **Dynamic tabs**: action/obs/reward panels update from generated environment metadata and runtime state.
- **Save workflow**: generated env stays in memory until `Save To Gallery`.
- **Gallery retrieval**: click saved env to reload and continue modifying via chat.
- **Training tab**: configurable algorithm + hyperparameters (`PPO/DQN/A2C`, timesteps, lr, gamma, epsilon, n_steps, batch_size).
- **Eval tab**: live rollout stream with action trace and rendered frames.

## API Surface
- `POST /api/chat`
- `GET /api/envs`
- `GET /api/envs/{id}`
- `POST /api/envs/{id}/save`
- `POST /api/envs/{id}/reset`
- `POST /api/envs/{id}/step`
- `GET /api/envs/{id}/state`
- `POST /api/train/{id}`
- `POST /api/eval/{id}`
- `GET /api/envs/{id}/download`
- WebSockets: `/ws/agents`, `/ws/train`, `/ws/eval`

## Demo Prompt
`Create a maze where a robot finds the exit`

## Troubleshooting
- `pygame` build error with `pkg-config not found`:
  - fixed by pinning `pygame==2.6.1` (wheel install) and running in Python 3.11.
- `No supported WebSocket library detected`:
  - run `pip install -r requirements.txt` to completion; previously this failed early at pygame.
- `ENOENT ... .next/server/middleware-manifest.json`:
  - stop frontend and run:
  - `rm -rf /Users/tarun/zero_rl/zerorl/frontend/.next`
  - `cd /Users/tarun/zero_rl/zerorl/frontend && npm run dev`
- Codex generation errors:
  - ensure `OPENAI_API_KEY` or `CODEX_API_KEY` exists in `/Users/tarun/zero_rl/zerorl/backend/.env`.
  - for slow responses, increase:
  - `CODEX_TIMEOUT_SEC=480` (or higher)
  - `CODEX_MAX_RETRIES=2` (or 3)
  - backend now uses streamed NDJSON bridge parsing to avoid invalid envelope failures.
