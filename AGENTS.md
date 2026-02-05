# ZeroRL - Codex Agent Instructions

## Project Context
ZeroRL is an AI-powered RL environment studio. Users describe environments in natural language and the system generates working Gymnasium environments with Pygame rendering, training scripts, and documentation.

## Technology Stack
- Frontend: Next.js 14 (App Router), TypeScript, Tailwind CSS, prompt-kit style chat UI
- Backend: FastAPI, Python 3.11+
- RL: Gymnasium, Stable-Baselines3, Pygame
- Agent orchestration: Parallel skill-driven generation + validator loop

## Code Quality Rules
1. Python code must include type hints.
2. Public functions/classes need docstrings.
3. FastAPI handlers should use async.
4. TypeScript components/hooks must be typed.

## Environment Rules
1. Generated env classes must inherit from `gymnasium.Env`.
2. Required methods: `__init__`, `reset`, `step`, `render`, `close`.
3. Rendering uses Pygame for 2D scenes.
4. Environments should pass `stable_baselines3.common.env_checker.check_env`.
5. Keep scope demo-friendly (2D grid/navigation tasks).

## File Conventions
- Generated env files: `/envs/{env_id}/`
- Expected files: `env.py`, `train.py`, `config.json`, `README.md`
- `env_id` should be UUID-based.

## Validation Pipeline
Each generated env must:
1. Parse (syntax check)
2. Import
3. Instantiate
4. Pass `check_env`
5. Run random rollout for 10 steps
6. Render one RGB frame

If validation fails:
1. Parse error
2. Apply targeted fix
3. Re-run validation
4. Stop after 3 fix attempts and surface error

## Parallel Agent Workflow
- Phase 1 (parallel): `env-architect`, `reward-engineer`, `space-designer`
- Phase 2 (serial): `code-validator` with fix loop
- Phase 3 (parallel): `docs-generator`, `trainer-config`

## UI Expectations
- Left pane: chat + live agent status
- Right pane: tabs for env/actions/obs/rewards/train/code/gallery
- Real-time status via WebSockets
- Real-time training updates via WebSockets
