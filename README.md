# 0RL — From Words to Worlds

Describe RL environments in natural language — 0RL builds working Gymnasium / Genesis environments with validation, training scripts, and live 3D preview.

```
"Create a 3-DOF robotic arm reaching task"
  → env.py + config.py + train.py + live viewer
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (Next.js)                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Chat UI  │  │ 3D Viewer    │  │ Code Viewer  │                  │
│  │ (prompt- │  │ (R3F/Three)  │  │ (file tabs)  │                  │
│  │  kit)    │  │ STL · URDF   │  │              │                  │
│  └────┬─────┘  └──────┬───────┘  └──────────────┘                  │
│       │               │                                              │
│       │ WebSocket      │ WebSocket (frames)                          │
└───────┼───────────────┼────────────────────────────────────────────┘
        │               │
┌───────┼───────────────┼────────────────────────────────────────────┐
│  Backend (FastAPI)    │                                              │
│       │               │                                              │
│  ┌────▼─────┐   ┌─────▼──────┐                                     │
│  │ /ws/chat │   │ /ws/env/   │                                      │
│  │ handler  │   │ frame      │                                      │
│  └────┬─────┘   │ streamer   │                                      │
│       │         └────────────┘                                      │
│  ┌────▼─────────────────────────────────────────────────────────┐   │
│  │  Orchestrator (ReAct agent)                                   │   │
│  │                                                               │   │
│  │  LLM ──► tool call ──► observe ──► repeat                    │   │
│  │                                                               │   │
│  │  Guardrails:                                                  │   │
│  │    • auto-validate after env.py write                         │   │
│  │    • fix loop (max 3 attempts)                                │   │
│  │    • escalate to eval_agent on failure                        │   │
│  └──┬───────────────────────────────────────────────────────────┘   │
│     │                                                                │
│  ┌──▼──────────────────────────────────────────────────────────┐    │
│  │  Tools                                                       │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐              │    │
│  │  │ file_write │ │ file_read  │ │ file_edit  │              │    │
│  │  │ shell      │ │ dir_list   │ │ code_search│              │    │
│  │  └────────────┘ └────────────┘ └────────────┘              │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐              │    │
│  │  │cad_generate│ │urdf_generate│ │ doc_lookup │              │    │
│  │  │ (OpenSCAD) │ │ (Artic-Any)│ │ (doc_agent)│              │    │
│  │  └────────────┘ └────────────┘ └────────────┘              │    │
│  │  ┌────────────┐                                             │    │
│  │  │  eval_env  │  8-stage validation pipeline                │    │
│  │  └────────────┘                                             │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

## Agent Flow

```
User prompt
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ Orchestrator │────►│ Plan (LLM thinks)│
│ (ReAct loop) │     └────────┬─────────┘
│              │              │
│              │     ┌────────▼─────────┐
│              │     │ Write env.py,    │
│              │     │ config.py, etc.  │
│              │     └────────┬─────────┘
│              │              │
│              │     ┌────────▼─────────┐     ┌──────────────┐
│              │     │ Auto-validate    │────►│ 8-stage      │
│              │     │ (guardrail)      │     │ pipeline:    │
│              │     └────────┬─────────┘     │ syntax →     │
│              │              │               │ import →     │
│              │         pass ╱╲ fail         │ instantiate →│
│              │             ╱  ╲             │ reset →      │
│              │     ┌──────╱    ╲──────┐     │ step →       │
│              │     │ Done │    │ Fix  │     │ render →     │
│              │     └──────┘    │ loop │     │ check_env    │
│              │                 │(≤3x) │     └──────────────┘
│              │                 └──┬───┘
│              │                    │
│              │     ┌──────────────▼──┐
│              │     │ Eval agent      │  (on 3rd failure)
│              │     │ (deep diagnosis)│
│              │     └─────────────────┘
└──────────────┘
```

## Validation Pipeline

Each generated environment passes through 8 stages:

| # | Stage | What it checks |
|---|-------|----------------|
| 1 | **file_check** | `env.py` exists, config files present |
| 2 | **syntax** | Python parses without errors |
| 3 | **import** | Module imports successfully |
| 4 | **instantiate** | `Env()` constructor runs |
| 5 | **reset** | `env.reset()` returns valid obs |
| 6 | **step** | `env.step(action)` runs 10 steps |
| 7 | **render** | `env.render()` produces an RGB frame |
| 8 | **check_env** | Stable-Baselines3 `check_env` passes |

Genesis environments use stages 1-6 + a Genesis-specific render check.

## Tools

| Tool | Input | Output |
|------|-------|--------|
| `cad_generate` | name + description + optional reference image | `.stl` file (via OpenSCAD + BOSL2) |
| `urdf_generate` | name + description + optional reference image | `.urdf` file (via Articulate-Anything) |
| `eval_env` | env_id | 8-stage validation report |
| `doc_lookup` | question | Answer from Gymnasium/Genesis docs |
| `file_write/read/edit` | path + content | File operations in `envs/` |
| `shell` | command | Sandboxed shell execution |
| `code_search` | query | Ripgrep search across project |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, React Three Fiber |
| Backend | FastAPI, WebSockets, uvicorn |
| Agent | LangChain (Anthropic / OpenAI / Google), ReAct pattern |
| RL | Gymnasium, Stable-Baselines3, Pygame |
| Physics | Genesis (GPU-accelerated simulation) |
| CAD | OpenSCAD + BOSL2, Articulate-Anything VLM pipeline |
| 3D Viewer | Three.js (STL + URDF rendering) |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/tarun-grid/zero_rl.git
cd zero_rl

# 2. Setup (creates venv, installs deps, clones docs)
./setup.sh

# 3. Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 4. Run
./start.sh

# Opens at http://127.0.0.1:3000
```

### Manual Setup

```bash
# Python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Docs (for doc_lookup tool)
git clone --depth 1 https://github.com/Farama-Foundation/Gymnasium.git docs/gymnasium-repo
git clone --depth 1 https://github.com/Genesis-Embodied-AI/genesis-doc.git docs/genesis-doc

# Run backend + frontend
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

### System Dependencies

| Dependency | Required | Install |
|-----------|----------|---------|
| Python 3.11+ | Yes | `brew install python` |
| Node.js 18+ | Yes | `brew install node` |
| OpenSCAD | For CAD tool | `brew install openscad` |
| ripgrep | For doc/code search | `brew install ripgrep` |

## Testing

```bash
source .venv/bin/activate

# Test file tools + eval pipeline (no LLM needed)
python tests/test_tools.py --test tools
python tests/test_tools.py --test eval

# Test full orchestrator pipeline (needs API key)
python tests/test_tools.py --test gym

# Test doc lookup (needs API key)
python -c "from core.tools.doc_tool import doc_lookup; print(doc_lookup.invoke({'question': 'How to define observation_space in Gymnasium?'}))"

# Test CAD generation (needs API key + OpenSCAD)
python -c "from core.tools.cad_generate import cad_generate; print(cad_generate.invoke({'name': 'test_cube', 'description': 'A simple cube with rounded edges, 50mm'}))"
```

## Project Structure

```
zero_rl/
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── routes/
│   │   ├── chat.py             # WebSocket chat handler
│   │   ├── envs.py             # Environment CRUD API
│   │   ├── assets.py           # Asset browser API
│   │   └── settings.py         # LLM provider settings
│   └── services/
│       ├── runner.py           # Env execution + frame streaming
│       └── run_env_subprocess.py  # Genesis subprocess runner
├── core/
│   ├── orchestrator.py         # ReAct agent with guardrails
│   ├── config.py               # Project config + model selection
│   ├── agents/
│   │   ├── doc_agent.py        # Documentation search agent
│   │   └── eval_agent.py       # Validation diagnosis agent
│   ├── prompts/
│   │   ├── orchestrator.py     # Main system prompt
│   │   ├── doc_agent.py        # Doc agent prompt
│   │   ├── eval_agent.py       # Eval agent prompt
│   │   └── skills/             # Reference env implementations
│   │       ├── genesis/        # Genesis env templates
│   │       └── gym/            # Gymnasium env templates
│   └── tools/
│       ├── cad_generate.py     # Text/image → OpenSCAD → STL
│       ├── urdf_generate.py    # Text/image → URDF XML
│       ├── eval_tool.py        # 8-stage validation pipeline
│       ├── doc_tool.py         # Doc lookup wrapper
│       ├── coding.py           # File ops + shell tools
│       └── openscad_libs/      # BOSL2 library for OpenSCAD
├── frontend/
│   └── src/
│       ├── app/page.tsx        # Main chat + viewer UI
│       ├── hooks/              # useChat, useEnvs, useAssets
│       ├── components/
│       │   ├── viewer/         # 3D viewers (STL, URDF, env frames)
│       │   ├── progress/       # Build progress visualizer
│       │   └── prompt-kit/     # Chat input components
│       └── lib/                # WebSocket manager, utilities
├── tests/
│   └── test_tools.py           # Tool + pipeline tests
├── scripts/
│   └── cad_model_comparison.py # LLM model comparison for CAD
├── setup.sh                    # One-command setup
├── start.sh                    # Launch frontend + backend
├── requirements.txt            # Python dependencies
└── .env                        # API keys (not committed)
```

## LLM Configuration

Configure via the settings panel in the UI, or set environment variables:

```bash
# Provider: anthropic | openai | google-genai
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6

# Per-tool overrides (optional)
CAD_PROVIDER=google-genai
CAD_MODEL=gemini-2.5-flash-preview-04-17
```

Supported models:
- **Anthropic**: claude-opus-4-6, claude-sonnet-4-6
- **OpenAI**: gpt-4o, gpt-4o-mini, o1-mini
- **Google**: gemini-2.5-flash-preview-04-17, gemini-2.0-flash

## References

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) — RL environment standard
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — GPU-accelerated physics simulation
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — RL training algorithms
- [Articulate-Anything](https://github.com/vlongle/articulate-anything) — VLM-driven URDF generation
- [CADAM](https://github.com/Adam-CAD/CADAM) — Text-to-CAD with OpenSCAD (methodology ported)
- [BOSL2](https://github.com/BelfrySCAD/BOSL2) — OpenSCAD parametric library
- [LangChain](https://github.com/langchain-ai/langchain) — LLM agent framework

## Citation

If you use this work in academic research, publications, blog posts, videos, or any derived projects, please cite:

```
@misc{reddi2025zero_rl,
  author       = {Tarun Reddi},
  title        = {0RL — From Words to Worlds},
  year         = {2025-2026},
  url          = {https://github.com/REDDITARUN/zero_rl}
}
```

Or in plain text:

> Tarun Reddi, "0RL — From Words to Worlds", 2025-2026. https://github.com/REDDITARUN/zero_rl

## License

MIT — see [LICENSE](LICENSE) for details. Citation is required for any published or public-facing use.
