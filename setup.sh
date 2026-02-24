#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[fail]${NC}  $*"; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          zero-rl  —  setup               ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── 1. System dependencies ──────────────────────────────────────────────────
info "Checking system dependencies..."

command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Python 3.11+."
command -v node    >/dev/null 2>&1 || fail "node not found. Install Node.js 18+."
command -v npm     >/dev/null 2>&1 || fail "npm not found."
command -v git     >/dev/null 2>&1 || fail "git not found."

PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
NODE_VER=$(node -v | sed 's/v//')
ok "python ${PYTHON_VER}, node ${NODE_VER}"

# Optional: OpenSCAD (for CAD tool)
if command -v openscad >/dev/null 2>&1; then
    ok "openscad found"
else
    warn "openscad not found — CAD generation will not work."
    warn "Install: brew install openscad  (macOS) / apt install openscad  (Linux)"
fi

# Optional: ripgrep (for doc search)
if command -v rg >/dev/null 2>&1; then
    ok "ripgrep found"
else
    warn "ripgrep (rg) not found — doc search will not work."
    warn "Install: brew install ripgrep  (macOS) / apt install ripgrep  (Linux)"
fi

# ── 2. Python virtual environment ──────────────────────────────────────────
info "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    ok "Created .venv"
else
    ok ".venv already exists"
fi
source .venv/bin/activate

# ── 3. Python dependencies ─────────────────────────────────────────────────
info "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "Python deps installed"

# ── 4. Frontend dependencies ───────────────────────────────────────────────
info "Installing frontend dependencies..."
cd frontend
npm install --silent 2>/dev/null
cd "$ROOT_DIR"
ok "Frontend deps installed"

# ── 5. Clone documentation repos (for doc_lookup tool) ─────────────────────
info "Setting up documentation repos..."
mkdir -p docs

if [ ! -d "docs/gymnasium-repo" ]; then
    info "  Cloning Gymnasium docs..."
    git clone --depth 1 https://github.com/Farama-Foundation/Gymnasium.git docs/gymnasium-repo 2>/dev/null
    ok "  Gymnasium docs cloned"
else
    ok "  Gymnasium docs already present"
fi

if [ ! -d "docs/genesis-doc" ]; then
    info "  Cloning Genesis docs..."
    git clone --depth 1 https://github.com/Genesis-Embodied-AI/genesis-doc.git docs/genesis-doc 2>/dev/null
    ok "  Genesis docs cloned"
else
    ok "  Genesis docs already present"
fi

# ── 6. Environment file ────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    warn "No .env file found. Creating template..."
    cat > .env << 'ENVEOF'
# At least one LLM provider key is required
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=

# Optional: override default provider/model
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-sonnet-4-6
ENVEOF
    warn ".env template created — add your API key(s) before running."
else
    ok ".env file exists"
fi

# ── 7. Create output directories ───────────────────────────────────────────
mkdir -p envs assets
ok "Output directories ready (envs/, assets/)"

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Setup complete!                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Run the app:  ${CYAN}./start.sh${NC}"
echo ""
