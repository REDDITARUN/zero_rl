"""Compare CAD generation quality across LLM providers/models.

Tests Gemini 3.1 Pro vs Claude Opus 4.6 on the same prompts,
measuring: compile success, STL size, token usage, latency.

Usage:
    cd /Users/tarun/Personal/zero_rl
    .venv/bin/python scripts/cad_model_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from core.tools.cad_generate import (
    STRICT_CODE_PROMPT,
    extract_openscad_code,
    parse_parameters,
    _compile_scad,
)
from core.config import ASSETS_DIR, get_api_key
from articulate_anything.utils.prompt_utils import setup_vlm_model


# ── Models to compare ────────────────────────────────────────────────────────

MODELS = [
    {
        "id": "gemini-3.1-pro",
        "provider": "google-genai",
        "model_name": "gemini-3.1-pro-preview",
        "temperature": 1.0,
        "max_tokens": 8192,
    },
    {
        "id": "claude-opus-4.6",
        "provider": "anthropic",
        "model_name": "claude-opus-4-6",
        "temperature": 0.8,
        "max_tokens": 8192,
    },
]

# ── Test prompts ─────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    {
        "name": "porsche_911",
        "description": (
            "A Porsche 911 sports car. Sleek aerodynamic body with iconic round "
            "headlights, sloped rear engine cover, wide rear fenders, spoiler, "
            "side mirrors, 5-spoke wheels. About 450mm long, 180mm wide, 130mm tall."
        ),
    },
    {
        "name": "chess_rook",
        "description": (
            "A chess rook piece. Cylindrical base tapering up, crenellated "
            "battlements on top with 4 notches. About 50mm tall, 25mm base diameter. "
            "Smooth and elegant."
        ),
    },
    {
        "name": "desk_lamp",
        "description": (
            "An adjustable desk lamp with a heavy round base, articulated arm "
            "with two segments and a pivot joint, and a cone-shaped shade. "
            "Base 80mm diameter, total height ~300mm when upright."
        ),
    },
]


@dataclass
class TrialResult:
    model_id: str
    prompt_name: str
    success: bool = False
    compile_time_s: float = 0.0
    llm_time_s: float = 0.0
    total_time_s: float = 0.0
    stl_size_bytes: int = 0
    num_params: int = 0
    scad_lines: int = 0
    error: str = ""
    scad_path: str = ""


def run_trial(model_cfg: dict, prompt: dict) -> TrialResult:
    """Run a single generation trial."""
    model_id = model_cfg["id"]
    name = f"{prompt['name']}_{model_id.replace('.', '_').replace('-', '_')}"
    description = prompt["description"]

    print(f"\n{'='*60}")
    print(f"  Model: {model_id}")
    print(f"  Prompt: {prompt['name']}")
    print(f"{'='*60}")

    asset_dir = ASSETS_DIR / name
    asset_dir.mkdir(parents=True, exist_ok=True)

    result = TrialResult(model_id=model_id, prompt_name=prompt["name"])

    try:
        api_key = get_api_key(model_cfg["provider"])
        vlm = setup_vlm_model(
            model_name=model_cfg["model_name"],
            system_instruction=STRICT_CODE_PROMPT,
            api_key=api_key,
        )

        t0 = time.time()
        response = vlm.generate_content(
            [description],
            generation_config={
                "temperature": model_cfg["temperature"],
                "max_tokens": model_cfg["max_tokens"],
            },
        )
        result.llm_time_s = round(time.time() - t0, 2)

        scad_code = extract_openscad_code(response.text)
        scad_path = asset_dir / f"{name}.scad"
        stl_path = asset_dir / f"{name}.stl"
        scad_path.write_text(scad_code, encoding="utf-8")
        result.scad_path = str(scad_path)
        result.scad_lines = len(scad_code.splitlines())

        params = parse_parameters(scad_code)
        result.num_params = len(params)
        if params:
            (asset_dir / "parameters.json").write_text(
                json.dumps(params, indent=2), encoding="utf-8"
            )

        t1 = time.time()
        ok, err = _compile_scad(scad_path, stl_path)
        result.compile_time_s = round(time.time() - t1, 2)

        if ok:
            result.success = True
            result.stl_size_bytes = stl_path.stat().st_size
        else:
            result.success = False
            result.error = err[:500]

    except Exception as e:
        result.success = False
        result.error = str(e)[:500]

    result.total_time_s = round(result.llm_time_s + result.compile_time_s, 2)
    return result


def print_summary(results: list[TrialResult]) -> None:
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("  COMPARISON RESULTS")
    print("=" * 80)

    header = f"{'Model':<20} {'Prompt':<15} {'OK?':<5} {'LLM(s)':<8} {'Build(s)':<9} {'Total(s)':<9} {'STL(KB)':<9} {'Lines':<6} {'Params':<7}"
    print(header)
    print("-" * len(header))

    for r in results:
        stl_kb = r.stl_size_bytes / 1024 if r.stl_size_bytes else 0
        status = "YES" if r.success else "FAIL"
        print(
            f"{r.model_id:<20} {r.prompt_name:<15} {status:<5} "
            f"{r.llm_time_s:<8.1f} {r.compile_time_s:<9.1f} {r.total_time_s:<9.1f} "
            f"{stl_kb:<9.0f} {r.scad_lines:<6} {r.num_params:<7}"
        )

    if any(not r.success for r in results):
        print("\nFailed trials:")
        for r in results:
            if not r.success:
                print(f"  {r.model_id} / {r.prompt_name}: {r.error[:200]}")

    # Per-model summary
    print("\n" + "-" * 40)
    for model in MODELS:
        mid = model["id"]
        model_results = [r for r in results if r.model_id == mid]
        successes = sum(1 for r in model_results if r.success)
        avg_llm = sum(r.llm_time_s for r in model_results) / len(model_results) if model_results else 0
        avg_total = sum(r.total_time_s for r in model_results) / len(model_results) if model_results else 0
        print(f"{mid}: {successes}/{len(model_results)} success, avg LLM={avg_llm:.1f}s, avg total={avg_total:.1f}s")


def main() -> None:
    results: list[TrialResult] = []

    for prompt in TEST_PROMPTS:
        for model in MODELS:
            r = run_trial(model, prompt)
            results.append(r)

            status = "SUCCESS" if r.success else f"FAIL: {r.error[:100]}"
            print(f"  -> {status} (LLM: {r.llm_time_s}s, compile: {r.compile_time_s}s)")

    print_summary(results)

    out_path = ASSETS_DIR / "model_comparison.json"
    out_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
