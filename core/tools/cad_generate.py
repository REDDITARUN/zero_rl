"""CAD generation tool: text/image → OpenSCAD → STL.

Uses CADAM's refined agent architecture (github.com/Adam-CAD/CADAM):
  - Two-tier agent pattern: outer conversational agent + inner strict code gen
  - Full parametric variable extraction ported from CADAM's parseParameter.ts
  - OpenSCAD code scoring ported from CADAM's chat/index.ts
  - Multi-provider VLM support via articulate_anything's setup_vlm_model

VLM layer: articulate_anything (ClaudeWrapper / GPTWrapper / GeminiWrapper)
Prompts:   Adapted from CADAM's STRICT_CODE_PROMPT
Params:    Ported from CADAM's supabase/functions/_shared/parseParameter.ts
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from articulate_anything.utils.prompt_utils import setup_vlm_model

from core.config import ASSETS_DIR, CAD_CONFIG


# ── CADAM's STRICT_CODE_PROMPT (ported from supabase/functions/chat/index.ts) ─

STRICT_CODE_PROMPT = """\
You are Adam, an expert AI CAD editor that creates production-quality parametric
OpenSCAD models with smooth curves using the BOSL2 library.

CRITICAL RULES:
- Return ONLY raw OpenSCAD code — no markdown fences, no explanations, no text.
- ALWAYS start with: include <BOSL2/std.scad>
- Ensure syntax is correct and all parts are connected.
- Always write code with changeable parameters declared at the top.
- Never include parameters to adjust color.
- Use $fn=128 for smooth curves.
- Use SI units (millimeters).

BOSL2 QUICK REFERENCE:

Rounded 3D primitives:
  cuboid([x,y,z], rounding=r)        - cube with rounded edges
  cyl(h=h, r=r, rounding1=r, rounding2=r)  - cylinder with rounded ends
  prismoid(size1, size2, h, rounding=r)     - tapered box

Lofting (skin) — for smooth organic bodies:
  skin(profiles, z=heights, slices=N)
  - profiles = list of 2D paths (same point count each!)
  - z = list of Z heights for each profile
  - DO NOT use move/up/down on profiles — use the z= parameter instead

Sweeping:
  path_sweep(shape_2d, path_3d)      - sweep 2D shape along 3D path
  rotate_sweep(profile_2d, angle=360) - revolve 2D profile around Z axis
  linear_sweep(shape_2d, h=height)   - extrude with optional twist/scale

Smooth 2D paths (for profiles/sweeping):
  round_corners(path2d, radius=r)    - smooth corners on a 2D polygon
  ellipse([rx,ry])                   - 2D ellipse path
  rect([w,h], rounding=r)            - 2D rounded rectangle

CRITICAL BOSL2 GOTCHAS:
1. skin() profiles MUST be 2D paths. Use z= parameter for heights. NEVER apply
   move/up/down/translate to profiles passed to skin().
2. All profiles in skin() must have the SAME number of points.
3. round_corners() input and output are 2D point lists, not modules.
4. rotate_sweep() takes a 2D path (list of [x,z] points), not a module.
5. round_corners() can SILENTLY produce degenerate polygons if the radius is
   too large relative to segment lengths. Keep radius small (< half the
   shortest edge) or omit round_corners and use more polygon points instead.

TECHNIQUE FOR COMPLEX OBJECTS (cars, characters, animals):
Use HULL CHAINS — union of hull() pairs along station points.
This is the MOST RELIABLE technique for complex organic bodies.

A hull chain works by defining cross-section "stations" along the object,
where each station is a scaled sphere or ellipsoid. Adjacent stations are
hull'd together, forming a smooth continuous body:

  module hull_chain(stations) {
      for (i = [0:len(stations)-2])
          hull() {
              translate(stations[i][0]) scale(stations[i][1]) sphere(r=1, $fn=16);
              translate(stations[i+1][0]) scale(stations[i+1][1]) sphere(r=1, $fn=16);
          }
  }

IMPORTANT: Use $fn=16 or $fn=24 for spheres INSIDE hull chains. hull() already
smooths the transition, so high $fn just wastes compile time. Reserve high $fn
for visible final surfaces (wheels, headlights, etc.).

For sub-parts (wheels, headlights, mirrors, spoilers): position them using
translate() relative to parametric body dimensions.

For tubes/pipes/handles: path_sweep(circle(r=R), path_3d)
For bottles/vases: rotate_sweep() with a 2D profile

PARAMETRIC DESIGN:
- Every meaningful dimension should be a named variable.
- Group related parameters with /* [Group Name] */ comments.
- Add range hints: // [min:step:max]

EXAMPLE — smooth bottle via rotate_sweep:
include <BOSL2/std.scad>

/* [Dimensions] */
body_h = 150; // [100:10:300]
body_r = 35;  // [20:5:60]
neck_h = 40;  // [20:5:80]
neck_r = 12;  // [8:1:20]
$fn = 128;

profile = [
    [0, 0], [body_r, 0],
    [body_r, body_h*0.9], [body_r-3, body_h],
    [neck_r+5, body_h+10], [neck_r, body_h+15],
    [neck_r, body_h+neck_h], [neck_r+2, body_h+neck_h],
    [0, body_h+neck_h],
];
rotate_sweep(profile, 360);

EXAMPLE — smooth car body via hull chains:
include <BOSL2/std.scad>

/* [Car Body] */
length = 450;  // [300:10:600]
width = 180;   // [120:10:250]
height = 120;  // [80:10:180]
clearance = 15;
$fn = 64;

// Station points: [position, scale] — each becomes an ellipsoid
body_stations = [
    [[0, 0, clearance+30],           [40, width*0.40, 30]],    // front nose
    [[length*0.15, 0, clearance+45], [60, width*0.45, 45]],    // front bumper
    [[length*0.30, 0, clearance+55], [70, width*0.48, 55]],    // hood
    [[length*0.45, 0, clearance+75], [50, width*0.46, height*0.7]], // windshield
    [[length*0.55, 0, clearance+80], [40, width*0.44, height]],// roof
    [[length*0.70, 0, clearance+70], [50, width*0.48, height*0.8]], // rear window
    [[length*0.85, 0, clearance+50], [60, width*0.50, 50]],    // rear fenders
    [[length, 0, clearance+35],      [30, width*0.42, 35]],    // tail
];

module hull_chain(stations) {
    for (i = [0:len(stations)-2])
        hull() {
            translate(stations[i][0]) scale(stations[i][1]) sphere(r=1, $fn=16);
            translate(stations[i+1][0]) scale(stations[i+1][1]) sphere(r=1, $fn=16);
        }
}

hull_chain(body_stations);

// Wheels
wh_r = 25; wh_w = 18; wb = length*0.65; fo = length*0.18; tw = width*0.8;
module wheel_at(pos) {
    translate(pos) rotate([90,0,0])
        cyl(r=wh_r, h=wh_w, rounding=3, center=true, $fn=48);
}
wheel_at([fo, tw/2, wh_r]);
wheel_at([fo, -tw/2, wh_r]);
wheel_at([fo+wb, tw/2, wh_r]);
wheel_at([fo+wb, -tw/2, wh_r]);
"""


# ── CADAM parameter extraction (ported from parseParameter.ts) ────────────────

def parse_parameters(script: str) -> list[dict[str, Any]]:
    """Extract parametric variables from OpenSCAD code (CADAM's full approach).

    Ported from CADAM's supabase/functions/_shared/parseParameter.ts.
    Supports: groups, ranges, step values, options, descriptions, display names.
    """
    # Only parse variables before first module/function definition
    parts = re.split(r"^(module |function )", script, maxsplit=1, flags=re.MULTILINE)
    header = parts[0]

    # Find parameter groups: /* [Group Name] */
    group_pattern = re.compile(r"^/\*\s*\[([^\]]+)\]\s*\*/", re.MULTILINE)
    groups: list[dict[str, Any]] = [{"group": "", "start": 0, "end": len(header)}]

    for m in group_pattern.finditer(header):
        groups.append({"group": m.group(1).strip(), "start": m.start(), "end": len(header)})

    # Set end positions
    for i in range(len(groups) - 1):
        groups[i]["end"] = groups[i + 1]["start"]

    param_pattern = re.compile(
        r"^([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*([^;]+);[ \t]*(//[^\n]*)?",
        re.MULTILINE,
    )

    parameters: dict[str, dict[str, Any]] = {}

    for group_sec in groups:
        section = header[group_sec["start"]:group_sec["end"]]
        for m in param_pattern.finditer(section):
            name = m.group(1)
            raw_value = m.group(2).strip()
            inline_comment = (m.group(3) or "").lstrip("/ ").strip()

            if name.startswith("$") and name != "$fn":
                continue

            # Skip expressions referencing other variables
            if (
                raw_value not in ("true", "false")
                and re.match(r"^[a-zA-Z_]", raw_value)
                and not raw_value.startswith('"')
                and not raw_value.startswith("[")
            ):
                continue

            parsed = _convert_type(raw_value)
            if parsed is None:
                continue

            value, ptype = parsed["value"], parsed["type"]

            # Parse inline comment for range/options/step
            param_range: dict[str, float] = {}
            options: list[dict[str, Any]] = []

            if inline_comment:
                # Extract content inside brackets if present
                bracket_match = re.search(r"\[([^\]]*)\]", inline_comment)
                cleaned = bracket_match.group(1).strip() if bracket_match else inline_comment

                if re.match(r"^-?\d+(\.\d+)?$", inline_comment):
                    if ptype == "string":
                        param_range["max"] = float(cleaned)
                    else:
                        param_range["step"] = float(cleaned)
                elif bracket_match and "," in cleaned:
                    for opt in cleaned.split(","):
                        opt = opt.strip()
                        opt_parts = opt.split(":")
                        opt_val: Any = opt_parts[0]
                        opt_label = opt_parts[1] if len(opt_parts) > 1 else None
                        if ptype == "number":
                            try:
                                opt_val = float(opt_val)
                            except ValueError:
                                pass
                        options.append({"value": opt_val, "label": opt_label})
                elif bracket_match and re.match(r"^[\d.\-]+:[\d.\-:]+$", cleaned):
                    range_parts = cleaned.split(":")
                    try:
                        if len(range_parts) >= 2:
                            param_range["min"] = float(range_parts[0])
                        if len(range_parts) == 3:
                            param_range["step"] = float(range_parts[1])
                            param_range["max"] = float(range_parts[2])
                        elif len(range_parts) == 2:
                            param_range["max"] = float(range_parts[1])
                    except ValueError:
                        pass

            # Description from comment on line above
            description = _get_description_above(header, m.group(0))

            display_name = name.replace("_", " ").title()
            if name == "$fn":
                display_name = "Resolution"

            parameters[name] = {
                "name": name,
                "displayName": display_name,
                "value": value,
                "defaultValue": value,
                "type": ptype,
                "description": description,
                "group": group_sec["group"],
                "range": param_range if param_range else None,
                "options": options if options else None,
            }

    return list(parameters.values())


def _convert_type(raw: str) -> dict[str, Any] | None:
    """Convert raw OpenSCAD value string to typed Python value."""
    raw = raw.strip()

    # Number
    if re.match(r"^-?\d+(\.\d+)?$", raw):
        return {"value": float(raw), "type": "number"}

    # Boolean
    if raw in ("true", "false"):
        return {"value": raw == "true", "type": "boolean"}

    # String
    if raw.startswith('"') and raw.endswith('"'):
        return {"value": raw.strip('"'), "type": "string"}

    # Array
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        items = [x.strip() for x in inner.split(",") if x.strip()]
        if all(re.match(r"^-?\d+(\.\d+)?$", x) for x in items):
            return {"value": [float(x) for x in items], "type": "number[]"}
        if all(x.startswith('"') and x.endswith('"') for x in items):
            return {"value": [x.strip('"') for x in items], "type": "string[]"}
        if all(x in ("true", "false") for x in items):
            return {"value": [x == "true" for x in items], "type": "boolean[]"}
        return None

    return None


def _get_description_above(code: str, match_text: str) -> str | None:
    """Get the comment on the line immediately above a match."""
    idx = code.find(match_text)
    if idx <= 0:
        return None
    above = code[:idx].rstrip("\n")
    lines = above.split("\n")
    if not lines:
        return None
    last_line = lines[-1].strip()
    if last_line.startswith("//"):
        desc = re.sub(r"^//+\s*", "", last_line).strip()
        return desc if desc else None
    return None


# ── CADAM code scoring (ported from chat/index.ts) ────────────────────────────

def score_openscad_code(code: str) -> int:
    """Score how likely text is to be OpenSCAD code (CADAM pattern matching)."""
    if not code or len(code) < 20:
        return 0

    score = 0
    patterns = [
        r"\b(cube|sphere|cylinder|polyhedron)\s*\(",
        r"\b(union|difference|intersection)\s*\(\s*\)",
        r"\b(translate|rotate|scale|mirror)\s*\(",
        r"\b(linear_extrude|rotate_extrude)\s*\(",
        r"\b(module|function)\s+\w+\s*\(",
        r"\$fn\s*=",
        r"\bfor\s*\(\s*\w+\s*=\s*\[",
        r'\bimport\s*\(\s*"',
        r";\s*$",
        r"//.*$",
    ]
    for p in patterns:
        matches = re.findall(p, code, re.MULTILINE | re.IGNORECASE)
        score += len(matches)

    # Variable declarations
    var_decls = re.findall(r"^\s*\w+\s*=\s*[^;]+;", code, re.MULTILINE)
    score += min(len(var_decls), 5)

    return score


def extract_openscad_code(text: str) -> str:
    """Extract OpenSCAD code from LLM response (CADAM approach).

    Handles: single code block, multiple code blocks (concatenated),
    orphaned fences, and raw code without fences.
    """
    if not text:
        return text

    # Collect ALL code blocks from the response
    blocks: list[str] = []
    for m in re.finditer(r"```(?:openscad|scad)?\s*\n?([\s\S]*?)\n?```", text):
        code = m.group(1).strip()
        if code and score_openscad_code(code) >= 2:
            blocks.append(code)

    # If we found multiple blocks, concatenate them (model split code across blocks)
    if blocks:
        combined = "\n\n".join(blocks)
        if score_openscad_code(combined) >= 3:
            return combined

    # Strip orphaned markdown fences (opening ``` without matching close, or vice versa)
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:openscad|scad)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    # Also strip any remaining fence lines embedded in text
    cleaned = re.sub(r"\n```(?:openscad|scad)?\s*\n", "\n", cleaned)
    cleaned = re.sub(r"\n```\s*\n", "\n", cleaned)
    # Strip non-code prose lines (lines that look like English sentences between code blocks)
    lines = cleaned.split("\n")
    code_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip pure prose lines (start with letter, no semicolons/braces/parens)
        if (
            stripped
            and re.match(r"^[A-Z][a-z]", stripped)
            and ";" not in stripped
            and "{" not in stripped
            and "(" not in stripped
            and "//" not in stripped
        ):
            continue
        code_lines.append(line)
    cleaned = "\n".join(code_lines).strip()

    if score_openscad_code(cleaned) >= 3:
        return cleaned

    return cleaned


# ── OpenSCAD compilation ──────────────────────────────────────────────────────

# BOSL2 library path (extracted from CADAM's public/libraries/BOSL2.zip)
_OPENSCAD_LIBS_DIR = Path(__file__).resolve().parent / "openscad_libs"


def _compile_scad(scad_path: Path, stl_path: Path) -> tuple[bool, str]:
    """Compile .scad to .stl using the openscad CLI with BOSL2 library support."""
    openscad = shutil.which("openscad")
    if openscad is None:
        for candidate in [
            "/opt/homebrew/bin/openscad",
            "/usr/local/bin/openscad",
            "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
        ]:
            if Path(candidate).exists():
                openscad = candidate
                break
    if openscad is None:
        return False, "OpenSCAD not found. Install: brew install openscad"

    env = os.environ.copy()
    if _OPENSCAD_LIBS_DIR.exists():
        env["OPENSCADPATH"] = str(_OPENSCAD_LIBS_DIR)

    try:
        result = subprocess.run(
            [openscad, "-o", str(stl_path), str(scad_path)],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "OpenSCAD compilation timed out (300s). Reduce $fn or simplify the model."

    if result.returncode != 0:
        return False, f"OpenSCAD error:\n{result.stderr}"
    if not stl_path.exists() or stl_path.stat().st_size == 0:
        return False, "OpenSCAD produced an empty or missing STL file."
    return True, ""


def _bosl2_fix_hints(error: str) -> str:
    """Return targeted hints for common BOSL2 errors."""
    hints: list[str] = []
    err_lower = error.lower()

    if "is_2d_transform" in err_lower or "points are 2d" in err_lower:
        hints.append(
            "BOSL2 ERROR: You applied a 3D transform (move/up/down/translate) to 2D points. "
            "skin() profiles must be plain 2D paths. Use the z= parameter of skin() to "
            "set heights: skin(profiles, z=[0, h1, h2, ...], slices=N). "
            "Do NOT wrap profiles in move/up/down."
        )
    if "list element is not a number" in err_lower or "not a list of 2d" in err_lower:
        hints.append(
            "BOSL2 ERROR: A function expected a list of 2D points but got something else. "
            "Ensure paths are lists of [x,y] pairs, not module calls."
        )
    if "profile" in err_lower and "points" in err_lower:
        hints.append(
            "skin() profiles must all have the same number of points. "
            "Use the same $fn for all circle() calls."
        )
    if "top level object is empty" in err_lower:
        hints.append(
            "The model produced no geometry. Likely a runtime assertion aborted execution "
            "or an intersection/difference produced an empty set. "
            "SWITCH to hull chain approach: define station points as [position, scale] pairs "
            "and use hull() between adjacent stations to build the body. "
            "Avoid intersection() of extruded profiles — coordinate mismatches cause empty results."
        )
    if "timed out" in err_lower:
        hints.append(
            "The model took too long to compile. Use $fn=16 for spheres inside hull() calls. "
            "Only use high $fn for visible surfaces like wheels. Reduce the number of hull "
            "chain stations. Use cuboid/cyl instead of hull(sphere) where a box/cylinder will do."
        )
    if not hints:
        hints.append(
            "Try a simpler approach: use hull chains (hull() between pairs of scaled spheres "
            "at station points) for organic bodies, or combine cuboid/cyl/sphere primitives. "
            "Avoid skin() and intersection() of extruded profiles — they are fragile."
        )
    return "\n".join(hints) + "\n\n"


# ── Image loading (file path or data-URL) ─────────────────────────────────────

def _load_image(source: str) -> Any:
    """Load a PIL Image from a file path or ``data:image/...;base64,...`` URL."""
    import base64 as _b64
    import io

    from PIL import Image as PILImage

    if source.startswith("data:"):
        m = re.match(r"data:image/\w+;base64,(.*)", source, re.DOTALL)
        if m:
            return PILImage.open(io.BytesIO(_b64.b64decode(m.group(1))))
        return None

    p = Path(source)
    return PILImage.open(p) if p.exists() else None


# ── Public API ────────────────────────────────────────────────────────────────

def generate_cad(
    name: str,
    description: str,
    image_path: str | None = None,
    max_retries: int = 2,
) -> tuple[bool, str, Path | None]:
    """Generate an STL file from a text description (+ optional image).

    Uses articulate_anything's VLM wrappers (Claude/GPT/Gemini) with
    CADAM's strict code generation prompt.
    LLM configured via CAD_PROVIDER / CAD_MODEL env vars (falls back to global).

    Returns (success, message, stl_path).
    """
    asset_dir = ASSETS_DIR / name
    asset_dir.mkdir(parents=True, exist_ok=True)

    vlm = setup_vlm_model(
        model_name=CAD_CONFIG.vlm_model_name,
        system_instruction=STRICT_CODE_PROMPT,
        api_key=CAD_CONFIG.api_key,
    )

    last_err = ""
    for attempt in range(max_retries + 1):
        prompt_parts: list[Any] = [description]

        if image_path:
            pil_img = _load_image(image_path)
            if pil_img:
                prompt_parts.append(pil_img)

        if attempt > 0 and last_err:
            fix_hints = _bosl2_fix_hints(last_err)
            prompt_parts = [
                f"The previous OpenSCAD code failed to compile:\n\n{last_err}\n\n"
                f"{fix_hints}"
                f"Fix the code. Keep all parametric variables. Return ONLY corrected "
                f"OpenSCAD code.\n\nOriginal request: {description}"
            ]

        response = vlm.generate_content(
            prompt_parts,
            generation_config={"temperature": CAD_CONFIG.temperature, "max_tokens": 8192},
        )
        scad_code = extract_openscad_code(response.text)

        scad_path = asset_dir / f"{name}.scad"
        stl_path = asset_dir / f"{name}.stl"
        scad_path.write_text(scad_code, encoding="utf-8")

        params = parse_parameters(scad_code)
        if params:
            params_path = asset_dir / "parameters.json"
            params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

        ok, err = _compile_scad(scad_path, stl_path)
        if ok:
            (asset_dir / "description.txt").write_text(description, encoding="utf-8")
            genesis_hint = (
                f'Genesis usage: gs.morphs.Mesh('
                f'file="assets/{name}/{name}.stl", scale=0.001, fixed=True)'
            )
            return (
                True,
                f"STL saved to assets/{name}/{name}.stl "
                f"({len(params)} parameters, {CAD_CONFIG}). "
                f"{genesis_hint}",
                stl_path,
            )

        last_err = err

    return False, f"Failed after {max_retries + 1} attempts. Last error:\n{last_err}", None


@tool
def cad_generate(name: str, description: str, image_path: str = "") -> str:
    """Generate a 3D object as an STL file from a text description.

    Uses CADAM's parametric OpenSCAD agent with multi-provider VLM support
    (Claude/GPT/Gemini). Compiles to STL and saves to assets/{name}/ with
    parameters.json and description.txt for reuse.

    Args:
        name: Short identifier for the asset (e.g. 'warehouse_shelf').
        description: What the object should look like and rough dimensions.
        image_path: Optional path to a reference image.
    """
    ok, msg, _ = generate_cad(name, description, image_path or None)
    return msg
