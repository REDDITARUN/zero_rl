"""URDF generation tool: text/image → URDF XML.

Uses Articulate-Anything (github.com/vlongle/articulate-anything) directly:
  - setup_vlm_model for multi-provider VLM (Claude/GPT/Gemini)
  - Agent base class pattern: system instruction → prompt → generate → parse
  - Actor-critic validation loop: generate → validate → fix
  - odio_urdf utilities for structural URDF validation
  - Steps for provenance tracking

No fallbacks — articulate_anything is a hard dependency.

References:
  - articulate-anything installed as editable: core/tools/articulate_repo/
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from articulate_anything.utils.prompt_utils import (
    setup_vlm_model,
    extract_code_from_string,
)
from articulate_anything.utils.utils import Steps, create_dir
from articulate_anything.api.odio_urdf_utils import urdf_to_odio

from core.config import ASSETS_DIR, URDF_CONFIG


# ── System instructions ──────────────────────────────────────────────────────

URDF_SYSTEM_INSTRUCTION = (
    "You are a robotics expert specializing in URDF (Unified Robot "
    "Description Format) file generation. You create precise, valid "
    "URDF XML files for robots and articulated objects.\n\n"
    "Your URDFs must:\n"
    "- Have a <robot> root element with a name attribute\n"
    "- Include <visual>, <collision>, and <inertial> for every link\n"
    "- Use correct inertia tensors (box/cylinder/sphere formulas)\n"
    "- Define proper joint limits, damping (0.1), and friction (0.1)\n"
    "- Use SI units (meters, kg, radians)\n"
    "- Be structurally valid (all parent/child links must exist)\n"
    "- Have a base_link as the root link\n\n"
    "Inertia formulas:\n"
    "- Box (m,x,y,z): ixx=m/12*(y²+z²), iyy=m/12*(x²+z²), izz=m/12*(x²+y²)\n"
    "- Cylinder (m,r,h): ixx=iyy=m/12*(3r²+h²), izz=m/2*r²\n"
    "- Sphere (m,r): ixx=iyy=izz=2/5*m*r²\n"
)

PLAN_PROMPT = """\
Given a description (and optional reference image), output a JSON plan for a URDF file.

The plan must include:
1. A list of links: name, geometry type (box/cylinder/sphere), dimensions, mass, color
2. A list of joints: name, type (fixed/revolute/prismatic/continuous), parent, child,
   origin_xyz, origin_rpy, axis, limits (lower/upper/effort/velocity)

Rules:
- Every robot needs a "base_link" as the root.
- All links must be connected through joints (tree structure, no loops).
- Include reasonable inertial properties.
- Use SI units (meters, kg, radians).

Output ONLY valid JSON.

Description: {description}
"""

GENERATE_PROMPT = """\
Generate a complete URDF XML file from this structured plan.

Rules:
- Root element: <robot name="...">
- Every link needs: <visual>, <collision>, <inertial>
- Joint limits, damping=0.1, friction=0.1 on all non-fixed joints
- Use standard URDF geometry (box size, cylinder radius+length, sphere radius)
- Colors via <material> elements
- Return ONLY URDF XML — no markdown, no explanations.

Plan:
{plan}

Original request: {description}
"""

FIX_PROMPT = """\
The previous URDF had these structural errors:
{errors}

Fix the URDF XML. Keep the same structure and intent.
Return ONLY the corrected URDF XML — no markdown, no explanations.

Original request: {description}
"""


# ── URDF validation ──────────────────────────────────────────────────────────

def _validate_urdf(xml_text: str) -> tuple[bool, list[str]]:
    """Structural validation of URDF XML + odio_urdf check."""
    errors: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return False, [f"XML parse error: {e}"]

    if root.tag != "robot":
        return False, [f"Root element is <{root.tag}>, expected <robot>"]

    links = root.findall("link")
    joints = root.findall("joint")
    if not links:
        return False, ["No <link> elements found"]

    link_names = {l.get("name") for l in links}
    for joint in joints:
        jname = joint.get("name", "unnamed")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            errors.append(f"Joint '{jname}' missing parent or child element")
            continue
        if parent.get("link") not in link_names:
            errors.append(f"Joint '{jname}' parent '{parent.get('link')}' not in links")
        if child.get("link") not in link_names:
            errors.append(f"Joint '{jname}' child '{child.get('link')}' not in links")

    for link in links:
        lname = link.get("name", "unnamed")
        if link.find("inertial") is None:
            errors.append(f"Link '{lname}' missing <inertial>")
        if link.find("visual") is None:
            errors.append(f"Link '{lname}' missing <visual>")

    # Articulate-anything's odio_urdf validation
    try:
        odio_repr = urdf_to_odio(xml_text)
    except Exception as exc:
        errors.append(f"odio_urdf validation failed: {exc}")

    return (
        not errors,
        errors if errors else [f"Valid: {len(links)} links, {len(joints)} joints"],
    )


# ── XML extraction ───────────────────────────────────────────────────────────

def _extract_xml(text: str) -> str:
    """Extract URDF XML from LLM response."""
    m = re.search(r"```(?:xml|urdf)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(<\?xml.*?</robot>|<robot[\s\S]*?</robot>)", text, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return text.strip()


def _extract_json(text: str) -> dict | None:
    """Extract JSON plan from LLM response."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    raw = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m2 = re.search(r"\{[\s\S]*\}", raw)
        if m2:
            try:
                return json.loads(m2.group())
            except json.JSONDecodeError:
                return None
    return None


# ── Image loading (file path or data-URL) ─────────────────────────────────────

def _load_image(source: str) -> Any:
    """Load a PIL Image from a file path or ``data:image/...;base64,...`` URL."""
    import base64
    import io

    from PIL import Image as PILImage

    if source.startswith("data:"):
        m = re.match(r"data:image/\w+;base64,(.*)", source, re.DOTALL)
        if m:
            return PILImage.open(io.BytesIO(base64.b64decode(m.group(1))))
        return None

    p = Path(source)
    return PILImage.open(p) if p.exists() else None


# ── Generation pipeline (Articulate-Anything agent pattern) ──────────────────

class URDFGenerator:
    """URDF generation using articulate_anything's VLM wrappers.

    Three-stage actor-critic pipeline:
      1. Plan: decompose object into links + joints (JSON)
      2. Generate: produce URDF XML from plan
      3. Critic/Fix: validate → fix loop
    """

    def __init__(self) -> None:
        self.vlm = setup_vlm_model(
            model_name=URDF_CONFIG.vlm_model_name,
            system_instruction=URDF_SYSTEM_INSTRUCTION,
            api_key=URDF_CONFIG.api_key,
        )
        self.steps = Steps()

    def generate(
        self,
        description: str,
        image_path: str | None = None,
        max_retries: int = 2,
    ) -> tuple[str | None, dict | None]:
        """Run the full plan → generate → validate → fix pipeline.

        Returns (urdf_xml, plan_dict) or (None, None) on failure.
        """
        # Stage 1: Plan
        plan_prompt = PLAN_PROMPT.format(description=description)
        plan_parts = self._build_prompt_parts(plan_prompt, image_path)
        plan_response = self.vlm.generate_content(
            plan_parts, generation_config={"temperature": URDF_CONFIG.temperature, "max_tokens": 4096}
        )
        plan = _extract_json(plan_response.text)
        self.steps.add_step("plan", plan)

        # Stage 2+3: Generate + Validate/Fix loop
        if plan:
            gen_text = GENERATE_PROMPT.format(
                plan=json.dumps(plan, indent=2), description=description
            )
        else:
            gen_text = f"Generate a complete URDF XML file for: {description}"

        for attempt in range(max_retries + 1):
            gen_parts = self._build_prompt_parts(gen_text, image_path if attempt == 0 else None)
            response = self.vlm.generate_content(
                gen_parts, generation_config={"temperature": URDF_CONFIG.temperature, "max_tokens": 8192}
            )
            urdf_xml = _extract_xml(response.text)
            ok, errors = _validate_urdf(urdf_xml)

            if ok:
                self.steps.add_step("urdf_generation", {
                    "urdf": urdf_xml,
                    "plan": plan,
                    "attempt": attempt + 1,
                })
                return urdf_xml, plan

            # Build fix prompt for next attempt
            if attempt < max_retries:
                gen_text = FIX_PROMPT.format(
                    errors="\n".join(errors), description=description
                )

        return None, plan

    def _build_prompt_parts(self, text: str, image_path: str | None) -> list[Any]:
        """Build prompt parts in articulate_anything format (str + PIL.Image)."""
        parts: list[Any] = [text]
        if image_path:
            pil_img = _load_image(image_path)
            if pil_img:
                parts.append(pil_img)
        return parts


# ── Public API ────────────────────────────────────────────────────────────────

def generate_urdf(
    name: str,
    description: str,
    image_path: str | None = None,
    max_retries: int = 2,
) -> tuple[bool, str, Path | None]:
    """Generate a URDF file using articulate_anything's VLM pipeline.

    Returns (success, message, urdf_path).
    """
    asset_dir = ASSETS_DIR / name
    asset_dir.mkdir(parents=True, exist_ok=True)

    generator = URDFGenerator()
    urdf_xml, plan = generator.generate(description, image_path, max_retries)

    if urdf_xml is None:
        return False, "Failed to generate valid URDF after all attempts.", None

    urdf_path = asset_dir / f"{name}.urdf"
    urdf_path.write_text(urdf_xml, encoding="utf-8")
    (asset_dir / "description.txt").write_text(description, encoding="utf-8")

    if plan:
        (asset_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    ok, msgs = _validate_urdf(urdf_xml)
    summary = msgs[0] if msgs else "Generated"
    return (
        True,
        f"URDF saved to assets/{name}/{name}.urdf ({URDF_CONFIG}). {summary}",
        urdf_path,
    )


@tool
def urdf_generate(name: str, description: str, image_path: str = "") -> str:
    """Generate a URDF robot/object description from text (+ optional image).

    Uses Articulate-Anything's multi-provider VLM pipeline with a three-stage
    actor-critic pattern: plan → generate → validate → fix.
    Supports Claude, GPT, and Gemini models.

    Args:
        name: Short identifier (e.g. 'simple_arm').
        description: What the robot/object should be, its joints, DOF, etc.
        image_path: Optional path to a reference image.
    """
    ok, msg, _ = generate_urdf(name, description, image_path or None)
    return msg
