"""Prompts for URDF generation (inspired by Articulate-Anything's approach).

Articulate-Anything uses a multi-stage actor-critic pipeline:
  1. Mesh retrieval — find base mesh
  2. Link articulation — decompose object into links
  3. Joint articulation — define joints between links
  4. Critic — verify correctness, iterate

We adapt this as a two-step LLM pipeline:
  Step 1: Plan the robot's links and joints (structured JSON)
  Step 2: Generate the full URDF XML from the plan
"""

URDF_PLAN_PROMPT = """\
You are a robotics engineer planning the structure of a robot or articulated
object for a URDF file.

Given a description (and optional reference image), output a JSON plan with:
1. A list of links (name, geometry type, dimensions, mass)
2. A list of joints (name, type, parent_link, child_link, axis, limits)

Joint types: fixed, revolute, prismatic, continuous
Geometry types: box, cylinder, sphere

Rules:
- Every robot needs a "base_link" as the root.
- All links must be connected through joints (tree structure, no loops).
- Include reasonable inertial properties.
- Use SI units (meters, kg, radians).
- Joint limits should be physically plausible.

Output ONLY valid JSON in this format:
{
  "name": "robot_name",
  "links": [
    {
      "name": "base_link",
      "geometry": {"type": "box", "size": [0.3, 0.3, 0.1]},
      "mass": 5.0,
      "color": [0.8, 0.8, 0.8, 1.0]
    }
  ],
  "joints": [
    {
      "name": "joint_1",
      "type": "revolute",
      "parent": "base_link",
      "child": "link_1",
      "origin_xyz": [0, 0, 0.05],
      "origin_rpy": [0, 0, 0],
      "axis": [0, 0, 1],
      "lower": -1.57,
      "upper": 1.57,
      "effort": 100,
      "velocity": 1.0
    }
  ]
}
"""

URDF_GENERATE_PROMPT = """\
You are a robotics engineer. Given a structured JSON plan for a robot,
generate a complete, valid URDF XML file.

Rules:
- Root element must be <robot name="...">
- Every link must have <visual>, <collision>, and <inertial> elements.
- Inertia tensors must be positive definite (use box/cylinder/sphere formulas).
- Joint limits, damping (0.1), and friction (0.1) on all non-fixed joints.
- Use standard URDF conventions for geometry (box size, cylinder radius+length, sphere radius).
- Colors via <material> elements.

Return ONLY the URDF XML — no markdown fences, no explanations.

Inertia formulas:
- Box (m, x, y, z): ixx=m/12*(y²+z²), iyy=m/12*(x²+z²), izz=m/12*(x²+y²)
- Cylinder (m, r, h): ixx=iyy=m/12*(3r²+h²), izz=m/2*r²
- Sphere (m, r): ixx=iyy=izz=2/5*m*r²
"""

URDF_FIX_PROMPT = """\
The previous URDF had these structural errors:

{errors}

Fix the URDF XML. Return ONLY the corrected URDF XML — no markdown fences.

Original request: {description}
"""
