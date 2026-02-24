"use client";

import { useMemo } from "react";
import * as THREE from "three";

// ── URDF XML → Three.js scene graph ─────────────────────────────────────────

interface URDFLink {
  name: string;
  visual?: {
    origin: { xyz: [number, number, number]; rpy: [number, number, number] };
    geometry:
      | { type: "box"; size: [number, number, number] }
      | { type: "cylinder"; radius: number; length: number }
      | { type: "sphere"; radius: number };
    color: [number, number, number, number];
  };
}

interface URDFJoint {
  name: string;
  type: string;
  parent: string;
  child: string;
  origin: { xyz: [number, number, number]; rpy: [number, number, number] };
}

function parseXYZ(el: Element | null): [number, number, number] {
  if (!el) return [0, 0, 0];
  const s = el.getAttribute("xyz") || "0 0 0";
  const p = s.trim().split(/\s+/).map(Number);
  return [p[0] || 0, p[1] || 0, p[2] || 0];
}

function parseRPY(el: Element | null): [number, number, number] {
  if (!el) return [0, 0, 0];
  const s = el.getAttribute("rpy") || "0 0 0";
  const p = s.trim().split(/\s+/).map(Number);
  return [p[0] || 0, p[1] || 0, p[2] || 0];
}

function parseColor(matEl: Element | null): [number, number, number, number] {
  if (!matEl) return [0.5, 0.5, 0.5, 1];
  const colorEl = matEl.querySelector("color");
  if (!colorEl) return [0.5, 0.5, 0.5, 1];
  const rgba = (colorEl.getAttribute("rgba") || "0.5 0.5 0.5 1")
    .trim()
    .split(/\s+/)
    .map(Number);
  return [rgba[0] ?? 0.5, rgba[1] ?? 0.5, rgba[2] ?? 0.5, rgba[3] ?? 1];
}

export function parseURDF(xml: string): { links: URDFLink[]; joints: URDFJoint[] } {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, "text/xml");
  const robot = doc.querySelector("robot");
  if (!robot) return { links: [], joints: [] };

  const materialMap = new Map<string, [number, number, number, number]>();
  robot.querySelectorAll(":scope > material").forEach((m) => {
    const name = m.getAttribute("name");
    if (name) materialMap.set(name, parseColor(m));
  });

  const links: URDFLink[] = [];
  robot.querySelectorAll("link").forEach((linkEl) => {
    const name = linkEl.getAttribute("name") || "unnamed";
    const vis = linkEl.querySelector("visual");
    if (!vis) {
      links.push({ name });
      return;
    }

    const originEl = vis.querySelector("origin");
    const geoEl = vis.querySelector("geometry");
    const matEl = vis.querySelector("material");

    let color = parseColor(matEl);
    if (matEl && !matEl.querySelector("color")) {
      const matName = matEl.getAttribute("name");
      if (matName && materialMap.has(matName)) {
        color = materialMap.get(matName)!;
      }
    }

    let geometry: URDFLink["visual"] extends infer V
      ? V extends { geometry: infer G }
        ? G
        : never
      : never;

    if (geoEl) {
      const box = geoEl.querySelector("box");
      const cyl = geoEl.querySelector("cylinder");
      const sph = geoEl.querySelector("sphere");

      if (box) {
        const sz = (box.getAttribute("size") || "0.1 0.1 0.1")
          .trim()
          .split(/\s+/)
          .map(Number) as [number, number, number];
        geometry = { type: "box", size: sz };
      } else if (cyl) {
        geometry = {
          type: "cylinder",
          radius: parseFloat(cyl.getAttribute("radius") || "0.05"),
          length: parseFloat(cyl.getAttribute("length") || "0.1"),
        };
      } else if (sph) {
        geometry = {
          type: "sphere",
          radius: parseFloat(sph.getAttribute("radius") || "0.05"),
        };
      } else {
        geometry = { type: "box", size: [0.05, 0.05, 0.05] };
      }
    } else {
      geometry = { type: "box", size: [0.05, 0.05, 0.05] };
    }

    links.push({
      name,
      visual: {
        origin: { xyz: parseXYZ(originEl), rpy: parseRPY(originEl) },
        geometry,
        color,
      },
    });
  });

  const joints: URDFJoint[] = [];
  robot.querySelectorAll("joint").forEach((jEl) => {
    const originEl = jEl.querySelector("origin");
    joints.push({
      name: jEl.getAttribute("name") || "unnamed",
      type: jEl.getAttribute("type") || "fixed",
      parent: jEl.querySelector("parent")?.getAttribute("link") || "",
      child: jEl.querySelector("child")?.getAttribute("link") || "",
      origin: { xyz: parseXYZ(originEl), rpy: parseRPY(originEl) },
    });
  });

  return { links, joints };
}

// ── Build Three.js group from parsed URDF ────────────────────────────────────

function buildLinkGroup(link: URDFLink): THREE.Group {
  const group = new THREE.Group();
  group.name = link.name;

  if (!link.visual) return group;

  const { origin, geometry, color } = link.visual;
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color[0], color[1], color[2]),
    metalness: 0.3,
    roughness: 0.6,
    transparent: color[3] < 1,
    opacity: color[3],
  });

  let geo: THREE.BufferGeometry;
  switch (geometry.type) {
    case "box":
      geo = new THREE.BoxGeometry(geometry.size[0], geometry.size[1], geometry.size[2]);
      break;
    case "cylinder":
      geo = new THREE.CylinderGeometry(geometry.radius, geometry.radius, geometry.length, 32);
      geo.rotateX(Math.PI / 2); // URDF: cylinder along Z; Three.js: along Y
      break;
    case "sphere":
      geo = new THREE.SphereGeometry(geometry.radius, 32, 24);
      break;
  }

  const mesh = new THREE.Mesh(geo, material);
  mesh.position.set(...origin.xyz);

  const euler = new THREE.Euler(origin.rpy[0], origin.rpy[1], origin.rpy[2], "XYZ");
  mesh.setRotationFromEuler(euler);

  group.add(mesh);
  return group;
}

export function buildURDFScene(urdfXml: string): THREE.Group {
  const { links, joints } = parseURDF(urdfXml);

  const linkGroups = new Map<string, THREE.Group>();
  for (const link of links) {
    linkGroups.set(link.name, buildLinkGroup(link));
  }

  const childSet = new Set(joints.map((j) => j.child));
  const root = new THREE.Group();
  root.name = "urdf_root";

  for (const joint of joints) {
    const parentGroup = linkGroups.get(joint.parent);
    const childGroup = linkGroups.get(joint.child);
    if (!parentGroup || !childGroup) continue;

    const jointGroup = new THREE.Group();
    jointGroup.name = `joint_${joint.name}`;
    jointGroup.position.set(...joint.origin.xyz);
    const euler = new THREE.Euler(
      joint.origin.rpy[0],
      joint.origin.rpy[1],
      joint.origin.rpy[2],
      "XYZ"
    );
    jointGroup.setRotationFromEuler(euler);
    jointGroup.add(childGroup);
    parentGroup.add(jointGroup);
  }

  for (const [name, group] of linkGroups) {
    if (!childSet.has(name)) {
      root.add(group);
    }
  }

  // Center the model
  const bbox = new THREE.Box3().setFromObject(root);
  const center = bbox.getCenter(new THREE.Vector3());
  root.position.sub(center);

  return root;
}

// ── React component ──────────────────────────────────────────────────────────

export function URDFMesh({ xml }: { xml: string }) {
  const scene = useMemo(() => buildURDFScene(xml), [xml]);

  return <primitive object={scene} />;
}
