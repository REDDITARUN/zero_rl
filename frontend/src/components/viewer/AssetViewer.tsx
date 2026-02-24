"use client";

import { Suspense, useState, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  GizmoHelper,
  GizmoViewcube,
  Stage,
  PerspectiveCamera,
} from "@react-three/drei";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { Loader2 } from "lucide-react";
import { URDFMesh } from "./URDFMesh";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function STLMesh({ geometry }: { geometry: THREE.BufferGeometry }) {
  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        color="#67e8f9"
        metalness={0.5}
        roughness={0.35}
        envMapIntensity={0.3}
      />
    </mesh>
  );
}

type ModelData =
  | { kind: "stl"; geometry: THREE.BufferGeometry }
  | { kind: "urdf"; xml: string };

function SceneContent({ model }: { model: ModelData }) {
  return (
    <>
      <PerspectiveCamera
        makeDefault
        position={[-80, 80, 80]}
        fov={45}
        near={0.001}
        far={2000}
      />
      <Stage environment={null} intensity={0.6}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={1.2} castShadow />
        <directionalLight position={[-5, 5, 5]} intensity={0.2} />
        <directionalLight position={[-5, 5, -5]} intensity={0.2} />
        <directionalLight position={[0, 5, 0]} intensity={0.2} />
        <directionalLight position={[-5, -5, -5]} intensity={0.6} />
        {model.kind === "stl" ? (
          <STLMesh geometry={model.geometry} />
        ) : (
          <URDFMesh xml={model.xml} />
        )}
      </Stage>
      <OrbitControls makeDefault enableDamping dampingFactor={0.05} />
      <GizmoHelper alignment="bottom-right" margin={[60, 70]}>
        <GizmoViewcube />
      </GizmoHelper>
    </>
  );
}

interface AssetViewerProps {
  assetId: string;
  filename: string;
}

export function AssetViewer({ assetId, filename }: AssetViewerProps) {
  const [model, setModel] = useState<ModelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const isURDF = filename.endsWith(".urdf");

  useEffect(() => {
    setLoading(true);
    setError("");
    setModel(null);

    const url = `${API}/api/assets/${encodeURIComponent(assetId)}/file/${encodeURIComponent(filename)}`;

    if (isURDF) {
      fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.text();
        })
        .then((xml) => setModel({ kind: "urdf", xml }))
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    } else {
      fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.arrayBuffer();
        })
        .then((buf) => {
          const loader = new STLLoader();
          const geo = loader.parse(buf);
          geo.center();
          geo.computeVertexNormals();
          setModel({ kind: "stl", geometry: geo });
        })
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }
  }, [assetId, filename, isURDF]);

  if (loading) {
    return (
      <div className="flex h-full min-h-[300px] items-center justify-center rounded-xl border border-border bg-black/30">
        <div className="text-center">
          <Loader2 size={20} className="mx-auto mb-2 animate-spin text-primary/50" />
          <p className="text-[10px] text-muted-foreground/40 font-mono">
            loading {isURDF ? "urdf" : "3d"} model...
          </p>
        </div>
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="flex h-full min-h-[300px] items-center justify-center rounded-xl border border-border bg-black/30">
        <p className="text-[10px] text-red-400/60 font-mono">{error || "failed to load model"}</p>
      </div>
    );
  }

  return (
    <div className="relative h-full min-h-[300px] overflow-hidden rounded-xl border border-border">
      <Canvas className="block h-full w-full" style={{ background: "#1a1a2e" }}>
        <Suspense fallback={null}>
          <SceneContent model={model} />
        </Suspense>
      </Canvas>
    </div>
  );
}
