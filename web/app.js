import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file");
const fileLabel = document.getElementById("file-label");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const metaWrap = document.getElementById("meta-wrap");
const emptyEl = document.getElementById("empty");
const hintEl = document.getElementById("hint");
const submitBtn = document.getElementById("submit");
const host = document.getElementById("canvas-host");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0xf3f3f3, 1);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf3f3f3);
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 5000);
camera.position.set(0.4, 0.3, 1.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 0.55);
key.position.set(2, 4, 3);
scene.add(key);

let pointsObj = null;
let viewer = null;
let useExternalViewer = false;

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  if (viewer) {
    viewer.setSize?.(w, h);
  }
}

window.addEventListener("resize", resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  if (!useExternalViewer) {
    controls.update();
    renderer.render(scene, camera);
  }
}
animate();

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  fileLabel.textContent = f ? f.name : "Drop video or images";
});

function setStatus(text, kind = "") {
  statusEl.hidden = false;
  statusEl.className = `status ${kind}`.trim();
  statusEl.textContent = text;
}

function clearScene() {
  if (pointsObj) {
    scene.remove(pointsObj);
    pointsObj.geometry.dispose();
    pointsObj.material.dispose();
    pointsObj = null;
  }
  if (viewer) {
    try {
      viewer.dispose();
    } catch (_) {
      /* ignore */
    }
    viewer = null;
  }
  useExternalViewer = false;
  // Ensure our canvas is visible again for point fallback
  renderer.domElement.style.display = "block";
}

function fitCameraToObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.1);
  const dist = maxDim * 1.8;
  controls.target.copy(center);
  camera.position.copy(center.clone().add(new THREE.Vector3(dist * 0.35, dist * 0.25, dist)));
  camera.near = Math.max(dist / 200, 0.01);
  camera.far = dist * 40;
  camera.updateProjectionMatrix();
  controls.update();
}

async function loadPointsPly(url) {
  clearScene();
  const loader = new PLYLoader();
  const geometry = await loader.loadAsync(url);
  geometry.computeBoundingBox();
  if (!geometry.getAttribute("color")) {
    const count = geometry.getAttribute("position").count;
    const colors = new Float32Array(count * 3);
    colors.fill(0.45);
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }
  const material = new THREE.PointsMaterial({
    size: 0.01,
    vertexColors: true,
    sizeAttenuation: true,
  });
  pointsObj = new THREE.Points(geometry, material);
  scene.add(pointsObj);
  fitCameraToObject(pointsObj);
  emptyEl.classList.add("hidden");
  hintEl.textContent = "Drag to orbit · scroll to zoom · right-drag to pan";
}

async function loadGaussianPly(url) {
  clearScene();
  renderer.domElement.style.display = "none";
  useExternalViewer = true;

  viewer = new GaussianSplats3D.Viewer({
    rootElement: host,
    cameraUp: [0, -1, -0.6],
    initialCameraPosition: [-2, -1.5, 2.5],
    initialCameraLookAt: [0, 0, 0],
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort: true,
  });

  await viewer.addSplatScene(url, {
    showLoadingUI: false,
    progressiveLoad: true,
  });

  emptyEl.classList.add("hidden");
  hintEl.textContent = "Drag to orbit · scroll to zoom";
  resize();
}

async function loadScene(jobId, preferred = "splat") {
  if (preferred === "splat") {
    try {
      await loadGaussianPly(`/api/jobs/${jobId}/gaussians.ply`);
      return "splat";
    } catch (err) {
      console.warn("Gaussian splat load failed, falling back to points", err);
    }
  }
  await loadPointsPly(`/api/jobs/${jobId}/points.ply`);
  return "points";
}

function formatElapsed(ms) {
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

async function pollJob(jobId) {
  const started = Date.now();
  for (;;) {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) throw new Error(`Status check failed (${res.status})`);
    const data = await res.json();
    const elapsed = formatElapsed(Date.now() - started);
    setStatus(`${(data.message || data.status || "Working").trim()} · ${elapsed}`);
    if (data.status === "succeeded") {
      metaWrap.hidden = false;
      metaEl.textContent = JSON.stringify(data.meta ?? data, null, 2);
      setStatus(`Loading scene into viewer… · ${elapsed}`);
      const mode = await loadScene(jobId, data.viewer || "splat");
      setStatus(
        mode === "splat"
          ? `Ready — Gaussian splat (WorldMirror 2.0)`
          : `Ready — point cloud fallback`,
        "ok"
      );
      return;
    }
    if (data.status === "failed") {
      throw new Error(data.message || "Reconstruction failed");
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) return;

  submitBtn.disabled = true;
  metaWrap.hidden = true;
  emptyEl.classList.remove("hidden");
  hintEl.textContent = "Working — keep this tab open";
  setStatus("Uploading…");

  try {
    const body = new FormData();
    body.append("file", file);
    body.append("max_frames", document.getElementById("max-frames").value);
    body.append("target_fps", document.getElementById("target-fps").value);
    body.append("target_size", "952");

    const res = await fetch("/api/jobs", { method: "POST", body });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `Upload failed (${res.status})`);
    }
    const { id } = await res.json();
    setStatus(
      "Queued — waiting for GPU. First WorldMirror start can take several minutes."
    );
    await pollJob(id);
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), "error");
    hintEl.textContent = "Upload a video to start high-quality reconstruction";
  } finally {
    submitBtn.disabled = false;
  }
});
