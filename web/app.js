import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file");
const fileLabel = document.getElementById("file-label");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const hintEl = document.getElementById("hint");
const submitBtn = document.getElementById("submit");
const host = document.getElementById("canvas-host");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 500);
camera.position.set(0.4, 0.3, 1.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

scene.add(new THREE.AmbientLight(0xffffff, 0.85));
const key = new THREE.DirectionalLight(0xfff2e0, 0.65);
key.position.set(2, 4, 3);
scene.add(key);

let pointsObj = null;

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  fileLabel.textContent = f ? f.name : "Choose video or images";
});

function setStatus(text, kind = "") {
  statusEl.hidden = false;
  statusEl.className = `status ${kind}`.trim();
  statusEl.textContent = text;
}

function clearScenePoints() {
  if (!pointsObj) return;
  scene.remove(pointsObj);
  pointsObj.geometry.dispose();
  pointsObj.material.dispose();
  pointsObj = null;
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

async function loadPly(url) {
  clearScenePoints();
  const loader = new PLYLoader();
  const geometry = await loader.loadAsync(url);
  geometry.computeBoundingBox();
  if (!geometry.getAttribute("color")) {
    const count = geometry.getAttribute("position").count;
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      colors[i * 3] = 0.55;
      colors[i * 3 + 1] = 0.75;
      colors[i * 3 + 2] = 0.8;
    }
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
  hintEl.textContent = "Drag to orbit · scroll to zoom · right-drag to pan";
}

async function pollJob(jobId) {
  for (;;) {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) throw new Error(`Status check failed (${res.status})`);
    const data = await res.json();
    setStatus(`${data.status}: ${data.message || ""}`);
    if (data.status === "succeeded") {
      metaEl.hidden = false;
      metaEl.textContent = JSON.stringify(data.meta ?? data, null, 2);
      await loadPly(`/api/jobs/${jobId}/scene.ply`);
      setStatus(`Ready — ${data.meta?.num_points ?? "?"} points`, "ok");
      return;
    }
    if (data.status === "failed") {
      throw new Error(data.message || "Reconstruction failed");
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) return;

  submitBtn.disabled = true;
  metaEl.hidden = true;
  setStatus("Uploading…");

  try {
    const body = new FormData();
    body.append("file", file);
    body.append("max_frames", document.getElementById("max-frames").value);
    body.append("target_fps", document.getElementById("target-fps").value);
    body.append("conf_thres", "50");

    const res = await fetch("/api/jobs", { method: "POST", body });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `Upload failed (${res.status})`);
    }
    const { id } = await res.json();
    setStatus(`queued: job ${id}`);
    await pollJob(id);
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), "error");
  } finally {
    submitBtn.disabled = false;
  }
});
