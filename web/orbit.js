import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const form = document.getElementById("orbit-form");
const fileInput = document.getElementById("file");
const fileLabel = document.getElementById("file-label");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const metaWrap = document.getElementById("meta-wrap");
const emptyEl = document.getElementById("empty");
const hintEl = document.getElementById("hint");
const submitBtn = document.getElementById("submit");
const host = document.getElementById("orbit-host");
const angleHud = document.getElementById("angle-hud");
const angleLabel = document.getElementById("angle-label");
const angleSlider = document.getElementById("angle-slider");
const viewCountEl = document.getElementById("view-count");
const filmstrip = document.getElementById("filmstrip");
const viewPreview = document.getElementById("view-preview");
const viewPreviewImg = document.getElementById("view-preview-img");
const viewPreviewCap = document.getElementById("view-preview-cap");
const providerEl = document.getElementById("provider");
const modelEl = document.getElementById("model");

const DEFAULT_MODELS = {
  gemini: "gemini-2.5-flash-image",
  openai: "gpt-image-1",
};

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0xffffff, 1);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const viewCam = new THREE.PerspectiveCamera(45, 1, 0.05, 80);
viewCam.position.set(3.1, 2.05, 3.4);

const controls = new OrbitControls(viewCam, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, 0.55, 0);

scene.add(new THREE.AmbientLight(0xffffff, 1));

const grid = new THREE.GridHelper(24, 48, 0xd8d8d8, 0xececec);
grid.position.y = 0;
scene.add(grid);

const shadow = new THREE.Mesh(
  new THREE.CircleGeometry(0.55, 48),
  new THREE.MeshBasicMaterial({
    color: 0x9a9a9a,
    transparent: true,
    opacity: 0.16,
  })
);
shadow.rotation.x = -Math.PI / 2;
shadow.position.y = 0.002;
scene.add(shadow);

const imageGroup = new THREE.Group();
scene.add(imageGroup);

const frustumGroup = new THREE.Group();
scene.add(frustumGroup);

let imagePlane = null;
let imageAspect = 1;
let localImageUrl = null;
let jobId = null;
let frames = [];
let currentAngle = 0;

function params() {
  return {
    increment: clampNum("increment", 5, 90, 10),
    elevation: clampNum("elevation", 0, 75, 12),
    start: clampNum("start-deg", 0, 350, 0),
    end: clampNum("end-deg", 10, 360, 360),
    radius: clampNum("radius", 0.6, 8, 2.4),
    fov: clampNum("fov", 20, 90, 45),
    provider: providerEl.value,
    model: modelEl.value.trim(),
    extra: document.getElementById("extra-prompt").value,
    apiKey: document.getElementById("api-key").value,
  };
}

function clampNum(id, min, max, fallback) {
  const n = Number(document.getElementById(id).value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function plannedAngles() {
  const { start, end, increment } = params();
  const out = [];
  for (let a = start; a < end - 1e-6; a += increment) {
    out.push(Number((a % 360).toFixed(4)));
    if (out.length > 72) break;
  }
  return out;
}

function refreshViewCount() {
  const angles = plannedAngles();
  const { increment } = params();
  viewCountEl.textContent = `${angles.length} views · ${increment}° steps`;
  const max = Math.max(0, params().end - increment);
  angleSlider.min = String(params().start);
  angleSlider.max = String(max);
  angleSlider.step = String(increment);
}

function cameraXYZ(azimuth, elevation, radius) {
  const az = THREE.MathUtils.degToRad(azimuth);
  const el = THREE.MathUtils.degToRad(elevation);
  return new THREE.Vector3(
    radius * Math.sin(az) * Math.cos(el),
    radius * Math.sin(el),
    radius * Math.cos(az) * Math.cos(el)
  );
}

function makeFrustum(fovDeg, aspect) {
  const depth = 0.58;
  const fov = THREE.MathUtils.degToRad(fovDeg);
  const h = 2 * Math.tan(fov / 2) * depth;
  const w = h * aspect;
  const origin = new THREE.Vector3(0, 0, 0);
  const corners = [
    new THREE.Vector3(-w / 2, -h / 2, -depth),
    new THREE.Vector3(w / 2, -h / 2, -depth),
    new THREE.Vector3(w / 2, h / 2, -depth),
    new THREE.Vector3(-w / 2, h / 2, -depth),
  ];
  const pts = [];
  for (const c of corners) pts.push(origin, c);
  for (let i = 0; i < 4; i += 1) pts.push(corners[i], corners[(i + 1) % 4]);

  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const lines = new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({ color: 0x5ba6ff })
  );

  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.055, 0.13, 14),
    new THREE.MeshBasicMaterial({ color: 0x3b82f6 })
  );
  cone.rotation.x = -Math.PI / 2;
  cone.position.set(0, 0.11, 0.02);

  const group = new THREE.Group();
  group.add(lines);
  group.add(cone);
  return group;
}

function rebuildFrustum() {
  while (frustumGroup.children.length) {
    const child = frustumGroup.children[0];
    frustumGroup.remove(child);
    child.traverse?.((n) => {
      n.geometry?.dispose?.();
      n.material?.dispose?.();
    });
  }
  frustumGroup.add(makeFrustum(params().fov, imageAspect));
  placeFrustum(currentAngle);
}

function placeFrustum(angle) {
  const { elevation, radius } = params();
  const look = new THREE.Vector3(0, imagePlane ? imagePlane.position.y : 0.6, 0);
  const pos = cameraXYZ(angle, elevation, radius);
  frustumGroup.position.copy(pos);
  frustumGroup.lookAt(look);
  currentAngle = angle;
  angleLabel.textContent = `${Math.round(angle)}°`;
  angleSlider.value = String(angle);
}

function setImageTexture(url, width, height) {
  imageAspect = width / Math.max(height, 1);
  const loader = new THREE.TextureLoader();
  loader.setCrossOrigin("anonymous");
  loader.load(url, (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    const h = 1.25;
    const w = h * imageAspect;
    while (imageGroup.children.length) {
      const child = imageGroup.children[0];
      imageGroup.remove(child);
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    }
    const border = new THREE.Mesh(
      new THREE.PlaneGeometry(w + 0.03, h + 0.03),
      new THREE.MeshBasicMaterial({ color: 0xb7d6ff, side: THREE.DoubleSide })
    );
    const photo = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })
    );
    photo.position.z = 0.002;
    imagePlane = photo;
    imagePlane.position.y = h / 2;
    border.position.y = h / 2;
    imageGroup.add(border);
    imageGroup.add(photo);
    shadow.scale.set(w * 1.05, 1, 0.55);
    rebuildFrustum();
    emptyEl.classList.add("hidden");
    angleHud.hidden = false;
  });
}

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  renderer.setSize(w, h, false);
  viewCam.aspect = w / Math.max(h, 1);
  viewCam.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, viewCam);
}

function setStatus(text, kind = "") {
  statusEl.hidden = false;
  statusEl.className = `status ${kind}`.trim();
  statusEl.textContent = text;
}

function formatElapsed(ms) {
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function frameForAngle(angle) {
  if (!frames.length) return null;
  return frames.reduce((best, f) => {
    if (!best) return f;
    return Math.abs(f.angle - angle) < Math.abs(best.angle - angle) ? f : best;
  }, null);
}

function showPreview(frame) {
  if (!frame || !jobId) {
    viewPreview.hidden = true;
    return;
  }
  viewPreview.hidden = false;
  viewPreviewImg.src = `/api/orbit/jobs/${jobId}/frames/${frame.index}`;
  viewPreviewCap.textContent =
    frame.source === "original" ? "Original · 0°" : `Generated · ${Math.round(frame.angle)}°`;
}

function renderFilmstrip() {
  filmstrip.hidden = frames.length === 0;
  filmstrip.innerHTML = "";
  for (const frame of frames) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "strip-item";
    btn.innerHTML = `<img alt="" src="/api/orbit/jobs/${jobId}/frames/${frame.index}" /><span>${Math.round(frame.angle)}°</span>`;
    btn.addEventListener("click", () => {
      placeFrustum(frame.angle);
      showPreview(frame);
    });
    filmstrip.appendChild(btn);
  }
}

["increment", "start-deg", "end-deg", "elevation", "radius", "fov"].forEach((id) => {
  document.getElementById(id).addEventListener("input", () => {
    refreshViewCount();
    rebuildFrustum();
  });
});

providerEl.addEventListener("change", () => {
  modelEl.value = DEFAULT_MODELS[providerEl.value] || modelEl.value;
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  fileLabel.textContent = file ? file.name : "Drop a still image";
  if (!file) return;
  if (localImageUrl) URL.revokeObjectURL(localImageUrl);
  localImageUrl = URL.createObjectURL(file);
  const probe = new Image();
  probe.onload = () => setImageTexture(localImageUrl, probe.width, probe.height);
  probe.src = localImageUrl;
  frames = [];
  jobId = null;
  filmstrip.hidden = true;
  viewPreview.hidden = true;
  hintEl.textContent = "Drag to orbit the diagram · slider moves the camera";
});

angleSlider.addEventListener("input", () => {
  const angle = Number(angleSlider.value);
  placeFrustum(angle);
  showPreview(frameForAngle(angle));
});

async function pollJob(id) {
  const started = Date.now();
  for (;;) {
    const res = await fetch(`/api/orbit/jobs/${id}`);
    if (!res.ok) throw new Error(`Status check failed (${res.status})`);
    const data = await res.json();
    const elapsed = formatElapsed(Date.now() - started);
    setStatus(`${(data.message || data.status || "Working").trim()} · ${elapsed}`);
    if (Array.isArray(data.meta?.frames)) {
      frames = data.meta.frames;
      renderFilmstrip();
      showPreview(frameForAngle(currentAngle));
    }
    if (data.status === "succeeded") {
      metaWrap.hidden = false;
      metaEl.textContent = JSON.stringify(data.meta ?? data, null, 2);
      frames = data.meta?.frames || frames;
      renderFilmstrip();
      showPreview(frameForAngle(currentAngle));
      setStatus(`Ready — ${frames.length} views`, "ok");
      return;
    }
    if (data.status === "failed") {
      throw new Error(data.message || "Orbit generation failed");
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) return;

  const p = params();
  const angles = plannedAngles();
  if (!angles.length) {
    setStatus("No views in that range", "error");
    return;
  }

  submitBtn.disabled = true;
  metaWrap.hidden = true;
  setStatus(`Uploading… ${angles.length} views queued`);

  try {
    const body = new FormData();
    body.append("file", file);
    body.append("increment_deg", String(p.increment));
    body.append("start_deg", String(p.start));
    body.append("end_deg", String(p.end));
    body.append("elevation_deg", String(p.elevation));
    body.append("radius", String(p.radius));
    body.append("fov_deg", String(p.fov));
    body.append("provider", p.provider);
    body.append("model", p.model);
    body.append("extra_prompt", p.extra);
    if (p.apiKey) body.append("api_key", p.apiKey);

    const res = await fetch("/api/orbit/jobs", { method: "POST", body });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `Upload failed (${res.status})`);
    }
    const created = await res.json();
    jobId = created.id;
    setStatus("Queued — generating views one angle at a time");
    await pollJob(jobId);
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), "error");
  } finally {
    submitBtn.disabled = false;
  }
});

window.addEventListener("resize", resize);
refreshViewCount();
resize();
animate();
