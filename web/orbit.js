import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { deleteOrbit, getOrbit, listOrbits, orbitSummary, saveOrbit } from "/assets/orbit_store.js";
import { downloadOrbitZip } from "/assets/orbit_zip.js";

const PARAM_KEY = "scene-space-orbit-params";
const LAST_ID_KEY = "scene-space-orbit-last";

const form = document.getElementById("orbit-form");
const fileInput = document.getElementById("file");
const fileLabel = document.getElementById("file-label");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const metaWrap = document.getElementById("meta-wrap");
const emptyEl = document.getElementById("empty");
const hintEl = document.getElementById("hint");
const submitBtn = document.getElementById("submit");
const downloadBtn = document.getElementById("download-orbit");
const host = document.getElementById("orbit-host");
const viewportEl = document.getElementById("orbit-viewport");
const angleHud = document.getElementById("angle-hud");
const angleLabel = document.getElementById("angle-label");
const angleSlider = document.getElementById("angle-slider");
const viewCountEl = document.getElementById("view-count");
const filmstrip = document.getElementById("filmstrip");
const providerEl = document.getElementById("provider");
const modelEl = document.getElementById("model");
const keyStatusEl = document.getElementById("key-status");
const apiKeyWrap = document.getElementById("api-key-wrap");
const savedWrap = document.getElementById("saved-wrap");
const savedList = document.getElementById("saved-list");
const orbitFocus = document.getElementById("orbit-focus");
const focusImg = document.getElementById("focus-img");
const focusAngle = document.getElementById("focus-angle");
const focusSource = document.getElementById("focus-source");

let serverProviders = { gemini: false, openai: false };

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
let originalBlob = null;
let originalName = "";
let sessionId = null;
let jobId = null;
let frames = [];
let currentAngle = 0;
let saveTimer = 0;
let wheelLock = 0;
let mapToken = 0;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

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
    contextMode: document.getElementById("context-mode").value,
    orderMode: document.getElementById("order-mode").value,
  };
}

function persistableParams() {
  const p = params();
  delete p.apiKey;
  return p;
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

function persistParams() {
  try {
    localStorage.setItem(PARAM_KEY, JSON.stringify(persistableParams()));
  } catch {
    /* quota / private mode */
  }
}

function restoreParams() {
  try {
    const saved = JSON.parse(localStorage.getItem(PARAM_KEY) || "null");
    if (!saved) return;
    const map = {
      increment: "increment",
      elevation: "elevation",
      start: "start-deg",
      end: "end-deg",
      radius: "radius",
      fov: "fov",
    };
    for (const [key, id] of Object.entries(map)) {
      if (saved[key] != null) document.getElementById(id).value = saved[key];
    }
    if (saved.provider) providerEl.value = saved.provider;
    if (saved.model) modelEl.value = saved.model;
    if (saved.extra != null) document.getElementById("extra-prompt").value = saved.extra;
    if (saved.contextMode) document.getElementById("context-mode").value = saved.contextMode;
    if (saved.orderMode) document.getElementById("order-mode").value = saved.orderMode;
  } catch {
    /* ignore bad cache */
  }
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

function frameSrc(frame) {
  return frame?.objectUrl || localImageUrl || "";
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

function setPlaneMap(url) {
  if (!url) return;
  const token = (mapToken += 1);
  const loader = new THREE.TextureLoader();
  loader.setCrossOrigin("anonymous");
  loader.load(url, (tex) => {
    if (token !== mapToken) {
      tex.dispose();
      return;
    }
    tex.colorSpace = THREE.SRGBColorSpace;
    if (!imagePlane?.material) return;
    const old = imagePlane.material.map;
    imagePlane.material.map = tex;
    imagePlane.material.needsUpdate = true;
    old?.dispose?.();
  });
}

function showViewing(on) {
  viewportEl.classList.toggle("is-viewing", on);
  orbitFocus.hidden = !on;
  emptyEl.hidden = on;
  emptyEl.classList.toggle("hidden", on);
  angleHud.hidden = !on;
  if (on) {
    hintEl.textContent = "Scroll or use the slider to step through angles";
  }
}

function showFocus(frame) {
  const url = frameSrc(frame) || localImageUrl;
  if (!url) {
    showViewing(false);
    return;
  }
  showViewing(true);
  focusImg.src = url;
  const angle = frame ? frame.angle : currentAngle;
  focusAngle.textContent = `${Math.round(angle)}°`;
  focusSource.textContent = frame
    ? frame.source === "original"
      ? "Original"
      : "Generated"
    : "Original";
  setPlaneMap(url);
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
    showFocus(frameForAngle(currentAngle));
    resize();
  });
}

function loadStill(blob, name, angle = 0) {
  if (localImageUrl) URL.revokeObjectURL(localImageUrl);
  originalBlob = blob;
  originalName = name || "still.jpg";
  localImageUrl = URL.createObjectURL(blob);
  fileLabel.textContent = originalName;
  fileInput.removeAttribute("required");
  const probe = new Image();
  probe.onload = () => {
    currentAngle = angle;
    setImageTexture(localImageUrl, probe.width, probe.height);
  };
  probe.src = localImageUrl;
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

function scrubAngles() {
  if (frames.length) {
    return [...frames].sort((a, b) => a.angle - b.angle).map((f) => f.angle);
  }
  return plannedAngles();
}

function selectAngle(angle) {
  placeFrustum(angle);
  const frame = frameForAngle(angle);
  showFocus(frame);
  highlightFilmstrip(frame);
}

function stepAngle(dir) {
  const angles = scrubAngles();
  if (!angles.length) return;
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < angles.length; i += 1) {
    const dist = Math.abs(angles[i] - currentAngle);
    if (dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  }
  const next = angles[(best + dir + angles.length) % angles.length];
  selectAngle(next);
}

function highlightFilmstrip(frame) {
  const index = frame?.index;
  for (const btn of filmstrip.querySelectorAll(".strip-item")) {
    btn.classList.toggle("is-current", Number(btn.dataset.index) === index);
  }
}

function renderFilmstrip() {
  filmstrip.hidden = frames.length === 0;
  downloadBtn.hidden = frames.length === 0;
  filmstrip.innerHTML = "";
  const current = frameForAngle(currentAngle);
  for (const frame of frames) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "strip-item";
    btn.dataset.index = String(frame.index);
    if (current && current.index === frame.index) btn.classList.add("is-current");
    const src = frameSrc(frame);
    btn.innerHTML = `<img alt="" src="${src}" /><span>${Math.round(frame.angle)}°</span>`;
    btn.addEventListener("click", () => selectAngle(frame.angle));
    filmstrip.appendChild(btn);
  }
}

function mergeFrames(incoming) {
  const prev = new Map(frames.map((f) => [Math.round(f.angle), f]));
  frames = incoming.map((frame) => {
    const old = prev.get(Math.round(frame.angle));
    if (old?.blob) {
      return { ...frame, blob: old.blob, objectUrl: old.objectUrl };
    }
    return { ...frame };
  });
}

async function hydrateFrame(frame) {
  if (frame.objectUrl) return frame;
  if (frame.blob) {
    frame.objectUrl = URL.createObjectURL(frame.blob);
    return frame;
  }
  if (!jobId || frame.index == null) return frame;
  const res = await fetch(`/api/orbit/jobs/${jobId}/frames/${frame.index}`);
  if (!res.ok) return frame;
  frame.blob = await res.blob();
  frame.objectUrl = URL.createObjectURL(frame.blob);
  return frame;
}

async function hydrateIncoming(incoming) {
  mergeFrames(incoming);
  await Promise.all(frames.map((frame) => hydrateFrame(frame)));
  renderFilmstrip();
  selectAngle(currentAngle);
  scheduleSave();
}

function currentSessionId() {
  return sessionId || jobId || "draft";
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => {
    persistCurrent().catch((err) => console.warn("Orbit save failed", err));
  }, 250);
}

async function persistCurrent() {
  if (!originalBlob) return;
  const id = currentSessionId();
  sessionId = id;
  const record = {
    id,
    jobId,
    fileName: originalName,
    params: persistableParams(),
    currentAngle,
    original: originalBlob,
    frames: frames
      .filter((f) => f.blob)
      .map((f) => ({
        index: f.index,
        angle: f.angle,
        source: f.source,
        context_angles: f.context_angles || [],
        file: f.file,
        blob: f.blob,
      })),
  };
  await saveOrbit(record);
  try {
    localStorage.setItem(LAST_ID_KEY, id);
  } catch {
    /* ignore */
  }
  await refreshSavedList();
}

function revokeFrameUrls(list) {
  for (const frame of list) {
    if (frame.objectUrl) URL.revokeObjectURL(frame.objectUrl);
  }
}

async function applyRecord(record) {
  if (!record?.original) return;
  revokeFrameUrls(frames);
  sessionId = record.id;
  jobId = record.jobId || record.id;
  originalName = record.fileName || "still.jpg";
  frames = (record.frames || []).map((f) => ({
    ...f,
    objectUrl: f.blob ? URL.createObjectURL(f.blob) : "",
  }));
  if (record.params) {
    localStorage.setItem(PARAM_KEY, JSON.stringify(record.params));
    restoreParams();
    refreshViewCount();
  }
  loadStill(record.original, originalName, record.currentAngle ?? 0);
  renderFilmstrip();
  selectAngle(record.currentAngle ?? 0);
  persistParams();
  setStatus(
    frames.length ? `Restored ${frames.length} views from this browser` : "Restored still from this browser",
    "ok"
  );
}

async function refreshSavedList() {
  const rows = await listOrbits();
  savedList.innerHTML = "";
  savedWrap.hidden = rows.length === 0;
  for (const row of rows) {
    const info = orbitSummary(row);
    const li = document.createElement("li");
    li.className = "saved-item";
    li.innerHTML = `
      <button type="button" class="saved-open" data-id="${escapeHtml(info.id)}">
        <strong>${escapeHtml(info.title)}</strong>
        <span>${escapeHtml(info.detail)}</span>
      </button>
      <button type="button" class="saved-delete" data-id="${escapeHtml(info.id)}" aria-label="Delete saved orbit">×</button>
    `;
    savedList.appendChild(li);
  }
}

["increment", "start-deg", "end-deg", "elevation", "radius", "fov"].forEach((id) => {
  document.getElementById(id).addEventListener("input", () => {
    refreshViewCount();
    rebuildFrustum();
    persistParams();
    scheduleSave();
  });
});

["context-mode", "order-mode", "provider", "model", "extra-prompt"].forEach((id) => {
  document.getElementById(id).addEventListener("change", persistParams);
});

function refreshKeyStatus() {
  const ready = [];
  if (serverProviders.gemini) ready.push("Gemini");
  if (serverProviders.openai) ready.push("OpenAI");
  if (ready.length) {
    keyStatusEl.textContent = `Using ${ready.join(" + ")} keys from the server`;
    apiKeyWrap.hidden = true;
  } else {
    keyStatusEl.textContent = "No server keys — paste an API key below";
    apiKeyWrap.hidden = false;
  }
}

async function loadServerKeys() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    serverProviders = data.providers || serverProviders;
  } catch (err) {
    console.warn("Could not read server key status", err);
  }
  refreshKeyStatus();
}

providerEl.addEventListener("change", () => {
  modelEl.value = DEFAULT_MODELS[providerEl.value] || modelEl.value;
  persistParams();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  revokeFrameUrls(frames);
  frames = [];
  jobId = null;
  sessionId = `draft-${Date.now()}`;
  filmstrip.hidden = true;
  downloadBtn.hidden = true;
  loadStill(file, file.name, params().start);
  persistParams();
  scheduleSave();
});

angleSlider.addEventListener("input", () => {
  selectAngle(Number(angleSlider.value));
  scheduleSave();
});

viewportEl.addEventListener(
  "wheel",
  (event) => {
    if (orbitFocus.hidden) return;
    if (event.target.closest(".filmstrip, #orbit-host")) return;
    event.preventDefault();
    const now = performance.now();
    if (now < wheelLock) return;
    if (Math.abs(event.deltaY) < 4) return;
    wheelLock = now + 90;
    stepAngle(event.deltaY > 0 ? 1 : -1);
    scheduleSave();
  },
  { passive: false }
);

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
    return;
  }
  if (event.key === "ArrowRight" || event.key === "ArrowDown") {
    event.preventDefault();
    stepAngle(1);
  }
  if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
    event.preventDefault();
    stepAngle(-1);
  }
});

savedList.addEventListener("click", async (event) => {
  const del = event.target.closest(".saved-delete");
  if (del) {
    await deleteOrbit(del.dataset.id);
    if (sessionId === del.dataset.id) {
      sessionId = null;
    }
    await refreshSavedList();
    return;
  }
  const open = event.target.closest(".saved-open");
  if (!open) return;
  const record = await getOrbit(open.dataset.id);
  if (record) await applyRecord(record);
});

downloadBtn.addEventListener("click", async () => {
  if (!frames.length) return;
  downloadBtn.disabled = true;
  try {
    await Promise.all(frames.map((frame) => hydrateFrame(frame)));
    await downloadOrbitZip({
      frames,
      params: persistableParams(),
      fileName: originalName,
    });
  } catch (err) {
    setStatus(err.message || String(err), "error");
  } finally {
    downloadBtn.disabled = false;
  }
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
      await hydrateIncoming(data.meta.frames);
    }
    if (data.status === "succeeded") {
      metaWrap.hidden = false;
      metaEl.textContent = JSON.stringify(data.meta ?? data, null, 2);
      if (Array.isArray(data.meta?.frames)) {
        await hydrateIncoming(data.meta.frames);
      }
      setStatus(`Ready — ${frames.length} views · saved in this browser`, "ok");
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
  const file = fileInput.files?.[0] || originalBlob;
  if (!file) {
    setStatus("Upload a still first", "error");
    return;
  }

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
    body.append("file", file, originalName || file.name || "still.jpg");
    body.append("increment_deg", String(p.increment));
    body.append("start_deg", String(p.start));
    body.append("end_deg", String(p.end));
    body.append("elevation_deg", String(p.elevation));
    body.append("radius", String(p.radius));
    body.append("fov_deg", String(p.fov));
    body.append("provider", p.provider);
    body.append("model", p.model);
    body.append("extra_prompt", p.extra);
    body.append("context_mode", p.contextMode);
    body.append("order_mode", p.orderMode);
    if (p.apiKey) body.append("api_key", p.apiKey);

    const res = await fetch("/api/orbit/jobs", { method: "POST", body });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `Upload failed (${res.status})`);
    }
    const created = await res.json();
    jobId = created.id;
    sessionId = created.id;
    setStatus("Queued — generating views one angle at a time");
    await pollJob(jobId);
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), "error");
  } finally {
    submitBtn.disabled = false;
  }
});

async function boot() {
  restoreParams();
  refreshViewCount();
  loadServerKeys();
  resize();
  animate();
  await refreshSavedList();
  let lastId = null;
  try {
    lastId = localStorage.getItem(LAST_ID_KEY);
  } catch {
    lastId = null;
  }
  const record = (lastId && (await getOrbit(lastId))) || (await listOrbits())[0];
  if (record) await applyRecord(record);
}

window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(host);
boot();
