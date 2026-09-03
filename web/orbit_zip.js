const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let c = i;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function dosDateTime(date = new Date()) {
  const year = Math.max(date.getFullYear() - 1980, 0);
  const dosTime =
    (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
  const dosDate =
    (year << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return { dosTime, dosDate };
}

function writeU16(view, offset, value) {
  view.setUint16(offset, value, true);
}

function writeU32(view, offset, value) {
  view.setUint32(offset, value, true);
}

export function frameDownloadName(frame) {
  const deg = String(Math.round(Number(frame.angle) || 0)).padStart(3, "0");
  const source = frame.source === "original" ? "original" : "generated";
  return `${deg}deg_${source}.jpg`;
}

export function frameCaption(frame) {
  const deg = `${Math.round(Number(frame.angle) || 0)}°`;
  const source = frame.source === "original" ? "Original" : "Generated";
  const ctx = Array.isArray(frame.context_angles) && frame.context_angles.length
    ? ` · ctx ${frame.context_angles.map((a) => `${Math.round(a)}°`).join(", ")}`
    : "";
  return `${deg} · ${source}${ctx}`;
}

export async function labeledJpeg(blob, caption) {
  const bitmap = await createImageBitmap(blob);
  const bar = 40;
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height + bar;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#111111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0);
  ctx.fillStyle = "#111111";
  ctx.fillRect(0, bitmap.height, canvas.width, bar);
  ctx.fillStyle = "#f4f4f4";
  ctx.font = `600 ${Math.max(14, Math.round(bitmap.width / 42))}px "DM Sans", sans-serif`;
  ctx.textBaseline = "middle";
  ctx.fillText(caption, 16, bitmap.height + bar / 2);
  bitmap.close?.();
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (out) => (out ? resolve(out) : reject(new Error("Could not label frame"))),
      "image/jpeg",
      0.92
    );
  });
}

export function buildZip(files) {
  const encoder = new TextEncoder();
  const { dosTime, dosDate } = dosDateTime();
  const locals = [];
  const centrals = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const data = file.data;
    const crc = crc32(data);
    const local = new Uint8Array(30 + nameBytes.length + data.length);
    const view = new DataView(local.buffer);
    writeU32(view, 0, 0x04034b50);
    writeU16(view, 4, 20);
    writeU16(view, 6, 0);
    writeU16(view, 8, 0);
    writeU16(view, 10, dosTime);
    writeU16(view, 12, dosDate);
    writeU32(view, 14, crc);
    writeU32(view, 18, data.length);
    writeU32(view, 22, data.length);
    writeU16(view, 26, nameBytes.length);
    writeU16(view, 28, 0);
    local.set(nameBytes, 30);
    local.set(data, 30 + nameBytes.length);
    locals.push(local);

    const central = new Uint8Array(46 + nameBytes.length);
    const cview = new DataView(central.buffer);
    writeU32(cview, 0, 0x02014b50);
    writeU16(cview, 4, 20);
    writeU16(cview, 6, 20);
    writeU16(cview, 8, 0);
    writeU16(cview, 10, 0);
    writeU16(cview, 12, dosTime);
    writeU16(cview, 14, dosDate);
    writeU32(cview, 16, crc);
    writeU32(cview, 20, data.length);
    writeU32(cview, 24, data.length);
    writeU16(cview, 28, nameBytes.length);
    writeU16(cview, 30, 0);
    writeU16(cview, 32, 0);
    writeU16(cview, 34, 0);
    writeU16(cview, 36, 0);
    writeU32(cview, 38, 0);
    writeU32(cview, 42, offset);
    central.set(nameBytes, 46);
    centrals.push(central);
    offset += local.length;
  }

  const centralSize = centrals.reduce((sum, part) => sum + part.length, 0);
  const end = new Uint8Array(22);
  const eview = new DataView(end.buffer);
  writeU32(eview, 0, 0x06054b50);
  writeU16(eview, 4, 0);
  writeU16(eview, 6, 0);
  writeU16(eview, 8, files.length);
  writeU16(eview, 10, files.length);
  writeU32(eview, 12, centralSize);
  writeU32(eview, 16, offset);
  writeU16(eview, 20, 0);

  return new Blob([...locals, ...centrals, end], { type: "application/zip" });
}

export async function downloadOrbitZip({ frames, params, fileName }) {
  const labeled = [];
  for (const frame of frames) {
    if (!frame.blob) continue;
    const name = frameDownloadName(frame);
    const jpeg = await labeledJpeg(frame.blob, frameCaption(frame));
    labeled.push({
      name,
      data: new Uint8Array(await jpeg.arrayBuffer()),
      meta: {
        file: name,
        angle: frame.angle,
        source: frame.source,
        context_angles: frame.context_angles || [],
      },
    });
  }
  if (!labeled.length) {
    throw new Error("No saved frames to download");
  }

  const manifest = {
    app: "scene-space",
    kind: "orbit",
    source: fileName || "still",
    params: params || {},
    n_views: labeled.length,
    frames: labeled.map((item) => item.meta),
  };
  const manifestBytes = new TextEncoder().encode(`${JSON.stringify(manifest, null, 2)}\n`);
  const zip = buildZip([
    ...labeled.map((item) => ({ name: item.name, data: item.data })),
    { name: "manifest.json", data: manifestBytes },
  ]);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(zip);
  a.download = `orbit-${stamp}.zip`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
