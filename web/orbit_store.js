const DB_NAME = "scene-space-orbit";
const DB_VERSION = 1;
const STORE = "orbits";
const MAX_ORBITS = 8;

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error("IndexedDB open failed"));
  });
}

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function listOrbits() {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, "readonly");
    const rows = (await requestToPromise(tx.objectStore(STORE).getAll())) || [];
    rows.sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
    return rows;
  } finally {
    db.close();
  }
}

export async function getOrbit(id) {
  if (!id) return null;
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, "readonly");
    return (await requestToPromise(tx.objectStore(STORE).get(id))) || null;
  } finally {
    db.close();
  }
}

export async function deleteOrbit(id) {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, "readwrite");
    await requestToPromise(tx.objectStore(STORE).delete(id));
  } finally {
    db.close();
  }
}

export async function saveOrbit(record) {
  const payload = {
    ...record,
    savedAt: Date.now(),
  };
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    await requestToPromise(store.put(payload));
    const rows = (await requestToPromise(store.getAll())) || [];
    rows.sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
    const extra = rows.slice(MAX_ORBITS);
    for (const row of extra) {
      await requestToPromise(store.delete(row.id));
    }
  } finally {
    db.close();
  }
  return payload;
}

export function orbitSummary(record) {
  const n = record.frames?.length || 0;
  const when = record.savedAt ? new Date(record.savedAt) : null;
  const time = when
    ? when.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : "Saved";
  const name = record.fileName || "Orbit";
  return {
    id: record.id,
    title: name,
    detail: n ? `${n} views · ${time}` : `Still only · ${time}`,
  };
}
