# Methods

Two ways to get a 3D-ish scene from photos. They are separate routes.

| Method | Route | Input | Output | Compute |
|--------|-------|-------|--------|---------|
| **Reconstruct** | `/` | Walkthrough video / multi-view photos | Navigable 3DGS (`gaussians.ply`) | Modal A100 · recon3d (VGGT + gsplat) |
| **Orbit views** | `/orbit` | One still | Novel views every N° around the subject | CPU · Gemini or OpenAI image model |

## Reconstruct

Photogrammetry-style: estimate cameras, train Gaussians on *your* frames. Best when the camera actually moves through the space. See `PIPELINE.md` and `MODELS.md`.

## Orbit views

Experiment: keep the original photo as a plane in a 3D diagram, move a camera frustum around it, and ask an image model to paint what that pose should look like.

Default: **10°** steps from 0° to 360° (36 views). Increment, start/end, elevation, radius, FOV, context, order, provider, model, and extra prompt are all editable.

**Context (default: original + nearest).** Each generated view is conditioned on the original photo plus the spatially nearest already-made views. That keeps identity locked to the still and local continuity with neighbors. Alternatives: original + previous (smoother chain, more drift) or original only.

**Order (default: bidirectional).** Generate ±increment from the original (`0, 10, 350, 20, 340…`) so the back view is fewer hops from the photo than a single 0→350 walk.

- **Gemini:** `gemini-2.5-flash-image` (image + instruction)
- **OpenAI:** `gpt-image-1` (image edit)

Angle 0° is the uploaded photo. Other angles hit the API. Keys come from local `.env` (`GEMINI_API_KEY`, `OPENAI_API_KEY`), loaded into Modal at deploy. They are never sent to the browser.

The main view is the flat still. Scroll, arrow keys, or the slider step through angles; the photo and the small 3D frustum both follow. Generated frames stay in IndexedDB on this browser (reload keeps them). **Download labeled orbit** writes a zip: `{angle}deg_{original|generated}.jpg` with a caption bar, plus `manifest.json`.

The 3D viewport is only a pose diagram (grid, image plane, blue frustum). It is not a reconstructed mesh.
