# Methods

Three routes. Reconstruct is static 3D. Orbit invents views from one still. Video 4D fits a time-varying scene.

| Method | Route | Input | Output | Compute |
|--------|-------|-------|--------|---------|
| **Reconstruct** | `/` | Walkthrough video / multi-view photos | Navigable 3DGS (`gaussians.ply`) | Modal A100 · recon3d (VGGT + gsplat) |
| **Orbit views** | `/orbit` | One still | Novel views every N° around the subject | CPU · Gemini or OpenAI image model |
| **Video 4D** | `/4d` | Monocular video (motion in the scene) | Time-sliced 4D-GS PLYs + slider | Modal A100 · VGGT + deform 4D-GS |

## Reconstruct

Photogrammetry-style: estimate cameras, train Gaussians on *your* frames. Best when the camera actually moves through the space. See `PIPELINE.md` and `MODELS.md`.

## Orbit views

Experiment: keep the original photo as a plane in a 3D diagram, move a camera frustum around it, and ask an image model to paint what that pose should look like.

Default: **10°** steps from 0° to 360° (36 views). Increment, start/end, elevation, radius, FOV, context, order, provider, model, and extra prompt are all editable.

**Context (default: adaptive).** A scene brief is written from the still first (front / right / back / left). Nearby angles still get the original photo. Past the **front lock** (default 60°) the original is dropped and only side/back neighbors are sent, so 180° is not a warped copy of the front. “Original + nearest” is the old always-lock behavior.

**Order (default: cardinal first).** Generate 0°, then 90°, 270°, 180°, then fill the steps in between. That plants real side and back anchors before interpolating. Bidirectional (`0, 10, 350…`) and sequential are still available.

Each view prompt is a change/preserve spec: camera heading changes; people (faces, clothes, placement), small objects, furniture, and lighting stay locked. OpenAI calls use `quality=high` (and `input_fidelity=high` on GPT Image 1.x).

Pick one image model from the dropdown (priced/realism order). Defaults: **GPT Image 2**, then **Nano Banana 2**.

- **OpenAI:** `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini`
- **Gemini:** `gemini-3.1-flash-image` (Nano Banana 2), `gemini-3-pro-image` (Pro), `gemini-2.5-flash-image`, `gemini-3.1-flash-lite-image`

Angle 0° is the uploaded photo. Other angles hit the API. Keys come from local `.env` (`GEMINI_API_KEY`, `OPENAI_API_KEY`), loaded into Modal at deploy. They are never sent to the browser.

The main view is the flat still. Scroll, arrow keys, or the slider step through angles; the photo and the small 3D frustum both follow. Generated frames stay in IndexedDB on this browser (reload keeps them). **Download labeled orbit** writes a zip: `{angle}deg_{original|generated}.jpg` with a caption bar, plus `manifest.json`.

The 3D viewport is only a pose diagram (grid, image plane, blue frustum). It is not a reconstructed mesh.

## Video 4D

Experiment: take a clip, estimate cameras with VGGT, then train canonical 3D Gaussians plus a deformation MLP (xyz, t) → Δxyz, Δquat, Δscale. That is the reconstruct half of [CAT4D](https://cat-4d.github.io/) / [4D-GS](https://github.com/hustvl/4DGaussians). CAT4D’s multi-view video diffusion is not public, so this route does **not** invent extra viewpoints.

Best clips: a few seconds, either a slow walk with people moving, or a mostly locked camera with motion in frame. A single still will be rejected.

Defaults: **24** frames at **4 fps**, **8000** steps (about a third static warmup with densify, then deform), **8** exported times. The viewer loads one PLY per time; Play steps through them. `gaussians.ply` is the middle slice so the existing splat URL still works.
