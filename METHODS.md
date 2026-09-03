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

Default: **10°** steps from 0° to 360° (36 views). Increment, start/end, elevation, radius, FOV, provider, model, and extra prompt are all editable.

- **Gemini:** `gemini-2.5-flash-image` (image + instruction)
- **OpenAI:** `gpt-image-1` (image edit)

Angle 0° is the uploaded photo. Other angles hit the API. Paste an API key in the form, or set `GEMINI_API_KEY` / `OPENAI_API_KEY` on the Modal app.

The 3D viewport is only a pose diagram (grid, image plane, blue frustum). It is not a reconstructed mesh.
