# Where the FishWise image metadata is

**Location:**  
`/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/fishwise_attributions/attributions.csv`

**How it was produced:**  
The script `scrape_attributions.py` in this folder requests one FishWise gallery page per Chaetodontidae species (`/Pictures/?SID={sid}&Find=0`), parses the HTML for `<img>` tags, and writes one row per image. No images are downloaded; only metadata is collected.

**Schema (columns):**

| Column         | Description |
|----------------|-------------|
| `species`     | Scientific name (e.g. *Chaetodon auriga*) |
| `photographer` | Attribution string (e.g. John Randall, Ken Graham) |
| `filename`    | Original filename on FishWise (e.g. `061570F000024W000001.jpg`) |
| `thumb_url`   | Thumbnail URL: `https://www.fishwisepro.com/pics/{JPG\|GIF}/TN/TN{filename}` |
| `full_url`    | **Full-size image URL** (use this for downloads): `https://www.fishwisepro.com/pics/{JPG\|GIF}/{filename}` |
| `format`      | `JPG` or `GIF` |
| `sid`         | FishWise species ID (integer) |

**Important:**  
- Use **`full_url`** as the download URL for each image.  
- Rows may repeat species and photographer; each row is one image.  
- Filenames follow a pattern like `{6-digit SID}F...W....{ext}`; the first 6 digits are the species ID (zero-padded).

**Related project files:**  
- Species list (130 Chaetodontidae): `../data/species_image_summary.csv`  
- SID cache (species → FishWise ID): `../data/fishwise/fishwise_sid_cache.json`  
- Existing download scripts (Bishop Museum, FishBase, etc.): `../scripts/download/` — they use `urllib`, retries, polite delays, and save under project dirs with names like `Genus_species_Source_originalname.ext`.

---

# Prompt for an LLM: FishWise image download strategy

**Copy everything below this line to hand off to an LLM.**

---

You are helping design and implement a **download strategy for FishWise Professional images** in an existing Python project. The project already has **metadata only** — no FishWise images have been downloaded yet.

## Project context

- **Repo root:** `/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026`
- **Goal:** Download Chaetodontidae (butterflyfish) images from FishWise Pro for research, with correct attribution and minimal load on the server.
- **Legal/ethics:** The project has been collecting metadata only “awaiting permission” from FishWise; a download script should be polite (rate limit, identifiable User-Agent), and the implementer should ensure they have or obtain permission before bulk downloading.

## Where the image metadata is

- **File:** `fishwise_attributions/attributions.csv` (under the repo root).
- **Columns:** `species`, `photographer`, `filename`, `thumb_url`, `full_url`, `format`, `sid`.
- **Download URL:** Use the **`full_url`** column. Each row is one image; multiple rows per species and per photographer are normal.
- **Example row:**  
  `species=Chaetodon auriga`, `photographer=John Randall`, `filename=061570F000024W000001.jpg`, `full_url=https://www.fishwisepro.com/pics/JPG/061570F000024W000001.jpg`

## Existing patterns in this repo

- Other image download scripts live in `scripts/download/` (e.g. `download_bishop_museum.py`, `download_additional_images.py`).
- They use **urllib** (no Selenium) for the actual download, with:
  - A **User-Agent** string (e.g. `Mozilla/5.0 (...)` or project-specific).
  - **Retries** (e.g. 2–3 attempts with backoff).
  - **Delay** between requests (e.g. 1–3 seconds) to avoid hammering the server.
  - **Timeout** (e.g. 20–30 s) per request.
- Saved files are named to avoid collisions and preserve attribution, e.g.  
  `Genus_species_FishWise_originalfilename.jpg`  
  and stored in a dedicated directory (e.g. `images_fishwise/` under the repo root or under `data/`).
- A **CSV manifest** is written listing at least: species, local filename, source URL, and optionally photographer and download success.

## What to produce

1. **Download strategy (short doc or checklist)** that covers:
   - Input: read `fishwise_attributions/attributions.csv` and use `full_url` (and optionally `species`, `photographer`, `filename`) for each row.
   - Output directory: e.g. `images_fishwise/` under the repo root; one file per image with a safe, unique name (e.g. species + source + original filename).
   - Rate limiting: delay between requests (e.g. 2–3 s), no parallel requests unless you explicitly recommend and justify it.
   - Retries and failure handling: how many retries, what to log, whether to write a CSV of failed URLs.
   - Skipping already-downloaded files (by path or by tracking in a manifest) so re-runs are idempotent.
   - Optional: progress (e.g. “Downloaded 45/1200”) or progress bar.
   - Attribution: how photographer/species are stored (in filename, in a manifest CSV, or both).

2. **Implementation:** A Python script that implements the strategy. Prefer:
   - **Location:** `scripts/download/download_fishwise_images.py` or `fishwise_attributions/download_fishwise_images.py`.
   - **Input:** Path to `attributions.csv` (default: `fishwise_attributions/attributions.csv` relative to repo root).
   - **Output:**
     - Image files in e.g. `images_fishwise/` (or a path the user can override via CLI).
     - A CSV manifest (e.g. `fishwise_attributions/fishwise_download_manifest.csv` or under `data/fishwise/`) with at least: `species`, `photographer`, `filename`, `local_path`, `full_url`, `download_success`.
   - **CLI:** Optional flags for `--delay`, `--dry-run` (print URLs only), `--limit N` (download only first N rows), and path overrides.
   - Reuse patterns from the repo’s existing download scripts (urllib, retries, delays, naming).

3. **Idempotency:** If a file already exists at the target path, skip downloading it again (or optionally overwrite via a flag).

Do not download images in the LLM environment; only produce the strategy description and the Python script. The user will run the script locally after confirming they have permission to download from FishWise.
