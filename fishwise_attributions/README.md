# FishWise Professional — Chaetodontidae attributions

Gentle, metadata-only scrape of FishWise Pro to collect **image attributions** (species, photographer, URLs) for Chaetodontidae. No images are downloaded.

- **`scrape_attributions.py`** — One request per species (`/Pictures/?SID={sid}&Find=0`), parses HTML for `alt="Species - Photographer"` and image URLs. Writes `attributions.csv`.
- **`fishwise_attributions.qmd`** — Quarto doc describing the **query design** and **scraping strategy**; run after scraping to summarize the CSV.

## Run the scrape

```bash
cd fishwise_attributions
python scrape_attributions.py --delay 2.5
```

Optional: `--dry-run` (print URLs only), `--delay 3` (slower).

## Render the doc

```bash
quarto render fishwise_attributions.qmd
```

SIDs come from `data/fishwise/fishwise_sid_cache.json` (see `scripts/download/update_fishwise_sid_cache.py`).
