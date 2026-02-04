#!/usr/bin/env python3
"""
Download additional Chaetodontidae fish images from FishBase and iNaturalist.

For each species in the combined species list (~130 species), this script:
1. Queries FishBase's FishPicsList.php API for the exact image filenames
   associated with each species (eliminates code collision problems)
2. Queries iNaturalist API for research-grade observations with photos
3. Deduplicates across ALL image directories using MD5 hashes
4. Produces master inventory and per-species summary CSVs

IMPORTANT: Uses the FishBase FishPicsList.php web service API instead of
blind suffix iteration. This eliminates the code collision problem where
different species share the same 5-letter image code (e.g., Charg =
Chaetodon argentatus AND Chanodichthys argentatus).

Images are saved to:
  - images_fishbase_extra/   (additional FishBase images not in Miyazawa dataset)
  - images_inaturalist/      (iNaturalist research-grade photos)
"""

import os
import re
import csv
import sys
import time
import json
import hashlib
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from io import BytesIO


# ---------- paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TREE_PATH = os.path.join(SCRIPT_DIR, "tree", "Butterflyfish_concat_final.tre")
if not os.path.exists(TREE_PATH):
    TREE_PATH = os.path.join(
        os.path.dirname(SCRIPT_DIR),
        "chaets-divergence-2026",
        "rooted-chaet-tree-vis",
        "Butterflyfish_concat_final.tre",
    )

COMBINED_CSV = os.path.join(SCRIPT_DIR, "combined_species_data.csv")
EXISTING_IMAGE_CSV = os.path.join(SCRIPT_DIR, "chaetodontidae_image_data.csv")
BISHOP_CSV = os.path.join(SCRIPT_DIR, "bishop_museum_image_data.csv")

# Image directories
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
BISHOP_DIR = os.path.join(SCRIPT_DIR, "images_bishop")
FISHBASE_EXTRA_DIR = os.path.join(SCRIPT_DIR, "images_fishbase_extra")
INATURALIST_DIR = os.path.join(SCRIPT_DIR, "images_inaturalist")

# Output files
INVENTORY_CSV = os.path.join(SCRIPT_DIR, "all_images_inventory.csv")
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "species_image_summary.csv")
DEDUP_CSV = os.path.join(SCRIPT_DIR, "dedup_report.csv")

# FishBase API and image base URLs
FISHBASE_API = "https://www.fishbase.se/webservice/photos/FishPicsList.php"
FISHBASE_IMG_BASE = "https://www.fishbase.se/images/species"

# iNaturalist API
INAT_API_BASE = "https://api.inaturalist.org/v1"


# ---------- 1. Parse tree for species ----------
def parse_tree_species(tree_path):
    """Extract Chaetodontidae species names from the NEXUS tree."""
    with open(tree_path) as f:
        content = f.read()

    taxa_match = re.search(r"taxlabels\s+(.*?)\s*;", content, re.DOTALL)
    if not taxa_match:
        raise ValueError("Could not find taxlabels in tree file")

    taxa = taxa_match.group(1).strip().split()

    chaet_genera = {
        "Amphichaetodon", "Chaetodon", "Chelmon", "Chelmonops",
        "Coradion", "Forcipiger", "Hemitaurichthys", "Heniochus",
        "Johnrandallia", "Parachaetodon", "Prognathodes", "Roa",
    }

    species_set = set()
    for taxon in taxa:
        parts = taxon.split("_")
        genus = parts[0].capitalize()
        if genus not in chaet_genera or len(parts) < 2:
            continue
        epithet = parts[1]
        species_name = f"{genus} {epithet}"
        species_set.add(species_name)

    return species_set


# ---------- 2. Read existing image data ----------
def read_existing_fishbase_filenames():
    """Get the set of FishBase image filenames already downloaded in images/ dir."""
    existing_fb = set()
    if os.path.exists(EXISTING_IMAGE_CSV):
        with open(EXISTING_IMAGE_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["source"] == "FishBase":
                    # Extract the FishBase filename from URL, e.g., Chvag_u2.jpg
                    url_file = row["image_url"].split("/")[-1]
                    existing_fb.add(url_file.lower())
    return existing_fb


def get_all_species():
    """Read the combined species list."""
    species_list = []
    with open(COMBINED_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            species_list.append(row["species"])
    return sorted(set(species_list))


# ---------- 3. Download helper ----------
def download_image(url, dest_path, retries=2):
    """Download a single image with retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as response:
                data = response.read()
                # Verify we got image data (not HTML error page)
                if len(data) < 500:
                    return False
                # Check that data looks like an image (JPEG/PNG/GIF magic bytes)
                if not (data[:2] == b'\xff\xd8'       # JPEG
                        or data[:4] == b'\x89PNG'      # PNG
                        or data[:4] == b'GIF8'):        # GIF
                    return False
                with open(dest_path, "wb") as f:
                    f.write(data)
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return False
    return False


# ---------- 4. FishBase API-based downloads ----------
def query_fishbase_api(genus, species):
    """Query FishBase FishPicsList.php API for exact image list.

    Returns list of dicts with keys: filename, url, author, type
    Only returns 'adult' type images (no stamps, diseases, larvae).
    """
    url = f"{FISHBASE_API}?Genus={genus}&Species={species}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        sp_elem = root.find("species")
        if sp_elem is None or sp_elem.get("status") != "valid":
            return []

        images = []
        for pic in root.findall("pictures"):
            pic_type = pic.get("type", "")
            # Only download adult fish photos, not stamps/diseases/larvae
            if pic_type != "adult":
                continue

            picname_elem = pic.find("picname")
            if picname_elem is None or picname_elem.text is None:
                continue
            picname = picname_elem.text.strip()

            author_elem = pic.find("author")
            author = author_elem.text.strip() if author_elem is not None and author_elem.text else ""

            # Use fishbase.se URL (more reliable than fishbase.org)
            img_url = f"{FISHBASE_IMG_BASE}/{picname}"

            images.append({
                "filename": picname,
                "url": img_url,
                "author": author,
                "type": pic_type,
            })

        return images

    except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError, OSError) as e:
        print(f"    API error: {e}")
        return []


def download_fishbase_extra(species_list):
    """Download additional FishBase images using the API for species verification."""
    os.makedirs(FISHBASE_EXTRA_DIR, exist_ok=True)

    # Get already-downloaded FishBase filenames
    existing_fb = read_existing_fishbase_filenames()

    # Also check what's already in the extra directory
    existing_extra = set()
    if os.path.exists(FISHBASE_EXTRA_DIR):
        for f in os.listdir(FISHBASE_EXTRA_DIR):
            existing_extra.add(f)

    results = []
    total_species = len(species_list)

    for idx, species in enumerate(species_list):
        parts = species.split()
        if len(parts) < 2:
            continue
        genus, epithet = parts[0], parts[1]

        print(f"  [{idx+1:3d}/{total_species}] {species}", end=" ", flush=True)

        # Query API for verified image list
        api_images = query_fishbase_api(genus, epithet)

        if not api_images:
            print("-> 0 on FishBase")
            time.sleep(0.3)
            continue

        n_new = 0
        for img_info in api_images:
            fb_filename = img_info["filename"]  # e.g., Charg_u3.jpg

            # Skip if already downloaded in original Miyazawa batch
            if fb_filename.lower() in existing_fb:
                continue

            # Build local filename
            safe_sp = species.replace(" ", "_")
            local_filename = f"{safe_sp}_FishBase_{fb_filename}"

            # Skip if already in extra dir
            if local_filename in existing_extra:
                n_new += 1
                results.append({
                    "species": species,
                    "filename": local_filename,
                    "source": "FishBase_extra",
                    "directory": "images_fishbase_extra",
                    "image_url": img_info["url"],
                    "author": img_info["author"],
                    "download_success": "yes",
                })
                continue

            dest_path = os.path.join(FISHBASE_EXTRA_DIR, local_filename)
            if download_image(img_info["url"], dest_path):
                n_new += 1
                results.append({
                    "species": species,
                    "filename": local_filename,
                    "source": "FishBase_extra",
                    "directory": "images_fishbase_extra",
                    "image_url": img_info["url"],
                    "author": img_info["author"],
                    "download_success": "yes",
                })
            time.sleep(0.15)

        total_for_sp = len(api_images)
        print(f"-> {total_for_sp} total on FishBase, {n_new} new downloads")
        time.sleep(0.3)

    return results


# ---------- 5. iNaturalist downloads ----------
def query_inaturalist(species_name, max_photos=10):
    """Query iNaturalist API for research-grade observations with photos."""
    params = (
        f"taxon_name={species_name.replace(' ', '+')}"
        f"&photos=true"
        f"&quality_grade=research"
        f"&per_page={min(max_photos * 2, 30)}"  # request extra in case some don't match
        f"&order_by=votes"
    )
    url = f"{INAT_API_BASE}/observations?{params}"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Chaetodontidae-image-project)",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        photos = []
        seen_photo_ids = set()
        for obs in data.get("results", []):
            # Verify the taxon matches
            taxon = obs.get("taxon", {})
            taxon_name = taxon.get("name", "")
            if taxon_name.lower() != species_name.lower():
                continue

            for op in obs.get("observation_photos", []):
                photo = op.get("photo", {})
                photo_id = photo.get("id")
                license_code = photo.get("license_code")

                # Skip photos without open licenses
                if not license_code:
                    continue

                if photo_id and photo_id not in seen_photo_ids:
                    seen_photo_ids.add(photo_id)
                    photo_url = photo.get("url", "")
                    large_url = photo_url.replace("/square.", "/large.")
                    photos.append({
                        "photo_id": photo_id,
                        "url": large_url,
                        "license": license_code,
                        "attribution": photo.get("attribution", ""),
                    })
                    if len(photos) >= max_photos:
                        break
            if len(photos) >= max_photos:
                break

        return photos

    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
        print(f"    API error: {e}")
        return []


def download_inaturalist(species_list):
    """Download iNaturalist images for all species."""
    os.makedirs(INATURALIST_DIR, exist_ok=True)

    # Check what's already downloaded
    existing_inat = set()
    if os.path.exists(INATURALIST_DIR):
        for f in os.listdir(INATURALIST_DIR):
            existing_inat.add(f)

    results = []
    total_species = len(species_list)

    for idx, species in enumerate(species_list):
        print(f"  [{idx+1:3d}/{total_species}] {species}", end=" ", flush=True)

        photos = query_inaturalist(species, max_photos=10)

        if not photos:
            print("-> 0 photos found")
            time.sleep(0.5)
            continue

        n_downloaded = 0
        for photo in photos:
            photo_id = photo["photo_id"]
            safe_sp = species.replace(" ", "_")
            local_filename = f"{safe_sp}_iNaturalist_{photo_id}.jpg"
            dest_path = os.path.join(INATURALIST_DIR, local_filename)

            if local_filename in existing_inat or os.path.exists(dest_path):
                n_downloaded += 1
                results.append({
                    "species": species,
                    "filename": local_filename,
                    "source": "iNaturalist",
                    "directory": "images_inaturalist",
                    "image_url": photo["url"],
                    "download_success": "yes",
                    "license": photo.get("license", ""),
                    "attribution": photo.get("attribution", ""),
                })
                continue

            if download_image(photo["url"], dest_path):
                n_downloaded += 1
                results.append({
                    "species": species,
                    "filename": local_filename,
                    "source": "iNaturalist",
                    "directory": "images_inaturalist",
                    "image_url": photo["url"],
                    "download_success": "yes",
                    "license": photo.get("license", ""),
                    "attribution": photo.get("attribution", ""),
                })
            time.sleep(0.1)

        print(f"-> {n_downloaded} photos")
        time.sleep(0.8)  # respect iNat rate limits

    return results


# ---------- 6. MD5 deduplication ----------
def compute_md5(filepath):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_full_inventory():
    """Build inventory of ALL images across all directories, with MD5 hashes."""
    inventory = []
    dirs_and_sources = [
        (IMAGES_DIR, "images"),
        (BISHOP_DIR, "images_bishop"),
        (FISHBASE_EXTRA_DIR, "images_fishbase_extra"),
        (INATURALIST_DIR, "images_inaturalist"),
    ]

    for dirpath, dirname in dirs_and_sources:
        if not os.path.exists(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                continue
            filepath = os.path.join(dirpath, fname)
            if not os.path.isfile(filepath):
                continue

            md5 = compute_md5(filepath)

            # Extract species from filename: Genus_species_Source_...
            parts = fname.split("_")
            if len(parts) >= 2:
                species = f"{parts[0]} {parts[1]}"
            else:
                species = fname

            # Determine source from filename
            if "_FishPix_" in fname:
                source = "FishPix"
            elif "_FishBase_" in fname:
                source = "FishBase"
            elif "_Bishop_" in fname:
                source = "BishopMuseum"
            elif "_iNaturalist_" in fname:
                source = "iNaturalist"
            else:
                source = "unknown"

            inventory.append({
                "species": species,
                "filename": fname,
                "source": source,
                "directory": dirname,
                "filepath": filepath,
                "md5_hash": md5,
                "is_duplicate": "no",
            })

    return inventory


def deduplicate(inventory):
    """Find and mark duplicate images based on MD5 hash.

    Priority: images/ > images_bishop/ > images_fishbase_extra/ > images_inaturalist/
    """
    dir_priority = {
        "images": 0,
        "images_bishop": 1,
        "images_fishbase_extra": 2,
        "images_inaturalist": 3,
    }

    by_hash = {}
    for item in inventory:
        h = item["md5_hash"]
        if h not in by_hash:
            by_hash[h] = []
        by_hash[h].append(item)

    dedup_log = []
    n_dups = 0

    for h, items in by_hash.items():
        if len(items) <= 1:
            continue

        items.sort(key=lambda x: dir_priority.get(x["directory"], 99))

        kept = items[0]
        for dup in items[1:]:
            dup["is_duplicate"] = "yes"
            n_dups += 1
            dedup_log.append({
                "md5_hash": h,
                "kept_file": os.path.join(kept["directory"], kept["filename"]),
                "kept_source": kept["source"],
                "duplicate_file": os.path.join(dup["directory"], dup["filename"]),
                "duplicate_source": dup["source"],
                "species": dup["species"],
            })

    return dedup_log, n_dups


# ---------- 7. Summary table ----------
def build_species_summary(inventory, tree_species):
    """Build per-species summary of image counts by source."""
    sp_data = {}
    for item in inventory:
        if item["is_duplicate"] == "yes":
            continue
        sp = item["species"]
        src = item["source"]
        dirn = item["directory"]
        if sp not in sp_data:
            sp_data[sp] = {
                "fishpix": 0, "fishbase": 0, "bishop": 0,
                "fishbase_extra": 0, "inaturalist": 0,
            }
        if src == "FishPix":
            sp_data[sp]["fishpix"] += 1
        elif src == "FishBase" and dirn == "images":
            sp_data[sp]["fishbase"] += 1
        elif src == "FishBase" and dirn == "images_fishbase_extra":
            sp_data[sp]["fishbase_extra"] += 1
        elif src == "BishopMuseum":
            sp_data[sp]["bishop"] += 1
        elif src == "iNaturalist":
            sp_data[sp]["inaturalist"] += 1

    tree_sp_lower = {s.lower() for s in tree_species}

    rows = []
    for sp in sorted(sp_data.keys()):
        counts = sp_data[sp]
        total = sum(counts.values())

        sp_lower = sp.lower()
        in_tree = sp_lower in tree_sp_lower

        sources = []
        if counts["fishpix"] > 0:
            sources.append("FishPix")
        if counts["fishbase"] > 0:
            sources.append("FishBase")
        if counts["bishop"] > 0:
            sources.append("BishopMuseum")
        if counts["fishbase_extra"] > 0:
            sources.append("FishBase_extra")
        if counts["inaturalist"] > 0:
            sources.append("iNaturalist")

        rows.append({
            "species": sp,
            "in_tree": "yes" if in_tree else "no",
            "n_fishpix": counts["fishpix"],
            "n_fishbase": counts["fishbase"],
            "n_bishop": counts["bishop"],
            "n_fishbase_extra": counts["fishbase_extra"],
            "n_inaturalist": counts["inaturalist"],
            "n_total": total,
            "sources": " + ".join(sources),
        })

    return rows


# ---------- 8. Validate existing images/ directory ----------
def validate_existing_fishbase_images():
    """Check if any images in images/ are from wrong species via API.

    This catches cases like Chpic_u2.jpg being Chaunax pictus
    rather than Chaetodon pictus.
    """
    if not os.path.exists(EXISTING_IMAGE_CSV):
        return []

    print("\n[2b] Validating existing FishBase images against API...")

    flagged = []
    species_checked = set()

    with open(EXISTING_IMAGE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source"] != "FishBase":
                continue
            species = row["species"]
            if species in species_checked:
                continue
            species_checked.add(species)

            parts = species.split()
            if len(parts) < 2:
                continue

            # Query API for this species
            api_images = query_fishbase_api(parts[0], parts[1])
            api_filenames = {img["filename"].lower() for img in api_images}

            # Check each downloaded file against API
            with open(EXISTING_IMAGE_CSV) as f2:
                reader2 = csv.DictReader(f2)
                for row2 in reader2:
                    if row2["species"] != species or row2["source"] != "FishBase":
                        continue
                    url_file = row2["image_url"].split("/")[-1].lower()
                    if url_file not in api_filenames:
                        flagged.append({
                            "species": species,
                            "filename": row2["filename"],
                            "fb_image_file": url_file,
                            "reason": "Not in FishBase API results for this species",
                        })
            time.sleep(0.3)

    if flagged:
        print(f"    WARNING: {len(flagged)} existing images may be wrong species:")
        for f in flagged:
            print(f"      {f['species']}: {f['fb_image_file']} - {f['reason']}")

    return flagged


# ---------- main ----------
def main():
    print("=" * 70)
    print("Additional Chaetodontidae Image Downloader v2")
    print("FishBase API-verified + iNaturalist + MD5 Deduplication")
    print("=" * 70)

    # Parse tree
    print("\n[1] Parsing phylogenetic tree...")
    tree_species = parse_tree_species(TREE_PATH)
    print(f"    Found {len(tree_species)} Chaetodontidae species in tree")

    # Get species list
    print("\n[2] Reading species list...")
    species_list = get_all_species()
    print(f"    {len(species_list)} unique species across all databases")

    # Validate existing images
    flagged = validate_existing_fishbase_images()

    # Download additional FishBase images (API-verified)
    print("\n[3] Downloading additional FishBase images (API-verified)...")
    print("-" * 50)
    fb_results = download_fishbase_extra(species_list)
    n_fb = sum(1 for r in fb_results if r["download_success"] == "yes")
    print(f"\n    Downloaded {n_fb} new verified FishBase images")

    # Download iNaturalist images
    print("\n[4] Downloading iNaturalist images...")
    print("-" * 50)
    inat_results = download_inaturalist(species_list)
    n_inat = sum(1 for r in inat_results if r["download_success"] == "yes")
    print(f"\n    Downloaded {n_inat} iNaturalist images")

    # Build full inventory with MD5 hashes
    print("\n[5] Building image inventory and computing MD5 hashes...")
    inventory = build_full_inventory()
    print(f"    Total images across all directories: {len(inventory)}")

    # Deduplicate
    print("\n[6] Checking for duplicates...")
    dedup_log, n_dups = deduplicate(inventory)
    print(f"    Found {n_dups} duplicate images")

    # Write dedup report
    with open(DEDUP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "md5_hash", "kept_file", "kept_source",
            "duplicate_file", "duplicate_source", "species",
        ])
        writer.writeheader()
        if dedup_log:
            writer.writerows(dedup_log)
    print(f"    Dedup report: {DEDUP_CSV}")

    # Write full inventory
    print("\n[7] Writing image inventory...")
    with open(INVENTORY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "species", "filename", "source", "directory", "md5_hash", "is_duplicate",
        ])
        writer.writeheader()
        for item in inventory:
            writer.writerow({
                "species": item["species"],
                "filename": item["filename"],
                "source": item["source"],
                "directory": item["directory"],
                "md5_hash": item["md5_hash"],
                "is_duplicate": item["is_duplicate"],
            })
    print(f"    Inventory: {INVENTORY_CSV}")

    # Build species summary
    print("\n[8] Building species summary...")
    summary_rows = build_species_summary(inventory, tree_species)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "species", "in_tree", "n_fishpix", "n_fishbase", "n_bishop",
            "n_fishbase_extra", "n_inaturalist", "n_total", "sources",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"    Summary: {SUMMARY_CSV}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_unique = sum(1 for item in inventory if item["is_duplicate"] == "no")
    print(f"  Total images (all sources):    {len(inventory)}")
    print(f"  Unique images (after dedup):   {total_unique}")
    print(f"  Duplicates found:              {n_dups}")
    print(f"  New FishBase images:           {n_fb}")
    print(f"  New iNaturalist images:        {n_inat}")
    print(f"  Species with images:           {len(summary_rows)}")
    if flagged:
        print(f"  Flagged existing images:       {len(flagged)} (potential wrong species)")

    # Show species with most reps
    top = sorted(summary_rows, key=lambda x: -x["n_total"])[:10]
    print(f"\n  Top 10 species by image count:")
    for r in top:
        print(f"    {r['species']:40s}  {r['n_total']:3d} images  ({r['sources']})")

    # Show species with fewest
    bottom = [r for r in summary_rows if r["in_tree"] == "yes"]
    bottom.sort(key=lambda x: x["n_total"])
    print(f"\n  Tree species with fewest images:")
    for r in bottom[:5]:
        print(f"    {r['species']:40s}  {r['n_total']:3d} images  ({r['sources']})")


if __name__ == "__main__":
    main()
