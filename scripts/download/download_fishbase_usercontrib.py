#!/usr/bin/env python3
"""
Download user-contributed photos from FishBase ThumbnailsSummary pages.

These are photos uploaded by FishBase users (not the official curated photos
from the API). They have CC licenses and include attribution metadata.

Output directory: images_fishbase_usercontrib/
Metadata saved to: fishbase_usercontrib_metadata.csv
"""

import os
import re
import csv
import time
import hashlib
import urllib.request
import urllib.error
from html.parser import HTMLParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "images_fishbase_usercontrib")
METADATA_CSV = os.path.join(SCRIPT_DIR, "fishbase_usercontrib_metadata.csv")
COMBINED_CSV = os.path.join(SCRIPT_DIR, "combined_species_data.csv")

# FishBase URLs
FISHBASE_BASE = "https://www.fishbase.se"
THUMBNAILS_URL = "https://www.fishbase.se/photos/ThumbnailsSummary.php?ID={}"
IMAGE_URL = "https://www.fishbase.se/tools/display_image.php?fw=n&imgName={}"


def parse_user_photos_regex(html):
    """Parse FishBase HTML to extract user-contributed photos using regex.

    User photos have patterns like:
    - imgName=1508794173_172.68.102.54.jpg (timestamp_IP.jpg)
    - Photo by PHOTOGRAPHER NAME
    - CC-BY-NC license links
    """
    photos = []

    # Find all user-contributed photo blocks
    # Pattern: display_image.php with timestamp-based filename, followed by "Photo by" attribution
    pattern = r"imgName=(\d+_[\d\.]+\.(jpg|png|gif))[^>]*>.*?Photo by ([^<]+)"

    for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
        filename = match.group(1)
        photographer = match.group(3).strip()

        # Look for license near this photo
        # Search backwards from match position for CC license
        start_pos = max(0, match.start() - 500)
        context = html[start_pos:match.end()]

        license_type = "unknown"
        if 'by-nc' in context.lower():
            license_type = "CC-BY-NC"
        elif 'by-sa' in context.lower():
            license_type = "CC-BY-SA"
        elif 'creativecommons.org/licenses/by/' in context.lower():
            license_type = "CC-BY"

        # Get autoctr
        autoctr_match = re.search(r'autoctr=(\d+)', context)
        autoctr = autoctr_match.group(1) if autoctr_match else ""

        photos.append({
            'filename': filename,
            'photographer': photographer,
            'license': license_type,
            'autoctr': autoctr,
        })

    return photos


class FishBasePhotoParser(HTMLParser):
    """Parse FishBase ThumbnailsSummary page to extract user-contributed photos."""

    def __init__(self):
        super().__init__()
        self.user_photos = []  # List of dicts with filename, license, autoctr
        self.current_photo = {}
        self.in_tooltip = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        # Look for user-contributed image patterns
        if tag == 'img' and 'src' in attrs_dict:
            src = attrs_dict['src']
            # User photos have timestamp pattern: display_image.php?...imgName=TIMESTAMP_IP.jpg
            if 'display_image.php' in src and 'imgName=' in src:
                match = re.search(r'imgName=(\d+_[\d\.]+\.(jpg|png|gif))', src, re.IGNORECASE)
                if match:
                    self.current_photo['filename'] = match.group(1)

        # Look for license info
        if tag == 'a' and 'href' in attrs_dict:
            href = attrs_dict['href']
            if 'creativecommons.org' in href:
                if 'by-nc' in href.lower():
                    self.current_photo['license'] = 'CC-BY-NC'
                elif 'by-sa' in href.lower():
                    self.current_photo['license'] = 'CC-BY-SA'
                elif '/by/' in href.lower():
                    self.current_photo['license'] = 'CC-BY'
                else:
                    self.current_photo['license'] = 'CC'

            # Get autoctr for attribution lookup
            if 'UploadedBy.php' in href:
                match = re.search(r'autoctr=(\d+)', href)
                if match:
                    self.current_photo['autoctr'] = match.group(1)

                    # Save completed photo entry
                    if 'filename' in self.current_photo:
                        self.user_photos.append(self.current_photo.copy())
                        self.current_photo = {}


def get_species_id_mapping():
    """Build mapping of species name to FishBase speccode."""
    # We'll need to query the API for each species
    import xml.etree.ElementTree as ET

    mapping = {}

    with open(COMBINED_CSV) as f:
        for row in csv.DictReader(f):
            species = row['species']
            parts = species.split()
            if len(parts) < 2:
                continue

            genus, epithet = parts[0], parts[1]
            url = f"https://www.fishbase.se/webservice/photos/FishPicsList.php?Genus={genus}&Species={epithet}"

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    xml_data = response.read()

                root = ET.fromstring(xml_data)
                speccode_elem = root.find(".//speccode")
                if speccode_elem is not None and speccode_elem.text:
                    mapping[species] = speccode_elem.text

            except Exception as e:
                print(f"  Warning: Could not get speccode for {species}: {e}")

            time.sleep(0.2)

    return mapping


def fetch_user_photos(species, speccode):
    """Fetch list of user-contributed photos for a species."""
    url = THUMBNAILS_URL.format(speccode)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8', errors='ignore')

        # Use regex parser instead of HTML parser
        photos = parse_user_photos_regex(html)
        return photos

    except Exception as e:
        print(f"  Error fetching photos for {species}: {e}")
        return []


def download_image(filename, dest_path):
    """Download a user-contributed image."""
    url = IMAGE_URL.format(filename)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()

            # Verify it's an image
            if len(data) < 1000:
                return False
            if not (data[:2] == b'\xff\xd8' or data[:4] == b'\x89PNG' or data[:4] == b'GIF8'):
                return False

            with open(dest_path, 'wb') as f:
                f.write(data)
            return True

    except Exception as e:
        return False


def compute_md5(filepath):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("="*70)
    print("FishBase User-Contributed Photo Downloader")
    print("="*70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load species list
    print("\n[1] Loading species list...")
    species_list = []
    with open(COMBINED_CSV) as f:
        for row in csv.DictReader(f):
            species_list.append(row['species'])
    species_list = sorted(set(species_list))
    print(f"    {len(species_list)} species")

    # Get species ID mapping
    print("\n[2] Getting FishBase species IDs...")
    speccode_map = get_species_id_mapping()
    print(f"    Got IDs for {len(speccode_map)} species")

    # Check what we already have
    existing = set()
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            existing.add(f)

    # Download user photos
    print("\n[3] Downloading user-contributed photos...")
    print("-"*50)

    all_metadata = []
    n_downloaded = 0
    n_skipped = 0
    n_species_with_photos = 0

    for idx, species in enumerate(species_list):
        if species not in speccode_map:
            continue

        speccode = speccode_map[species]
        safe_species = species.replace(" ", "_")

        print(f"  [{idx+1:3d}/{len(species_list)}] {species}", end=" ", flush=True)

        # Fetch user photo list
        user_photos = fetch_user_photos(species, speccode)

        if not user_photos:
            print("-> 0 user photos")
            time.sleep(0.3)
            continue

        n_new = 0
        for photo in user_photos:
            orig_filename = photo.get('filename', '')
            if not orig_filename:
                continue

            # Build local filename
            local_filename = f"{safe_species}_FishBaseUser_{orig_filename}"

            if local_filename in existing:
                n_skipped += 1
                all_metadata.append({
                    'species': species,
                    'filename': local_filename,
                    'source': 'FishBase_UserContrib',
                    'orig_filename': orig_filename,
                    'license': photo.get('license', 'unknown'),
                    'photographer': photo.get('photographer', ''),
                    'autoctr': photo.get('autoctr', ''),
                    'speccode': speccode,
                })
                continue

            dest_path = os.path.join(OUTPUT_DIR, local_filename)
            if download_image(orig_filename, dest_path):
                n_new += 1
                n_downloaded += 1
                all_metadata.append({
                    'species': species,
                    'filename': local_filename,
                    'source': 'FishBase_UserContrib',
                    'orig_filename': orig_filename,
                    'license': photo.get('license', 'unknown'),
                    'photographer': photo.get('photographer', ''),
                    'autoctr': photo.get('autoctr', ''),
                    'speccode': speccode,
                })

            time.sleep(0.15)

        if n_new > 0 or len(user_photos) > 0:
            n_species_with_photos += 1

        print(f"-> {len(user_photos)} found, {n_new} new")
        time.sleep(0.3)

    # Save metadata
    print(f"\n[4] Saving metadata...")
    if all_metadata:
        with open(METADATA_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'species', 'filename', 'source', 'orig_filename',
                'license', 'photographer', 'autoctr', 'speccode'
            ])
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"    Saved to {METADATA_CSV}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Species with user photos: {n_species_with_photos}")
    print(f"  New images downloaded: {n_downloaded}")
    print(f"  Already had: {n_skipped}")
    print(f"  Total metadata entries: {len(all_metadata)}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
