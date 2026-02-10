#!/usr/bin/env python3
"""
Web-based visual review app for Randall/FishBase duplicate candidates.

Usage:
    python3 review_duplicates_app.py

Then open http://localhost:5002 in a browser.
"""

import os
import csv
import io
from datetime import datetime

import sys
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, send_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'utils'))
from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR, species_to_dirname, ensure_dir,
)

app = Flask(__name__)

# Paths
CANDIDATES_CSV = os.path.join(GMM_DIR, "randall_dup_candidates.csv")
CONFIRMED_CSV = os.path.join(GMM_DIR, "randall_dup_confirmed.csv")

# In-memory state
_candidates = []
_confirmed = {}


def load_candidates():
    """Load candidate duplicates from CSV."""
    if not os.path.exists(CANDIDATES_CSV):
        return []
    candidates = []
    with open(CANDIDATES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            candidates.append(row)
    return candidates


def load_confirmed():
    """Load confirmed duplicates from CSV."""
    if not os.path.exists(CONFIRMED_CSV):
        return {}
    confirmed = {}
    with open(CONFIRMED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["fishbase_file"], row["bishop_file"])
            confirmed[key] = row
    return confirmed


def save_confirmed():
    """Save confirmed duplicates to CSV."""
    ensure_dir(os.path.dirname(CONFIRMED_CSV))
    fields = ["species", "fishbase_file", "bishop_file", "is_duplicate",
              "reviewed_at", "notes"]

    with open(CONFIRMED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(_confirmed.values(), key=lambda x: x["species"]):
            w.writerow(row)


def load_image(path):
    """Load image, handling RGBA properly."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (180, 180, 180))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert('RGB')


@app.route("/")
def index():
    # Get counts
    n_total = len(_candidates)
    n_reviewed = len(_confirmed)
    n_dups = sum(1 for c in _confirmed.values() if c["is_duplicate"] == "yes")
    n_not = n_reviewed - n_dups
    n_remaining = n_total - n_reviewed

    # Get next unreviewed index
    next_idx = 0
    for i, c in enumerate(_candidates):
        key = (c["fishbase_file"], c["bishop_file"])
        if key not in _confirmed:
            next_idx = i
            break

    return render_template_string(INDEX_HTML,
                                  n_total=n_total,
                                  n_reviewed=n_reviewed,
                                  n_dups=n_dups,
                                  n_not=n_not,
                                  n_remaining=n_remaining,
                                  next_idx=next_idx)


@app.route("/review/<int:idx>")
def review(idx):
    if idx < 0 or idx >= len(_candidates):
        return "Invalid index", 404

    c = _candidates[idx]
    key = (c["fishbase_file"], c["bishop_file"])
    status = _confirmed.get(key, {})

    # Find prev/next unreviewed
    prev_idx = None
    next_idx = None
    for i in range(idx - 1, -1, -1):
        k = (_candidates[i]["fishbase_file"], _candidates[i]["bishop_file"])
        if k not in _confirmed:
            prev_idx = i
            break
    for i in range(idx + 1, len(_candidates)):
        k = (_candidates[i]["fishbase_file"], _candidates[i]["bishop_file"])
        if k not in _confirmed:
            next_idx = i
            break

    # Count remaining
    n_remaining = sum(1 for c2 in _candidates
                      if (c2["fishbase_file"], c2["bishop_file"]) not in _confirmed)

    return render_template_string(REVIEW_HTML,
                                  idx=idx,
                                  n_total=len(_candidates),
                                  n_remaining=n_remaining,
                                  candidate=c,
                                  status=status,
                                  prev_idx=prev_idx,
                                  next_idx=next_idx)


@app.route("/image/<species_dir>/<filename>")
def serve_image(species_dir, filename):
    # Try segmented first, then original
    seg_path = os.path.join(SEGMENTED_DIR, species_dir, filename)
    if os.path.exists(seg_path):
        path = seg_path
    else:
        # Try to find in inventory directories
        for subdir in ["images", "images_bishop", "images_fishbase_extra"]:
            test_path = os.path.join(SCRIPT_DIR, subdir, filename)
            if os.path.exists(test_path):
                path = test_path
                break
        else:
            return "Not found", 404

    try:
        img = load_image(path)
        # Resize for display (max 400px height)
        w, h = img.size
        if h > 400:
            scale = 400 / h
            new_w = max(1, int(w * scale))
            img = img.resize((new_w, 400), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return f"Error: {e}", 500


@app.route("/api/mark", methods=["POST"])
def api_mark():
    data = request.get_json()
    idx = data.get("idx", 0)
    is_dup = data.get("is_duplicate", False)
    notes = data.get("notes", "")

    if idx < 0 or idx >= len(_candidates):
        return jsonify({"error": "invalid index"}), 400

    c = _candidates[idx]
    key = (c["fishbase_file"], c["bishop_file"])

    _confirmed[key] = {
        "species": c["species"],
        "fishbase_file": c["fishbase_file"],
        "bishop_file": c["bishop_file"],
        "is_duplicate": "yes" if is_dup else "no",
        "reviewed_at": datetime.now().isoformat(),
        "notes": notes,
    }
    save_confirmed()

    # Find next unreviewed
    next_idx = None
    for i in range(idx + 1, len(_candidates)):
        k = (_candidates[i]["fishbase_file"], _candidates[i]["bishop_file"])
        if k not in _confirmed:
            next_idx = i
            break

    return jsonify({"ok": True, "next_idx": next_idx})


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Randall/FishBase Duplicate Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #f0f0f0; color: #222; padding: 40px; max-width: 600px; margin: 0 auto; }
  h1 { margin-bottom: 20px; }
  .stats { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
  .stats p { margin: 8px 0; font-size: 16px; }
  .progress-bar { background: #ddd; border-radius: 8px; height: 24px; margin: 16px 0;
                  overflow: hidden; position: relative; }
  .progress-fill { background: #4caf50; height: 100%; }
  .progress-text { position: absolute; top: 0; left: 0; right: 0; text-align: center;
                   line-height: 24px; font-size: 13px; font-weight: 600; }
  .btn { display: inline-block; padding: 12px 24px; background: #1a73e8; color: white;
         text-decoration: none; border-radius: 6px; font-size: 16px; font-weight: 600; }
  .btn:hover { background: #1565c0; }
  .note { margin-top: 20px; font-size: 14px; color: #666; }
</style>
</head>
<body>
<h1>Randall/FishBase Duplicate Review</h1>

<div class="stats">
  <p><strong>Total candidates:</strong> {{ n_total }}</p>
  <p><strong>Reviewed:</strong> {{ n_reviewed }}</p>
  <p><strong>Confirmed duplicates:</strong> {{ n_dups }}</p>
  <p><strong>Not duplicates:</strong> {{ n_not }}</p>
  <p><strong>Remaining:</strong> {{ n_remaining }}</p>

  <div class="progress-bar">
    <div class="progress-fill" style="width: {{ (n_reviewed / n_total * 100)|round(1) }}%"></div>
    <div class="progress-text">{{ n_reviewed }} / {{ n_total }}</div>
  </div>
</div>

{% if n_remaining > 0 %}
<a href="/review/{{ next_idx }}" class="btn">Start Review &rarr;</a>
{% else %}
<p style="font-size: 18px; color: #2e7d32;">All candidates reviewed!</p>
{% endif %}

<p class="note">
  FishBase often uses Randall's photos from Bishop Museum with different compression.
  Review each pair and mark whether they're the same photo.
</p>
</body>
</html>"""


REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Review Pair {{ idx + 1 }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #e8e8e8; color: #222; }
  .header { background: #333; color: white; padding: 12px 20px; display: flex;
            align-items: center; justify-content: space-between; }
  .header a { color: #adf; text-decoration: none; }
  .header .title { font-size: 18px; }
  .nav-btn { background: #555; color: white; padding: 6px 14px; border-radius: 4px;
             text-decoration: none; font-size: 13px; }
  .nav-btn:hover { background: #666; }
  .nav-btn.disabled { background: #444; color: #777; pointer-events: none; }

  .content { padding: 20px; max-width: 1200px; margin: 0 auto; }
  .species-name { font-size: 24px; font-style: italic; margin-bottom: 8px; }
  .distance { font-size: 14px; color: #666; margin-bottom: 20px; }
  .confidence-high { color: #2e7d32; font-weight: 600; }
  .confidence-medium { color: #e65100; font-weight: 600; }

  .comparison { display: flex; gap: 20px; margin-bottom: 20px; }
  .image-card { flex: 1; background: white; border-radius: 8px; overflow: hidden;
                box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
  .image-card .label { background: #333; color: white; padding: 8px 12px; font-size: 14px;
                       font-weight: 600; }
  .image-card .label.bishop { background: #2e7d32; }
  .image-card .label.fishbase { background: #1565c0; }
  .image-card .img-container { background: #b4b4b4; padding: 10px; text-align: center;
                               min-height: 300px; display: flex; align-items: center;
                               justify-content: center; }
  .image-card img { max-width: 100%; max-height: 400px; }
  .image-card .filename { padding: 8px 12px; font-size: 12px; color: #666;
                          word-break: break-all; }

  .actions { background: white; padding: 20px; border-radius: 8px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
  .actions h3 { margin-bottom: 12px; }
  .btn-group { display: flex; gap: 12px; margin-bottom: 16px; }
  .btn { padding: 12px 24px; border: none; border-radius: 6px; font-size: 16px;
         font-weight: 600; cursor: pointer; }
  .btn-dup { background: #c62828; color: white; }
  .btn-dup:hover { background: #b71c1c; }
  .btn-not { background: #2e7d32; color: white; }
  .btn-not:hover { background: #1b5e20; }
  .btn-skip { background: #eee; color: #333; }
  .btn-skip:hover { background: #ddd; }

  .status { margin-top: 12px; padding: 8px 12px; border-radius: 4px; font-size: 14px; }
  .status-dup { background: #ffcdd2; color: #c62828; }
  .status-not { background: #c8e6c9; color: #2e7d32; }

  .kbd-hint { margin-top: 16px; font-size: 12px; color: #888; }
  kbd { background: #eee; padding: 2px 6px; border-radius: 3px; border: 1px solid #ccc; }
</style>
</head>
<body>

<div class="header">
  <div>
    {% if prev_idx is not none %}
      <a href="/review/{{ prev_idx }}" class="nav-btn">&larr; Prev</a>
    {% else %}
      <span class="nav-btn disabled">&larr; Prev</span>
    {% endif %}
    <a href="/" style="margin-left: 12px;">Index</a>
  </div>
  <span class="title">Pair {{ idx + 1 }} of {{ n_total }} ({{ n_remaining }} remaining)</span>
  <div>
    {% if next_idx is not none %}
      <a href="/review/{{ next_idx }}" class="nav-btn">Next &rarr;</a>
    {% else %}
      <span class="nav-btn disabled">Next &rarr;</span>
    {% endif %}
  </div>
</div>

<div class="content">
  <div class="species-name">{{ candidate.species }}</div>
  <div class="distance">
    Hamming distance: {{ candidate.min_distance }}
    <span class="{{ 'confidence-high' if candidate.confidence == 'high' else 'confidence-medium' }}">
      ({{ candidate.confidence }} confidence)
    </span>
  </div>

  <div class="comparison">
    <div class="image-card">
      <div class="label bishop">Bishop Museum (Randall)</div>
      <div class="img-container">
        <img src="/image/{{ candidate.species|replace(' ', '_') }}/{{ candidate.bishop_file|replace('.jpg', '.png') }}"
             onerror="this.src='/image/{{ candidate.species|replace(' ', '_') }}/{{ candidate.bishop_file }}'">
      </div>
      <div class="filename">{{ candidate.bishop_file }}</div>
    </div>

    <div class="image-card">
      <div class="label fishbase">FishBase</div>
      <div class="img-container">
        <img src="/image/{{ candidate.species|replace(' ', '_') }}/{{ candidate.fishbase_file|replace('.jpg', '.png') }}"
             onerror="this.src='/image/{{ candidate.species|replace(' ', '_') }}/{{ candidate.fishbase_file }}'">
      </div>
      <div class="filename">{{ candidate.fishbase_file }}</div>
    </div>
  </div>

  <div class="actions">
    <h3>Is this the same photo?</h3>

    {% if status %}
      <div class="status {{ 'status-dup' if status.is_duplicate == 'yes' else 'status-not' }}">
        Already marked as: {{ 'DUPLICATE' if status.is_duplicate == 'yes' else 'NOT duplicate' }}
      </div>
    {% endif %}

    <div class="btn-group">
      <button class="btn btn-dup" onclick="mark(true)">Yes, Duplicate</button>
      <button class="btn btn-not" onclick="mark(false)">No, Different</button>
      {% if next_idx is not none %}
        <a href="/review/{{ next_idx }}" class="btn btn-skip">Skip</a>
      {% endif %}
    </div>

    <div class="kbd-hint">
      <kbd>Y</kbd> = duplicate &nbsp;
      <kbd>N</kbd> = not duplicate &nbsp;
      <kbd>&rarr;</kbd> = skip to next
    </div>
  </div>
</div>

<script>
const IDX = {{ idx }};

function mark(isDup) {
  fetch('/api/mark', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx: IDX, is_duplicate: isDup, notes: ''})
  }).then(r => r.json()).then(data => {
    if (data.next_idx !== null) {
      window.location.href = '/review/' + data.next_idx;
    } else {
      window.location.href = '/';
    }
  });
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'y' || e.key === 'Y') {
    mark(true);
  } else if (e.key === 'n' || e.key === 'N') {
    mark(false);
  } else if (e.key === 'ArrowRight') {
    {% if next_idx is not none %}
    window.location.href = '/review/{{ next_idx }}';
    {% endif %}
  } else if (e.key === 'ArrowLeft') {
    {% if prev_idx is not none %}
    window.location.href = '/review/{{ prev_idx }}';
    {% endif %}
  }
});
</script>
</body>
</html>"""


def init_app():
    global _candidates, _confirmed
    print("Loading candidates...")
    _candidates = load_candidates()
    print(f"  {len(_candidates)} candidates")

    print("Loading confirmed...")
    _confirmed = load_confirmed()
    n_dups = sum(1 for c in _confirmed.values() if c["is_duplicate"] == "yes")
    print(f"  {len(_confirmed)} reviewed ({n_dups} duplicates)")


def main():
    init_app()
    print("\nStarting duplicate review app at http://localhost:5002")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=5002, debug=False)


if __name__ == "__main__":
    main()
