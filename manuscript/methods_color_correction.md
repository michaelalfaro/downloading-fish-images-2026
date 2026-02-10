# Methods: Image Color Standardization

## Image Sources and Color Bias Assessment

We assembled a dataset of 2,001 Chaetodontidae images from four primary sources: the Bishop Museum's John E. Randall fish photograph collection (n=175), FishBase (n=625), FishPix (n=73), and iNaturalist (n=1,128). Preliminary analysis revealed systematic color differences between sources that could confound downstream color pattern analyses.

### Establishing a Reference Standard

John E. Randall's fish photographs, housed at the Bishop Museum, represent a uniquely standardized dataset. Over his career spanning several decades, Randall employed consistent photographic equipment and studio lighting conditions across thousands of specimens collected worldwide. This methodological consistency makes the Randall collection an ideal reference standard for color calibration.

We identified Randall photographs in our dataset through two mechanisms: (1) all 175 images from the Bishop Museum JER collection, and (2) 163 FishBase images confirmed as Randall photographs through perceptual hash-based duplicate detection (using difference hash and perceptual hash algorithms with Hamming distance threshold ≤15). This yielded a combined reference population of 338 Randall images.

### Color Distribution Analysis by Source

We quantified color characteristics in the CIELAB color space, which separates luminance (L*) from chromaticity (a*, b*) and provides perceptually uniform color differences. For each segmented fish image, we computed the mean a* and b* values across all body pixels, where a* represents the red-green axis (positive = red, negative = green) and b* represents the yellow-blue axis (positive = yellow, negative = blue).

**Table 1. Color characteristics by image source**

| Source | N | Mean a* | SD a* | Mean b* | SD b* | Interpretation |
|--------|---|---------|-------|---------|-------|----------------|
| Randall (Reference) | 338 | +2.87 | 4.84 | +14.25 | 12.43 | Studio lighting, neutral color balance |
| FishBase (non-Randall) | 461 | -4.28 | 8.53 | +19.87 | 17.02 | Heterogeneous sources, slight green-yellow bias |
| FishPix | 73 | -6.31 | 9.55 | +9.59 | 19.38 | Blue-green cast consistent with underwater photography |

### FishPix Color Cast and Correction

The FishPix database, maintained by the Kanagawa Prefectural Museum of Natural History and the National Museum of Nature and Science (Japan), comprises over 217,000 fish photographs contributed primarily by amateur SCUBA divers, with additional contributions from ichthyologists (https://fishpix.kahaku.go.jp). The crowdsourced underwater photography origin explains the consistent blue-green color cast observed in our analysis (mean a* = -6.31, mean b* = +9.59), as underwater images characteristically lose red and yellow wavelengths with depth.

We applied a global color correction to FishPix images to align their color distribution with the Randall reference standard. The correction was computed as the difference between Randall and FishPix population means:

- Δa* = +9.18 (shift toward red, away from green)
- Δb* = +4.62 (shift toward yellow, away from blue)

Correction was applied in CIELAB space to the a* and b* channels of segmented fish body pixels only, preserving both luminance (L*) and original background transparency. Post-correction, FishPix images showed mean values of a* = +2.46 and b* = +13.97, closely matching the Randall reference (a* = +2.87, b* = +14.25).

### FishBase Non-Randall Images

The non-Randall FishBase images (n=461) showed high variance in both color channels (SD a* = 8.53, SD b* = 17.02), reflecting their heterogeneous origins from numerous contributors using diverse photographic equipment and conditions. Given this high variance, we did not apply global color correction to these images, as such correction would improve some images while degrading others. Instead, we retain source information as a covariate for downstream analyses and recommend source-stratified analyses when color fidelity is critical.

### Duplicate Detection Between Sources

Perceptual hash comparison revealed that 165 Bishop Museum images had matching duplicates in FishBase (where FishBase had obtained copies of Randall photographs with attribution). These duplicates were identified despite different file compression, as traditional MD5 hash comparison failed to detect them. To avoid pseudoreplication, we retained the Bishop Museum originals and marked FishBase duplicates for exclusion, prioritizing the primary archival source.

## Effect Size Assessment

To evaluate whether the observed color differences represented meaningful systematic bias versus natural variation, we computed Cohen's d effect sizes for FishPix relative to the Randall reference:

- a* channel: d = 1.21 (large effect)
- b* channel: d = 0.28 (small effect)

The large effect size for the a* channel (green-red axis) indicates that the FishPix color bias substantially exceeds natural inter-image variation, supporting the application of systematic correction. The smaller effect for b* reflects higher natural variance in the yellow-blue dimension across fish species.

## References

- FishPix Database: https://fishpix.kahaku.go.jp/fishimage-e/
- Bishop Museum John E. Randall Collection: https://pbs.bishopmuseum.org/images/JER/
- FishBase: https://www.fishbase.org
