# Randall Image Analysis Report

## Summary

**Date:** February 2025

### Hypothesis Testing

**Hypothesis 1:** All Bishop Museum/Randall images are also on FishBase credited to Randall
- **Result: ✅ SUPPORTED**
- All 97 species in Bishop collection have corresponding Randall images on FishBase
- Total of 254 Randall images across these species on FishBase

**Hypothesis 2:** FishBase Randall images are ONLY from the Bishop collection
- **Result: ❌ NOT SUPPORTED**
- Found 15 additional species with Randall images on FishBase not in Bishop collection
- However, **we already have all 15 of these images** in our FishBase downloads

---

## Species with Randall Images NOT in Bishop Collection

These 15 species have Randall-credited images on FishBase but are not in our `images_bishop/` directory. However, we have already downloaded these images as part of our FishBase collection.

| Species | Randall File | Location | In Exemplar? | Exemplar Used |
|---------|--------------|----------|--------------|---------------|
| Chaetodon burgessi | Chbur_u0.jpg | images_fishbase_extra | Yes | Chbur_u5 (different image) |
| Chaetodon capistratus | Chcap_u0.jpg | images_fishbase_extra | Yes | Chcap_u9 (different image) |
| Chaetodon interruptus | Chuni_u2.jpg, Chuni_u3.jpg | images_fishbase_extra | Yes | **Chuni_u2 (Randall!)** |
| Chaetodon oxycephalus | Choxy_u0.jpg | images_fishbase_extra | Yes | Choxy_ua (different image) |
| Chaetodon quadrimaculatus | Chqua_u2.jpg | images_fishbase_extra | Yes | Chqua_j0 (juvenile) |
| Chaetodon sedentarius | Chsed_u1.jpg | images | Yes | **Chsed_u1 (Randall!)** |
| Chaetodon striatus | Chstr_u0.jpg | images | Yes | Chstr_ua (different image) |
| Chaetodon triangulum | Chtri_ui.jpg | images | Yes | Chtri_ub (different image) |
| Hemitaurichthys thompsoni | Hetho_u0.jpg | images_fishbase_extra | Yes | **Hetho_u0 (Randall!)** |
| Johnrandallia nigrirostris | Jonig_u0.jpg | images_fishbase_extra | Yes | Jonig_u2 (different image) |
| Prognathodes aculeatus | Chacu_u1.jpg | images_fishbase_extra | Yes | Pracu_u0 (different image) |
| Roa excelsa | Chexc_u0.jpg | images | Yes | **Chexc_u0 (Randall!)** |
| Roa jayakari | Chjay_u1.jpg | images | Yes | Chjay_u0 (different image) |
| Roa modesta | Chmod_u1.jpg | images | Yes | **Chmod_u1 (Randall!)** |

### Key Finding

5 of 14 species (36%) are already using Randall images as exemplars:
- Chaetodon interruptus (Chuni_u2)
- Chaetodon sedentarius (Chsed_u1)
- Hemitaurichthys thompsoni (Hetho_u0)
- Roa excelsa (Chexc_u0)
- Roa modesta (Chmod_u1)

The other 9 species have Randall images available but a different FishBase image was selected as exemplar.

---

## Recommendations

### 1. Update Metadata
Add a `is_randall` flag to the inventory to identify Randall-credited images from FishBase.

### 2. Reconsider Exemplars
For these 9 species, consider switching to the Randall image as exemplar (better color fidelity):
- Chaetodon burgessi: Chbur_u0 → replace Chbur_u5
- Chaetodon capistratus: Chcap_u0 → replace Chcap_u9
- Chaetodon oxycephalus: Choxy_u0 → replace Choxy_ua
- Chaetodon quadrimaculatus: Chqua_u2 → replace Chqua_j0
- Chaetodon striatus: Chstr_u0 → replace Chstr_ua
- Chaetodon triangulum: Chtri_ui → replace Chtri_ub
- Johnrandallia nigrirostris: Jonig_u0 → replace Jonig_u2
- Prognathodes aculeatus: Chacu_u1 → replace Pracu_u0
- Roa jayakari: Chjay_u1 → replace Chjay_u0

### 3. No Downloads Needed
All Randall images are already in our collection - no additional downloads required.

---

## Image Counts by Source

| Directory | Total Images | Contains Randall? |
|-----------|--------------|-------------------|
| images_bishop/ | 178 | Yes (all are Randall) |
| images/ | 131 | Yes (some FishBase images are Randall) |
| images_fishbase_extra/ | 568 | Yes (some are Randall) |
| images_fishbase_usercontrib/ | 722 | No (user-contributed) |
| images_inaturalist/ | 1,125 | No (citizen science) |

### Total Randall Images in Collection
- Bishop Museum: 178 images (97 species)
- FishBase Randall (non-Bishop): ~17 images (14 species)
- **Total: ~195 Randall images**

---

## Technical Notes

1. **Filename Convention:** Randall images on FishBase use codes like `Chaur_u0.jpg` where the species abbreviation is followed by `_u#`

2. **API Limitation:** The FishBase ropensci API does not return image data. Web scraping was required to identify Randall credits.

3. **Taxonomy Changes:** Some old species codes (e.g., `Chuni` for C. interruptus, `Chacu` for P. aculeatus) reflect historical genus assignments.

4. **Color Quality:** Randall images are photographed out of water with controlled lighting - ideal for color analysis reference.
