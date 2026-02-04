#!/usr/bin/env Rscript
# Plot rooted Butterflyfish tree with Chaetodontidae fish images (3 columns)
# Uses freshly downloaded images from FishBase, FishPix, Bishop Museum, and iNaturalist
# Images are matched to tree tips via tip_image_lookup.csv
# Column 1: Best image (Bishop Museum preferred)
# Column 2: Second best (usually iNaturalist)
# Column 3: Third best (usually FishPix/FishBase)

library(ape)
library(treeio)
library(ggtree)
library(ggimage)
library(ggplot2)

# --- Paths ---
script_dir <- "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/tree"
tree_file  <- file.path(script_dir, "Butterflyfish_concat_final.tre")
lookup_csv <- file.path(script_dir, "tip_image_lookup.csv")
out_pdf    <- file.path(script_dir, "chaet_tree_with_images.pdf")

# --- Read and root tree ---
tree <- read.nexus(tree_file)
tree <- root(tree, outgroup = "Morone_saxatilis_12185", resolve.root = TRUE)

cat("Tree has", Ntip(tree), "tips\n")

# --- Define Chaetodontidae genera ---
chaet_genera <- tolower(c(
  "Amphichaetodon", "Chaetodon", "Chelmon", "Chelmonops",
  "Coradion", "Forcipiger", "Hemitaurichthys", "Heniochus",
  "Johnrandallia", "Parachaetodon", "Prognathodes", "Roa"
))

# --- Read image lookup table (now with 3 columns) ---
lookup <- read.csv(lookup_csv, stringsAsFactors = FALSE)

# --- Build tip metadata ---
tips <- tree$tip.label
tip_data <- data.frame(
  label = tips,
  genus = tolower(sub("_.*", "", tips)),
  stringsAsFactors = FALSE
)
tip_data$is_chaet <- tip_data$genus %in% chaet_genera

# Merge with image lookup
tip_data <- merge(tip_data, lookup, by.x = "label", by.y = "tip_label", all.x = TRUE)

# Replace "NA" strings with actual NA for all 3 image columns
for (col in c("img_path", "img_path_2", "img_path_3")) {
  tip_data[[col]][tip_data[[col]] == "NA"] <- NA_character_
  # Verify image files exist
  tip_data[[col]] <- ifelse(
    !is.na(tip_data[[col]]) & file.exists(tip_data[[col]]),
    tip_data[[col]],
    NA_character_
  )
}

n_chaet    <- sum(tip_data$is_chaet)
n_img1     <- sum(!is.na(tip_data$img_path))
n_img2     <- sum(!is.na(tip_data$img_path_2))
n_img3     <- sum(!is.na(tip_data$img_path_3))
cat("Chaetodontidae tips:", n_chaet, "\n")
cat("  Column 1 images:", n_img1, "\n")
cat("  Column 2 images:", n_img2, "\n")
cat("  Column 3 images:", n_img3, "\n")

missing <- tip_data$label[tip_data$is_chaet & is.na(tip_data$img_path)]
if (length(missing) > 0) {
  cat("  Missing all images:", paste(missing, collapse = ", "), "\n")
}

# --- Build the tree plot (cladogram) ---
p <- ggtree(tree, layout = "rectangular", branch.length = "none", size = 0.3) %<+% tip_data

# Get the x-range of the cladogram to set good offsets
tree_data <- p$data
max_x <- max(tree_data$x, na.rm = TRUE)

# Text labels for all tips
p <- p + geom_tiplab(
  size     = 1.8,
  offset   = max_x * 0.01
)

# Column 1: Best image (Bishop Museum priority)
p <- p + geom_tiplab(
  aes(image  = img_path,
      subset = !is.na(img_path)),
  geom     = "image",
  offset   = max_x * 0.35,
  align    = max_x * 0.35,
  linetype = "dashed",
  linesize = 0.2,
  size     = 0.003,
  asp      = 2.0
)

# Column 2: Second best image (usually iNaturalist)
p <- p + geom_tiplab(
  aes(image  = img_path_2,
      subset = !is.na(img_path_2)),
  geom     = "image",
  offset   = max_x * 0.55,
  align    = max_x * 0.55,
  linetype = NA,
  size     = 0.003,
  asp      = 2.0
)

# Column 3: Third best image (usually FishPix/FishBase)
p <- p + geom_tiplab(
  aes(image  = img_path_3,
      subset = !is.na(img_path_3)),
  geom     = "image",
  offset   = max_x * 0.75,
  align    = max_x * 0.75,
  linetype = NA,
  size     = 0.003,
  asp      = 2.0
)

# Expand x-axis to fit labels + 3 image columns
p <- p +
  xlim(NA, max_x * 3.0) +
  coord_cartesian(clip = "off") +
  theme(plot.margin = margin(10, 200, 10, 10, unit = "pt"))

# --- Save PDF ---
n_tips    <- Ntip(tree)
pdf_h     <- max(30, n_tips * 0.5)
pdf_w     <- 22

cat("Saving PDF:", out_pdf, "\n")
cat("  dimensions:", pdf_w, "x", pdf_h, "inches\n")

ggsave(
  out_pdf,
  plot   = p,
  width  = pdf_w,
  height = pdf_h,
  limitsize = FALSE
)

cat("Done!\n")
