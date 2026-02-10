#!/usr/bin/env Rscript
# Plot rooted Butterflyfish tree with SEGMENTED fish images (single column)
# Uses segmented images (background removed) from FishBase sources only
# Images are matched to tree tips via tip_image_lookup_segmented.csv

library(ape)
library(treeio)
library(ggtree)
library(ggimage)
library(ggplot2)

# --- Paths ---
script_dir <- "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/tree"
tree_file  <- file.path(script_dir, "Butterflyfish_concat_final.tre")
lookup_csv <- file.path(script_dir, "tip_image_lookup_segmented.csv")
out_pdf    <- file.path(script_dir, "chaet_tree_segmented_fishbase.pdf")

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

# --- Read image lookup table (single column for segmented images) ---
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

# Replace "NA" strings with actual NA
tip_data$img_path[tip_data$img_path == "NA"] <- NA_character_

# Verify image files exist
tip_data$img_path <- ifelse(
  !is.na(tip_data$img_path) & file.exists(tip_data$img_path),
  tip_data$img_path,
  NA_character_
)

n_chaet    <- sum(tip_data$is_chaet)
n_img      <- sum(!is.na(tip_data$img_path))
cat("Chaetodontidae tips:", n_chaet, "\n")
cat("  Tips with images:", n_img, "\n")

missing <- tip_data$label[tip_data$is_chaet & is.na(tip_data$img_path)]
if (length(missing) > 0) {
  cat("  Missing images:", paste(missing, collapse = ", "), "\n")
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

# Single column: Segmented FishBase image
p <- p + geom_tiplab(
  aes(image  = img_path,
      subset = !is.na(img_path)),
  geom     = "image",
  offset   = max_x * 0.35,
  align    = max_x * 0.35,
  linetype = "dashed",
  linesize = 0.2,
  size     = 0.004,  # Slightly larger since only one column
  asp      = 2.0
)

# Expand x-axis to fit labels + 1 image column
p <- p +
  xlim(NA, max_x * 2.0) +
  coord_cartesian(clip = "off") +
  theme(plot.margin = margin(10, 150, 10, 10, unit = "pt"))

# Add title
p <- p + ggtitle("Chaetodontidae Phylogeny with Segmented FishBase Images")

# --- Save PDF ---
n_tips    <- Ntip(tree)
pdf_h     <- max(30, n_tips * 0.45)
pdf_w     <- 16

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
