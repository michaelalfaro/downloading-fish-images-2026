#!/usr/bin/env Rscript
# 04_run_pavo_grayscale.R
#
# Run pavo adjacency analysis on GRAYSCALE exemplar images.
# Tests robustness of pattern metrics (m, A, Jc, Jt) to color removal.
# Chromatic metrics (m_dS) expected to differ.
#
# Outputs 6 key metrics per species:
#   m     - overall transition density
#   A     - aspect ratio of transitions
#   Jc    - color class diversity (Simpson's)
#   Jt    - transition diversity (Simpson's)
#   m_dS  - mean chromatic boundary strength (will be near 0 for grayscale)
#   m_dL  - mean achromatic boundary strength

library(tidyverse)
library(pavo)

# ============================================================
# Paths
# ============================================================

pilot_dir <- "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/pilot_analysis"
output_dir <- file.path(pilot_dir, "data")
figure_dir <- file.path(pilot_dir, "figures")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

# Input
exemplar_csv <- file.path(output_dir, "exemplar_inventory.csv")

# Output
adjacency_csv <- file.path(output_dir, "pavo_adjacency_gray.csv")
classified_rds <- file.path(output_dir, "classified_images_gray.rds")

# ============================================================
# Helper functions (from ICB 2019 code)
# ============================================================

# Calculate euclidean color distance
rgb_euc_dist <- function(rgb_table_altered, c1, c2) {
  col1_diff <- rgb_table_altered[c1, "col1"] - rgb_table_altered[c2, "col1"]
  col2_diff <- rgb_table_altered[c1, "col2"] - rgb_table_altered[c2, "col2"]
  euc_dist <- sqrt(col1_diff^2 + col2_diff^2)
  return(as.numeric(euc_dist))
}

# Calculate luminance distance
rgb_lum_dist <- function(rgb_table_altered, c1, c2) {
  lum_diff <- rgb_table_altered[c1, "lum"] - rgb_table_altered[c2, "lum"]
  lum_dist <- sqrt(lum_diff^2)
  return(as.numeric(lum_dist))
}

# Calculate euclidean and luminance distances for a classified image
calc_euc_lum_dists <- function(class_fish) {
  # Extract RGB values for n colors
  class_fish_rgb <- attr(class_fish, 'classRGB')

  # Transform to color opponent coordinates + luminance
  class_fish_rgb_altered <- class_fish_rgb %>%
    rownames_to_column(var = "col_num") %>%
    as_tibble() %>%
    mutate(
      col1 = (R - G) / (R + G + 0.001),  # Add small value to avoid div by zero
      col2 = (G - B) / (G + B + 0.001),
      lum = R + G + B
    ) %>%
    select(col1, col2, lum) %>%
    as.data.frame()

  # Get all pairwise color comparisons
  n_colors <- nrow(class_fish_rgb)
  if (n_colors < 2) {
    # Return empty dataframe if only 1 color
    return(data.frame(c1 = integer(), c2 = integer(), dS = numeric(), dL = numeric()))
  }

  combos <- t(combn(1:n_colors, 2)) %>%
    as_tibble() %>%
    transmute(c1 = as.numeric(V1), c2 = as.numeric(V2)) %>%
    rowwise() %>%
    mutate(
      dS = rgb_euc_dist(class_fish_rgb_altered, c1, c2),
      dL = rgb_lum_dist(class_fish_rgb_altered, c1, c2)
    ) %>%
    as.data.frame()

  return(combos)
}

# ============================================================
# Main analysis
# ============================================================

cat("=" , rep("=", 59), "\n", sep = "")
cat("Pavo Adjacency Analysis - GRAYSCALE Images\n")
cat("=" , rep("=", 59), "\n\n", sep = "")

# Load exemplar inventory
cat("Loading exemplar inventory...\n")
exemplars <- read.csv(exemplar_csv, stringsAsFactors = FALSE)
cat("  Species:", nrow(exemplars), "\n")
cat("  Gestalt k range:", range(exemplars$gestalt_k), "\n\n")

# Process each exemplar image
cat("Processing images...\n")
results_list <- list()
classified_list <- list()
n_success <- 0
n_error <- 0

for (i in 1:nrow(exemplars)) {
  species <- exemplars$species[i]
  img_path <- exemplars$grayscale_path[i]  # Use grayscale path
  k <- exemplars$gestalt_k[i]

  cat(sprintf("  [%d/%d] %s (k=%d)... ", i, nrow(exemplars), species, k))

  tryCatch({
    # Load image
    img <- getimg(img_path, max.size = 3)

    # Classify with species-specific k
    classified <- classify(img, kcols = k)
    classified_list[[species]] <- classified

    # Calculate color distances
    dists <- calc_euc_lum_dists(classified)

    # Calculate adjacency statistics
    if (nrow(dists) > 0) {
      adj_stats <- adjacent(classimg = classified, coldists = dists, xpts = 100, xscale = 100)

      # Extract key metrics
      results_list[[species]] <- data.frame(
        species = species,
        gestalt_k = k,
        m = adj_stats$m,
        m_r = adj_stats$m_r,
        m_c = adj_stats$m_c,
        A = adj_stats$A,
        Sc = adj_stats$Sc,
        St = adj_stats$St,
        Jc = adj_stats$Jc,
        Jt = adj_stats$Jt,
        m_dS = adj_stats$m_dS,
        s_dS = adj_stats$s_dS,
        cv_dS = adj_stats$cv_dS,
        m_dL = adj_stats$m_dL,
        s_dL = adj_stats$s_dL,
        cv_dL = adj_stats$cv_dL,
        stringsAsFactors = FALSE
      )
      n_success <- n_success + 1
      cat("OK\n")
    } else {
      cat("SKIP (single color)\n")
      n_error <- n_error + 1
    }
  }, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    n_error <<- n_error + 1
  })
}

# Combine results
cat("\nCombining results...\n")
adjacency_data <- bind_rows(results_list)
cat("  Successful analyses:", n_success, "\n")
cat("  Errors/Skipped:", n_error, "\n")

# Save results
cat("\nSaving outputs...\n")
write.csv(adjacency_data, adjacency_csv, row.names = FALSE)
cat("  Adjacency data:", adjacency_csv, "\n")

saveRDS(classified_list, classified_rds)
cat("  Classified images:", classified_rds, "\n")

# ============================================================
# Summary statistics
# ============================================================

cat("\n" , rep("=", 59), "\n", sep = "")
cat("Summary Statistics\n")
cat(rep("=", 59), "\n\n", sep = "")

# Key metrics summary
key_metrics <- c("m", "A", "Jc", "Jt", "m_dS", "m_dL")

cat("Key metrics (6 core variables):\n")
for (metric in key_metrics) {
  vals <- adjacency_data[[metric]]
  cat(sprintf("  %s: mean=%.3f, sd=%.3f, range=[%.3f, %.3f]\n",
              metric, mean(vals, na.rm=T), sd(vals, na.rm=T),
              min(vals, na.rm=T), max(vals, na.rm=T)))
}

# Gestalt k breakdown
cat("\nResults by gestalt k:\n")
k_summary <- adjacency_data %>%
  group_by(gestalt_k) %>%
  summarise(
    n = n(),
    mean_m = mean(m, na.rm = TRUE),
    mean_Jc = mean(Jc, na.rm = TRUE)
  )
print(k_summary)

# Quick correlation check
cat("\nCorrelation matrix (key metrics):\n")
cor_mat <- cor(adjacency_data[, key_metrics], use = "pairwise.complete.obs")
print(round(cor_mat, 2))

cat("\nDone! Ready for phylogenetic comparative analyses.\n")
