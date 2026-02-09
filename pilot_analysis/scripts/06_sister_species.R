#!/usr/bin/env Rscript
# 06_sister_species.R
#
# Sister species color divergence analysis (Hemingson 2019 hypotheses)
# Tests: color divergence ~ range overlap × range symmetry

library(ape)
library(phytools)
library(tidyverse)
library(readxl)

# ============================================================
# Paths
# ============================================================

pilot_dir <- "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/pilot_analysis"
chaets_dir <- "/Users/michaelalfaro/Dropbox/git/chaets-divergence-2026"

# Input
sister_xlsx <- file.path(chaets_dir, "sister species info", "Sister species pairs Chaetodontidae.xlsx")
adjacency_csv <- file.path(pilot_dir, "data", "pavo_adjacency_color.csv")
tree_path <- file.path(pilot_dir, "trees", "pilot_tree_pruned.tre")

# Output
results_dir <- file.path(pilot_dir, "results")
matched_csv <- file.path(pilot_dir, "data", "sister_pairs_matched.csv")
divergence_csv <- file.path(results_dir, "sister_divergence_stats.csv")

# ============================================================
# Helper functions
# ============================================================

# Clean species name
clean_species <- function(sp) {
  sp <- trimws(sp)
  sp <- gsub("\\s+", " ", sp)
  return(sp)
}

# Expand abbreviated species names (e.g., "C. lunula" -> "Chaetodon lunula")
expand_species <- function(sp, genus = "Chaetodon") {
  if (grepl("^[A-Z]\\. ", sp)) {
    sp <- gsub("^[A-Z]\\. ", paste0(genus, " "), sp)
  }
  return(clean_species(sp))
}

# Calculate color dissimilarity (Euclidean distance in 6D space)
calc_color_dissim <- function(metrics1, metrics2) {
  vars <- c("m", "A", "Jc", "Jt", "m_dS", "m_dL")
  diff <- metrics1[vars] - metrics2[vars]
  return(sqrt(sum(diff^2, na.rm = TRUE)))
}

# ============================================================
# Load data
# ============================================================

cat("=" , rep("=", 59), "\n", sep = "")
cat("Sister Species Color Divergence Analysis\n")
cat("=" , rep("=", 59), "\n\n", sep = "")

# Load sister pairs
cat("Loading sister species pairs...\n")
sister_raw <- read_xlsx(sister_xlsx, col_names = FALSE)
colnames(sister_raw) <- c("sp1", "sp2")
cat("  Raw pairs:", nrow(sister_raw), "\n")

# Clean and expand species names
sister_pairs <- sister_raw %>%
  mutate(
    sp1_clean = sapply(sp1, function(x) expand_species(clean_species(x))),
    sp2_clean = sapply(sp2, function(x) expand_species(clean_species(x), genus = ifelse(grepl("^[A-Z]", x), strsplit(x, " ")[[1]][1], "Chaetodon")))
  )

# Load adjacency data
cat("Loading adjacency data...\n")
adj_data <- read.csv(adjacency_csv, stringsAsFactors = FALSE)
adj_species <- adj_data$species
cat("  Species with color data:", length(adj_species), "\n")

# ============================================================
# Match sister pairs to data
# ============================================================

cat("\nMatching sister pairs to exemplar data...\n")

matched_pairs <- list()

for (i in 1:nrow(sister_pairs)) {
  sp1 <- sister_pairs$sp1_clean[i]
  sp2 <- sister_pairs$sp2_clean[i]

  # Try various name formats
  sp1_match <- sp1 %in% adj_species
  sp2_match <- sp2 %in% adj_species

  if (sp1_match && sp2_match) {
    # Both species have exemplar data
    metrics1 <- adj_data[adj_data$species == sp1, ]
    metrics2 <- adj_data[adj_data$species == sp2, ]

    dissim <- calc_color_dissim(metrics1, metrics2)

    matched_pairs[[length(matched_pairs) + 1]] <- data.frame(
      sp1 = sp1,
      sp2 = sp2,
      color_dissimilarity = dissim,
      sp1_m = metrics1$m,
      sp2_m = metrics2$m,
      sp1_Jc = metrics1$Jc,
      sp2_Jc = metrics2$Jc,
      both_have_data = TRUE,
      stringsAsFactors = FALSE
    )
  }
}

matched_df <- bind_rows(matched_pairs)
n_matched <- nrow(matched_df)

cat("  Pairs with both species having exemplars:", n_matched, "\n")

# Save matched pairs
write.csv(matched_df, matched_csv, row.names = FALSE)
cat("  Saved:", matched_csv, "\n")

# ============================================================
# Summary statistics
# ============================================================

cat("\n" , rep("-", 59), "\n", sep = "")
cat("Summary Statistics\n")
cat(rep("-", 59), "\n\n", sep = "")

if (n_matched > 0) {
  cat("Color dissimilarity among sister pairs:\n")
  cat(sprintf("  Mean: %.3f\n", mean(matched_df$color_dissimilarity)))
  cat(sprintf("  SD: %.3f\n", sd(matched_df$color_dissimilarity)))
  cat(sprintf("  Range: [%.3f, %.3f]\n",
              min(matched_df$color_dissimilarity),
              max(matched_df$color_dissimilarity)))

  # Identify most and least divergent pairs
  cat("\nMost divergent sister pairs:\n")
  top_divergent <- matched_df %>%
    arrange(desc(color_dissimilarity)) %>%
    head(5)
  for (i in 1:nrow(top_divergent)) {
    cat(sprintf("  %s vs %s: %.3f\n",
                top_divergent$sp1[i], top_divergent$sp2[i],
                top_divergent$color_dissimilarity[i]))
  }

  cat("\nLeast divergent sister pairs:\n")
  least_divergent <- matched_df %>%
    arrange(color_dissimilarity) %>%
    head(5)
  for (i in 1:nrow(least_divergent)) {
    cat(sprintf("  %s vs %s: %.3f\n",
                least_divergent$sp1[i], least_divergent$sp2[i],
                least_divergent$color_dissimilarity[i]))
  }

  # Save summary stats
  summary_stats <- data.frame(
    metric = c("n_pairs", "mean_dissim", "sd_dissim", "min_dissim", "max_dissim"),
    value = c(n_matched,
              mean(matched_df$color_dissimilarity),
              sd(matched_df$color_dissimilarity),
              min(matched_df$color_dissimilarity),
              max(matched_df$color_dissimilarity))
  )
  write.csv(summary_stats, divergence_csv, row.names = FALSE)
  cat("\n  Saved:", divergence_csv, "\n")
} else {
  cat("  No matched sister pairs found.\n")
}

cat("\nNote: Range overlap and symmetry data would need to be added\n")
cat("for full Hemingson hypothesis testing (PGLS regression).\n")

cat("\nDone!\n")
