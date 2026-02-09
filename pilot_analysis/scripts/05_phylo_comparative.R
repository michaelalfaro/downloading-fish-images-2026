#!/usr/bin/env Rscript
# 05_phylo_comparative.R
#
# Phylogenetic comparative analyses replicating Alfaro et al. 2019 ICB paper:
# 1. Phylogenetic PCA on color pattern metrics
# 2. Blomberg's K (phylogenetic signal)
# 3. DTT (Disparity Through Time)
# 4. Node Heights Test

library(ape)
library(phytools)
library(geiger)
library(tidyverse)

# ============================================================
# Paths
# ============================================================

pilot_dir <- "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/pilot_analysis"

# Input
tree_path <- file.path(pilot_dir, "trees", "pilot_tree_pruned.tre")
adjacency_csv <- file.path(pilot_dir, "data", "pavo_adjacency_color.csv")

# Output
results_dir <- file.path(pilot_dir, "results")
dir.create(results_dir, showWarnings = FALSE)

pca_csv <- file.path(pilot_dir, "data", "phylo_pca_scores.csv")
blomberg_csv <- file.path(results_dir, "blomberg_k_results.csv")
dtt_csv <- file.path(results_dir, "dtt_mdi_results.csv")
nodeheights_csv <- file.path(results_dir, "node_heights_results.csv")

# ============================================================
# Helper functions
# ============================================================

# Clean species name for tree matching
clean_species_for_tree <- function(sp) {
  gsub(" ", "_", sp)
}

# Clean tree tip to get species name only (Genus_epithet)
clean_tree_tip <- function(tip) {
  parts <- strsplit(tip, "_")[[1]]
  if (length(parts) >= 2) {
    return(paste(parts[1], parts[2], sep = "_"))
  }
  return(tip)
}

# Node heights test (from phytools)
run_node_heights_test <- function(tree, trait, nsim = 1000) {
  # Calculate contrasts
  pic_vals <- pic(trait, tree)

  # Get heights of internal nodes where contrasts are calculated
  # pic returns contrasts at internal nodes, ordered by node number
  n_tips <- length(tree$tip.label)
  internal_node_nums <- as.numeric(names(pic_vals))

  # Get the depth of each internal node
  node_heights <- node.depth.edgelength(tree)
  contrast_heights <- node_heights[internal_node_nums]

  # Regression of abs(contrasts) vs node heights
  model <- lm(abs(pic_vals) ~ contrast_heights)

  # Get slope and p-value
  slope <- coef(model)[2]
  p_value <- summary(model)$coefficients[2, 4]

  return(list(slope = slope, p_value = p_value, r_squared = summary(model)$r.squared))
}

# ============================================================
# Load data
# ============================================================

cat("=" , rep("=", 59), "\n", sep = "")
cat("Phylogenetic Comparative Analyses\n")
cat("=" , rep("=", 59), "\n\n", sep = "")

# Load tree
cat("Loading tree...\n")
tree <- read.tree(tree_path)
cat("  Tips:", length(tree$tip.label), "\n")

# Load adjacency data
cat("Loading adjacency data...\n")
adj_data <- read.csv(adjacency_csv, stringsAsFactors = FALSE)
adj_data$tree_name <- clean_species_for_tree(adj_data$species)
cat("  Species:", nrow(adj_data), "\n")

# Match tree to data
cat("\nMatching tree to data...\n")

# Clean tree tips to species names
tree_species <- sapply(tree$tip.label, clean_tree_tip)
names(tree_species) <- tree$tip.label

# Create mapping from clean species name to original tree tip
species_to_tip <- setNames(names(tree_species), tree_species)

# Find matches
in_both <- intersect(tree_species, adj_data$tree_name)
cat("  Matched species:", length(in_both), "\n")

# Get the original tree tips that matched
matched_tips <- species_to_tip[in_both]

# Prune tree to matched species
if (length(matched_tips) < length(tree$tip.label)) {
  tips_to_drop <- setdiff(tree$tip.label, matched_tips)
  tree <- drop.tip(tree, tips_to_drop)
  cat("  Pruned tree tips:", length(tree$tip.label), "\n")
}

# Rename tree tips to clean species names for easier matching
tree$tip.label <- sapply(tree$tip.label, clean_tree_tip)

# Filter and reorder data to match tree
adj_data <- adj_data %>%
  filter(tree_name %in% tree$tip.label) %>%
  arrange(match(tree_name, tree$tip.label))

rownames(adj_data) <- adj_data$tree_name

cat("  Final matched:", nrow(adj_data), "\n")

# ============================================================
# 1. Phylogenetic PCA
# ============================================================

cat("\n" , rep("-", 59), "\n", sep = "")
cat("1. Phylogenetic PCA\n")
cat(rep("-", 59), "\n", sep = "")

# Key metrics for PCA (log-transform as in 2019 paper)
pca_vars <- c("m", "A", "Jc", "Jt", "m_dS", "m_dL")
pca_data <- adj_data[, pca_vars]

# Log transform (add small constant to avoid log(0))
pca_data_log <- log(pca_data + 0.001)
rownames(pca_data_log) <- adj_data$tree_name

# Run phylogenetic PCA
cat("Running phylogenetic PCA...\n")
phylo_pca <- phyl.pca(tree, pca_data_log, method = "lambda", mode = "cov")

# Extract scores
pca_scores <- as.data.frame(phylo_pca$S)
colnames(pca_scores) <- paste0("PC", 1:ncol(pca_scores))
pca_scores$species <- rownames(pca_scores)

# Calculate variance explained
eigenvalues <- diag(phylo_pca$Eval)
var_explained <- eigenvalues / sum(eigenvalues) * 100

cat("  Variance explained:\n")
for (i in 1:min(3, length(var_explained))) {
  cat(sprintf("    PC%d: %.1f%%\n", i, var_explained[i]))
}

# Save PCA scores
write.csv(pca_scores, pca_csv, row.names = FALSE)
cat("  Saved:", pca_csv, "\n")

# ============================================================
# 2. Blomberg's K (Phylogenetic Signal)
# ============================================================

cat("\n" , rep("-", 59), "\n", sep = "")
cat("2. Blomberg's K (Phylogenetic Signal)\n")
cat(rep("-", 59), "\n", sep = "")

# Test K for PC1-3 and raw metrics
test_vars <- c("PC1", "PC2", "PC3", pca_vars)
k_results <- list()

for (var in test_vars) {
  if (var %in% colnames(pca_scores)) {
    trait <- setNames(pca_scores[[var]], pca_scores$species)
  } else {
    trait <- setNames(adj_data[[var]], adj_data$tree_name)
  }

  # Ensure trait matches tree
  trait <- trait[tree$tip.label]

  # Calculate K
  k_test <- phylosig(tree, trait, method = "K", test = TRUE, nsim = 1000)

  k_results[[var]] <- data.frame(
    variable = var,
    K = k_test$K,
    p_value = k_test$P,
    stringsAsFactors = FALSE
  )

  cat(sprintf("  %s: K = %.3f, p = %.4f\n", var, k_test$K, k_test$P))
}

k_df <- bind_rows(k_results)
write.csv(k_df, blomberg_csv, row.names = FALSE)
cat("  Saved:", blomberg_csv, "\n")

# ============================================================
# 3. DTT (Disparity Through Time)
# ============================================================

cat("\n" , rep("-", 59), "\n", sep = "")
cat("3. DTT (Disparity Through Time)\n")
cat(rep("-", 59), "\n", sep = "")

dtt_results <- list()

for (pc in c("PC1", "PC2", "PC3")) {
  trait <- setNames(pca_scores[[pc]], pca_scores$species)
  trait <- trait[tree$tip.label]

  # Run DTT
  dtt_out <- dtt(tree, trait, nsim = 1000, calculateMDIp = TRUE, plot = FALSE)

  dtt_results[[pc]] <- data.frame(
    variable = pc,
    MDI = dtt_out$MDI,
    MDI_pvalue = dtt_out$MDIpVal,
    stringsAsFactors = FALSE
  )

  cat(sprintf("  %s: MDI = %.3f, p = %.4f\n", pc, dtt_out$MDI, dtt_out$MDIpVal))
}

dtt_df <- bind_rows(dtt_results)
write.csv(dtt_df, dtt_csv, row.names = FALSE)
cat("  Saved:", dtt_csv, "\n")

# ============================================================
# 4. Node Heights Test
# ============================================================

cat("\n" , rep("-", 59), "\n", sep = "")
cat("4. Node Heights Test (Rate Acceleration)\n")
cat(rep("-", 59), "\n", sep = "")

nh_results <- list()

for (pc in c("PC1", "PC2", "PC3")) {
  trait <- setNames(pca_scores[[pc]], pca_scores$species)
  trait <- trait[tree$tip.label]

  # Run node heights test
  nh_out <- run_node_heights_test(tree, trait)

  nh_results[[pc]] <- data.frame(
    variable = pc,
    slope = nh_out$slope,
    p_value = nh_out$p_value,
    r_squared = nh_out$r_squared,
    stringsAsFactors = FALSE
  )

  direction <- ifelse(nh_out$slope > 0, "acceleration", "deceleration")
  cat(sprintf("  %s: slope = %.4f (%s), p = %.4f\n",
              pc, nh_out$slope, direction, nh_out$p_value))
}

nh_df <- bind_rows(nh_results)
write.csv(nh_df, nodeheights_csv, row.names = FALSE)
cat("  Saved:", nodeheights_csv, "\n")

# ============================================================
# Summary
# ============================================================

cat("\n" , rep("=", 59), "\n", sep = "")
cat("Summary\n")
cat(rep("=", 59), "\n", sep = "")

cat("\nKey findings:\n")

# Blomberg K summary
sig_k <- k_df %>% filter(p_value < 0.05)
if (nrow(sig_k) > 0) {
  cat("  Significant phylogenetic signal (K):\n")
  for (i in 1:nrow(sig_k)) {
    cat(sprintf("    %s: K=%.3f (p=%.4f)\n",
                sig_k$variable[i], sig_k$K[i], sig_k$p_value[i]))
  }
} else {
  cat("  No significant phylogenetic signal detected\n")
}

# DTT summary
sig_dtt <- dtt_df %>% filter(MDI_pvalue < 0.05)
if (nrow(sig_dtt) > 0) {
  cat("\n  Significant DTT (MDI):\n")
  for (i in 1:nrow(sig_dtt)) {
    direction <- ifelse(sig_dtt$MDI[i] > 0, "higher than expected", "lower than expected")
    cat(sprintf("    %s: MDI=%.3f (%s, p=%.4f)\n",
                sig_dtt$variable[i], sig_dtt$MDI[i], direction, sig_dtt$MDI_pvalue[i]))
  }
}

# Node heights summary
sig_nh <- nh_df %>% filter(p_value < 0.05)
if (nrow(sig_nh) > 0) {
  cat("\n  Significant rate changes (node heights):\n")
  for (i in 1:nrow(sig_nh)) {
    direction <- ifelse(sig_nh$slope[i] > 0, "accelerating", "decelerating")
    cat(sprintf("    %s: %s toward present (p=%.4f)\n",
                sig_nh$variable[i], direction, sig_nh$p_value[i]))
  }
}

cat("\nDone! Results saved to:", results_dir, "\n")
