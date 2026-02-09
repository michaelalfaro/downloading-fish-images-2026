#!/usr/bin/env Rscript
# 02_rescale_tree.R
#
# Rescale the phylogenomic tree using a time-calibrated reference tree.
# - Loads dated tree (FinalDatedChaetTree_no_duplicates.tre)
# - Loads phylogenomic tree (Butterflyfish_concat_final.tre)
# - Rescales phylogenomic tree to absolute time
# - Prunes to Chaetodontidae species with exemplars
# - Saves rescaled and pruned trees

library(ape)
library(phytools)

# ============================================================
# Paths
# ============================================================

# Project directories
pilot_dir <- "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/pilot_analysis"
chaets_dir <- "/Users/michaelalfaro/Dropbox/git/chaets-divergence-2026"

# Input trees
dated_tree_path <- "/Users/michaelalfaro/Dropbox/FinalDatedChaetTree_no_duplicates.tre"
phylo_tree_path <- file.path(chaets_dir, "Butterflyfish_concat_final.tre")

# Input data
exemplar_csv <- file.path(pilot_dir, "data", "exemplar_inventory.csv")

# Output trees
output_dir <- file.path(pilot_dir, "trees")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

rescaled_tree_path <- file.path(output_dir, "pilot_tree_rescaled.tre")
pruned_tree_path <- file.path(output_dir, "pilot_tree_pruned.tre")
log_path <- file.path(output_dir, "tree_scaling_log.txt")

# ============================================================
# Helper functions
# ============================================================

# Clean tip labels to species name only (Genus_epithet)
clean_tip_label <- function(label) {
  # Remove specimen IDs and suffixes
  # E.g., "Chaetodon_auriga_CHD01" -> "Chaetodon_auriga"
  # E.g., "Chaetodon_auriga2" -> "Chaetodon_auriga"

  parts <- strsplit(label, "_")[[1]]
  if (length(parts) >= 2) {
    genus <- parts[1]
    epithet <- gsub("[0-9]+[a-z]*$", "", parts[2])  # Remove trailing numbers
    return(paste(genus, epithet, sep = "_"))
  }
  return(label)
}

# Standardize tip labels (title case genus, lowercase epithet)
standardize_label <- function(label) {
  parts <- strsplit(label, "_")[[1]]
  if (length(parts) >= 2) {
    genus <- paste0(toupper(substr(parts[1], 1, 1)),
                    tolower(substr(parts[1], 2, nchar(parts[1]))))
    epithet <- tolower(parts[2])
    return(paste(genus, epithet, sep = "_"))
  }
  return(label)
}

# Check if tip is Chaetodontidae
is_chaetodontidae <- function(label) {
  chaet_genera <- c("amphichaetodon", "chaetodon", "chelmon", "chelmonops",
                    "coradion", "forcipiger", "hemitaurichthys", "heniochus",
                    "johnrandallia", "parachaetodon", "prognathodes", "roa")
  genus <- tolower(strsplit(label, "_")[[1]][1])
  return(genus %in% chaet_genera)
}

# ============================================================
# Load and process trees
# ============================================================

cat("=" , rep("=", 59), "\n", sep = "")
cat("Tree Rescaling for Pilot Analysis\n")
cat("=" , rep("=", 59), "\n\n", sep = "")

# Start log
log_lines <- c(
  paste("Tree Rescaling Log"),
  paste("Timestamp:", Sys.time()),
  ""
)

# Load dated tree
cat("Loading dated tree...\n")
dated_tree <- read.nexus(dated_tree_path)
cat("  Tips:", length(dated_tree$tip.label), "\n")
log_lines <- c(log_lines, paste("Dated tree tips:", length(dated_tree$tip.label)))

# Load phylogenomic tree
cat("Loading phylogenomic tree...\n")
phylo_tree <- read.nexus(phylo_tree_path)
cat("  Tips:", length(phylo_tree$tip.label), "\n")
log_lines <- c(log_lines, paste("Phylogenomic tree tips:", length(phylo_tree$tip.label)))

# Load exemplar data
cat("Loading exemplar inventory...\n")
exemplars <- read.csv(exemplar_csv, stringsAsFactors = FALSE)
exemplar_species <- gsub(" ", "_", exemplars$species)
cat("  Exemplar species:", length(exemplar_species), "\n\n")
log_lines <- c(log_lines, paste("Exemplar species:", length(exemplar_species)), "")

# ============================================================
# Clean and match tip labels
# ============================================================

cat("Standardizing tip labels...\n")

# Clean dated tree tips
dated_tips_clean <- sapply(dated_tree$tip.label, function(x) {
  standardize_label(clean_tip_label(x))
})
names(dated_tips_clean) <- dated_tree$tip.label

# Clean phylogenomic tree tips
phylo_tips_clean <- sapply(phylo_tree$tip.label, function(x) {
  standardize_label(clean_tip_label(x))
})
names(phylo_tips_clean) <- phylo_tree$tip.label

# Find Chaetodontidae tips in phylogenomic tree
chaet_tips <- phylo_tree$tip.label[sapply(phylo_tree$tip.label, is_chaetodontidae)]
cat("  Chaetodontidae in phylogenomic tree:", length(chaet_tips), "\n")

# Find common species between dated and phylogenomic trees
dated_species <- unique(dated_tips_clean)
phylo_chaet_species <- unique(phylo_tips_clean[chaet_tips])

common_species <- intersect(dated_species, phylo_chaet_species)
cat("  Common species between trees:", length(common_species), "\n")
log_lines <- c(log_lines, paste("Common species:", length(common_species)))

# ============================================================
# Rescale using simple approach
# ============================================================

cat("\nRescaling tree...\n")

# Strategy: Calculate average rate (subs/site per Ma) from common clades
# Then multiply phylogenomic branch lengths by rate to get time

# Get root age from dated tree (total tree height)
dated_height <- max(nodeHeights(dated_tree))
cat("  Dated tree height (Ma):", round(dated_height, 2), "\n")

# Get phylogenomic tree height
phylo_height <- max(nodeHeights(phylo_tree))
cat("  Phylogenomic tree height (subs/site):", round(phylo_height, 4), "\n")

# Simple scaling factor approach
# We'll scale the phylogenomic tree to match the dated tree's crown age
# First, prune to Chaetodontidae only

chaet_tree <- drop.tip(phylo_tree,
                       phylo_tree$tip.label[!phylo_tree$tip.label %in% chaet_tips])
cat("  Chaetodontidae subtree tips:", length(chaet_tree$tip.label), "\n")

# Get Chaetodontidae crown height from dated tree
# Find chaetodontidae tips in dated tree
dated_chaet_tips <- dated_tree$tip.label[sapply(dated_tree$tip.label, function(x) {
  is_chaetodontidae(clean_tip_label(x))
})]

if (length(dated_chaet_tips) > 1) {
  # Get MRCA of Chaetodontidae in dated tree
  dated_chaet_mrca <- findMRCA(dated_tree, dated_chaet_tips)
  dated_chaet_height <- max(nodeHeights(dated_tree)[,2]) -
                        nodeHeights(dated_tree)[dated_chaet_mrca - length(dated_tree$tip.label), 2]
  cat("  Chaetodontidae crown age (Ma):", round(dated_chaet_height, 2), "\n")
} else {
  dated_chaet_height <- dated_height * 0.5  # Fallback
}

# Calculate scaling factor
chaet_phylo_height <- max(nodeHeights(chaet_tree))
scale_factor <- dated_chaet_height / chaet_phylo_height
cat("  Scaling factor:", round(scale_factor, 4), "Ma per subs/site\n")
log_lines <- c(log_lines,
               paste("Scaling factor:", round(scale_factor, 4)),
               paste("Chaetodontidae crown age:", round(dated_chaet_height, 2), "Ma"))

# Apply scaling to branch lengths
rescaled_tree <- chaet_tree
rescaled_tree$edge.length <- chaet_tree$edge.length * scale_factor

cat("  Rescaled tree height (Ma):", round(max(nodeHeights(rescaled_tree)), 2), "\n")

# ============================================================
# Prune to exemplar species
# ============================================================

cat("\nPruning to exemplar species...\n")

# Map exemplar species to tree tips
# Find the best matching tip for each exemplar species
matched_tips <- c()
matched_species <- c()

for (ex_sp in exemplar_species) {
  ex_clean <- standardize_label(ex_sp)

  # Find matching tips in rescaled tree
  tree_clean <- sapply(rescaled_tree$tip.label, function(x) {
    standardize_label(clean_tip_label(x))
  })

  matches <- names(tree_clean)[tree_clean == ex_clean]

  if (length(matches) > 0) {
    matched_tips <- c(matched_tips, matches[1])  # Take first match
    matched_species <- c(matched_species, ex_sp)
  }
}

cat("  Exemplar species matched to tree:", length(matched_tips), "of", length(exemplar_species), "\n")
log_lines <- c(log_lines,
               "",
               paste("Exemplars matched:", length(matched_tips), "of", length(exemplar_species)))

# Prune tree to matched tips
if (length(matched_tips) > 0) {
  tips_to_drop <- rescaled_tree$tip.label[!rescaled_tree$tip.label %in% matched_tips]
  pruned_tree <- drop.tip(rescaled_tree, tips_to_drop)
  cat("  Pruned tree tips:", length(pruned_tree$tip.label), "\n")
} else {
  cat("  WARNING: No matches found! Using full Chaetodontidae tree.\n")
  pruned_tree <- rescaled_tree
}

# ============================================================
# Save trees
# ============================================================

cat("\nSaving trees...\n")

# Save rescaled Chaetodontidae tree
write.tree(rescaled_tree, file = rescaled_tree_path)
cat("  Rescaled tree:", rescaled_tree_path, "\n")

# Save pruned tree (exemplars only)
write.tree(pruned_tree, file = pruned_tree_path)
cat("  Pruned tree:", pruned_tree_path, "\n")

# Write log
log_lines <- c(log_lines,
               "",
               paste("Rescaled tree saved:", rescaled_tree_path),
               paste("Pruned tree saved:", pruned_tree_path),
               "",
               "Matched tips:",
               matched_tips)

writeLines(log_lines, log_path)
cat("  Log:", log_path, "\n")

# ============================================================
# Summary
# ============================================================

cat("\n" , rep("=", 59), "\n", sep = "")
cat("Summary\n")
cat(rep("=", 59), "\n", sep = "")
cat("  Rescaled tree: ", length(rescaled_tree$tip.label), " tips (all Chaetodontidae)\n", sep = "")
cat("  Pruned tree: ", length(pruned_tree$tip.label), " tips (exemplars only)\n", sep = "")
cat("  Tree height: ", round(max(nodeHeights(pruned_tree)), 2), " Ma\n", sep = "")
cat("\nDone!\n")
