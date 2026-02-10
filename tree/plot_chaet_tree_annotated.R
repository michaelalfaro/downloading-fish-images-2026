#!/usr/bin/env Rscript
# Plot Chaetodontidae phylogeny with:
# - Lineage-colored terminal branches
# - Species names with lineage prefixes
# - Segmented fish images
# - Paraphyly annotations for problematic clades
# - Major clade labels

library(ape)
library(treeio)
library(ggtree)
library(ggimage)
library(ggplot2)
library(dplyr)

# --- Paths ---
script_dir <- "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/tree"
tree_file  <- file.path(script_dir, "Butterflyfish_concat_final.tre")
lookup_csv <- file.path(script_dir, "tip_image_lookup_segmented.csv")
out_pdf    <- file.path(script_dir, "chaet_tree_annotated.pdf")

# --- Read and root tree ---
tree <- read.nexus(tree_file)
tree <- root(tree, outgroup = "Morone_saxatilis_12185", resolve.root = TRUE)

cat("Tree has", Ntip(tree), "tips\n")

# =============================================================================
# LINEAGE DEFINITIONS
# Based on Bellwood et al. (2010), Fessler & Westneat (2007), and molecular phylogenetics
# =============================================================================

# Major clades within Chaetodontidae
# 1. BANNERFISHES (Heniochus clade) - includes Heniochus, Hemitaurichthys, Johnrandallia
# 2. LONG-SNOUTS (Forcipiger clade) - includes Forcipiger
# 3. CORALFISH (Chelmon clade) - includes Chelmon, Chelmonops, Coradion
# 4. DEEP BANNERFISHES - Amphichaetodon
# 5. PROGNATHODES - deep-water Chaetodon relatives
# 6. ROA - recently split from Chaetodon
# 7. PARACHAETODON - now nested in Chaetodon
# 8. CHAETODON - the main butterflyfish radiation (multiple subclades)

# Define lineage assignments for each Chaetodontidae genus/species
lineage_definitions <- list(
  # Bannerfishes and allies
  "Bannerfishes" = c("Heniochus", "Hemitaurichthys", "Johnrandallia"),

  # Long-snouted butterflyfish
  "Forcipiger" = c("Forcipiger"),

  # Coralfish clade
  "Coralfish" = c("Chelmon", "Chelmonops", "Coradion"),

  # Deep bannerfishes
  "Amphichaetodon" = c("Amphichaetodon"),

  # Prognathodes - Atlantic deep-water group
  "Prognathodes" = c("Prognathodes"),

  # Roa - split from Chaetodon
  "Roa" = c("Roa"),

  # Main Chaetodon radiation - we'll subdivide further
  "Chaetodon" = c("Chaetodon", "Parachaetodon")
)

# Chaetodon subgroups based on tree topology (informal names based on patterns)
# These represent major clades within Chaetodon
chaetodon_subgroups <- list(
  # Vagabond group (auriga, vagabundus, pictus, etc.)
  "Vagabond" = c("auriga", "vagabundus", "pictus", "decussatus", "adiergastos",
                 "auripes", "fasciatus", "lunula", "collare"),

  # Saddleback group (ephippium, xanthocephalus, semeion)
  "Saddleback" = c("ephippium", "xanthocephalus", "semeion", "dialeucos",
                   "mesoleucos", "nigropunctatus"),

  # Lineolatus group (lineolatus, falcula, ulietensis, etc.)
  "Lineolatus" = c("lineolatus", "falcula", "ulietensis", "oxycephalus",
                   "semilarvatus", "rafflesii"),

  # Melannotus group
  "Melannotus" = c("melannotus", "ocellicaudus", "gardineri", "selene", "leucopleura"),

  # Atlantic group (capistratus, striatus, humeralis, ocellatus)
  "Atlantic" = c("capistratus", "striatus", "humeralis", "ocellatus", "robustus"),

  # Rainfordi group (octofasciatus, rainfordi, aureofasciatus)
  "Rainfordi" = c("rainfordi", "octofasciatus", "aureofasciatus"),

  # Trifascialis group (trifascialis, bennetti, speculum, plebeius, tricinctus, zanzibarensis)
  "Trifascialis" = c("trifascialis", "bennetti", "speculum", "plebeius",
                     "tricinctus", "zanzibarensis"),

  # Baronessa group (baronessa, triangulum, larvatus)
  "Baronessa" = c("baronessa", "triangulum", "larvatus"),

  # Ornate group (ornatissimus, meyeri, reticulatus, trifasciatus, lunulatus, austriacus, melapterus)
  "Ornate" = c("ornatissimus", "meyeri", "reticulatus", "trifasciatus",
               "lunulatus", "austriacus", "melapterus"),

  # Citrinellus group (citrinellus, punctatofasciatus, guttatissimus, etc.)
  "Citrinellus" = c("citrinellus", "punctatofasciatus", "guttatissimus", "multicinctus",
                    "pelewensis", "miliaris", "assarius", "guentheri", "litus",
                    "blackburnii", "quadrimaculatus", "sedentarius", "sanctaehelenae"),

  # Tinkeri group (tinkeri, burgessi, declivis, mitratus, nippon, fremblii)
  "Tinkeri" = c("tinkeri", "burgessi", "declivis", "mitratus", "nippon", "fremblii",
                "argentatus", "madagaskariensis", "mertensii", "xanthurus", "paucifasciatus"),

  # Kleinii group
  "Kleinii" = c("kleinii", "trichrous", "interruptus", "unimaculatus")
)

# =============================================================================
# COLOR SCHEMES
# =============================================================================

# Major lineage colors (saturated, distinct hues)
lineage_colors <- c(
  "Bannerfishes"  = "#E31A1C",  # Red
  "Forcipiger"    = "#FF7F00",  # Orange
  "Coralfish"     = "#6A3D9A",  # Purple
  "Amphichaetodon"= "#B15928",  # Brown
  "Prognathodes"  = "#1F78B4",  # Blue
  "Roa"           = "#33A02C",  # Green
  "Chaetodon"     = "#FB9A99",  # Light red (will use subgroup colors)
  "Outgroup"      = "#999999"   # Gray
)

# Chaetodon subgroup colors (hues of pink/red/orange)
chaetodon_colors <- c(
  "Vagabond"    = "#E41A1C",  # Red
  "Saddleback"  = "#FF6B6B",  # Light red
  "Lineolatus"  = "#FC8D62",  # Salmon
  "Melannotus"  = "#E78AC3",  # Pink
  "Atlantic"    = "#984EA3",  # Purple
  "Rainfordi"   = "#A6D854",  # Yellow-green
  "Trifascialis"= "#FFD92F",  # Yellow
  "Baronessa"   = "#E5C494",  # Tan
  "Ornate"      = "#B3B3B3",  # Light gray
  "Citrinellus" = "#66C2A5",  # Teal
  "Tinkeri"     = "#8DA0CB",  # Periwinkle
  "Kleinii"     = "#A6CEE3",  # Light blue
  "Other"       = "#FDBF6F"   # Light orange
)

# =============================================================================
# ASSIGN LINEAGES TO TIPS
# =============================================================================

assign_lineage <- function(tip_label) {
  parts <- strsplit(tip_label, "_")[[1]]
  genus <- parts[1]
  genus <- tools::toTitleCase(tolower(genus))

  # Check each lineage definition
  for (lineage in names(lineage_definitions)) {
    if (genus %in% lineage_definitions[[lineage]]) {
      return(lineage)
    }
  }
  return("Outgroup")
}

assign_chaetodon_subgroup <- function(tip_label) {
  parts <- strsplit(tip_label, "_")[[1]]
  genus <- parts[1]
  genus <- tools::toTitleCase(tolower(genus))

  if (!(genus %in% c("Chaetodon", "Parachaetodon"))) {
    return(NA)
  }

  # Get species epithet
  if (length(parts) < 2) return("Other")
  epithet <- tolower(parts[2])

  # Check each subgroup
  for (subgroup in names(chaetodon_subgroups)) {
    if (epithet %in% chaetodon_subgroups[[subgroup]]) {
      return(subgroup)
    }
  }
  return("Other")
}

get_display_name <- function(tip_label, lineage, subgroup) {
  parts <- strsplit(tip_label, "_")[[1]]
  genus <- tools::toTitleCase(tolower(parts[1]))

  if (length(parts) >= 2) {
    epithet <- parts[2]
  } else {
    epithet <- ""
  }

  # Build display name with lineage prefix
  if (lineage == "Chaetodon" && !is.na(subgroup)) {
    prefix <- paste0("[", subgroup, "] ")
  } else if (lineage != "Outgroup") {
    prefix <- paste0("[", lineage, "] ")
  } else {
    prefix <- ""
  }

  species_name <- paste(genus, epithet)
  return(paste0(prefix, species_name))
}

get_tip_color <- function(lineage, subgroup) {
  if (lineage == "Chaetodon" && !is.na(subgroup)) {
    return(chaetodon_colors[subgroup])
  } else {
    return(lineage_colors[lineage])
  }
}

# =============================================================================
# BUILD TIP DATA
# =============================================================================

# Read image lookup
lookup <- read.csv(lookup_csv, stringsAsFactors = FALSE)

# Build tip metadata
tips <- tree$tip.label
tip_data <- data.frame(
  label = tips,
  stringsAsFactors = FALSE
)

# Assign lineages and colors
tip_data$lineage <- sapply(tip_data$label, assign_lineage)
tip_data$subgroup <- sapply(tip_data$label, assign_chaetodon_subgroup)
tip_data$display_name <- mapply(get_display_name, tip_data$label, tip_data$lineage, tip_data$subgroup)
tip_data$tip_color <- mapply(get_tip_color, tip_data$lineage, tip_data$subgroup)

# Define Chaetodontidae genera for filtering
chaet_genera <- tolower(c(
  "Amphichaetodon", "Chaetodon", "Chelmon", "Chelmonops",
  "Coradion", "Forcipiger", "Hemitaurichthys", "Heniochus",
  "Johnrandallia", "Parachaetodon", "Prognathodes", "Roa"
))

tip_data$genus <- tolower(sub("_.*", "", tip_data$label))
tip_data$is_chaet <- tip_data$genus %in% chaet_genera

# Merge with image lookup
tip_data <- merge(tip_data, lookup, by.x = "label", by.y = "tip_label", all.x = TRUE)

# Replace "NA" strings with actual NA
tip_data$img_path[tip_data$img_path == "NA"] <- NA_character_
tip_data$img_path <- ifelse(
  !is.na(tip_data$img_path) & file.exists(tip_data$img_path),
  tip_data$img_path,
  NA_character_
)

# Summary stats
n_chaet <- sum(tip_data$is_chaet)
n_img <- sum(!is.na(tip_data$img_path))
cat("Chaetodontidae tips:", n_chaet, "\n")
cat("Tips with images:", n_img, "\n")

# Lineage breakdown
cat("\nLineage breakdown:\n")
print(table(tip_data$lineage[tip_data$is_chaet]))

cat("\nChaetodon subgroups:\n")
print(table(tip_data$subgroup[!is.na(tip_data$subgroup)]))

# =============================================================================
# PARAPHYLY ANNOTATIONS
# Based on traditional taxonomy vs. molecular phylogeny
# =============================================================================

# Known paraphyletic issues in traditional Chaetodon subgenera:
# 1. Exornator (traditional) - polyphyletic across tree
# 2. Subgenus Chaetodon - polyphyletic
# 3. Rabdophorus - mostly monophyletic but some issues

# We'll mark these as annotations if we can identify them from tree structure
# For now, we'll add manual annotations for known problem areas

paraphyly_notes <- data.frame(
  node_tip = c(
    "Parachaetodon_ocellatus_PW1643",  # Nested within Chaetodon
    "Chaetodon_robustus_CRB01"         # Sister to all other Chaetodon (Atlantic isolate)
  ),
  note = c(
    "Parachaetodon nested in Chaetodon",
    "Atlantic basal Chaetodon"
  ),
  stringsAsFactors = FALSE
)

# =============================================================================
# BUILD THE TREE PLOT
# =============================================================================

# Create the tree with colored edges
p <- ggtree(tree, layout = "rectangular", branch.length = "none", size = 0.4) %<+% tip_data

# Get tree data for positioning
tree_data <- p$data
max_x <- max(tree_data$x, na.rm = TRUE)

# Add colored tip labels with lineage prefixes
p <- p + geom_tiplab(
  aes(label = display_name, color = tip_color),
  size = 1.6,
  offset = max_x * 0.01,
  show.legend = FALSE
) +
scale_color_identity()

# Add fish images
p <- p + geom_tiplab(
  aes(image = img_path, subset = !is.na(img_path)),
  geom = "image",
  offset = max_x * 0.55,
  align = max_x * 0.55,
  linetype = "dashed",
  linesize = 0.15,
  size = 0.0035,
  asp = 2.0
)

# Expand x-axis
p <- p +
  xlim(NA, max_x * 2.2) +
  coord_cartesian(clip = "off") +
  theme(plot.margin = margin(10, 180, 10, 10, unit = "pt"))

# Add title
p <- p + ggtitle("Chaetodontidae Phylogeny with Lineage Annotations",
                 subtitle = "Lineage prefixes in brackets; tips colored by clade")

# =============================================================================
# ADD CLADE LABELS AND BARS
# =============================================================================

# Add clade labels for major lineages
lineages_to_label <- c("Bannerfishes", "Forcipiger", "Coralfish",
                        "Amphichaetodon", "Prognathodes", "Roa")

for (lin in lineages_to_label) {
  lin_tips <- tip_data$label[tip_data$lineage == lin]
  if (length(lin_tips) >= 2) {
    mrca_node <- getMRCA(tree, lin_tips)
    if (!is.null(mrca_node)) {
      p <- p + geom_cladelab(
        node = mrca_node,
        label = lin,
        color = lineage_colors[lin],
        offset = max_x * 1.05,
        fontsize = 2.5,
        barsize = 1.2,
        angle = 0,
        offset.text = 0.01,
        align = TRUE
      )
    }
  }
}

# Add Chaetodon clade label (the big one)
chaetodon_tips <- tip_data$label[tip_data$lineage == "Chaetodon"]
if (length(chaetodon_tips) >= 2) {
  mrca_node <- getMRCA(tree, chaetodon_tips)
  if (!is.null(mrca_node)) {
    p <- p + geom_cladelab(
      node = mrca_node,
      label = "Chaetodon sensu lato",
      color = "#E31A1C",
      offset = max_x * 1.05,
      fontsize = 2.5,
      barsize = 1.5,
      angle = 0,
      offset.text = 0.01,
      align = TRUE
    )
  }
}

# =============================================================================
# ADD PARAPHYLY ANNOTATIONS
# =============================================================================

# Get plot data with y-positions
plot_data <- p$data

# Find y-positions for specific tips
parachaetodon_row <- plot_data[plot_data$label == "Parachaetodon_ocellatus_PW1643" & !is.na(plot_data$label), ]
robustus_row <- plot_data[plot_data$label == "Chaetodon_robustus_CRB01" & !is.na(plot_data$label), ]

# Add annotations as text labels with arrows
if (nrow(parachaetodon_row) > 0) {
  para_y <- parachaetodon_row$y[1]
  p <- p + annotate("segment",
           x = max_x * 0.38, xend = max_x * 0.48,
           y = para_y, yend = para_y,
           color = "red", arrow = arrow(length = unit(0.1, "cm"))) +
    annotate("text", x = max_x * 0.25, y = para_y,
             label = "Parachaetodon nested in Chaetodon",
             color = "red", size = 1.8, hjust = 0, fontface = "italic")
}

if (nrow(robustus_row) > 0) {
  rob_y <- robustus_row$y[1]
  p <- p + annotate("segment",
           x = max_x * 0.38, xend = max_x * 0.48,
           y = rob_y, yend = rob_y,
           color = "darkblue", arrow = arrow(length = unit(0.1, "cm"))) +
    annotate("text", x = max_x * 0.25, y = rob_y,
             label = "Basal Atlantic Chaetodon",
             color = "darkblue", size = 1.8, hjust = 0, fontface = "italic")
}

# =============================================================================
# LEGEND
# =============================================================================

# Create a separate legend plot or add annotations
# For now, the colors in labels serve as the legend

# =============================================================================
# SAVE PDF
# =============================================================================

n_tips <- Ntip(tree)
pdf_h <- max(35, n_tips * 0.5)
pdf_w <- 20

cat("\nSaving PDF:", out_pdf, "\n")
cat("  dimensions:", pdf_w, "x", pdf_h, "inches\n")

ggsave(
  out_pdf,
  plot = p,
  width = pdf_w,
  height = pdf_h,
  limitsize = FALSE
)

cat("Done!\n")

# =============================================================================
# COLOR LEGEND AS SEPARATE FILE
# =============================================================================

# Create a legend showing all lineage colors
legend_data <- data.frame(
  lineage = c(names(lineage_colors)[names(lineage_colors) != "Chaetodon"],
              paste0("Chaetodon: ", names(chaetodon_colors))),
  color = c(lineage_colors[names(lineage_colors) != "Chaetodon"],
            chaetodon_colors),
  stringsAsFactors = FALSE
)

legend_plot <- ggplot(legend_data, aes(x = 1, y = rev(seq_along(lineage)), fill = color)) +
  geom_tile(width = 0.3, height = 0.8) +
  geom_text(aes(label = lineage, x = 1.25), hjust = 0, size = 3) +
  scale_fill_identity() +
  xlim(0.7, 3) +
  theme_void() +
  ggtitle("Lineage Color Legend")

legend_pdf <- file.path(script_dir, "chaet_lineage_legend.pdf")
ggsave(legend_pdf, plot = legend_plot, width = 5, height = 8)
cat("Legend saved to:", legend_pdf, "\n")
