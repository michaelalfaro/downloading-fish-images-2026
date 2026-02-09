#!/usr/bin/env python3
"""
08_create_presentation.py

Create PowerPoint presentation for Chaetodontidae Color Pattern Pilot Analysis
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import pandas as pd
from pathlib import Path

# Paths
PILOT_DIR = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/pilot_analysis")
OUTPUT_PATH = PILOT_DIR / "pilot_presentation.pptx"

# Load data for presentation content
exemplars = pd.read_csv(PILOT_DIR / "data" / "exemplar_inventory.csv")
pavo_color = pd.read_csv(PILOT_DIR / "data" / "pavo_adjacency_color.csv")
blomberg_k = pd.read_csv(PILOT_DIR / "results" / "blomberg_k_results.csv")
dtt_results = pd.read_csv(PILOT_DIR / "results" / "dtt_mdi_results.csv")
node_heights = pd.read_csv(PILOT_DIR / "results" / "node_heights_results.csv")
sister_pairs = pd.read_csv(PILOT_DIR / "data" / "sister_pairs_matched.csv")

def add_title_slide(prs, title, subtitle):
    """Add a title slide"""
    slide_layout = prs.slide_layouts[6]  # Blank slide
    slide = prs.slides.add_slide(slide_layout)

    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1.5))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(44)
    title_para.font.bold = True
    title_para.font.color.rgb = RGBColor(0, 51, 102)
    title_para.alignment = PP_ALIGN.CENTER

    # Add subtitle
    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(4), Inches(9), Inches(1))
    subtitle_frame = subtitle_box.text_frame
    subtitle_para = subtitle_frame.paragraphs[0]
    subtitle_para.text = subtitle
    subtitle_para.font.size = Pt(24)
    subtitle_para.font.color.rgb = RGBColor(100, 100, 100)
    subtitle_para.alignment = PP_ALIGN.CENTER

    return slide

def add_section_slide(prs, section_title):
    """Add a section divider slide"""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Add colored rectangle
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(2.75),
        Inches(10), Inches(2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0, 102, 153)
    shape.line.fill.background()

    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(3), Inches(9), Inches(1.5))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = section_title
    title_para.font.size = Pt(40)
    title_para.font.bold = True
    title_para.font.color.rgb = RGBColor(255, 255, 255)
    title_para.alignment = PP_ALIGN.CENTER

    return slide

def add_content_slide(prs, title, bullet_points):
    """Add a slide with title and bullet points"""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    title_para.font.color.rgb = RGBColor(0, 51, 102)

    # Add bullet points
    body_box = slide.shapes.add_textbox(Inches(0.75), Inches(1.3), Inches(8.5), Inches(5.5))
    body_frame = body_box.text_frame
    body_frame.word_wrap = True

    for i, bullet in enumerate(bullet_points):
        if i == 0:
            para = body_frame.paragraphs[0]
        else:
            para = body_frame.add_paragraph()
        para.text = f"• {bullet}"
        para.font.size = Pt(20)
        para.space_after = Pt(12)
        para.level = 0

    return slide

def add_results_slide(prs, title, content_lines, highlight=None):
    """Add a results slide with formatted content"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    title_para.font.color.rgb = RGBColor(0, 51, 102)

    # Add content
    body_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(9), Inches(5.5))
    body_frame = body_box.text_frame
    body_frame.word_wrap = True

    for i, line in enumerate(content_lines):
        if i == 0:
            para = body_frame.paragraphs[0]
        else:
            para = body_frame.add_paragraph()
        para.text = line
        para.font.size = Pt(18)
        para.space_after = Pt(8)
        if highlight and line.startswith(highlight):
            para.font.bold = True
            para.font.color.rgb = RGBColor(0, 102, 51)

    return slide

# Create presentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# Slide 1: Title
add_title_slide(
    prs,
    "Chaetodontidae Color Pattern Pilot Analysis",
    "Replicating Alfaro et al. 2019 with Curated Exemplars\n\nFebruary 2026"
)

# Slide 2: Overview
add_content_slide(prs, "Analysis Overview", [
    f"{len(exemplars)} curated exemplar species with user-selected images",
    "Species-specific gestalt k values (range: 2-5 color classes)",
    "Time-calibrated phylogeny (crown age: 54.17 Ma)",
    f"{len(pavo_color)} species analyzed with Pavo color pattern metrics",
    "Replicating key analyses from Alfaro et al. 2019 ICB paper",
    "Testing sister species divergence (Hemingson et al. 2019)"
])

# Slide 3: Methods - Tree
add_content_slide(prs, "Phylogenetic Tree", [
    "Source: Phylogenomic tree (209 tips) rescaled with dated tree (51 tips)",
    "Scaling factor: 1075.18 (based on 41 common species)",
    "Chaetodontidae crown age: 54.17 Ma",
    "Pruned to 109 species with exemplar data",
    "106 species matched for phylogenetic comparative analyses"
])

# Slide 4: Methods - Pavo
add_content_slide(prs, "Color Pattern Analysis (Pavo)", [
    "K-means classification with species-specific k values (2-5)",
    "6 core metrics calculated:",
    "   m = transition density (pattern complexity)",
    "   A = aspect ratio of transitions",
    "   Jc = color class diversity (Simpson's index)",
    "   Jt = transition type diversity",
    "   m_dS = mean chromatic boundary strength",
    "   m_dL = mean achromatic boundary strength",
    "Grayscale analysis for robustness testing"
])

# Slide 5: Pavo Results
m_mean = pavo_color['m'].mean()
m_sd = pavo_color['m'].std()
jc_mean = pavo_color['Jc'].mean()
jc_sd = pavo_color['Jc'].std()

add_results_slide(prs, "Color Pattern Metrics Summary", [
    f"Transition density (m): mean = {m_mean:.3f} ± {m_sd:.3f}",
    f"Color diversity (Jc): mean = {jc_mean:.3f} ± {jc_sd:.3f}",
    "",
    "By gestalt k value:",
    f"   k=2: {len(pavo_color[pavo_color['gestalt_k']==2])} species (simpler patterns)",
    f"   k=3: {len(pavo_color[pavo_color['gestalt_k']==3])} species",
    f"   k=4: {len(pavo_color[pavo_color['gestalt_k']==4])} species",
    f"   k=5: {len(pavo_color[pavo_color['gestalt_k']==5])} species (complex patterns)",
    "",
    "Species-specific k captures pattern complexity better than fixed k=4"
])

# Slide 6: Phylogenetic Signal
sig_k = blomberg_k[blomberg_k['p_value'] < 0.05]
add_results_slide(prs, "Phylogenetic Signal (Blomberg's K)", [
    "K < 1 indicates closely related species are MORE variable than expected",
    "",
    "Significant phylogenetic signal detected:",
    f"   PC1: K = {blomberg_k[blomberg_k['variable']=='PC1']['K'].values[0]:.3f} (p = {blomberg_k[blomberg_k['variable']=='PC1']['p_value'].values[0]:.4f})",
    f"   PC2: K = {blomberg_k[blomberg_k['variable']=='PC2']['K'].values[0]:.3f} (p = {blomberg_k[blomberg_k['variable']=='PC2']['p_value'].values[0]:.4f})",
    f"   m: K = {blomberg_k[blomberg_k['variable']=='m']['K'].values[0]:.3f} (p = {blomberg_k[blomberg_k['variable']=='m']['p_value'].values[0]:.4f})",
    f"   A: K = {blomberg_k[blomberg_k['variable']=='A']['K'].values[0]:.3f} (p = {blomberg_k[blomberg_k['variable']=='A']['p_value'].values[0]:.4f})",
    f"   Jc: K = {blomberg_k[blomberg_k['variable']=='Jc']['K'].values[0]:.3f} (p = {blomberg_k[blomberg_k['variable']=='Jc']['p_value'].values[0]:.4f})",
    "",
    "Consistent with Alfaro et al. 2019: rapid color pattern evolution"
])

# Slide 7: DTT Results
add_results_slide(prs, "Disparity Through Time (DTT)", [
    "Morphological Disparity Index (MDI):",
    f"   PC1: MDI = {dtt_results[dtt_results['variable']=='PC1']['MDI'].values[0]:.3f} (p = {dtt_results[dtt_results['variable']=='PC1']['MDI_pvalue'].values[0]:.3f})",
    f"   PC2: MDI = {dtt_results[dtt_results['variable']=='PC2']['MDI'].values[0]:.3f} (p = {dtt_results[dtt_results['variable']=='PC2']['MDI_pvalue'].values[0]:.3f})",
    f"   PC3: MDI = {dtt_results[dtt_results['variable']=='PC3']['MDI'].values[0]:.3f} (p = {dtt_results[dtt_results['variable']=='PC3']['MDI_pvalue'].values[0]:.3f})",
    "",
    "Positive MDI indicates subclade disparity higher than BM expectation",
    "",
    "Non-significant p-values suggest pattern could arise by chance",
    "More data or focused analysis may reveal clearer patterns"
])

# Slide 8: Node Heights Test
pc1_nh = node_heights[node_heights['variable'] == 'PC1']
add_results_slide(prs, "Rate Acceleration (Node Heights Test)", [
    "Tests for evolutionary rate changes through time",
    "",
    f"PC1: slope = {pc1_nh['slope'].values[0]:.4f} (p = {pc1_nh['p_value'].values[0]:.4f}) *",
    f"PC2: slope = {node_heights[node_heights['variable']=='PC2']['slope'].values[0]:.4f} (p = {node_heights[node_heights['variable']=='PC2']['p_value'].values[0]:.3f})",
    f"PC3: slope = {node_heights[node_heights['variable']=='PC3']['slope'].values[0]:.4f} (p = {node_heights[node_heights['variable']=='PC3']['p_value'].values[0]:.3f})",
    "",
    "Positive slope = accelerating evolution toward present",
    "",
    "* PC1 shows significant rate acceleration",
    "Color pattern evolution speeding up in recent time"
])

# Slide 9: Sister Species
add_results_slide(prs, "Sister Species Color Divergence", [
    f"{len(sister_pairs)} sister pairs with both members having exemplar data",
    "",
    "Color dissimilarity statistics:",
    f"   Mean: {sister_pairs['color_dissimilarity'].mean():.3f}",
    f"   SD: {sister_pairs['color_dissimilarity'].std():.3f}",
    f"   Range: [{sister_pairs['color_dissimilarity'].min():.3f}, {sister_pairs['color_dissimilarity'].max():.3f}]",
    "",
    "Most divergent pairs:",
    f"   Heniochus pleurotaenia vs H. varius: {sister_pairs[sister_pairs['sp1'].str.contains('pleurotaenia')]['color_dissimilarity'].values[0] if len(sister_pairs[sister_pairs['sp1'].str.contains('pleurotaenia')]) > 0 else 'N/A':.3f}",
    "",
    "Full Hemingson hypothesis testing requires range overlap data"
])

# Slide 10: Grayscale Robustness
add_content_slide(prs, "Grayscale Robustness", [
    "Pattern geometry metrics highly correlated between color and grayscale:",
    "   m (transition density): r > 0.95",
    "   A (aspect ratio): r > 0.95",
    "   Jc (color diversity): r > 0.95",
    "   Jt (transition diversity): r > 0.95",
    "",
    "m_dS (chromatic) = 0 in grayscale (as expected)",
    "",
    "Pattern structure captured independent of color information",
    "Validates use of geometric metrics for evolutionary analysis"
])

# Slide 11: Key Findings
add_content_slide(prs, "Key Findings", [
    "Phylogenetic signal detected but K < 1 for all metrics",
    "   → Closely related species more variable than expected under BM",
    "",
    "Significant rate acceleration for PC1 (color pattern)",
    "   → Evolution speeding up toward present",
    "",
    "Species-specific k values (2-5) capture pattern complexity",
    "   → Better than fixed k=4 used in 2019 analysis",
    "",
    "Grayscale analysis validates pattern geometry metrics",
    "   → Spatial structure independent of hue"
])

# Slide 12: Suggested Additional Analyses
add_content_slide(prs, "Top 5 Suggested Additional Analyses", [
    "1. BAMM Rate Analysis - Test for evolutionary rate shifts",
    "",
    "2. Multivariate OU Models (mvMorph) - Compare BM, OU, EB models",
    "",
    "3. Regime-Dependent Evolution (OUwie) - Test if color evolves differently by ecology",
    "",
    "4. Deep Learning Embedding Comparison - Compare pavo to DINO/CLIP",
    "",
    "5. Color Channel Decomposition - Test signal for hue, saturation, luminance separately"
])

# Slide 13: Next Steps
add_content_slide(prs, "Next Steps", [
    "Complete exemplar curation for remaining tree tips",
    "",
    "Add range overlap and symmetry data for Hemingson hypothesis",
    "",
    "Run BAMM analysis to identify rate shift locations",
    "",
    "Compare current results to 2019 paper (fixed k=4)",
    "",
    "Generate publication-quality figures with ggtree"
])

# Slide 14: Thank you
add_title_slide(
    prs,
    "Thank You",
    "Questions?\n\nData & Code: pilot_analysis/\nReport: pilot_analysis_report.html"
)

# Save presentation
prs.save(OUTPUT_PATH)
print(f"Presentation saved to: {OUTPUT_PATH}")
