# Taxonomy and Source Reconciliation Report

**Generated:** 2026-02-09 17:50:54

## Summary

This report documents the reconciliation of taxonomy and source attribution between
Bishop Museum (Randall) and FishBase image collections.

---

## Taxonomy Issues Resolved

### Roa Genus (formerly Chaetodon)

The Bishop Museum collection uses the old genus assignment "Chaetodon" for three species
that are now placed in genus Roa:

| Bishop Name | Accepted Name | Resolution |
|-------------|---------------|------------|
| Chaetodon excelsa | Roa excelsa | Merged (same species) |
| Chaetodon jayakari | Roa jayakari | No Bishop image (FishBase only) |
| Chaetodon modesta | Roa modesta | No Bishop image (FishBase only) |

**Finding:** The Bishop `Chaetodon_excelsa` image and FishBase `Roa_excelsa` image
are the SAME Randall photograph. We now use the Bishop version with updated taxonomy.

### Zanzibar Butterflyfish Spelling

| Variant | Source | Resolution |
|---------|--------|------------|
| Chaetodon zanzibariensis | Bishop Museum | Preferred (correct spelling) |
| Chaetodon zanzibarensis | FishBase | Synonym, removed as duplicate |

**Note:** The correct spelling is "zanzibarensis" (from Zanzibar). We keep the Bishop
image and remove the FishBase duplicate entry.

---

## Image Source Attribution

### Bishop Museum / Randall Images

All 178 images in `images_bishop/` are from the John Randall collection at Bishop Museum.
These are photographed in controlled conditions and represent the gold standard for
color reference.

### FishBase Randall Images

The following species have Randall-credited images on FishBase that are NOT in the
Bishop collection (i.e., FishBase-only Randall images):

| Species | FishBase File | Notes |
|---------|---------------|-------|
| Chaetodon burgessi | Chbur_u0.jpg | Alternative to current exemplar |
| Chaetodon capistratus | Chcap_u0.jpg | Alternative to current exemplar |
| Chaetodon interruptus | Chuni_u2.jpg | **Currently used as exemplar** |
| Chaetodon oxycephalus | Choxy_u0.jpg | Alternative available |
| Chaetodon quadrimaculatus | Chqua_u2.jpg | Currently using juvenile (j0) |
| Chaetodon sedentarius | Chsed_u1.jpg | **Currently used as exemplar** |
| Chaetodon striatus | Chstr_u0.jpg | Alternative available |
| Chaetodon triangulum | Chtri_ui.jpg | Alternative available |
| Hemitaurichthys thompsoni | Hetho_u0.jpg | **Currently used as exemplar** |
| Johnrandallia nigrirostris | Jonig_u0.jpg | Alternative available |
| Prognathodes aculeatus | Chacu_u1.jpg | Alternative available |
| Roa excelsa | Chexc_u0.jpg | Same as Bishop image |
| Roa jayakari | Chjay_u1.jpg | Randall, no Bishop equivalent |
| Roa modesta | Chmod_u1.jpg | **Currently used as exemplar** |

---

## Duplicate Images

Images that appear in both Bishop and FishBase collections (same Randall photograph):

| Species | Bishop File | FishBase File | Hash Distance |
|---------|-------------|---------------|---------------|
| Amphichaetodon howensis | Amphichaetodon_howensis_Bishop_-369771762.jpg | Amphichaetodon_howensis_FishBase_Amhow_u3.jpg | 2 |
| Amphichaetodon howensis | Amphichaetodon_howensis_Bishop_-1464681654.jpg | Amphichaetodon_howensis_FishBase_Amhow_u2.jpg | 5 |
| Amphichaetodon howensis | Amphichaetodon_howensis_Bishop_472868867.jpg | Amphichaetodon_howensis_FishBase_Amhow_u1.jpg | 2 |
| Amphichaetodon melbae | Amphichaetodon_melbae_Bishop_-1115161976.jpg | Amphichaetodon_melbae_FishBase_Ammel_u0.jpg | 4 |
| Chaetodon adiergastos | Chaetodon_adiergastos_Bishop_1979529705.jpg | Chaetodon_adiergastos_FishBase_Chadi_u2.jpg | 45 |
| Chaetodon adiergastos | Chaetodon_adiergastos_Bishop_1979529705.jpg | Chaetodon_adiergastos_FishBase_Chadi_u1.jpg | 1 |
| Chaetodon andamanensis | Chaetodon_andamanensis_Bishop_-322272266.jpg | Chaetodon_andamanensis_FishBase_Chand_u0.jpg | 2 |
| Chaetodon argentatus | Chaetodon_argentatus_Bishop_1074668308.jpg | Chaetodon_argentatus_FishBase_Charg_u2.jpg | 9 |
| Chaetodon argentatus | Chaetodon_argentatus_Bishop_1101542975.jpg | Chaetodon_argentatus_FishBase_Charg_u1.jpg | 0 |
| Chaetodon aureofasciatus | Chaetodon_aureofasciatus_Bishop_1459626629.jpg | Chaetodon_aureofasciatus_FishBase_Chaur_u1.jpg | 1 |
| Chaetodon aureofasciatus | Chaetodon_aureofasciatus_Bishop_1459626629.jpg | Chaetodon_aureofasciatus_FishBase_Chaur_ut.jpg | 49 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_1547714619.jpg | Chaetodon_auriga_FishBase_Chaur_u9.jpg | 4 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_843681634.jpg | Chaetodon_auriga_FishBase_Chaur_u8.jpg | 5 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_-2122348146.jpg | Chaetodon_auriga_FishBase_Chaur_ud.jpg | 40 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_-2122348146.jpg | Chaetodon_auriga_FishBase_Chaur_u0.jpg | 30 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_1449897185.jpg | Chaetodon_auriga_FishBase_Chaur_u0.jpg | 46 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_-2122348146.jpg | Chaetodon_auriga_FishBase_Chaur_u5.jpg | 1 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_1449897185.jpg | Chaetodon_auriga_FishBase_Chaur_u5.jpg | 35 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_-2122348146.jpg | Chaetodon_auriga_FishBase_Chaur_u4.jpg | 39 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_1449897185.jpg | Chaetodon_auriga_FishBase_Chaur_u4.jpg | 51 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_-2122348146.jpg | Chaetodon_auriga_FishBase_Chaur_u6.jpg | 36 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_1449897185.jpg | Chaetodon_auriga_FishBase_Chaur_u6.jpg | 4 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_415255648.jpg | Chaetodon_auriga_FishBase_Chaur_u7.jpg | 0 |
| Chaetodon auriga | Chaetodon_auriga_Bishop_1988149751.jpg | Chaetodon_auriga_FishBase_Chaur_ua.jpg | 3 |
| Chaetodon auripes | Chaetodon_auripes_Bishop_-346327811.jpg | Chaetodon_auripes_FishBase_Chaur_ub.jpg | 4 |
| Chaetodon auripes | Chaetodon_auripes_Bishop_1839596140.jpg | Chaetodon_auripes_FishBase_Chaur_uc.jpg | 1 |
| Chaetodon auripes | Chaetodon_auripes_Bishop_1839596140.jpg | Chaetodon_auripes_FishBase_Chaur_u10.jpg | 46 |
| Chaetodon auripes | Chaetodon_auripes_Bishop_1839596140.jpg | Chaetodon_auripes_FishBase_Chaur_uo.jpg | 46 |
| Chaetodon austriacus | Chaetodon_austriacus_Bishop_-1737564550.jpg | Chaetodon_austriacus_FishBase_Chaus_u1.jpg | 3 |
| Chaetodon baronessa | Chaetodon_baronessa_Bishop_601798515.jpg | Chaetodon_baronessa_FishBase_Chbar_u4.jpg | 3 |
| Chaetodon baronessa | Chaetodon_baronessa_Bishop_918345433.jpg | Chaetodon_baronessa_FishBase_Chbar_u1.jpg | 3 |
| Chaetodon baronessa | Chaetodon_baronessa_Bishop_1152880952.jpg | Chaetodon_baronessa_FishBase_Chbar_u0.jpg | 33 |
| Chaetodon baronessa | Chaetodon_baronessa_Bishop_-2122794458.jpg | Chaetodon_baronessa_FishBase_Chbar_u0.jpg | 3 |
| Chaetodon baronessa | Chaetodon_baronessa_Bishop_1152880952.jpg | Chaetodon_baronessa_FishBase_Chbar_u3.jpg | 3 |
| Chaetodon baronessa | Chaetodon_baronessa_Bishop_-2122794458.jpg | Chaetodon_baronessa_FishBase_Chbar_u3.jpg | 33 |
| Chaetodon bennetti | Chaetodon_bennetti_Bishop_711591108.jpg | Chaetodon_bennetti_FishBase_Chben_u2.jpg | 1 |
| Chaetodon bennetti | Chaetodon_bennetti_Bishop_-2064196433.jpg | Chaetodon_bennetti_FishBase_Chben_u2.jpg | 23 |
| Chaetodon bennetti | Chaetodon_bennetti_Bishop_711591108.jpg | Chaetodon_bennetti_FishBase_Chben_u1.jpg | 23 |
| Chaetodon bennetti | Chaetodon_bennetti_Bishop_-2064196433.jpg | Chaetodon_bennetti_FishBase_Chben_u1.jpg | 1 |
| Chaetodon bennetti | Chaetodon_bennetti_Bishop_-2064196433.jpg | Chaetodon_bennetti_FishBase_Chben_u6.jpg | 47 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_1288243346.jpg | Chaetodon_blackburnii_FishBase_Chbla_u0.jpg | 42 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_-410407509.jpg | Chaetodon_blackburnii_FishBase_Chbla_u0.jpg | 35 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_300609141.jpg | Chaetodon_blackburnii_FishBase_Chbla_u0.jpg | 7 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_1288243346.jpg | Chaetodon_blackburnii_FishBase_Chbla_f0.jpg | 20 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_-410407509.jpg | Chaetodon_blackburnii_FishBase_Chbla_f0.jpg | 1 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_300609141.jpg | Chaetodon_blackburnii_FishBase_Chbla_f0.jpg | 37 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_1288243346.jpg | Chaetodon_blackburnii_FishBase_Chbla_u1.jpg | 1 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_-410407509.jpg | Chaetodon_blackburnii_FishBase_Chbla_u1.jpg | 18 |
| Chaetodon blackburnii | Chaetodon_blackburnii_Bishop_300609141.jpg | Chaetodon_blackburnii_FishBase_Chbla_u1.jpg | 44 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_1043818087.jpg | Chaetodon_citrinellus_FishBase_Chcit_u5.jpg | 28 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_-1937929700.jpg | Chaetodon_citrinellus_FishBase_Chcit_u5.jpg | 0 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_-1736280642.jpg | Chaetodon_citrinellus_FishBase_Chcit_u5.jpg | 19 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_1043818087.jpg | Chaetodon_citrinellus_FishBase_Chcit_u4.jpg | 2 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_-1937929700.jpg | Chaetodon_citrinellus_FishBase_Chcit_u4.jpg | 28 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_-1736280642.jpg | Chaetodon_citrinellus_FishBase_Chcit_u4.jpg | 21 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_1043818087.jpg | Chaetodon_citrinellus_FishBase_Chcit_u0.jpg | 39 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_-1736280642.jpg | Chaetodon_citrinellus_FishBase_Chcit_u0.jpg | 48 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_-1191499504.jpg | Chaetodon_citrinellus_FishBase_Chcit_u1.jpg | 5 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_1043818087.jpg | Chaetodon_citrinellus_FishBase_Chcit_u3.jpg | 24 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_-1937929700.jpg | Chaetodon_citrinellus_FishBase_Chcit_u3.jpg | 18 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_-1736280642.jpg | Chaetodon_citrinellus_FishBase_Chcit_u3.jpg | 3 |
| Chaetodon citrinellus | Chaetodon_citrinellus_Bishop_520787921.jpg | Chaetodon_citrinellus_FishBase_Chcit_u2.jpg | 8 |
| Chaetodon collare | Chaetodon_collare_Bishop_1999007469.jpg | Chaetodon_collare_FishBase_Chcol_u2.jpg | 40 |
| Chaetodon collare | Chaetodon_collare_Bishop_1403678122.jpg | Chaetodon_collare_FishBase_Chcol_u2.jpg | 0 |
| Chaetodon collare | Chaetodon_collare_Bishop_1999007469.jpg | Chaetodon_collare_FishBase_Chcol_u3.jpg | 4 |
| Chaetodon collare | Chaetodon_collare_Bishop_1403678122.jpg | Chaetodon_collare_FishBase_Chcol_u3.jpg | 38 |
| Chaetodon daedalma | Chaetodon_daedalma_Bishop_1811677923.jpg | Chaetodon_daedalma_FishBase_Chdae_u1.jpg | 5 |
| Chaetodon declivis | Chaetodon_declivis_Bishop_1376440265.jpg | Chaetodon_declivis_FishBase_Chdec_u2.jpg | 32 |
| Chaetodon declivis | Chaetodon_declivis_Bishop_-274435096.jpg | Chaetodon_declivis_FishBase_Chdec_u2.jpg | 1 |
| Chaetodon declivis | Chaetodon_declivis_Bishop_1376440265.jpg | Chaetodon_declivis_FishBase_Chdec_u3.jpg | 3 |
| Chaetodon declivis | Chaetodon_declivis_Bishop_-274435096.jpg | Chaetodon_declivis_FishBase_Chdec_u3.jpg | 34 |
| Chaetodon declivis | Chaetodon_declivis_Bishop_-405463978.jpg | Chaetodon_declivis_FishBase_Chdec_u1.jpg | 9 |
| Chaetodon decussatus | Chaetodon_decussatus_Bishop_527122207.jpg | Chaetodon_decussatus_FishBase_Chdec_u4.jpg | 3 |
| Chaetodon dialeucos | Chaetodon_dialeucos_Bishop_1297085044.jpg | Chaetodon_dialeucos_FishBase_Chdia_u0.jpg | 1 |
| Chaetodon dolosus | Chaetodon_dolosus_Bishop_-1008609691.jpg | Chaetodon_dolosus_FishBase_Chdol_u1.jpg | 3 |
| Chaetodon ephippium | Chaetodon_ephippium_Bishop_-1248760382.jpg | Chaetodon_ephippium_FishBase_Cheph_u0.jpg | 42 |
| Chaetodon ephippium | Chaetodon_ephippium_Bishop_-1179871461.jpg | Chaetodon_ephippium_FishBase_Cheph_uf.jpg | 46 |
| Chaetodon ephippium | Chaetodon_ephippium_Bishop_-1248760382.jpg | Chaetodon_ephippium_FishBase_Cheph_u3.jpg | 27 |
| Chaetodon ephippium | Chaetodon_ephippium_Bishop_-1179871461.jpg | Chaetodon_ephippium_FishBase_Cheph_u3.jpg | 0 |
| Chaetodon ephippium | Chaetodon_ephippium_Bishop_-1248760382.jpg | Chaetodon_ephippium_FishBase_Cheph_u5.jpg | 1 |
| Chaetodon ephippium | Chaetodon_ephippium_Bishop_-1179871461.jpg | Chaetodon_ephippium_FishBase_Cheph_u5.jpg | 28 |
| Chaetodon falcula | Chaetodon_falcula_Bishop_1821335233.jpg | Chaetodon_falcula_FishBase_Chfal_u4.jpg | 49 |
| Chaetodon falcula | Chaetodon_falcula_Bishop_1821335233.jpg | Chaetodon_falcula_FishBase_Chfal_u1.jpg | 4 |
| Chaetodon fasciatus | Chaetodon_fasciatus_Bishop_-1886056978.jpg | Chaetodon_fasciatus_FishBase_Chfas_u1.jpg | 38 |
| Chaetodon fasciatus | Chaetodon_fasciatus_Bishop_2100774615.jpg | Chaetodon_fasciatus_FishBase_Chfas_u1.jpg | 4 |
| Chaetodon fasciatus | Chaetodon_fasciatus_Bishop_-1886056978.jpg | Chaetodon_fasciatus_FishBase_Chfas_u6.jpg | 12 |
| Chaetodon fasciatus | Chaetodon_fasciatus_Bishop_2100774615.jpg | Chaetodon_fasciatus_FishBase_Chfas_u6.jpg | 42 |
| Chaetodon flavirostris | Chaetodon_flavirostris_Bishop_1864146380.jpg | Chaetodon_flavirostris_FishBase_Chfla_u3.jpg | 2 |
| Chaetodon flavocoronatus | Chaetodon_flavocoronatus_Bishop_-1261632291.jpg | Chaetodon_flavocoronatus_FishBase_Chfla_u4.jpg | 28 |
| Chaetodon flavocoronatus | Chaetodon_flavocoronatus_Bishop_-1599575846.jpg | Chaetodon_flavocoronatus_FishBase_Chfla_u4.jpg | 1 |
| Chaetodon flavocoronatus | Chaetodon_flavocoronatus_Bishop_-1261632291.jpg | Chaetodon_flavocoronatus_FishBase_Chfla_u5.jpg | 3 |
| Chaetodon flavocoronatus | Chaetodon_flavocoronatus_Bishop_-1599575846.jpg | Chaetodon_flavocoronatus_FishBase_Chfla_u5.jpg | 32 |
| Chaetodon fremblii | Chaetodon_fremblii_Bishop_5130835.jpg | Chaetodon_fremblii_FishBase_Chfre_u0.jpg | 46 |
| Chaetodon fremblii | Chaetodon_fremblii_Bishop_5130835.jpg | Chaetodon_fremblii_FishBase_Chfre_u1.jpg | 3 |
| Chaetodon gardineri | Chaetodon_gardineri_Bishop_-1706450280.jpg | Chaetodon_gardineri_FishBase_Chgar_u1.jpg | 4 |
| Chaetodon gardineri | Chaetodon_gardineri_Bishop_-1706450280.jpg | Chaetodon_gardineri_FishBase_Chgar_u0.jpg | 36 |
| Chaetodon guentheri | Chaetodon_guentheri_Bishop_-184146759.jpg | Chaetodon_guentheri_FishBase_Chgue_u1.jpg | 2 |
| Chaetodon guttatissimus | Chaetodon_guttatissimus_Bishop_-397235578.jpg | Chaetodon_guttatissimus_FishBase_Chgut_u0.jpg | 33 |
| Chaetodon guttatissimus | Chaetodon_guttatissimus_Bishop_-1623433841.jpg | Chaetodon_guttatissimus_FishBase_Chgut_u0.jpg | 2 |
| Chaetodon guttatissimus | Chaetodon_guttatissimus_Bishop_-397235578.jpg | Chaetodon_guttatissimus_FishBase_Chgut_u1.jpg | 4 |
| Chaetodon guttatissimus | Chaetodon_guttatissimus_Bishop_-1623433841.jpg | Chaetodon_guttatissimus_FishBase_Chgut_u1.jpg | 31 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_977649394.jpg | Chaetodon_kleinii_FishBase_Chkle_u4.jpg | 3 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_-1577854379.jpg | Chaetodon_kleinii_FishBase_Chkle_u4.jpg | 25 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_308997156.jpg | Chaetodon_kleinii_FishBase_Chkle_u4.jpg | 20 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_977649394.jpg | Chaetodon_kleinii_FishBase_Chkle_u6.jpg | 44 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_-1577854379.jpg | Chaetodon_kleinii_FishBase_Chkle_u6.jpg | 44 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_308997156.jpg | Chaetodon_kleinii_FishBase_Chkle_u6.jpg | 35 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_977649394.jpg | Chaetodon_kleinii_FishBase_Chkle_u2.jpg | 21 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_-1577854379.jpg | Chaetodon_kleinii_FishBase_Chkle_u2.jpg | 15 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_308997156.jpg | Chaetodon_kleinii_FishBase_Chkle_u2.jpg | 0 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_977649394.jpg | Chaetodon_kleinii_FishBase_Chkle_u1.jpg | 26 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_-1577854379.jpg | Chaetodon_kleinii_FishBase_Chkle_u1.jpg | 0 |
| Chaetodon kleinii | Chaetodon_kleinii_Bishop_308997156.jpg | Chaetodon_kleinii_FishBase_Chkle_u1.jpg | 15 |
| Chaetodon larvatus | Chaetodon_larvatus_Bishop_1783626891.jpg | Chaetodon_larvatus_FishBase_Chlar_u1.jpg | 5 |
| Chaetodon leucopleura | Chaetodon_leucopleura_Bishop_951338608.jpg | Chaetodon_leucopleura_FishBase_Chleu_u5.jpg | 2 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_1526268337.jpg | Chaetodon_lineolatus_FishBase_Chlin_u2.jpg | 3 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_-879271097.jpg | Chaetodon_lineolatus_FishBase_Chlin_u2.jpg | 25 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_1821004318.jpg | Chaetodon_lineolatus_FishBase_Chlin_u2.jpg | 33 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_1526268337.jpg | Chaetodon_lineolatus_FishBase_Chlin_u6.jpg | 23 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_-879271097.jpg | Chaetodon_lineolatus_FishBase_Chlin_u6.jpg | 1 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_1821004318.jpg | Chaetodon_lineolatus_FishBase_Chlin_u6.jpg | 35 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_1526268337.jpg | Chaetodon_lineolatus_FishBase_Chlin_u5.jpg | 34 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_-879271097.jpg | Chaetodon_lineolatus_FishBase_Chlin_u5.jpg | 34 |
| Chaetodon lineolatus | Chaetodon_lineolatus_Bishop_1821004318.jpg | Chaetodon_lineolatus_FishBase_Chlin_u5.jpg | 2 |
| Chaetodon litus | Chaetodon_litus_Bishop_1983788749.jpg | Chaetodon_litus_FishBase_Chlit_u0.jpg | 4 |
| Chaetodon litus | Chaetodon_litus_Bishop_2072919420.jpg | Chaetodon_litus_FishBase_Chlit_j0.jpg | 1 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_543936522.jpg | Chaetodon_lunula_FishBase_Chlun_u6.jpg | 0 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_2036473160.jpg | Chaetodon_lunula_FishBase_Chlun_u6.jpg | 38 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_-530805309.jpg | Chaetodon_lunula_FishBase_Chlun_u7.jpg | 2 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_2036473160.jpg | Chaetodon_lunula_FishBase_Chlun_u7.jpg | 40 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_-530805309.jpg | Chaetodon_lunula_FishBase_Chlun_u0.jpg | 47 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_2036473160.jpg | Chaetodon_lunula_FishBase_Chlun_u0.jpg | 43 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_1720571305.jpg | Chaetodon_lunula_FishBase_Chlun_j0.jpg | 7 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_543936522.jpg | Chaetodon_lunula_FishBase_Chlun_u8.jpg | 39 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_-530805309.jpg | Chaetodon_lunula_FishBase_Chlun_u8.jpg | 41 |
| Chaetodon lunula | Chaetodon_lunula_Bishop_2036473160.jpg | Chaetodon_lunula_FishBase_Chlun_u8.jpg | 5 |
| Chaetodon lunulatus | Chaetodon_lunulatus_Bishop_1113667583.jpg | Chaetodon_lunulatus_FishBase_Chlun_ul.jpg | 48 |
| Chaetodon lunulatus | Chaetodon_lunulatus_Bishop_1113667583.jpg | Chaetodon_lunulatus_FishBase_Chlun_ui.jpg | 32 |
| Chaetodon lunulatus | Chaetodon_lunulatus_Bishop_1113667583.jpg | Chaetodon_lunulatus_FishBase_Chlun_ub.jpg | 4 |
| Chaetodon lunulatus | Chaetodon_lunulatus_Bishop_1378116790.jpg | Chaetodon_lunulatus_FishBase_Chlun_uc.jpg | 5 |
| Chaetodon madagaskariensis | Chaetodon_madagaskariensis_Bishop_1210446292.jpg | Chaetodon_madagaskariensis_FishBase_Chmad_u1.jpg | 1 |
| Chaetodon madagaskariensis | Chaetodon_madagaskariensis_Bishop_1210446292.jpg | Chaetodon_madagaskariensis_FishBase_Chmer_u5.jpg | 38 |
| Chaetodon marleyi | Chaetodon_marleyi_Bishop_384825925.jpg | Chaetodon_marleyi_FishBase_Chmar_ua.jpg | 2 |
| Chaetodon marleyi | Chaetodon_marleyi_Bishop_384825925.jpg | Chaetodon_marleyi_FishBase_Chmar_ub.jpg | 49 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1381464288.jpg | Chaetodon_melannotus_FishBase_Chmel_u8.jpg | 3 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1126507013.jpg | Chaetodon_melannotus_FishBase_Chmel_u8.jpg | 41 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_403079329.jpg | Chaetodon_melannotus_FishBase_Chmel_u8.jpg | 40 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1381464288.jpg | Chaetodon_melannotus_FishBase_Chmel_u4.jpg | 43 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1689715678.jpg | Chaetodon_melannotus_FishBase_Chmel_u4.jpg | 43 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1126507013.jpg | Chaetodon_melannotus_FishBase_Chmel_u4.jpg | 41 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_403079329.jpg | Chaetodon_melannotus_FishBase_Chmel_u4.jpg | 2 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1689715678.jpg | Chaetodon_melannotus_FishBase_Chmel_u6.jpg | 0 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1126507013.jpg | Chaetodon_melannotus_FishBase_Chmel_u6.jpg | 40 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_403079329.jpg | Chaetodon_melannotus_FishBase_Chmel_u6.jpg | 43 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1381464288.jpg | Chaetodon_melannotus_FishBase_Chmel_u7.jpg | 44 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1689715678.jpg | Chaetodon_melannotus_FishBase_Chmel_u7.jpg | 42 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_-1126507013.jpg | Chaetodon_melannotus_FishBase_Chmel_u7.jpg | 6 |
| Chaetodon melannotus | Chaetodon_melannotus_Bishop_403079329.jpg | Chaetodon_melannotus_FishBase_Chmel_u7.jpg | 43 |
| Chaetodon melapterus | Chaetodon_melapterus_Bishop_690928567.jpg | Chaetodon_melapterus_FishBase_Chmel_u9.jpg | 7 |
| Chaetodon melapterus | Chaetodon_melapterus_Bishop_-738901426.jpg | Chaetodon_melapterus_FishBase_Chmel_u9.jpg | 45 |
| Chaetodon melapterus | Chaetodon_melapterus_Bishop_690928567.jpg | Chaetodon_melapterus_FishBase_Chmel_ua.jpg | 48 |
| Chaetodon melapterus | Chaetodon_melapterus_Bishop_-738901426.jpg | Chaetodon_melapterus_FishBase_Chmel_ua.jpg | 8 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_-727881428.jpg | Chaetodon_mertensii_FishBase_Chmer_u3.jpg | 39 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_2125964090.jpg | Chaetodon_mertensii_FishBase_Chmer_u3.jpg | 3 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_1869466931.jpg | Chaetodon_mertensii_FishBase_Chmer_u3.jpg | 37 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_992543933.jpg | Chaetodon_mertensii_FishBase_Chmer_u3.jpg | 28 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_-727881428.jpg | Chaetodon_mertensii_FishBase_Chmer_u2.jpg | 42 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_2125964090.jpg | Chaetodon_mertensii_FishBase_Chmer_u2.jpg | 36 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_1869466931.jpg | Chaetodon_mertensii_FishBase_Chmer_u2.jpg | 0 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_992543933.jpg | Chaetodon_mertensii_FishBase_Chmer_u2.jpg | 17 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_-727881428.jpg | Chaetodon_mertensii_FishBase_Chmer_u4.jpg | 37 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_2125964090.jpg | Chaetodon_mertensii_FishBase_Chmer_u4.jpg | 25 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_1869466931.jpg | Chaetodon_mertensii_FishBase_Chmer_u4.jpg | 19 |
| Chaetodon mertensii | Chaetodon_mertensii_Bishop_992543933.jpg | Chaetodon_mertensii_FishBase_Chmer_u4.jpg | 2 |
| Chaetodon mesoleucos | Chaetodon_mesoleucos_Bishop_-899095560.jpg | Chaetodon_mesoleucos_FishBase_Chmes_u0.jpg | 1 |
| Chaetodon meyeri | Chaetodon_meyeri_Bishop_-146742631.jpg | Chaetodon_meyeri_FishBase_Chmey_u0.jpg | 0 |
| Chaetodon miliaris | Chaetodon_miliaris_Bishop_725437158.jpg | Chaetodon_miliaris_FishBase_Chmil_u0.jpg | 42 |
| Chaetodon miliaris | Chaetodon_miliaris_Bishop_1720766063.jpg | Chaetodon_miliaris_FishBase_Chmil_u0.jpg | 44 |
| Chaetodon miliaris | Chaetodon_miliaris_Bishop_725437158.jpg | Chaetodon_miliaris_FishBase_Chmil_u2.jpg | 2 |
| Chaetodon miliaris | Chaetodon_miliaris_Bishop_1720766063.jpg | Chaetodon_miliaris_FishBase_Chmil_u2.jpg | 14 |
| Chaetodon miliaris | Chaetodon_miliaris_Bishop_725437158.jpg | Chaetodon_miliaris_FishBase_Chmil_u1.jpg | 13 |
| Chaetodon miliaris | Chaetodon_miliaris_Bishop_1720766063.jpg | Chaetodon_miliaris_FishBase_Chmil_u1.jpg | 1 |
| Chaetodon mitratus | Chaetodon_mitratus_Bishop_-777256060.jpg | Chaetodon_mitratus_FishBase_Chmit_u1.jpg | 4 |
| Chaetodon multicinctus | Chaetodon_multicinctus_Bishop_-1285212619.jpg | Chaetodon_multicinctus_FishBase_Chmul_u3.jpg | 4 |
| Chaetodon multicinctus | Chaetodon_multicinctus_Bishop_-1145166510.jpg | Chaetodon_multicinctus_FishBase_Chmul_u3.jpg | 29 |
| Chaetodon multicinctus | Chaetodon_multicinctus_Bishop_-1285212619.jpg | Chaetodon_multicinctus_FishBase_Chmul_u2.jpg | 26 |
| Chaetodon multicinctus | Chaetodon_multicinctus_Bishop_-1145166510.jpg | Chaetodon_multicinctus_FishBase_Chmul_u2.jpg | 1 |
| Chaetodon nigropunctatus | Chaetodon_nigropunctatus_Bishop_276052843.jpg | Chaetodon_nigropunctatus_FishBase_Chnig_u2.jpg | 25 |
| Chaetodon nigropunctatus | Chaetodon_nigropunctatus_Bishop_1435538384.jpg | Chaetodon_nigropunctatus_FishBase_Chnig_u2.jpg | 1 |
| Chaetodon nigropunctatus | Chaetodon_nigropunctatus_Bishop_276052843.jpg | Chaetodon_nigropunctatus_FishBase_Chnig_u1.jpg | 51 |
| Chaetodon nigropunctatus | Chaetodon_nigropunctatus_Bishop_1435538384.jpg | Chaetodon_nigropunctatus_FishBase_Chnig_u1.jpg | 49 |
| Chaetodon nigropunctatus | Chaetodon_nigropunctatus_Bishop_276052843.jpg | Chaetodon_nigropunctatus_FishBase_Chnig_u4.jpg | 1 |
| Chaetodon nigropunctatus | Chaetodon_nigropunctatus_Bishop_1435538384.jpg | Chaetodon_nigropunctatus_FishBase_Chnig_u4.jpg | 27 |
| Chaetodon nippon | Chaetodon_nippon_Bishop_-751379567.jpg | Chaetodon_nippon_FishBase_Chnip_u1.jpg | 5 |
| Chaetodon ocellicaudus | Chaetodon_ocellicaudus_Bishop_958979710.jpg | Chaetodon_ocellicaudus_FishBase_Choce_u4.jpg | 3 |
| Chaetodon ocellicaudus | Chaetodon_ocellicaudus_Bishop_958979710.jpg | Chaetodon_ocellicaudus_FishBase_Choce_u1.jpg | 42 |
| Chaetodon octofasciatus | Chaetodon_octofasciatus_Bishop_1838436391.jpg | Chaetodon_octofasciatus_FishBase_Choct_u2.jpg | 3 |
| Chaetodon ornatissimus | Chaetodon_ornatissimus_Bishop_164962524.jpg | Chaetodon_ornatissimus_FishBase_Chorn_u5.jpg | 40 |
| Chaetodon ornatissimus | Chaetodon_ornatissimus_Bishop_553414317.jpg | Chaetodon_ornatissimus_FishBase_Chorn_u5.jpg | 46 |
| Chaetodon ornatissimus | Chaetodon_ornatissimus_Bishop_164962524.jpg | Chaetodon_ornatissimus_FishBase_Chorn_u2.jpg | 31 |
| Chaetodon ornatissimus | Chaetodon_ornatissimus_Bishop_553414317.jpg | Chaetodon_ornatissimus_FishBase_Chorn_u2.jpg | 3 |
| Chaetodon ornatissimus | Chaetodon_ornatissimus_Bishop_164962524.jpg | Chaetodon_ornatissimus_FishBase_Chorn_u3.jpg | 0 |
| Chaetodon ornatissimus | Chaetodon_ornatissimus_Bishop_553414317.jpg | Chaetodon_ornatissimus_FishBase_Chorn_u3.jpg | 30 |
| Chaetodon ornatissimus | Chaetodon_ornatissimus_Bishop_164962524.jpg | Chaetodon_ornatissimus_FishBase_Chorn_uk.jpg | 34 |
| Chaetodon ornatissimus | Chaetodon_ornatissimus_Bishop_553414317.jpg | Chaetodon_ornatissimus_FishBase_Chorn_uk.jpg | 32 |
| Chaetodon paucifasciatus | Chaetodon_paucifasciatus_Bishop_-1576082838.jpg | Chaetodon_paucifasciatus_FishBase_Chpau_u1.jpg | 1 |
| Chaetodon pelewensis | Chaetodon_pelewensis_Bishop_-460781405.jpg | Chaetodon_pelewensis_FishBase_Chpel_u1.jpg | 2 |
| Chaetodon plebeius | Chaetodon_plebeius_Bishop_-531604202.jpg | Chaetodon_plebeius_FishBase_Chple_u3.jpg | 24 |
| Chaetodon plebeius | Chaetodon_plebeius_Bishop_-243381367.jpg | Chaetodon_plebeius_FishBase_Chple_u3.jpg | 5 |
| Chaetodon plebeius | Chaetodon_plebeius_Bishop_-113106721.jpg | Chaetodon_plebeius_FishBase_Chple_u5.jpg | 14 |
| Chaetodon plebeius | Chaetodon_plebeius_Bishop_-531604202.jpg | Chaetodon_plebeius_FishBase_Chple_u4.jpg | 5 |
| Chaetodon plebeius | Chaetodon_plebeius_Bishop_-243381367.jpg | Chaetodon_plebeius_FishBase_Chple_u4.jpg | 22 |
| Chaetodon punctatofasciatus | Chaetodon_punctatofasciatus_Bishop_1458053428.jpg | Chaetodon_punctatofasciatus_FishBase_Chpun_u3.jpg | 2 |
| Chaetodon rafflesii | Chaetodon_rafflesii_Bishop_497192485.jpg | Chaetodon_rafflesii_FishBase_Chraf_u2.jpg | 9 |
| Chaetodon rafflesii | Chaetodon_rafflesii_Bishop_-779601278.jpg | Chaetodon_rafflesii_FishBase_Chraf_u2.jpg | 45 |
| Chaetodon rafflesii | Chaetodon_rafflesii_Bishop_497192485.jpg | Chaetodon_rafflesii_FishBase_Chraf_u3.jpg | 46 |
| Chaetodon rafflesii | Chaetodon_rafflesii_Bishop_-779601278.jpg | Chaetodon_rafflesii_FishBase_Chraf_u3.jpg | 2 |
| Chaetodon rafflesii | Chaetodon_rafflesii_Bishop_497192485.jpg | Chaetodon_rafflesii_FishBase_Chraf_u0.jpg | 50 |
| Chaetodon rafflesii | Chaetodon_rafflesii_Bishop_-779601278.jpg | Chaetodon_rafflesii_FishBase_Chraf_u0.jpg | 24 |
| Chaetodon rainfordi | Chaetodon_rainfordi_Bishop_-805104421.jpg | Chaetodon_rainfordi_FishBase_Chrai_u5.jpg | 50 |
| Chaetodon rainfordi | Chaetodon_rainfordi_Bishop_-805104421.jpg | Chaetodon_rainfordi_FishBase_Chrai_u3.jpg | 26 |
| Chaetodon rainfordi | Chaetodon_rainfordi_Bishop_-805104421.jpg | Chaetodon_rainfordi_FishBase_Chrai_u0.jpg | 9 |
| Chaetodon reticulatus | Chaetodon_reticulatus_Bishop_-2063111040.jpg | Chaetodon_reticulatus_FishBase_Chret_u4.jpg | 9 |
| Chaetodon selene | Chaetodon_selene_Bishop_-1379331455.jpg | Chaetodon_selene_FishBase_Chsel_u0.jpg | 0 |
| Chaetodon selene | Chaetodon_selene_Bishop_-1379331455.jpg | Chaetodon_selene_FishBase_Chsel_u1.jpg | 51 |
| Chaetodon semeion | Chaetodon_semeion_Bishop_-1205919593.jpg | Chaetodon_semeion_FishBase_Chsem_u0.jpg | 36 |
| Chaetodon semeion | Chaetodon_semeion_Bishop_238560942.jpg | Chaetodon_semeion_FishBase_Chsem_u0.jpg | 4 |
| Chaetodon semeion | Chaetodon_semeion_Bishop_-1205919593.jpg | Chaetodon_semeion_FishBase_Chsem_u3.jpg | 4 |
| Chaetodon semeion | Chaetodon_semeion_Bishop_238560942.jpg | Chaetodon_semeion_FishBase_Chsem_u3.jpg | 34 |
| Chaetodon semeion | Chaetodon_semeion_Bishop_-1205919593.jpg | Chaetodon_semeion_FishBase_Chsem_j1.jpg | 46 |
| Chaetodon semeion | Chaetodon_semeion_Bishop_238560942.jpg | Chaetodon_semeion_FishBase_Chsem_j1.jpg | 44 |
| Chaetodon semilarvatus | Chaetodon_semilarvatus_Bishop_-947886964.jpg | Chaetodon_semilarvatus_FishBase_Chsem_u2.jpg | 2 |
| Chaetodon smithi | Chaetodon_smithi_Bishop_-608924262.jpg | Chaetodon_smithi_FishBase_Chsmi_u1.jpg | 3 |
| Chaetodon smithi | Chaetodon_smithi_Bishop_-756583267.jpg | Chaetodon_smithi_FishBase_Chsmi_u0.jpg | 3 |
| Chaetodon smithi | Chaetodon_smithi_Bishop_-1166721005.jpg | Chaetodon_smithi_FishBase_Chsmi_u3.jpg | 2 |
| Chaetodon speculum | Chaetodon_speculum_Bishop_65885528.jpg | Chaetodon_speculum_FishBase_Chspe_u0.jpg | 3 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_1959828294.jpg | Chaetodon_tinkeri_FishBase_Chtin_u5.jpg | 46 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_-1989865351.jpg | Chaetodon_tinkeri_FishBase_Chtin_u2.jpg | 26 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_1959828294.jpg | Chaetodon_tinkeri_FishBase_Chtin_u2.jpg | 0 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_-408146097.jpg | Chaetodon_tinkeri_FishBase_Chtin_u2.jpg | 50 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_-1989865351.jpg | Chaetodon_tinkeri_FishBase_Chtin_u3.jpg | 45 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_1959828294.jpg | Chaetodon_tinkeri_FishBase_Chtin_u3.jpg | 37 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_-1989865351.jpg | Chaetodon_tinkeri_FishBase_Chtin_u1.jpg | 2 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_1959828294.jpg | Chaetodon_tinkeri_FishBase_Chtin_u1.jpg | 26 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_1959828294.jpg | Chaetodon_tinkeri_FishBase_Chtin_u0.jpg | 49 |
| Chaetodon tinkeri | Chaetodon_tinkeri_Bishop_-408146097.jpg | Chaetodon_tinkeri_FishBase_Chtin_u0.jpg | 1 |
| Chaetodon trichrous | Chaetodon_trichrous_Bishop_-192591132.jpg | Chaetodon_trichrous_FishBase_Chtri_ud.jpg | 2 |
| Chaetodon tricinctus | Chaetodon_tricinctus_Bishop_28770837.jpg | Chaetodon_tricinctus_FishBase_Chtri_u5.jpg | 3 |
| Chaetodon trifascialis | Chaetodon_trifascialis_Bishop_-500274613.jpg | Chaetodon_trifascialis_FishBase_Chtri_u7.jpg | 5 |
| Chaetodon trifascialis | Chaetodon_trifascialis_Bishop_-1790297166.jpg | Chaetodon_trifascialis_FishBase_Chtri_u7.jpg | 38 |
| Chaetodon trifascialis | Chaetodon_trifascialis_Bishop_-500274613.jpg | Chaetodon_trifascialis_FishBase_Chtri_u4.jpg | 42 |
| Chaetodon trifascialis | Chaetodon_trifascialis_Bishop_-1790297166.jpg | Chaetodon_trifascialis_FishBase_Chtri_u4.jpg | 11 |
| Chaetodon trifasciatus | Chaetodon_trifasciatus_Bishop_-781709008.jpg | Chaetodon_trifasciatus_FishBase_Chtri_ut.jpg | 36 |
| Chaetodon trifasciatus | Chaetodon_trifasciatus_Bishop_-781709008.jpg | Chaetodon_trifasciatus_FishBase_Chtri_uc.jpg | 4 |
| Chaetodon ulietensis | Chaetodon_ulietensis_Bishop_-201975074.jpg | Chaetodon_ulietensis_FishBase_Chuli_u1.jpg | 0 |
| Chaetodon ulietensis | Chaetodon_ulietensis_Bishop_-1430510223.jpg | Chaetodon_ulietensis_FishBase_Chuli_u1.jpg | 31 |
| Chaetodon ulietensis | Chaetodon_ulietensis_Bishop_-201975074.jpg | Chaetodon_ulietensis_FishBase_Chuli_u2.jpg | 30 |
| Chaetodon ulietensis | Chaetodon_ulietensis_Bishop_-1430510223.jpg | Chaetodon_ulietensis_FishBase_Chuli_u2.jpg | 5 |
| Chaetodon unimaculatus | Chaetodon_unimaculatus_Bishop_-538563321.jpg | Chaetodon_unimaculatus_FishBase_Chuni_u1.jpg | 0 |
| Chaetodon unimaculatus | Chaetodon_unimaculatus_Bishop_-2035665860.jpg | Chaetodon_unimaculatus_FishBase_Chuni_u1.jpg | 27 |
| Chaetodon wiebeli | Chaetodon_wiebeli_Bishop_532247427.jpg | Chaetodon_wiebeli_FishBase_Chwie_u1.jpg | 2 |
| Chaetodon wiebeli | Chaetodon_wiebeli_Bishop_532247427.jpg | Chaetodon_wiebeli_FishBase_Chwie_j0.jpg | 49 |
| Chaetodon xanthocephalus | Chaetodon_xanthocephalus_Bishop_-1021247480.jpg | Chaetodon_xanthocephalus_FishBase_Chxan_u5.jpg | 1 |
| Chaetodon xanthocephalus | Chaetodon_xanthocephalus_Bishop_-1021247480.jpg | Chaetodon_xanthocephalus_FishBase_Chxan_u2.jpg | 44 |
| Chaetodon xanthocephalus | Chaetodon_xanthocephalus_Bishop_-1021247480.jpg | Chaetodon_xanthocephalus_FishBase_Chxan_um.jpg | 41 |
| Chaetodon xanthurus | Chaetodon_xanthurus_Bishop_-858508951.jpg | Chaetodon_xanthurus_FishBase_Chxan_u6.jpg | 0 |
| Chaetodon zanzibarensis | Chaetodon_zanzibariensis_Bishop_-1292827274.jpg | Chaetodon_zanzibarensis_FishBase_Chzan_u1.jpg | 43 |
| Chaetodon zanzibarensis | Chaetodon_zanzibariensis_Bishop_-1292827274.jpg | Chaetodon_zanzibarensis_FishBase_Chzan_u2.jpg | 40 |
| Chaetodon zanzibarensis | Chaetodon_zanzibariensis_Bishop_-1292827274.jpg | Chaetodon_zanzibarensis_FishBase_Chzan_u0.jpg | 2 |
| Chelmon marginalis | Chelmon_marginalis_Bishop_24220165.jpg | Chelmon_marginalis_FishBase_Chmar_u2.jpg | 5 |
| Chelmon marginalis | Chelmon_marginalis_Bishop_-336691052.jpg | Chelmon_marginalis_FishBase_Chmar_u5.jpg | 1 |
| Chelmon marginalis | Chelmon_marginalis_Bishop_-1899628097.jpg | Chelmon_marginalis_FishBase_Chmar_u5.jpg | 23 |
| Chelmon marginalis | Chelmon_marginalis_Bishop_-336691052.jpg | Chelmon_marginalis_FishBase_Chmar_u4.jpg | 26 |
| Chelmon marginalis | Chelmon_marginalis_Bishop_-1899628097.jpg | Chelmon_marginalis_FishBase_Chmar_u4.jpg | 20 |
| Chelmon muelleri | Chelmon_muelleri_Bishop_1919363298.jpg | Chelmon_muelleri_FishBase_Chmue_u2.jpg | 2 |
| Chelmon muelleri | Chelmon_muelleri_Bishop_425540539.jpg | Chelmon_muelleri_FishBase_Chmue_u1.jpg | 3 |
| Chelmon rostratus | Chelmon_rostratus_Bishop_-1060273696.jpg | Chelmon_rostratus_FishBase_Chros_u5.jpg | 24 |
| Chelmon rostratus | Chelmon_rostratus_Bishop_-1630595999.jpg | Chelmon_rostratus_FishBase_Chros_u5.jpg | 1 |
| Chelmon rostratus | Chelmon_rostratus_Bishop_-1060273696.jpg | Chelmon_rostratus_FishBase_Chros_u4.jpg | 1 |
| Chelmon rostratus | Chelmon_rostratus_Bishop_-1630595999.jpg | Chelmon_rostratus_FishBase_Chros_u4.jpg | 24 |
| Chelmonops truncatus | Chelmonops_truncatus_Bishop_-473926281.jpg | Chelmonops_truncatus_FishBase_Chtru_u1.jpg | 3 |
| Coradion altivelis | Coradion_altivelis_Bishop_-1400875907.jpg | Coradion_altivelis_FishBase_Coalt_u0.jpg | 9 |
| Coradion altivelis | Coradion_altivelis_Bishop_1025347564.jpg | Coradion_altivelis_FishBase_Coalt_u1.jpg | 5 |
| Coradion chrysozonus | Coradion_chrysozonus_Bishop_1721541363.jpg | Coradion_chrysozonus_FishBase_Cochr_u4.jpg | 11 |
| Coradion chrysozonus | Coradion_chrysozonus_Bishop_213329914.jpg | Coradion_chrysozonus_FishBase_Cochr_u5.jpg | 3 |
| Coradion chrysozonus | Coradion_chrysozonus_Bishop_-1821725095.jpg | Coradion_chrysozonus_FishBase_Cochr_u1.jpg | 7 |
| Coradion chrysozonus | Coradion_chrysozonus_Bishop_28243640.jpg | Coradion_chrysozonus_FishBase_Cochr_u3.jpg | 7 |
| Coradion melanopus | Coradion_melanopus_Bishop_1537514406.jpg | Coradion_melanopus_FishBase_Comel_u2.jpg | 9 |
| Coradion melanopus | Coradion_melanopus_Bishop_1397670468.jpg | Coradion_melanopus_FishBase_Comel_u2.jpg | 39 |
| Coradion melanopus | Coradion_melanopus_Bishop_-1537131019.jpg | Coradion_melanopus_FishBase_Comel_u2.jpg | 42 |
| Coradion melanopus | Coradion_melanopus_Bishop_726040623.jpg | Coradion_melanopus_FishBase_Comel_u2.jpg | 47 |
| Coradion melanopus | Coradion_melanopus_Bishop_1537514406.jpg | Coradion_melanopus_FishBase_Comel_u1.jpg | 45 |
| Coradion melanopus | Coradion_melanopus_Bishop_-1537131019.jpg | Coradion_melanopus_FishBase_Comel_u1.jpg | 40 |
| Coradion melanopus | Coradion_melanopus_Bishop_726040623.jpg | Coradion_melanopus_FishBase_Comel_u1.jpg | 1 |
| Coradion melanopus | Coradion_melanopus_Bishop_1537514406.jpg | Coradion_melanopus_FishBase_Comel_u0.jpg | 37 |
| Coradion melanopus | Coradion_melanopus_Bishop_1397670468.jpg | Coradion_melanopus_FishBase_Comel_u0.jpg | 21 |
| Coradion melanopus | Coradion_melanopus_Bishop_-1537131019.jpg | Coradion_melanopus_FishBase_Comel_u0.jpg | 6 |
| Coradion melanopus | Coradion_melanopus_Bishop_726040623.jpg | Coradion_melanopus_FishBase_Comel_u0.jpg | 41 |
| Forcipiger flavissimus | Forcipiger_flavissimus_Bishop_-1224605166.jpg | Forcipiger_flavissimus_FishBase_Fofla_u4.jpg | 24 |
| Forcipiger flavissimus | Forcipiger_flavissimus_Bishop_-1548318421.jpg | Forcipiger_flavissimus_FishBase_Fofla_u4.jpg | 1 |
| Forcipiger flavissimus | Forcipiger_flavissimus_Bishop_-1224605166.jpg | Forcipiger_flavissimus_FishBase_Fofla_uc.jpg | 35 |
| Forcipiger flavissimus | Forcipiger_flavissimus_Bishop_-1548318421.jpg | Forcipiger_flavissimus_FishBase_Fofla_uc.jpg | 30 |
| Forcipiger flavissimus | Forcipiger_flavissimus_Bishop_-1224605166.jpg | Forcipiger_flavissimus_FishBase_Fofla_u3.jpg | 0 |
| Forcipiger flavissimus | Forcipiger_flavissimus_Bishop_-1548318421.jpg | Forcipiger_flavissimus_FishBase_Fofla_u3.jpg | 23 |
| Forcipiger longirostris | Forcipiger_longirostris_Bishop_2122975038.jpg | Forcipiger_longirostris_FishBase_Folon_u1.jpg | 4 |
| Forcipiger longirostris | Forcipiger_longirostris_Bishop_545316689.jpg | Forcipiger_longirostris_FishBase_Folon_u2.jpg | 2 |
| Forcipiger longirostris | Forcipiger_longirostris_Bishop_2048048784.jpg | Forcipiger_longirostris_FishBase_Folon_u0.jpg | 0 |
| Forcipiger longirostris | Forcipiger_longirostris_Bishop_2048048784.jpg | Forcipiger_longirostris_FishBase_Folon_j0.jpg | 38 |
| Hemitaurichthys multispinosus | Hemitaurichthys_multispinosus_Bishop_1514469863.jpg | Hemitaurichthys_multispinosus_FishBase_Hemul_u0.jpg | 1 |
| Hemitaurichthys polylepis | Hemitaurichthys_polylepis_Bishop_224753564.jpg | Hemitaurichthys_polylepis_FishBase_Hepol_uc.jpg | 50 |
| Hemitaurichthys polylepis | Hemitaurichthys_polylepis_Bishop_224753564.jpg | Hemitaurichthys_polylepis_FishBase_Hepol_u2.jpg | 3 |
| Hemitaurichthys polylepis | Hemitaurichthys_polylepis_Bishop_224753564.jpg | Hemitaurichthys_polylepis_FishBase_Hepol_ue.jpg | 51 |
| Hemitaurichthys zoster | Hemitaurichthys_zoster_Bishop_1598828141.jpg | Hemitaurichthys_zoster_FishBase_Hezos_u0.jpg | 6 |
| Heniochus acuminatus | Heniochus_acuminatus_Bishop_2027043176.jpg | Heniochus_acuminatus_FishBase_Heacu_u2.jpg | 2 |
| Heniochus acuminatus | Heniochus_acuminatus_Bishop_1965411939.jpg | Heniochus_acuminatus_FishBase_Heacu_u1.jpg | 3 |
| Heniochus acuminatus | Heniochus_acuminatus_Bishop_1817843498.jpg | Heniochus_acuminatus_FishBase_Heacu_u4.jpg | 2 |
| Heniochus chrysostomus | Heniochus_chrysostomus_Bishop_1854375753.jpg | Heniochus_chrysostomus_FishBase_Hechr_u0.jpg | 2 |
| Heniochus diphreutes | Heniochus_diphreutes_Bishop_1453219286.jpg | Heniochus_diphreutes_FishBase_Hedip_u2.jpg | 3 |
| Heniochus intermedius | Heniochus_intermedius_Bishop_-20578450.jpg | Heniochus_intermedius_FishBase_Heint_u1.jpg | 3 |
| Heniochus monoceros | Heniochus_monoceros_Bishop_-980348940.jpg | Heniochus_monoceros_FishBase_Hemon_u3.jpg | 5 |
| Heniochus monoceros | Heniochus_monoceros_Bishop_1235534495.jpg | Heniochus_monoceros_FishBase_Hemon_u2.jpg | 0 |
| Heniochus monoceros | Heniochus_monoceros_Bishop_1205143013.jpg | Heniochus_monoceros_FishBase_Hemon_u4.jpg | 4 |
| Heniochus pleurotaenia | Heniochus_pleurotaenia_Bishop_-1006778558.jpg | Heniochus_pleurotaenia_FishBase_Heple_u0.jpg | 7 |
| Heniochus singularius | Heniochus_singularius_Bishop_1959970624.jpg | Heniochus_singularius_FishBase_Hesin_u1.jpg | 2 |
| Heniochus singularius | Heniochus_singularius_Bishop_2065781403.jpg | Heniochus_singularius_FishBase_Hesin_u3.jpg | 4 |
| Heniochus varius | Heniochus_varius_Bishop_2014348865.jpg | Heniochus_varius_FishBase_Hevar_u2.jpg | 39 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-731143075.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u0.jpg | 49 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-154342580.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u0.jpg | 48 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-506807721.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u0.jpg | 51 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-731143075.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u2.jpg | 21 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-154342580.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u2.jpg | 2 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-506807721.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u2.jpg | 15 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-731143075.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u3.jpg | 27 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-154342580.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u3.jpg | 16 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-506807721.jpg | Parachaetodon_ocellatus_FishBase_Paoce_u3.jpg | 1 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-731143075.jpg | Parachaetodon_ocellatus_FishBase_Paoce_j0.jpg | 2 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-154342580.jpg | Parachaetodon_ocellatus_FishBase_Paoce_j0.jpg | 25 |
| Parachaetodon ocellatus | Parachaetodon_ocellatus_Bishop_-506807721.jpg | Parachaetodon_ocellatus_FishBase_Paoce_j0.jpg | 30 |
| Roa excelsa | Chaetodon_excelsa_Bishop_853070272.jpg | Roa_excelsa_FishBase_Chexc_u0.jpg | 3 |

---

## Changes Made

### Exemplar File Updates

- Merged Chaetodon excelsa into Roa excelsa (kept Bishop)
- Merged Chaetodon zanzibariensis into Chaetodon zanzibarensis (kept Bishop)
- Switched to Bishop: Chaetodon adiergastos
- Switched to Bishop: Chaetodon andamanensis
- Switched to Bishop: Chaetodon argentatus
- Switched to Bishop: Chaetodon aureofasciatus
- Switched to Bishop: Chaetodon auriga
- Switched to Bishop: Chaetodon auripes
- Switched to Bishop: Chaetodon austriacus
- Switched to Bishop: Chaetodon baronessa
- Switched to Bishop: Chaetodon bennetti
- Switched to Bishop: Chaetodon blackburnii
- Switched to Bishop: Chaetodon citrinellus
- Switched to Bishop: Chaetodon declivis
- Switched to Bishop: Chaetodon dialeucos
- Switched to Bishop: Chaetodon flavocoronatus
- Switched to Bishop: Chaetodon gardineri
- Switched to Bishop: Chaetodon guttatissimus
- Switched to Bishop: Chaetodon larvatus
- Switched to Bishop: Chaetodon leucopleura
- Switched to Bishop: Chaetodon trichrous
- Switched to Bishop: Chelmon marginalis
- Switched to Bishop: Chelmon muelleri
- Switched to Bishop: Chelmon rostratus
- Switched to Bishop: Chelmonops truncatus
- Switched to Bishop: Coradion altivelis
- Switched to Bishop: Coradion chrysozonus
- Switched to Bishop: Coradion melanopus
- Switched to Bishop: Forcipiger flavissimus
- Switched to Bishop: Forcipiger longirostris
- Switched to Bishop: Hemitaurichthys multispinosus
- Switched to Bishop: Hemitaurichthys polylepis
- Switched to Bishop: Hemitaurichthys zoster
- Switched to Bishop: Heniochus acuminatus
- Switched to Bishop: Heniochus chrysostomus
- Switched to Bishop: Heniochus diphreutes
- Switched to Bishop: Heniochus intermedius
- Switched to Bishop: Heniochus monoceros
- Switched to Bishop: Heniochus pleurotaenia
- Switched to Bishop: Heniochus singularius
- Switched to Bishop: Heniochus varius
- Switched to Bishop: Parachaetodon ocellatus

### Gestalt K File Updates

- Merged Chaetodon excelsa into Roa excelsa
- Merged Chaetodon zanzibariensis into Chaetodon zanzibarensis

---

## Image Counts by Source

| Source | Total Images | Randall? |
|--------|--------------|----------|
| bishop | 178 | All |
| fishbase | 131 | 113 images |
| fishbase_extra | 568 | 544 images |
| fishbase_usercontrib | 722 | 0 images |
| inaturalist | 1125 | 0 images |

---

## Recommendations

1. **Use Bishop images when available** - Better color fidelity for analysis
2. **Mark FishBase-only Randall images** - Useful for species without Bishop coverage
3. **Consider updating exemplars** - For species where Randall alternative exists
4. **Run color correction on non-Randall images** - Especially iNaturalist (57% underwater)

---

## Technical Notes

1. **Hash comparison** uses perceptual hashing (16x16 grayscale, binary threshold)
2. **Duplicate threshold**: Hamming distance < 52 (20% of 256 bits)
3. **Taxonomy follows FishBase accepted names** as of February 2025

