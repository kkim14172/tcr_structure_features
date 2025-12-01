reinitialize
bg_color white
load /Users/kkim14/Library/CloudStorage/OneDrive-InsideMDAnderson/project/tcr_structure_embedding/killing_assay/tcrmodel2_out/tcr462/ranked_0_proj.pdb, tcr462

set auto_zoom, off
set antialias, 2
set ray_trace_mode, 1
set ray_shadow, off
set depth_cue, 0
set cartoon_transparency, 0.0
set stick_radius, 0.18
set dash_width, 3.0
set dash_gap, 0.35
set label_font_id, 7
set label_size, 16


select mhcA, chain A
extract mhcA, mhcA
select pep,  chain C
extract pep, pep
select tcrA, chain D
extract tcrA, tcrA
select tcrB, chain E
extract tcrB, tcrB

hide everything, all
show cartoon, mhcA or tcrA or tcrB
show sticks, pep

set_color mhc_yellow, [240,200,20]
set_color pep_purple, [165,55,165]
set_color tcrA_teal,   [0, 160, 160]
set_color tcrB_orange, [230, 120, 40]
color tcrA_teal,   tcrA
color tcrB_orange, tcrB
color mhc_yellow, mhcA
color pep_purple, pep

set cartoon_transparency, 0.7, mhcA
set cartoon_transparency, 0.4, tcrA
set cartoon_transparency, 0.4, tcrB
set stick_transparency, 0.3, pep

label (pep and name CA), "%s%s" % ({'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}.get(resn, 'X'),resi)
set label_color, red, (pep and name CA)

# Make peptide thicker, similar to licorice
alter pep, vdw=1.2
rebuild

select pep_heavy,  pep and not elem H
select tcrA_heavy, tcrA and not elem H
select tcrB_heavy, tcrB and not elem H
select mhcA_heavy, mhcA and not elem H

# ==== Peptide–TCRα contacts (heavy atoms) ====
# select tcrA_contact_pep, byres (tcrA_heavy within 3 of pep_heavy)
# create tcrA_contact_pep, tcrA_contact_pep
# select tcrB_contact_pep, byres (tcrB_heavy within 3 of pep_heavy)
# create tcrB_contact_pep, tcrB_contact_pep

# label (tcrA_contact_pep and name CA), "%s%s(%s)" % ({'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}.get(resn, 'X'),resi,chain)
# set label_color, marine, (tcrA_contact_pep and name CA)
# label (tcrB_contact_pep and name CA), "%s%s(%s)" % ({'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}.get(resn, 'X'),resi,chain)
# set label_color, tcrB_stick_dark, (tcrB_contact_pep and name CA)

# show sticks, tcrA_contact_pep
# color marine, tcrA_contact_pep
# set stick_transparency, 0.3, tcrA_contact_pep

# show sticks, tcrB_contact_pep
# set_color tcrB_stick_dark, [180,70,10]
# color tcrB_stick_dark, tcrB_contact_pep
# set stick_transparency, 0.3, tcrB_contact_pep



# ==== Peptide–TCRα contacts (heavy atoms, dashed in gold) ====
dist hydro_pep_a, pep_heavy, tcrA_heavy, mode=2, cutoff=5
dist vdw_pep_a, pep_heavy, tcrA_heavy, mode=8, cutoff=1.0
dist hydro_pep_b, pep_heavy, tcrB_heavy, mode=2, cutoff=5
dist vdw_pep_b, pep_heavy, tcrB_heavy, mode=8, cutoff=1.0


color marine, hydro_pep_a
set dash_color, marine, hydro_pep_a
set dash_transparency, 0.0, hydro_pep_a

color tcrB_stick_dark, hydro_pep_b
set dash_color, tcrB_stick_dark, hydro_pep_b
set dash_transparency, 0.0, hydro_pep_b

color marine, vdw_pep_a
set dash_color, marine, vdw_pep_a
set dash_transparency, 0.7, vdw_pep_a

color tcrB_stick_dark, vdw_pep_b
set dash_color, tcrB_stick_dark, vdw_pep_b
set dash_transparency, 0.7, vdw_pep_b

# ==== MHC–TCRα contacts (heavy atoms, dashed in gold) ====
dist hydro_mhc_a, mhcA_heavy, tcrA_heavy, mode=2, cutoff=5
dist vdw_mhc_a, mhcA_heavy, tcrA_heavy, mode=8, cutoff=1.0
dist hydro_mhc_b, mhcA_heavy, tcrB_heavy, mode=2, cutoff=5
dist vdw_mhc_b, mhcA_heavy, tcrB_heavy, mode=8, cutoff=1.0

color marine, hydro_mhc_a
set dash_color, marine, hydro_mhc_a
set dash_transparency, 0.0, hydro_mhc_a

color tcrB_stick_dark, hydro_mhc_b
set dash_color, tcrB_stick_dark, hydro_mhc_b
set dash_transparency, 0.0, hydro_mhc_b

color marine, vdw_mhc_a
set dash_color, marine, vdw_mhc_a
set dash_transparency, 0.7, vdw_mhc_a

color tcrB_stick_dark, vdw_mhc_b
set dash_color, tcrB_stick_dark, vdw_mhc_b
set dash_transparency, 0.7, vdw_mhc_b

# select tcrA_contact_mhc, byres (tcrA_heavy within 3 of mhcA_heavy)
# select mhc_contact_tcrA, byres (mhcA_heavy within 3 of tcrA_heavy)
# select tcrB_contact_mhc, byres (tcrB_heavy within 3 of mhcA_heavy)
# select mhc_contact_tcrB, byres (mhcA_heavy within 3 of tcrB_heavy)

# label (tcrA_contact_mhc and name CA), "%s%s(%s)" % ({'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}.get(resn, 'X'),resi,chain)
# set label_color, blue, (tcrA_contact_mhc and name CA)
# label (tcrB_contact_mhc and name CA), "%s%s(%s)" % ({'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}.get(resn, 'X'),resi,chain)
# set label_color, orange, (tcrB_contact_mhc and name CA)
# label (mhc_contact_tcrA and name CA), "%s%s(%s)" % ({'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}.get(resn, 'X'),resi,chain)
# set label_color, blue, (mhc_contact_tcrA and name CA)
# label (mhc_contact_tcrB and name CA), "%s%s(%s)" % ({'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}.get(resn, 'X'),resi,chain)
# set label_color, orange, (mhc_contact_tcrB and name CA)


# show sticks, tcrA_contact_mhc
# color marine, tcrA_contact_mhc
# set stick_transparency, 0.3, tcrA_contact_mhc

# show sticks, tcrB_contact_mhc
# color tcrB_stick_dark, tcrB_contact_mhc
# set stick_transparency, 0.3, tcrB_contact_mhc

# show sticks, mhc_contact_tcrA
# color mhc_yellow, mhc_contact_tcrA
# set stick_transparency, 0.3, mhc_contact_tcrA

# show sticks, mhc_contact_tcrB
# color mhc_yellow, mhc_contact_tcrB
# set stick_transparency, 0.3, mhc_contact_tcrB

set_color cdr1b_cyanblue, [60, 180, 250]
set_color cdr2b_tealgreen, [0, 160, 130]
set_color cdr3b_indigo, [40, 60, 160]

set_color cdr1a_gold, [250, 200, 60]
set_color cdr2a_orange, [240, 120, 40]
set_color cdr3a_crimson, [200, 40, 60]


color cdr1a_gold, cdr1a
color cdr2a_orange, cdr2a
color cdr3a_crimson, cdr3a
color cdr1b_cyanblue, cdr1b
color cdr2b_tealgreen, cdr2b
color cdr3b_indigo, cdr3b

set label_outline_color, black
set label_size, 11
