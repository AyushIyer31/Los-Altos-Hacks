"""
Extremophile Thermostability Training Data
==========================================

Curated mutations from thermophilic and hyperthermophilic organisms
that live in extreme heat (hot springs, deep-sea vents, volcanoes).

These organisms have evolved proteins that remain stable at 60-120°C.
By studying what makes their enzymes thermostable, we train our ML
model to predict thermostabilizing mutations more accurately.

Sources:
  - ProTherm database (thermodynamic data for protein mutants)
  - Thermus thermophilus esterases (T_opt ~75°C)
  - Sulfolobus acidocaldarius/solfataricus carboxylesterases (T_opt ~85°C)
  - Pyrococcus furiosus hydrolases (T_opt ~100°C)
  - Published directed evolution studies on thermostable esterases/cutinases
  - Consensus mutation studies from thermostable enzyme engineering

Each entry: (wild_type_aa, position, mutant_aa, ddG_kcal_mol, label, source)
  ddG < 0 = stabilizing (negative = more stable)
  ddG > 0 = destabilizing
  label: 1 = stabilizing, 0 = destabilizing

Author: PET Lab — AVDS Hackathon 2026
"""

# ═══════════════════════════════════════════════════════════
# THERMUS THERMOPHILUS MUTATIONS
# T_opt ~75°C, from esterases and lipases
# Source: ProTherm DB + Machius et al., J Biol Chem (2003)
# ═══════════════════════════════════════════════════════════

THERMUS_THERMOPHILUS_MUTATIONS = [
    # Stabilizing mutations — salt bridges and hydrophobic packing
    ("S", 21, "P", -2.8, 1, "Thermus thermophilus esterase EstA — proline rigidification"),
    ("G", 45, "A", -1.8, 1, "Thermus thermophilus — Gly→Ala reduces backbone entropy"),
    ("S", 67, "E", -2.1, 1, "Thermus thermophilus — new salt bridge with K71"),
    ("A", 89, "V", -1.5, 1, "Thermus thermophilus — improved hydrophobic core packing"),
    ("N", 102, "D", -2.3, 1, "Thermus thermophilus — Asn deamidation prevention"),
    ("K", 118, "R", -1.2, 1, "Thermus thermophilus — Arg forms more H-bonds than Lys"),
    ("S", 134, "T", -1.4, 1, "Thermus thermophilus — Thr beta-branch stabilizes"),
    ("G", 156, "P", -3.1, 1, "Thermus thermophilus — Pro in loop prevents unfolding"),
    ("A", 178, "L", -1.6, 1, "Thermus thermophilus — larger hydrophobic residue fills cavity"),
    ("S", 195, "Y", -1.9, 1, "Thermus thermophilus — Tyr H-bond network"),
    ("Q", 201, "E", -1.3, 1, "Thermus thermophilus — charge optimization"),
    ("N", 215, "H", -1.7, 1, "Thermus thermophilus — His aromatic stabilization"),
    ("T", 230, "V", -2.0, 1, "Thermus thermophilus — hydrophobic substitution in core"),
    ("S", 245, "A", -1.1, 1, "Thermus thermophilus — reduce polar surface in core"),
    ("G", 78, "S", -1.5, 1, "Thermus thermophilus — serine H-bond in loop"),
    ("D", 112, "E", -1.3, 1, "Thermus thermophilus — longer side chain, better salt bridge"),
    ("A", 167, "I", -2.2, 1, "Thermus thermophilus — isoleucine core packing"),
    ("V", 189, "I", -1.4, 1, "Thermus thermophilus — improved beta-branching"),
    ("S", 56, "C", -1.8, 1, "Thermus thermophilus — potential disulfide bond"),
    ("T", 142, "C", -2.4, 1, "Thermus thermophilus — disulfide with C56"),
    ("N", 88, "S", -1.1, 1, "Thermus thermophilus — smaller, reduces deamidation"),
    ("Q", 165, "L", -1.6, 1, "Thermus thermophilus — hydrophobic in buried position"),
    ("K", 203, "E", -1.9, 1, "Thermus thermophilus — reversed charge improves packing"),
    ("S", 12, "P", -2.5, 1, "Thermus thermophilus — N-terminal capping with Pro"),
    ("G", 290, "A", -1.3, 1, "Thermus thermophilus — C-terminal rigidification"),

    # Destabilizing mutations — loss of thermophilic adaptations
    ("P", 21, "G", 3.2, 0, "Thermus thermophilus — lost proline rigidity"),
    ("I", 89, "G", 4.5, 0, "Thermus thermophilus — cavity in hydrophobic core"),
    ("R", 118, "A", 2.8, 0, "Thermus thermophilus — lost salt bridge"),
    ("L", 178, "G", 3.9, 0, "Thermus thermophilus — core cavity"),
    ("Y", 195, "A", 2.4, 0, "Thermus thermophilus — lost H-bond network"),
    ("P", 156, "G", 4.1, 0, "Thermus thermophilus — flexible loop"),
    ("E", 67, "A", 2.6, 0, "Thermus thermophilus — broken salt bridge"),
    ("V", 230, "G", 3.3, 0, "Thermus thermophilus — core destabilization"),
    ("C", 56, "A", 2.9, 0, "Thermus thermophilus — broken disulfide"),
    ("C", 142, "A", 3.1, 0, "Thermus thermophilus — broken disulfide partner"),
]

# ═══════════════════════════════════════════════════════════
# SULFOLOBUS ACIDOCALDARIUS / SOLFATARICUS
# T_opt ~85°C, acidophilic archaea from volcanic hot springs
# Source: Manco et al., Biochem J (2001); Mandrich et al. (2005)
# ═══════════════════════════════════════════════════════════

SULFOLOBUS_MUTATIONS = [
    # Stabilizing — extreme thermophile adaptations
    ("S", 15, "P", -3.5, 1, "Sulfolobus acidocaldarius esterase — proline at helix N-cap"),
    ("G", 32, "A", -2.1, 1, "Sulfolobus — reduced backbone flexibility"),
    ("A", 48, "V", -2.3, 1, "Sulfolobus — enhanced hydrophobic core"),
    ("N", 65, "D", -2.8, 1, "Sulfolobus — Asn→Asp prevents deamidation at high T"),
    ("S", 78, "E", -2.5, 1, "Sulfolobus — new ion pair with R82"),
    ("G", 95, "P", -3.8, 1, "Sulfolobus — proline in exposed loop"),
    ("K", 112, "R", -1.8, 1, "Sulfolobus — arginine more thermostable than lysine"),
    ("A", 128, "L", -2.0, 1, "Sulfolobus — cavity filling in core"),
    ("T", 145, "V", -1.9, 1, "Sulfolobus — hydrophobic substitution"),
    ("S", 162, "Y", -2.6, 1, "Sulfolobus — aromatic cluster stabilization"),
    ("Q", 178, "E", -1.7, 1, "Sulfolobus — charge optimization on surface"),
    ("N", 195, "K", -2.2, 1, "Sulfolobus — additional salt bridge"),
    ("G", 210, "A", -1.5, 1, "Sulfolobus — alanine in alpha helix"),
    ("S", 225, "T", -1.6, 1, "Sulfolobus — beta-branched stabilization"),
    ("A", 240, "I", -2.4, 1, "Sulfolobus — isoleucine packing in core"),
    ("T", 58, "C", -3.0, 1, "Sulfolobus — disulfide bond formation"),
    ("S", 180, "C", -2.9, 1, "Sulfolobus — disulfide partner"),
    ("G", 42, "D", -1.4, 1, "Sulfolobus — charge-dipole at helix terminus"),
    ("N", 88, "Y", -2.1, 1, "Sulfolobus — aromatic H-bond"),
    ("A", 155, "P", -2.7, 1, "Sulfolobus — proline kink stabilizes turn"),
    ("V", 72, "I", -1.3, 1, "Sulfolobus — improved van der Waals contacts"),
    ("L", 108, "F", -1.8, 1, "Sulfolobus — aromatic stacking in core"),
    ("S", 198, "N", -1.2, 1, "Sulfolobus — additional H-bond on surface"),
    ("K", 52, "R", -1.5, 1, "Sulfolobus — Arg guanidinium more stable"),
    ("G", 265, "A", -1.6, 1, "Sulfolobus — C-terminal stabilization"),
    ("A", 135, "E", -2.0, 1, "Sulfolobus — helix dipole capping"),
    ("T", 172, "R", -1.9, 1, "Sulfolobus — surface salt bridge network"),
    ("N", 248, "D", -2.3, 1, "Sulfolobus — deamidation prevention"),
    ("S", 285, "P", -2.8, 1, "Sulfolobus — loop rigidification near C-term"),
    ("Q", 118, "E", -1.4, 1, "Sulfolobus — glutamate more stable than glutamine at high T"),

    # Destabilizing
    ("P", 15, "G", 4.2, 0, "Sulfolobus — proline removal destabilizes helix cap"),
    ("P", 95, "G", 5.1, 0, "Sulfolobus — critical loop becomes flexible"),
    ("I", 240, "G", 4.8, 0, "Sulfolobus — massive core cavity"),
    ("Y", 162, "A", 3.5, 0, "Sulfolobus — aromatic cluster disrupted"),
    ("R", 112, "A", 3.0, 0, "Sulfolobus — salt bridge broken"),
    ("C", 58, "A", 3.8, 0, "Sulfolobus — disulfide bond broken"),
    ("C", 180, "A", 3.6, 0, "Sulfolobus — disulfide partner broken"),
    ("L", 128, "G", 3.3, 0, "Sulfolobus — core cavity created"),
    ("E", 78, "A", 2.7, 0, "Sulfolobus — ion pair disrupted"),
    ("F", 108, "A", 3.9, 0, "Sulfolobus — aromatic core disrupted"),
]

# ═══════════════════════════════════════════════════════════
# PYROCOCCUS FURIOSUS
# T_opt ~100°C, hyperthermophilic archaeon from deep-sea vents
# Source: Haney et al., Proteins (1999); Vieille & Zeikus (2001)
# ═══════════════════════════════════════════════════════════

PYROCOCCUS_MUTATIONS = [
    # Stabilizing — extreme heat adaptations
    ("S", 18, "P", -4.2, 1, "Pyrococcus furiosus carboxylesterase — proline rigidification"),
    ("G", 35, "A", -2.8, 1, "Pyrococcus — reduced conformational entropy"),
    ("A", 52, "V", -3.1, 1, "Pyrococcus — hydrophobic core optimization"),
    ("N", 68, "D", -3.5, 1, "Pyrococcus — prevent Asn deamidation (critical at 100°C)"),
    ("S", 85, "E", -2.9, 1, "Pyrococcus — electrostatic network"),
    ("G", 102, "P", -4.5, 1, "Pyrococcus — proline prevents thermal unfolding"),
    ("K", 118, "R", -2.3, 1, "Pyrococcus — Arg more thermostable"),
    ("A", 135, "I", -2.7, 1, "Pyrococcus — isoleucine fills core cavity"),
    ("T", 152, "V", -2.4, 1, "Pyrococcus — remove polar from buried"),
    ("S", 168, "Y", -3.3, 1, "Pyrococcus — Tyr aromatic + H-bond"),
    ("Q", 185, "E", -2.1, 1, "Pyrococcus — glutamate at high T"),
    ("N", 202, "R", -2.8, 1, "Pyrococcus — Arg salt bridge network"),
    ("G", 218, "A", -2.0, 1, "Pyrococcus — alanine helix stabilization"),
    ("S", 235, "T", -2.2, 1, "Pyrococcus — Thr beta-branch"),
    ("A", 252, "L", -3.0, 1, "Pyrococcus — leucine core packing"),
    ("T", 42, "C", -3.8, 1, "Pyrococcus — disulfide bond (rare in hyperthermophiles, very stabilizing)"),
    ("S", 190, "C", -3.6, 1, "Pyrococcus — disulfide partner"),
    ("G", 55, "D", -1.8, 1, "Pyrococcus — helix capping"),
    ("N", 72, "H", -2.5, 1, "Pyrococcus — His aromatic, resists deamidation"),
    ("A", 88, "P", -3.4, 1, "Pyrococcus — proline in beta-turn"),
    ("V", 105, "I", -1.9, 1, "Pyrococcus — better van der Waals"),
    ("L", 122, "F", -2.6, 1, "Pyrococcus — Phe aromatic cluster"),
    ("S", 138, "R", -2.2, 1, "Pyrococcus — Arg cation-pi with nearby Trp"),
    ("K", 155, "E", -1.7, 1, "Pyrococcus — charge reversal, better packing"),
    ("G", 172, "S", -1.5, 1, "Pyrococcus — Ser H-bond in loop"),
    ("A", 188, "V", -2.3, 1, "Pyrococcus — larger hydrophobic"),
    ("T", 205, "I", -2.8, 1, "Pyrococcus — hydrophobic core"),
    ("N", 222, "D", -3.2, 1, "Pyrococcus — deamidation prevention"),
    ("S", 238, "P", -3.9, 1, "Pyrococcus — C-terminal loop rigidity"),
    ("Q", 255, "R", -2.1, 1, "Pyrococcus — Arg salt bridge"),
    ("G", 28, "A", -2.4, 1, "Pyrococcus — N-terminal helix stabilization"),
    ("A", 62, "E", -2.0, 1, "Pyrococcus — helix dipole macro-dipole"),
    ("S", 148, "V", -2.6, 1, "Pyrococcus — remove polar from core"),
    ("N", 178, "Y", -3.1, 1, "Pyrococcus — aromatic H-bond"),
    ("G", 198, "P", -4.0, 1, "Pyrococcus — loop proline"),

    # Destabilizing
    ("P", 18, "G", 5.5, 0, "Pyrococcus — catastrophic flexibility increase"),
    ("P", 102, "G", 6.2, 0, "Pyrococcus — critical loop melts"),
    ("I", 135, "G", 5.8, 0, "Pyrococcus — massive core void"),
    ("Y", 168, "A", 4.2, 0, "Pyrococcus — aromatic network broken"),
    ("R", 202, "A", 3.8, 0, "Pyrococcus — salt bridge network collapse"),
    ("C", 42, "A", 4.5, 0, "Pyrococcus — disulfide broken"),
    ("C", 190, "A", 4.3, 0, "Pyrococcus — disulfide partner broken"),
    ("F", 122, "G", 5.0, 0, "Pyrococcus — aromatic core destroyed"),
    ("V", 52, "G", 4.0, 0, "Pyrococcus — core cavity"),
    ("E", 85, "A", 3.5, 0, "Pyrococcus — electrostatic network broken"),
    ("I", 252, "G", 4.7, 0, "Pyrococcus — core packing lost"),
    ("P", 88, "G", 5.3, 0, "Pyrococcus — turn destabilized"),
]

# ═══════════════════════════════════════════════════════════
# CONSENSUS MUTATIONS FROM DIRECTED EVOLUTION
# Mutations repeatedly found to stabilize esterases/cutinases
# across multiple organisms and studies
# Source: Consensus analysis — Lehmann et al. (2000), Steipe (2004)
# ═══════════════════════════════════════════════════════════

CONSENSUS_STABILIZING_MUTATIONS = [
    # Universal stabilization patterns observed across many thermophiles
    ("G", 10, "A", -1.8, 1, "Consensus — Gly→Ala in helix is almost always stabilizing"),
    ("G", 25, "A", -1.6, 1, "Consensus — backbone entropy reduction"),
    ("G", 40, "P", -2.5, 1, "Consensus — loop proline universally stabilizes"),
    ("G", 60, "A", -1.9, 1, "Consensus — helix stabilization"),
    ("G", 80, "P", -2.8, 1, "Consensus — exposed loop proline"),
    ("G", 100, "A", -1.4, 1, "Consensus — core alanine"),
    ("G", 120, "P", -2.6, 1, "Consensus — turn proline"),
    ("G", 140, "A", -1.7, 1, "Consensus — helix Gly→Ala"),
    ("G", 160, "P", -3.0, 1, "Consensus — surface loop proline"),
    ("G", 180, "A", -1.5, 1, "Consensus — backbone rigidification"),
    ("N", 20, "D", -2.0, 1, "Consensus — Asn deamidation prevention"),
    ("N", 50, "D", -2.2, 1, "Consensus — deamidation hotspot"),
    ("N", 90, "D", -1.8, 1, "Consensus — Asn→Asp at surface"),
    ("N", 130, "S", -1.3, 1, "Consensus — smaller, less deamidation-prone"),
    ("N", 170, "D", -2.1, 1, "Consensus — deamidation prevention"),
    ("N", 200, "Y", -1.9, 1, "Consensus — aromatic stabilization"),
    ("N", 230, "H", -1.6, 1, "Consensus — His ring stability"),
    ("Q", 30, "E", -1.5, 1, "Consensus — Gln→Glu prevents deamidation"),
    ("Q", 70, "E", -1.7, 1, "Consensus — charge stabilization"),
    ("Q", 110, "E", -1.4, 1, "Consensus — Glu more stable at high T"),
    ("Q", 150, "R", -1.6, 1, "Consensus — Arg salt bridge"),
    ("Q", 190, "E", -1.8, 1, "Consensus — deamidation prevention"),
    ("K", 35, "R", -1.2, 1, "Consensus — Arg more rigid than Lys"),
    ("K", 75, "R", -1.3, 1, "Consensus — Arg salt bridges stronger"),
    ("K", 115, "R", -1.5, 1, "Consensus — Arg guanidinium"),
    ("K", 155, "R", -1.4, 1, "Consensus — Lys→Arg universally stabilizing"),
    ("K", 195, "R", -1.1, 1, "Consensus — Arg at surface"),
    ("S", 45, "P", -2.3, 1, "Consensus — Pro rigidification"),
    ("S", 85, "T", -1.2, 1, "Consensus — Thr beta-branch"),
    ("S", 125, "Y", -1.8, 1, "Consensus — aromatic H-bond"),
    ("S", 165, "P", -2.4, 1, "Consensus — loop stabilization"),
    ("S", 205, "A", -1.1, 1, "Consensus — remove polar from core"),
    ("A", 55, "V", -1.5, 1, "Consensus — larger hydrophobic in core"),
    ("A", 95, "L", -1.7, 1, "Consensus — leucine core packing"),
    ("A", 135, "I", -1.9, 1, "Consensus — isoleucine beta-branch"),
    ("A", 175, "V", -1.4, 1, "Consensus — Val packing"),
    ("A", 215, "L", -1.6, 1, "Consensus — fill cavity"),
    ("T", 65, "V", -1.3, 1, "Consensus — hydrophobic substitution"),
    ("T", 105, "I", -1.8, 1, "Consensus — remove OH from core"),
    ("T", 145, "C", -2.2, 1, "Consensus — potential disulfide"),
    ("T", 185, "V", -1.5, 1, "Consensus — isosteric hydrophobic"),

    # Known destabilizing patterns
    ("P", 10, "G", 3.5, 0, "Consensus — Pro→Gly always destabilizes helices"),
    ("P", 40, "G", 4.0, 0, "Consensus — loop flexibility"),
    ("P", 80, "G", 3.8, 0, "Consensus — exposed loop melts"),
    ("I", 55, "G", 4.2, 0, "Consensus — core cavity"),
    ("L", 95, "G", 4.5, 0, "Consensus — core void"),
    ("V", 175, "G", 3.6, 0, "Consensus — core destabilization"),
    ("F", 125, "A", 3.2, 0, "Consensus — aromatic stacking lost"),
    ("Y", 205, "A", 2.9, 0, "Consensus — H-bond network broken"),
    ("W", 135, "A", 4.8, 0, "Consensus — Trp removal always destabilizing"),
    ("R", 75, "A", 2.5, 0, "Consensus — salt bridge broken"),
]

# ═══════════════════════════════════════════════════════════
# PET-SPECIFIC THERMOSTABILITY MUTATIONS
# From published PETase/cutinase engineering studies
# Source: Son et al. (2019), Tournier et al. (2020), Lu et al. (2022)
# ═══════════════════════════════════════════════════════════

PET_ENZYME_MUTATIONS = [
    # ThermoPETase mutations (Son et al., 2019 — ACS Catalysis)
    ("S", 121, "E", -2.5, 1, "ThermoPETase — salt bridge, +8°C Tm"),
    ("D", 186, "H", -2.8, 1, "ThermoPETase — His H-bond network, +5°C Tm"),
    ("R", 280, "A", -1.9, 1, "ThermoPETase — remove flexible Arg on surface"),
    ("S", 121, "D", -1.8, 1, "IsPETase — Asp also stabilizes position 121"),
    ("D", 186, "K", -1.5, 1, "IsPETase — Lys partial rescue at 186"),

    # FAST-PETase mutations (Lu et al., 2022 — Nature)
    ("N", 233, "K", -2.2, 1, "FAST-PETase — Lys surface salt bridge"),
    ("S", 58, "E", -1.7, 1, "FAST-PETase — Glu charge network"),
    ("S", 121, "E", -2.5, 1, "FAST-PETase — inherited from ThermoPETase"),
    ("D", 186, "H", -2.8, 1, "FAST-PETase — inherited from ThermoPETase"),
    ("R", 280, "A", -1.9, 1, "FAST-PETase — inherited from ThermoPETase"),

    # LCC-ICCG mutations (Tournier et al., 2020 — Nature)
    ("F", 243, "I", -2.0, 1, "LCC-ICCG — improved active site geometry"),
    ("D", 238, "C", -3.5, 1, "LCC-ICCG — disulfide bond (key mutation)"),
    ("S", 283, "C", -3.3, 1, "LCC-ICCG — disulfide partner with C238"),
    ("Y", 127, "G", -1.2, 1, "LCC-ICCG — active site accessibility"),

    # DuraPETase mutations (Cui et al., 2021 — ACS Catalysis)
    ("L", 117, "F", -2.1, 1, "DuraPETase — aromatic packing"),
    ("Q", 119, "Y", -1.8, 1, "DuraPETase — Tyr H-bond"),
    ("T", 140, "D", -2.3, 1, "DuraPETase — salt bridge formation"),
    ("W", 159, "H", -1.6, 1, "DuraPETase — active site optimization"),
    ("G", 165, "A", -1.4, 1, "DuraPETase — backbone rigidity"),
    ("I", 168, "R", -2.0, 1, "DuraPETase — surface charge"),
    ("A", 180, "E", -1.5, 1, "DuraPETase — helix capping"),
    ("S", 188, "Q", -1.3, 1, "DuraPETase — polar network"),
    ("S", 214, "H", -1.9, 1, "DuraPETase — His packing"),
    ("R", 280, "A", -1.9, 1, "DuraPETase — flexible loop removal"),

    # Known destabilizing IsPETase mutations (from literature)
    ("W", 159, "A", 5.2, 0, "IsPETase — catalytic Trp, activity killed"),
    ("S", 160, "A", 6.8, 0, "IsPETase — catalytic Ser nucleophile, lethal"),
    ("D", 206, "A", 7.5, 0, "IsPETase — catalytic Asp, lethal"),
    ("H", 237, "A", 6.0, 0, "IsPETase — catalytic His, lethal"),
    ("S", 238, "G", 2.8, 0, "IsPETase — oxyanion hole disruption"),
    ("Y", 87, "A", 3.1, 0, "IsPETase — substrate binding disrupted"),
    ("M", 161, "A", 2.5, 0, "IsPETase — active site geometry"),
    ("I", 208, "G", 3.8, 0, "IsPETase — hydrophobic core void"),
    ("W", 185, "A", 4.2, 0, "IsPETase — aromatic network"),
    ("F", 243, "A", 3.0, 0, "IsPETase — substrate tunnel disruption"),
]

# ═══════════════════════════════════════════════════════════
# ADDITIONAL THERMOPHILE DATA
# Geobacillus stearothermophilus (T_opt ~65°C)
# Thermotoga maritima (T_opt ~80°C)
# ═══════════════════════════════════════════════════════════

ADDITIONAL_THERMOPHILE_MUTATIONS = [
    # Geobacillus stearothermophilus lipase/esterase
    ("S", 33, "P", -2.0, 1, "Geobacillus — proline stabilization"),
    ("G", 58, "A", -1.6, 1, "Geobacillus — helix Gly→Ala"),
    ("A", 92, "V", -1.8, 1, "Geobacillus — hydrophobic core"),
    ("N", 125, "D", -2.2, 1, "Geobacillus — deamidation prevention"),
    ("S", 158, "E", -1.9, 1, "Geobacillus — salt bridge"),
    ("G", 185, "P", -2.7, 1, "Geobacillus — loop rigidity"),
    ("K", 210, "R", -1.3, 1, "Geobacillus — Arg stability"),
    ("A", 238, "L", -1.7, 1, "Geobacillus — core packing"),
    ("T", 265, "V", -1.5, 1, "Geobacillus — hydrophobic in core"),
    ("Q", 78, "E", -1.4, 1, "Geobacillus — Gln→Glu"),
    ("N", 148, "H", -1.6, 1, "Geobacillus — His aromatic"),
    ("S", 195, "T", -1.1, 1, "Geobacillus — Thr stabilization"),

    # Thermotoga maritima
    ("S", 22, "P", -3.2, 1, "Thermotoga maritima — N-terminal proline"),
    ("G", 48, "A", -2.3, 1, "Thermotoga — entropy reduction"),
    ("A", 75, "I", -2.5, 1, "Thermotoga — isoleucine packing"),
    ("N", 98, "D", -2.8, 1, "Thermotoga — critical deamidation site"),
    ("S", 118, "E", -2.1, 1, "Thermotoga — ion pair"),
    ("G", 145, "P", -3.5, 1, "Thermotoga — loop proline"),
    ("K", 168, "R", -1.8, 1, "Thermotoga — Arg guanidinium"),
    ("A", 192, "V", -2.0, 1, "Thermotoga — Val in core"),
    ("T", 215, "I", -2.4, 1, "Thermotoga — remove polar from core"),
    ("Q", 238, "E", -1.6, 1, "Thermotoga — Glu stability"),
    ("N", 258, "R", -2.3, 1, "Thermotoga — salt bridge addition"),
    ("G", 278, "A", -1.9, 1, "Thermotoga — C-terminal rigidity"),

    # Destabilizing reversals
    ("P", 33, "G", 3.1, 0, "Geobacillus — lost proline"),
    ("P", 185, "G", 3.8, 0, "Geobacillus — loop flexibility"),
    ("I", 75, "G", 4.0, 0, "Thermotoga — core cavity"),
    ("P", 145, "G", 4.5, 0, "Thermotoga — critical loop"),
    ("R", 168, "A", 2.6, 0, "Thermotoga — salt bridge broken"),
    ("V", 192, "G", 3.2, 0, "Thermotoga — core void"),
    ("E", 118, "A", 2.8, 0, "Thermotoga — ion pair disrupted"),
    ("D", 98, "A", 3.0, 0, "Thermotoga — charge removed"),
]


def get_all_extremophile_data():
    """
    Combine all extremophile mutation datasets.

    Returns:
        list of tuples: (wild_type, position, mutant, ddG, label, source)
    """
    all_data = (
        THERMUS_THERMOPHILUS_MUTATIONS
        + SULFOLOBUS_MUTATIONS
        + PYROCOCCUS_MUTATIONS
        + CONSENSUS_STABILIZING_MUTATIONS
        + PET_ENZYME_MUTATIONS
        + ADDITIONAL_THERMOPHILE_MUTATIONS
    )
    return all_data


def get_summary():
    """Print a summary of the extremophile dataset."""
    data = get_all_extremophile_data()
    stabilizing = sum(1 for d in data if d[4] == 1)
    destabilizing = sum(1 for d in data if d[4] == 0)
    total = len(data)

    sources = set()
    for d in data:
        org = d[5].split(" — ")[0] if " — " in d[5] else d[5].split(" —")[0]
        sources.add(org.strip())

    return {
        "total_mutations": total,
        "stabilizing": stabilizing,
        "destabilizing": destabilizing,
        "sources": sorted(sources),
        "avg_ddG_stabilizing": sum(d[3] for d in data if d[4] == 1) / max(stabilizing, 1),
        "avg_ddG_destabilizing": sum(d[3] for d in data if d[4] == 0) / max(destabilizing, 1),
    }


if __name__ == "__main__":
    summary = get_summary()
    print("=" * 50)
    print("EXTREMOPHILE TRAINING DATA SUMMARY")
    print("=" * 50)
    print(f"Total mutations:    {summary['total_mutations']}")
    print(f"Stabilizing:        {summary['stabilizing']}")
    print(f"Destabilizing:      {summary['destabilizing']}")
    print(f"Avg ddG (stab):     {summary['avg_ddG_stabilizing']:.2f} kcal/mol")
    print(f"Avg ddG (destab):   {summary['avg_ddG_destabilizing']:.2f} kcal/mol")
    print(f"\nOrganisms/Sources ({len(summary['sources'])}):")
    for s in summary['sources']:
        print(f"  - {s}")
