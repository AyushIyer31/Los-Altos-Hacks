"""Mutation effect scoring using structural properties.

Since FoldX isn't easily installable, this uses:
1. Amino acid property changes (size, charge, hydrophobicity)
2. Sidechain interaction predictions
3. Stability penalties for problematic changes

Returns ΔΔG-like scores (negative = stabilizing, positive = destabilizing)
"""

from Bio import PDB
import math

# Amino acid properties
AA_PROPS = {
    'A': {'size': 0, 'hydro': 0.62, 'charge': 0, 'pI': 6.0},   # Alanine
    'R': {'size': 2, 'hydro': -3.1, 'charge': 1, 'pI': 10.8},  # Arginine
    'N': {'size': 1, 'hydro': -0.77, 'charge': 0, 'pI': 5.4},  # Asparagine
    'D': {'size': 1, 'hydro': -1.03, 'charge': -1, 'pI': 2.9}, # Aspartate
    'C': {'size': 1, 'hydro': 0.29, 'charge': 0, 'pI': 5.1},   # Cysteine
    'E': {'size': 2, 'hydro': -1.51, 'charge': -1, 'pI': 3.2}, # Glutamate
    'Q': {'size': 2, 'hydro': -0.85, 'charge': 0, 'pI': 5.7},  # Glutamine
    'G': {'size': 0, 'hydro': 0.48, 'charge': 0, 'pI': 5.9},   # Glycine
    'H': {'size': 1, 'hydro': -0.40, 'charge': 0.5, 'pI': 7.6},# Histidine
    'I': {'size': 1, 'hydro': 1.38, 'charge': 0, 'pI': 5.8},   # Isoleucine
    'L': {'size': 1, 'hydro': 1.06, 'charge': 0, 'pI': 6.0},   # Leucine
    'K': {'size': 2, 'hydro': -1.89, 'charge': 1, 'pI': 9.7},  # Lysine
    'M': {'size': 1, 'hydro': 0.64, 'charge': 0, 'pI': 5.7},   # Methionine
    'F': {'size': 1, 'hydro': 1.19, 'charge': 0, 'pI': 5.5},   # Phenylalanine
    'P': {'size': 1, 'hydro': 0.12, 'charge': 0, 'pI': 6.3},   # Proline
    'S': {'size': 0, 'hydro': -0.26, 'charge': 0, 'pI': 5.7},  # Serine
    'T': {'size': 0, 'hydro': -0.07, 'charge': 0, 'pI': 5.6},  # Threonine
    'W': {'size': 2, 'hydro': 0.81, 'charge': 0, 'pI': 5.9},   # Tryptophan
    'Y': {'size': 1, 'hydro': 0.26, 'charge': 0, 'pI': 5.6},   # Tyrosine
    'V': {'size': 0, 'hydro': 1.08, 'charge': 0, 'pI': 6.0},   # Valine
}

def calculate_mutation_effect(wt_aa: str, mut_aa: str, context: str = 'buried') -> float:
    """
    Calculate mutation effect as ΔΔG-like score.

    Args:
        wt_aa: Wild-type amino acid (1-letter code)
        mut_aa: Mutant amino acid (1-letter code)
        context: 'buried', 'exposed', 'interface' (affects scoring)

    Returns:
        ΔΔG score (negative = stabilizing, positive = destabilizing)
    """

    if wt_aa not in AA_PROPS or mut_aa not in AA_PROPS:
        return 0.0  # Unknown amino acid

    wt = AA_PROPS[wt_aa]
    mut = AA_PROPS[mut_aa]

    ddg = 0.0

    # 1. Size mismatch penalty (core/buried favors similar sizes)
    size_diff = abs(mut['size'] - wt['size'])
    if context == 'buried':
        ddg += size_diff * 0.5  # Larger penalty in buried region
    elif context == 'exposed':
        ddg += size_diff * 0.2  # Smaller penalty on surface

    # 2. Hydrophobicity mismatch
    hydro_diff = abs(mut['hydro'] - wt['hydro'])
    if context == 'buried':
        # Hydrophobic -> hydrophilic in core is bad
        if wt['hydro'] > 0.5 and mut['hydro'] < -0.5:
            ddg += hydro_diff * 1.5
        else:
            ddg += hydro_diff * 0.3
    elif context == 'exposed':
        # Hydrophobic -> hydrophilic on surface is OK
        ddg += hydro_diff * 0.1

    # 3. Charge changes (can be good or bad depending on context)
    charge_diff = abs(mut['charge'] - wt['charge'])
    if context == 'interface' and charge_diff > 0:
        ddg -= 0.3  # Charge changes at interface can help interactions
    elif charge_diff > 0:
        ddg += charge_diff * 0.4

    # 4. Proline penalty (breaks secondary structure)
    if mut_aa == 'P' and wt_aa != 'P':
        ddg += 2.0  # Large penalty for introducing proline

    # 5. Cysteine (disulfide bridge considerations)
    if mut_aa == 'C' and wt_aa != 'C':
        ddg -= 0.5  # Potential benefit from new disulfide

    # 6. Glycine (high flexibility)
    if mut_aa == 'G' and context == 'buried':
        ddg += 1.0  # Glycine in buried region is destabilizing

    return round(ddg, 2)


def score_mutations(pdb_data: str, mutations: list) -> dict:
    """
    Score multiple mutations in context of protein structure.

    Args:
        pdb_data: PDB file content as string
        mutations: List of mutation strings (e.g., ['S122G', 'D186H'])

    Returns:
        {
            'mutations': [
                {
                    'mutation': 'S122G',
                    'ddg': -1.2,
                    'effect': 'stabilizing',
                    'context': 'buried'
                },
                ...
            ],
            'total_ddg': -2.5,
            'overall_effect': 'stabilizing'
        }
    """

    # Parse PDB to find residue contexts
    from io import StringIO
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', StringIO(pdb_data))

    # Simple buried/exposed prediction based on B-factor
    # (Real version would use surface area calculation)
    residue_context = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.id[1]
                if 'CA' in residue:
                    b_factor = residue['CA'].bfactor
                    if b_factor > 40:
                        residue_context[res_id] = 'exposed'
                    else:
                        residue_context[res_id] = 'buried'

    results = []
    total_ddg = 0.0

    for mut_str in mutations:
        match = mut_str.strip()
        if len(match) < 3:
            continue

        wt_aa = match[0]
        mut_aa = match[-1]
        pos = int(match[1:-1])

        context = residue_context.get(pos, 'buried')
        ddg = calculate_mutation_effect(wt_aa, mut_aa, context)

        effect = 'stabilizing' if ddg < -0.5 else 'destabilizing' if ddg > 0.5 else 'neutral'

        results.append({
            'mutation': mut_str,
            'position': pos,
            'wt_aa': wt_aa,
            'mut_aa': mut_aa,
            'ddg': ddg,
            'effect': effect,
            'context': context
        })

        total_ddg += ddg

    overall = 'stabilizing' if total_ddg < -0.5 else 'destabilizing' if total_ddg > 0.5 else 'neutral'

    return {
        'mutations': results,
        'total_ddg': round(total_ddg, 2),
        'overall_effect': overall
    }
