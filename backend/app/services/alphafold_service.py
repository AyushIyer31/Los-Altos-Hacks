"""AlphaFold Server API integration for structure prediction."""

import requests
import json
import time
import hashlib
from pathlib import Path

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/predict"
STRUCTURE_CACHE_DIR = Path(__file__).parent.parent / "cached_structures"
STRUCTURE_CACHE_DIR.mkdir(exist_ok=True)

def _get_cache_path(sequence: str) -> Path:
    """Get cache file path for a sequence."""
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    return STRUCTURE_CACHE_DIR / f"{seq_hash}.pdb"

def _get_plddt_path(sequence: str) -> Path:
    """Get pLDDT cache file path."""
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    return STRUCTURE_CACHE_DIR / f"{seq_hash}_plddt.json"

def predict_structure(sequence: str, name: str = "protein") -> dict:
    """
    Predict protein structure using AlphaFold Server API.

    Args:
        sequence: Amino acid sequence
        name: Protein name for the job

    Returns:
        {
            'pdb_data': PDB structure string,
            'plddt': average pLDDT score,
            'plddt_per_residue': list of pLDDT values,
            'job_id': AlphaFold Server job ID,
            'cached': whether result was cached
        }
    """

    cache_path = _get_cache_path(sequence)
    plddt_path = _get_plddt_path(sequence)

    # Check cache first
    if cache_path.exists() and plddt_path.exists():
        print(f"[AlphaFold] Cache hit for sequence (length {len(sequence)})")
        with open(cache_path) as f:
            pdb_data = f.read()
        with open(plddt_path) as f:
            plddt_data = json.load(f)
        return {
            'pdb_data': pdb_data,
            'plddt': plddt_data.get('average', 60),
            'plddt_per_residue': plddt_data.get('per_residue', []),
            'cached': True
        }

    print(f"[AlphaFold] Submitting structure prediction for sequence (length {len(sequence)})")

    try:
        # Submit prediction job
        response = requests.post(
            ALPHAFOLD_API,
            json={
                "sequence": sequence,
                "name": name[:50]  # Limit name length
            },
            timeout=30
        )
        response.raise_for_status()

        job_data = response.json()
        job_id = job_data.get('jobId')

        if not job_id:
            print(f"[AlphaFold] Error: No job ID returned")
            return None

        print(f"[AlphaFold] Job submitted: {job_id}, waiting for results...")

        # Poll for results (max 15 min)
        max_attempts = 90
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            time.sleep(10)  # Wait 10 seconds between polls

            # Check job status
            status_response = requests.get(
                f"{ALPHAFOLD_API}/{job_id}",
                timeout=30
            )

            if status_response.status_code == 200:
                result = status_response.json()

                if result.get('status') == 'COMPLETED':
                    print(f"[AlphaFold] Job {job_id} completed in {attempt * 10}s")

                    # Extract PDB data and pLDDT
                    pdb_data = result.get('pdb', '')
                    plddt_scores = result.get('plddt', [])
                    avg_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 60

                    # Cache results
                    with open(cache_path, 'w') as f:
                        f.write(pdb_data)
                    with open(plddt_path, 'w') as f:
                        json.dump({
                            'average': avg_plddt,
                            'per_residue': plddt_scores
                        }, f)

                    return {
                        'pdb_data': pdb_data,
                        'plddt': avg_plddt,
                        'plddt_per_residue': plddt_scores,
                        'job_id': job_id,
                        'cached': False
                    }

                elif result.get('status') == 'FAILED':
                    print(f"[AlphaFold] Job {job_id} failed")
                    return None

                else:
                    print(f"[AlphaFold] Job {job_id} still running... ({attempt*10}s elapsed)")

            elif status_response.status_code == 404:
                print(f"[AlphaFold] Job {job_id} not found, retrying...")
                continue

            else:
                print(f"[AlphaFold] Status check error: {status_response.status_code}")

        print(f"[AlphaFold] Job {job_id} timeout after {max_attempts * 10}s")
        return None

    except requests.exceptions.RequestException as e:
        print(f"[AlphaFold] API error: {e}")
        return None
    except Exception as e:
        print(f"[AlphaFold] Unexpected error: {e}")
        return None


def predict_structures_batch(sequences: list, names: list = None) -> list:
    """
    Predict structures for multiple sequences in parallel-friendly way.

    Args:
        sequences: List of amino acid sequences
        names: Optional list of protein names

    Returns:
        List of structure predictions (may have None entries if failed)
    """
    results = []
    for i, seq in enumerate(sequences):
        name = names[i] if names and i < len(names) else f"protein_{i}"
        result = predict_structure(seq, name)
        results.append(result)
    return results
