"""
Structure resolution service.

Finds the best available 3D structure for a protein given any combination
of metadata identifiers, querying all available databases in priority order.

Priority (highest to lowest):
  1. Direct structure ID
     - AlphaFold "AF-<UniProt>-F1" format  -> AlphaFold EBI
     - 4-char RCSB PDB ID + known UniProt  -> AlphaFold EBI first, then RCSB
     - 4-char RCSB PDB ID (unknown UniProt) -> RCSB crystal structure
  2. UniProt accession -> AlphaFold EBI, then RCSB search
  3. Gene name + optional organism -> UniProt REST search -> step 2
  4. Protein name + optional organism -> UniProt REST search -> step 2
  5. Sequence similarity -> RCSB sequence search (last resort)
"""

import os
import pathlib
from typing import TypedDict

import httpx

_AF_DIR  = pathlib.Path(os.path.dirname(__file__)).parent / "alphafold_structures"
_PDB_DIR = pathlib.Path(os.path.dirname(__file__)).parent / "cached_data" / "structures"

# Confirmed PDB-ID -> UniProt mappings for common preset proteins.
# Keeps the resolver from hitting RCSB when AlphaFold is preferred.
_KNOWN_PDB_TO_UNIPROT: dict[str, str] = {
    "5XJH": "A0A0K8P6T7",  # IsPETase (Ideonella sakaiensis)
    "6EQE": "A0A0K8P6T7",
    "6IJ6": "A0A0K8P6T7",  # ThermoPETase
    "6KLD": "A0A0K8P6T7",
    "7SH4": "A0A0K8P6T7",  # FAST-PETase
    "4EB0": "G9BY57",       # LCC
    "7VVE": "G9BY57",
}

# Common organism name -> NCBI taxonomy ID for UniProt queries
_TAXON_IDS: dict[str, str] = {
    "H. sapiens": "9606", "Homo sapiens": "9606", "human": "9606",
    "I. sakaiensis": "1907928", "Ideonella sakaiensis": "1907928",
    "T. fusca": "2571", "Thermobifida fusca": "2571",
    "E. coli": "562", "Escherichia coli": "562",
    "S. cerevisiae": "4932", "Saccharomyces cerevisiae": "4932",
    "M. musculus": "10090", "Mus musculus": "10090", "mouse": "10090",
}


class ResolvedStructure(TypedDict):
    pdb_text: str
    source: str              # Human-readable: "AlphaFold DB - FUS (Homo sapiens), UniProt P35637"
    source_type: str         # "alphafold" | "rcsb"
    is_experimental: bool    # True for crystal/cryo-EM, False for AlphaFold
    resolved_pdb_id: str     # "AF-P35637-F1" or "5XJH"
    resolved_uniprot_id: str | None
    method: str              # "ALPHAFOLD PREDICTION" | "X-RAY DIFFRACTION" | "ELECTRON MICROSCOPY"
    resolution: float | None  # Crystal/cryo-EM resolution in Angstroms
    plddt_avg: float | None   # AlphaFold average pLDDT (0-100)


async def resolve(
    *,
    pdb_id: str | None = None,
    uniprot_id: str | None = None,
    gene: str | None = None,
    name: str | None = None,
    organism: str | None = None,
    sequence: str | None = None,
) -> ResolvedStructure | None:
    """
    Find the best 3D structure for a protein using any available metadata.
    Returns None if no structure can be found from any source.
    """

    # ── 1. Direct structure ID ───────────────────────────────────────────────
    if pdb_id:
        pid = pdb_id.strip()

        # AlphaFold AF-<UniProt>-F1 format -> extract UniProt directly
        if pid.upper().startswith("AF-"):
            parts = pid.split("-")
            uid = parts[1].upper() if len(parts) >= 2 else None
            if uid:
                result = await _alphafold(uid)
                if result:
                    return result

        # Standard 4-char RCSB PDB ID — prefer the experimental structure
        elif len(pid) == 4 and pid.isalnum():
            result = await _rcsb(pid.upper())
            if result:
                return result
            # RCSB unavailable — fall back to AlphaFold if UniProt known
            uid = _KNOWN_PDB_TO_UNIPROT.get(pid.upper()) or (
                uniprot_id.strip().upper() if uniprot_id else None
            )
            if uid:
                result = await _alphafold(uid)
                if result:
                    return result

    # ── 2. UniProt accession ─────────────────────────────────────────────────
    if uniprot_id:
        uid = uniprot_id.strip().upper()
        result = await _alphafold(uid)
        if result:
            return result
        result = await _rcsb_by_uniprot(uid)
        if result:
            return result

    # ── 3. Gene name + optional organism ─────────────────────────────────────
    if gene:
        uid = await _uniprot_search(gene=gene, organism=organism)
        if uid:
            result = await _alphafold(uid)
            if result:
                return result
            result = await _rcsb_by_uniprot(uid)
            if result:
                return result

    # ── 4. Protein name + optional organism ──────────────────────────────────
    if name:
        uid = await _uniprot_search(name=name, organism=organism)
        if uid:
            result = await _alphafold(uid)
            if result:
                return result
            result = await _rcsb_by_uniprot(uid)
            if result:
                return result

    # ── 5. Sequence similarity (last resort) ─────────────────────────────────
    if sequence and len(sequence) >= 20:
        result = await _rcsb_by_sequence(sequence)
        if result:
            return result

    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _alphafold(uniprot: str) -> ResolvedStructure | None:
    """Return AlphaFold structure for a UniProt accession. Caches locally."""
    local = _AF_DIR / f"{uniprot}.pdb"
    if local.exists():
        pdb_text = local.read_text()
        return ResolvedStructure(
            pdb_text=pdb_text,
            source=f"AlphaFold DB (UniProt {uniprot})",
            source_type="alphafold",
            is_experimental=False,
            resolved_pdb_id=f"AF-{uniprot}-F1",
            resolved_uniprot_id=uniprot,
            method="ALPHAFOLD PREDICTION",
            resolution=None,
            plddt_avg=_plddt_avg(pdb_text),
        )
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            api = await c.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}")
            if api.status_code != 200:
                return None
            entries = api.json()
            if not entries:
                return None
            entry   = entries[0]
            pdb_url = entry.get("pdbUrl")
            if not pdb_url:
                return None
            pdb_r = await c.get(pdb_url)
            if pdb_r.status_code != 200:
                return None
            pdb_text = pdb_r.text
            _AF_DIR.mkdir(parents=True, exist_ok=True)
            local.write_text(pdb_text)

            gene_label = entry.get("gene") or entry.get("uniprotDescription") or uniprot
            org_label  = entry.get("organismScientificName", "")
            plddt_val  = entry.get("globalMetricValue") or _plddt_avg(pdb_text)
            detail     = gene_label + (f" ({org_label})" if org_label else "")

            return ResolvedStructure(
                pdb_text=pdb_text,
                source=f"AlphaFold DB - {detail}, UniProt {uniprot}",
                source_type="alphafold",
                is_experimental=False,
                resolved_pdb_id=f"AF-{uniprot}-F1",
                resolved_uniprot_id=uniprot,
                method="ALPHAFOLD PREDICTION",
                resolution=None,
                plddt_avg=round(float(plddt_val), 1) if plddt_val else None,
            )
    except Exception:
        return None


async def _rcsb(pdb_id: str) -> ResolvedStructure | None:
    """Fetch a structure from RCSB PDB. Caches locally."""
    _PDB_DIR.mkdir(parents=True, exist_ok=True)
    cache = _PDB_DIR / f"rcsb_{pdb_id}.pdb"
    if cache.exists():
        return ResolvedStructure(
            pdb_text=cache.read_text(),
            source=f"RCSB PDB {pdb_id} (cached)",
            source_type="rcsb",
            is_experimental=True,
            resolved_pdb_id=pdb_id,
            resolved_uniprot_id=None,
            method="EXPERIMENTAL",
            resolution=None,
            plddt_avg=None,
        )
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
            if r.status_code != 200:
                return None
            pdb_text = r.text

            resolution: float | None = None
            method = "EXPERIMENTAL"
            try:
                meta = await c.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
                if meta.status_code == 200:
                    mj = meta.json()
                    exp = mj.get("exptl", [{}])[0]
                    method = exp.get("method", "EXPERIMENTAL")
                    refine = mj.get("refine", [{}])
                    if refine and refine[0].get("ls_d_res_high"):
                        resolution = float(refine[0]["ls_d_res_high"])
            except Exception:
                pass

            cache.write_text(pdb_text)
            res_str = f", {resolution:.1f} A" if resolution else ""
            return ResolvedStructure(
                pdb_text=pdb_text,
                source=f"RCSB PDB {pdb_id} ({method.lower()}{res_str})",
                source_type="rcsb",
                is_experimental=True,
                resolved_pdb_id=pdb_id,
                resolved_uniprot_id=None,
                method=method,
                resolution=resolution,
                plddt_avg=None,
            )
    except Exception:
        return None


async def _rcsb_by_uniprot(uniprot_id: str) -> ResolvedStructure | None:
    """Search RCSB for structures associated with a UniProt accession."""
    try:
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers"
                                 ".reference_sequence_identifiers.database_accession",
                    "operator": "in",
                    "negation": False,
                    "value": [uniprot_id],
                },
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": 3},
                "sort": [{"sort_by": "score", "direction": "desc"}],
                "results_content_type": ["experimental"],
            },
        }
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
            if r.status_code != 200:
                return None
            hits = r.json().get("result_set", [])
            if not hits:
                return None
            return await _rcsb(hits[0]["identifier"])
    except Exception:
        return None


async def _uniprot_search(
    gene: str | None = None,
    name: str | None = None,
    organism: str | None = None,
) -> str | None:
    """Return the best UniProt accession for a gene/protein name + organism."""
    parts: list[str] = []
    if gene:
        parts.append(f"gene:{gene}")
    elif name:
        parts.append(f"protein_name:{name}")
    if not parts:
        return None
    taxon = _TAXON_IDS.get(organism or "")
    if taxon:
        parts.append(f"organism_id:{taxon}")
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": " AND ".join(parts),
                    "format": "json",
                    "size": 1,
                    "fields": "accession",
                },
            )
            if r.status_code != 200:
                return None
            results = r.json().get("results", [])
            if results:
                return results[0]["primaryAccession"]
    except Exception:
        return None
    return None


async def _rcsb_by_sequence(sequence: str) -> ResolvedStructure | None:
    """Search RCSB by sequence similarity (>=90% identity)."""
    try:
        query = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 1,
                    "identity_cutoff": 0.9,
                    "sequence_type": "protein",
                    "value": sequence[:400],
                },
            },
            "return_type": "entry",
            "request_options": {"results_content_type": ["experimental"]},
        }
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
            if r.status_code != 200:
                return None
            hits = r.json().get("result_set", [])
            if not hits:
                return None
            return await _rcsb(hits[0]["identifier"])
    except Exception:
        return None


def _plddt_avg(pdb_text: str) -> float | None:
    """Compute mean pLDDT from B-factor column of CA atoms."""
    vals: list[float] = []
    for line in pdb_text.splitlines():
        if line[:4] == "ATOM" and line[12:16].strip() == "CA":
            try:
                b = float(line[60:66])
                if 0.0 <= b <= 100.0:
                    vals.append(b)
            except (ValueError, IndexError):
                continue
    return round(sum(vals) / len(vals), 1) if vals else None
