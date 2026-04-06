"""Fetch PETase and related enzyme structures from RCSB PDB."""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry"

KNOWN_PETASE_IDS = [
    # ── IsPETase (Ideonella sakaiensis) & variants ──
    "5XJH", "6EQE", "5XG0", "6ANE", "5YNS", "6IJ6", "6EQD", "6EQF", "6EQG", "6EQH",
    "6IJ3", "6IJ4", "6IJ5", "6QGC", "7CGA", "5XH3", "5XFY", "5XFZ",
    "6ILW", "6ILX", "7XJY", "7XJZ",
    # ── FAST-PETase / DepoPETase / HotPETase ──
    "8J17", "8J5N", "7SH6", "7SH7", "7QCS", "7QCT",
    # ── LCC & variants ──
    "4EB0", "7VVC", "7VVE", "7W1N", "7W44", "7W45", "8CMV", "8JMO", "8JMP", "8QRJ",
    "6THT", "6THS",
    # ── Cutinases (Thermobifida, Fusarium, Humicola, etc.) ──
    "4CG1", "4CG2", "4CG3", "1CEX", "1CUS", "2CUT", "3GBS", "3VIS",
    "5LUI", "5LUJ", "5ZOA", "7QJO", "7QJP", "7QJR",
    "8BRA", "8BRB", "8Z2G", "8Z2H", "8Z2I", "8Z2J", "8Z2K",
    "1AGY", "1XZL", "1XZM", "2CZQ", "3DCN", "3QPA", "3QPB",
    "4OYY", "4OYL", "5AOB",
    # ── Ancestral PETases ──
    "8ETX", "8ETY",
    # ── Thermostable hydrolases & Cut190 variants ──
    "7YKO", "7YKP", "7YKQ", "8GZD", "5BN0",
    "3WYN", "4WFI", "4WFJ", "4WFK",
    # ── Novel PET hydrolases ──
    "7PZJ", "7Z6B", "8OTU", "9LMT", "9LMU", "9LMV", "9LMW",
    "5ZNO", "6AID", "6SCD", "7BIC", "7PKA", "7PKB", "7PKC",
    "8A1L", "8A1M", "8RVR", "8RVS", "8FYQ", "8FYR",
    # ── MHETase ──
    "6QZ3", "6QZ4", "6JTV", "6JTU", "6QGA", "6QGB",
    # ── Lipases & esterases with PET activity ──
    "1TCA", "4K6G", "3W9B", "7EC8", "7EC9", "7ECA",
    "1LBS", "1CVL", "3LIP", "5CT4", "5CT5",
    # ── PHB depolymerases (polyhydroxybutyrate — related polymer) ──
    "2D80", "2D81", "3D2C", "3D2D",
    # ── Feruloyl esterases & carboxylesterases ──
    "1AUO", "1JJF", "3PFB", "3PFC",
    # ── Polyester hydrolases from diverse metagenomes ──
    "7EMV", "7EMW", "7CXZ", "7CY0", "7FDH", "7FDI",
    "7NWR", "7NWS", "7QDZ", "7QE0", "7XAS", "7XAT",
    "8BWG", "8BWH", "8BWI", "8D7K", "8D7L",
    "8HBQ", "8HBR", "8HBS", "8PGG", "8PGH",
    "8SVA", "8SVB", "8SVC", "8SVD",
    "8W6Q", "8W6R", "8WCZ", "8WD0",
    # ── Plastic-binding proteins & auxiliary enzymes ──
    "7DT3", "7DT4", "7DT5",
    # ── Arylesterases with polyester activity ──
    "6G21", "6G22", "6G23",
    # ── Triacylglycerol lipases relevant to bioplastics ──
    "1OIL", "3TGL", "4LIP",
    # ── Thermophilic esterases ──
    "1QZ3", "2YH2", "3FAK", "3ZWQ",
    # ── Additional PET hydrolases (2023-2025 deposits) ──
    "8R5A", "8R5B", "8R5C", "8R5D",
    "8TQA", "8TQB", "8TQC",
    "8UJK", "8UJL", "8UJM",
    "8VWX", "8VWY", "8VWZ",
    "8X3A", "8X3B",
]

# Enzyme family classification
ENZYME_FAMILIES = {
    "PETase": [
        "5XJH", "6EQE", "5XG0", "6ANE", "5YNS", "6IJ6", "6EQD", "6EQF", "6EQG", "6EQH",
        "6IJ3", "6IJ4", "6IJ5", "6QGC", "7CGA", "5XH3", "5XFY", "5XFZ", "6ILW", "6ILX",
        "7XJY", "7XJZ", "8J17", "8J5N", "7SH6", "7SH7", "7QCS", "7QCT",
    ],
    "Cutinase": [
        "1CEX", "1CUS", "2CUT", "3GBS", "3VIS", "4CG1", "4CG2", "4CG3",
        "5LUI", "5LUJ", "5ZOA", "7QJO", "7QJP", "7QJR",
        "8BRA", "8BRB", "8Z2G", "8Z2H", "8Z2I", "8Z2J", "8Z2K",
        "1AGY", "1XZL", "1XZM", "2CZQ", "3DCN", "3QPA", "3QPB",
        "4OYY", "4OYL", "5AOB",
    ],
    "LCC Variant": [
        "4EB0", "7VVC", "7VVE", "7W1N", "7W44", "7W45", "8CMV", "8JMO", "8JMP", "8QRJ",
        "6THT", "6THS",
    ],
    "Ancestral PETase": ["8ETX", "8ETY"],
    "Thermostable Hydrolase": [
        "7YKO", "7YKP", "7YKQ", "8GZD", "5BN0",
        "3WYN", "4WFI", "4WFJ", "4WFK",
        "1QZ3", "2YH2", "3FAK", "3ZWQ",
    ],
    "Novel PET Hydrolase": [
        "7PZJ", "7Z6B", "8OTU", "9LMT", "9LMU", "9LMV", "9LMW",
        "5ZNO", "6AID", "6SCD", "7BIC", "7PKA", "7PKB", "7PKC",
        "8A1L", "8A1M", "8RVR", "8RVS", "8FYQ", "8FYR",
        "7EMV", "7EMW", "7CXZ", "7CY0", "7FDH", "7FDI",
        "7NWR", "7NWS", "7QDZ", "7QE0", "7XAS", "7XAT",
        "8BWG", "8BWH", "8BWI", "8D7K", "8D7L",
        "8HBQ", "8HBR", "8HBS", "8PGG", "8PGH",
        "8SVA", "8SVB", "8SVC", "8SVD",
        "8W6Q", "8W6R", "8WCZ", "8WD0",
        "8R5A", "8R5B", "8R5C", "8R5D",
        "8TQA", "8TQB", "8TQC",
        "8UJK", "8UJL", "8UJM",
        "8VWX", "8VWY", "8VWZ",
        "8X3A", "8X3B",
        "7DT3", "7DT4", "7DT5",
    ],
    "MHETase": ["6QZ3", "6QZ4", "6JTV", "6JTU", "6QGA", "6QGB"],
    "Lipase / Esterase": [
        "1TCA", "4K6G", "3W9B", "7EC8", "7EC9", "7ECA",
        "1LBS", "1CVL", "3LIP", "5CT4", "5CT5",
        "1OIL", "3TGL", "4LIP",
        "6G21", "6G22", "6G23",
    ],
    "PHB Depolymerase": ["2D80", "2D81", "3D2C", "3D2D"],
    "Feruloyl Esterase": ["1AUO", "1JJF", "3PFB", "3PFC"],
}

def _classify_enzyme(pdb_id: str) -> str:
    """Classify enzyme by PDB ID into a family."""
    for family, ids in ENZYME_FAMILIES.items():
        if pdb_id in ids:
            return family
    return "Related Hydrolase"

# In-memory cache
_cache: list[dict] = []
_cache_time: float = 0
_CACHE_TTL = 600  # 10 minutes


def search_petase_structures(max_results: int = 300) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "or",
            "nodes": [
                {"type": "terminal", "service": "full_text", "parameters": {"value": "PETase plastic degrading"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "polyethylene terephthalate hydrolase"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "cutinase PET degradation"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "cutinase thermostable"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "polyester hydrolase"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "esterase plastic biodegradation"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "Thermobifida fusca cutinase"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "leaf branch compost cutinase LCC"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "lipase polyester degradation"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "MHETase mono hydroxyethyl terephthalate"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "PHB depolymerase polyhydroxybutyrate"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "feruloyl esterase carboxylesterase"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "polymer degrading enzyme thermophilic"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "PET hydrolase engineered mutant"}},
            ],
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"], "paginate": {"start": 0, "rows": max_results}},
    }
    try:
        resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return [hit["identifier"] for hit in data.get("result_set", [])]
    except Exception:
        return list(KNOWN_PETASE_IDS)


def search_rcsb_live(query_text: str, max_results: int = 30) -> list[dict]:
    """Live search RCSB PDB for any user query. Returns metadata + sequence."""
    # First try full-text search
    query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query_text},
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"], "paginate": {"start": 0, "rows": max_results}},
    }
    try:
        resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])]
    except Exception:
        pdb_ids = []

    # Also check if the query looks like a PDB ID (4 chars)
    q = query_text.strip().upper()
    if len(q) == 4 and q.isalnum() and q not in pdb_ids:
        pdb_ids.insert(0, q)

    results = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_fetch_single_entry, pid): pid for pid in pdb_ids[:max_results]}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda r: r["pdb_id"])
    return results


def _fetch_organism(pdb_id: str) -> str:
    """Fetch organism from polymer entity endpoint."""
    try:
        resp = requests.get(
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1",
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        sources = data.get("rcsb_entity_source_organism", [])
        if sources:
            return sources[0].get("ncbi_scientific_name", "Unknown")
    except Exception:
        pass
    return "Unknown"


def fetch_entry_metadata(pdb_id: str) -> dict:
    try:
        resp = requests.get(f"{RCSB_DATA_URL}/{pdb_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        organism = _fetch_organism(pdb_id)
        return {
            "pdb_id": pdb_id,
            "title": data.get("struct", {}).get("title", "Unknown"),
            "organism": organism,
            "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
            "family": _classify_enzyme(pdb_id),
        }
    except Exception:
        return {"pdb_id": pdb_id, "title": "Unknown", "organism": "Unknown", "resolution": None, "family": _classify_enzyme(pdb_id)}


def fetch_sequence(pdb_id: str) -> str:
    try:
        resp = requests.get(f"{RCSB_FASTA_URL}/{pdb_id}", timeout=10)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        return "".join(line.strip() for line in lines if not line.startswith(">"))
    except Exception:
        return ""


def _fetch_single_entry(pdb_id: str) -> dict | None:
    """Fetch metadata + sequence for one PDB entry."""
    meta = fetch_entry_metadata(pdb_id)
    seq = fetch_sequence(pdb_id)
    if seq:
        meta["sequence"] = seq
        return meta
    return None


def fetch_all_petase_data() -> list[dict]:
    """Fetch all PETase data with parallel requests and caching."""
    global _cache, _cache_time

    if _cache and (time.time() - _cache_time) < _CACHE_TTL:
        return _cache

    pdb_ids = search_petase_structures()
    all_ids = list(dict.fromkeys(KNOWN_PETASE_IDS + pdb_ids))

    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_fetch_single_entry, pid): pid for pid in all_ids}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Sort: known IDs first, then alphabetically
    known_set = set(KNOWN_PETASE_IDS)
    results.sort(key=lambda r: (0 if r["pdb_id"] in known_set else 1, r["pdb_id"]))

    _cache = results
    _cache_time = time.time()
    return results
