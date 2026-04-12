"""Latent space optimization for PETase enzyme engineering.

Uses trained XGBoost classifier (94.4% CV accuracy) and amino acid property
analysis to identify beneficial mutations. Scores candidates using the
classifier's probability estimates.
"""

import numpy as np
from .amino_acid_props import (
    CATALYTIC_RESIDUES, THERMOSTABILITY_HOTSPOTS,
    HYDROPHOBICITY, SIZE, CHARGE, FLEXIBILITY,
)

# Base weights for the combined fitness score
BASE_STABILITY_WEIGHT = 0.5
BASE_ACTIVITY_WEIGHT = 0.5

TEMP_BASELINE = 40.0
TEMP_SCALE = 0.006

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _get_temp_weights(target_temp: float) -> tuple[float, float]:
    """Calculate stability/activity weights based on target temperature."""
    temp_shift = max(0.0, target_temp - TEMP_BASELINE) * TEMP_SCALE
    stability_w = min(0.85, BASE_STABILITY_WEIGHT + temp_shift)
    activity_w = max(0.15, BASE_ACTIVITY_WEIGHT - temp_shift)
    return stability_w, activity_w


def _get_hotspot_bonus(target_temp: float) -> float:
    if target_temp <= 50:
        return 0.01
    elif target_temp <= 65:
        return 0.03
    else:
        return 0.05


def _scan_beneficial_mutations(sequence: str, top_k: int = 50) -> list[dict]:
    """Scan single-point mutations using the trained classifier (vectorized).

    Uses raw numpy arrays to score all ~6000 mutations in a single batch,
    then only builds result dicts for the beneficial ones (~50).
    """
    from . import trained_classifier as _clf
    _clf.train_model()

    catalytic_set = set(CATALYTIC_RESIDUES.values())
    aa_set = set(AMINO_ACIDS)

    # Build all mutation tuples at once
    mutation_tuples = []
    mutation_meta = []  # parallel list of (pos, wt_aa, mut_aa)
    for pos in range(len(sequence)):
        wt_aa = sequence[pos]
        if wt_aa not in aa_set or pos in catalytic_set:
            continue
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            mutation_tuples.append((wt_aa, pos + 1, mut_aa))
            mutation_meta.append((pos, wt_aa, mut_aa))

    # Single batch — returns raw numpy arrays, no dict overhead
    all_ddg, all_prob = _clf.predict_mutations_batch_raw(mutation_tuples, sequence=sequence)

    if len(all_ddg) == 0:
        return []

    # Filter beneficial (ddg < 0 and prob > 0.5) using numpy masks
    beneficial_mask = (all_ddg < 0) & (all_prob > 0.5)
    beneficial_indices = np.where(beneficial_mask)[0]

    # Only build dicts for the beneficial mutations (~50 vs ~6000)
    mutations = []
    for idx in beneficial_indices:
        pos, wt_aa, mut_aa = mutation_meta[idx]
        prob = float(all_prob[idx])
        score = prob + (0.1 if pos in THERMOSTABILITY_HOTSPOTS else 0.0)
        mutations.append({
            "position": pos,
            "wild_type": wt_aa,
            "mutant": mut_aa,
            "score": score,
            "confidence": round(prob, 4),
            "label": f"{wt_aa}{pos + 1}{mut_aa}",
        })

    mutations.sort(key=lambda x: x["score"], reverse=True)
    return mutations[:top_k]


def _score_candidate(sequence: str, original: str, score_cache: dict = None) -> tuple[float, float]:
    """Score a candidate using cached mutation scores (no re-prediction).

    Returns (stability_score, activity_score) both in 0-1 range.
    score_cache maps mutation label (e.g. "S121E") to probability_beneficial.
    """
    mutations = []
    for i, (wt, mt) in enumerate(zip(original, sequence)):
        if wt != mt:
            mutations.append((wt, i + 1, mt))

    if not mutations:
        return 0.5, 0.5

    catalytic_vals = set(CATALYTIC_RESIDUES.values())
    probs = []
    active_site_probs = []

    for wt, pos, mt in mutations:
        label = f"{wt}{pos}{mt}"
        if score_cache and label in score_cache:
            prob = score_cache[label]
        else:
            # Fallback: quick single prediction only if not in cache
            from . import trained_classifier as _clf
            pred = _clf.predict_mutation(wt, pos, mt, sequence=original)
            prob = pred["probability_beneficial"]
        probs.append(prob)
        near_active = any(abs((pos - 1) - center) <= 5 for center in catalytic_vals)
        if near_active:
            active_site_probs.append(prob)

    stability_score = float(np.mean(probs))

    if active_site_probs:
        activity_score = float(np.mean(active_site_probs))
    else:
        activity_score = 0.5 + 0.2 * float(np.mean(probs))

    return stability_score, activity_score


def _combine_beneficial_mutations(
    sequence: str,
    beneficial_mutations: list[dict],
    num_mutations: int = 3,
) -> str:
    seq_list = list(sequence)
    used_positions = set()
    applied = 0

    for mut in beneficial_mutations:
        pos = mut["position"]
        if pos in used_positions:
            continue
        if pos in CATALYTIC_RESIDUES.values():
            continue
        seq_list[pos] = mut["mutant"]
        used_positions.add(pos)
        applied += 1
        if applied >= num_mutations:
            break

    return "".join(seq_list)


# In-memory cache for optimization results (avoids re-scanning on repeated runs)
_optimize_cache: dict[str, dict] = {}
_OPTIMIZE_CACHE_MAX = 20


def optimize(
    sequence: str,
    num_candidates: int = 10,
    optimization_steps: int = 50,
    target_temp: float = 60.0,
) -> dict:
    """Run optimization to generate improved PETase candidates.

    Uses the trained XGBoost classifier and amino acid property analysis.
    Results are cached in memory so repeated requests for the same
    sequence/temperature return instantly.
    """
    # Check result cache
    cache_key = f"{sequence}:{num_candidates}:{target_temp}"
    if cache_key in _optimize_cache:
        return _optimize_cache[cache_key]

    # Step 1: Scan for beneficial single mutations (single batch call)
    beneficial = _scan_beneficial_mutations(sequence, top_k=optimization_steps)

    if not beneficial:
        return {
            "original_sequence": sequence,
            "candidates": [],
            "latent_space_summary": {"beneficial_mutations_found": 0},
        }

    # Build score cache from step 1 — reused everywhere, no re-prediction
    score_cache = {m["label"]: m["score"] for m in beneficial}
    confidence_cache = {m["label"]: m["confidence"] for m in beneficial}

    # Step 2: Deduplicate by position
    best_per_position = {}
    for mut in beneficial:
        pos = mut["position"]
        if pos not in best_per_position or mut["score"] > best_per_position[pos]["score"]:
            best_per_position[pos] = mut
    unique_muts = sorted(best_per_position.values(), key=lambda x: x["score"], reverse=True)

    # Step 3: Generate candidates
    candidates = []
    seen_seqs = set()
    catalytic_vals = set(CATALYTIC_RESIDUES.values())

    # Single mutants
    for mut in unique_muts:
        pos = mut["position"]
        if pos in catalytic_vals:
            continue
        seq_list = list(sequence)
        seq_list[pos] = mut["mutant"]
        candidate_seq = "".join(seq_list)
        if candidate_seq != sequence and candidate_seq not in seen_seqs:
            mutations = [f"{mut['wild_type']}{pos + 1}{mut['mutant']}"]
            candidates.append({"sequence": candidate_seq, "mutations": mutations, "num_mutations": 1})
            seen_seqs.add(candidate_seq)

    # Multi-mutant combos
    max_combo = min(6, len(unique_muts))
    for n_muts in range(2, max_combo + 1):
        for start in range(0, len(unique_muts) - n_muts + 1):
            subset = unique_muts[start: start + n_muts]
            candidate_seq = _combine_beneficial_mutations(sequence, subset, num_mutations=n_muts)
            if candidate_seq == sequence or candidate_seq in seen_seqs:
                continue
            mutations = []
            for i, (wt, mt) in enumerate(zip(sequence, candidate_seq)):
                if wt != mt:
                    mutations.append(f"{wt}{i + 1}{mt}")
            if mutations:
                candidates.append({"sequence": candidate_seq, "mutations": mutations, "num_mutations": len(mutations)})
                seen_seqs.add(candidate_seq)
            if len(candidates) >= optimization_steps:
                break
        if len(candidates) >= optimization_steps:
            break

    # Step 4: Pre-rank using cached scores (zero prediction cost)
    pre_rank_hotspot = 0.5 if target_temp >= 60 else 0.2
    for cand in candidates:
        cand["mutation_score_sum"] = sum(
            score_cache.get(mut, 0) for mut in cand["mutations"]
        )
        hotspot_bonus = sum(
            pre_rank_hotspot for m in cand["mutations"]
            if int(m[1:-1]) - 1 in THERMOSTABILITY_HOTSPOTS
        )
        cand["mutation_score_sum"] += hotspot_bonus

    candidates.sort(key=lambda x: x["mutation_score_sum"], reverse=True)
    top_to_score = candidates[:num_candidates + 5]

    # Score candidates using cached scores (no re-prediction)
    stability_w, activity_w = _get_temp_weights(target_temp)
    wt_stability, wt_activity = 0.5, 0.5
    wt_combined = stability_w * wt_stability + activity_w * wt_activity
    hotspot_per_mut = _get_hotspot_bonus(target_temp)

    scored = []
    for cand in top_to_score:
        stability, activity = _score_candidate(cand["sequence"], sequence, score_cache=score_cache)
        combined = stability_w * stability + activity_w * activity

        hotspot_bonus = sum(
            hotspot_per_mut for m in cand["mutations"]
            if int(m[1:-1]) - 1 in THERMOSTABILITY_HOTSPOTS
        )
        combined += hotspot_bonus

        scored.append({
            "sequence": cand["sequence"],
            "mutations": cand["mutations"],
            "predicted_stability_score": round(stability, 6),
            "predicted_activity_score": round(activity, 6),
            "combined_score": round(combined, 6),
        })

    # Step 5: Rank and return
    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    top_candidates = scored[:num_candidates]

    for i, cand in enumerate(top_candidates):
        cand["rank"] = i + 1

    # Step 6: Add explainability, literature validation, and classifier predictions
    # All use cached scores — no ML re-prediction needed
    from . import explainability as _explain
    from . import literature_validation as _litval

    for cand in top_candidates:
        explanation = _explain.explain_candidate(
            cand["mutations"], esm_scores=score_cache
        )
        cand["explanations"] = explanation["mutation_explanations"]
        cand["overall_strategy"] = explanation["overall_strategy"]

        validation = _litval.validate_mutations(cand["mutations"])
        cand["literature_validation"] = {
            "exact_matches": validation["exact_matches"],
            "position_matches": validation["position_matches"],
            "novel_predictions": validation["novel_predictions"],
            "variant_overlaps": validation["variant_overlaps"],
            "validation_score": validation["validation_score"],
            "summary": validation["summary"],
        }

        # Build classifier prediction from cached scores (no re-prediction)
        per_mutation = []
        beneficial_count = 0
        total_ddg = 0.0
        for mut_label in cand["mutations"]:
            prob = score_cache.get(mut_label, 0.5)
            conf = confidence_cache.get(mut_label, 0.5)
            ddg = float(-np.log(prob / max(1 - prob, 1e-6)))  # inverse sigmoid
            is_ben = bool(ddg < 0)
            if is_ben:
                beneficial_count += 1
            total_ddg += ddg
            per_mutation.append({
                "mutation": mut_label,
                "predicted_beneficial": is_ben,
                "predicted_ddg": round(ddg, 4),
                "confidence": round(conf, 4),
                "probability_beneficial": round(prob, 4),
            })

        avg_conf = np.mean([p["confidence"] for p in per_mutation]) if per_mutation else 0.0
        cand["classifier_prediction"] = {
            "all_beneficial": beneficial_count == len(per_mutation),
            "beneficial_count": beneficial_count,
            "total": len(per_mutation),
            "total_predicted_ddg": round(total_ddg, 4),
            "average_confidence": round(float(avg_conf), 4),
            "per_mutation": per_mutation,
        }

    # Latent space summary
    coords_2d = [[0.0, 0.0]]
    for cand in top_candidates[:5]:
        x = (cand["predicted_stability_score"] - 0.5) * 4
        y = (cand["predicted_activity_score"] - 0.5) * 4
        coords_2d.append([round(x, 4), round(y, 4)])

    from . import trained_classifier as _clf
    training_info = _clf.get_training_metrics()

    result = {
        "original_sequence": sequence,
        "candidates": top_candidates,
        "wild_type_score": round(wt_combined, 6),
        "latent_space_summary": {
            "wild_type_score": round(wt_combined, 6),
            "beneficial_mutations_found": len(beneficial),
            "candidates_explored": len(candidates),
            "top_mutations": [m["label"] for m in beneficial[:10]],
            "latent_coordinates": coords_2d,
            "labels": ["wild_type"] + [f"candidate_{i+1}" for i in range(len(coords_2d) - 1)],
        },
        "classifier_info": {
            "model_type": training_info.get("model_type", "XGBClassifier + ESM-2"),
            "training_samples": training_info.get("training_samples", 0),
            "cv_accuracy": training_info.get("cv_accuracy_mean", 0),
            "feature_importances": training_info.get("feature_importances", {}),
        },
    }

    # Cache result for instant repeat requests
    if len(_optimize_cache) >= _OPTIMIZE_CACHE_MAX:
        _optimize_cache.pop(next(iter(_optimize_cache)))
    _optimize_cache[cache_key] = result

    return result
