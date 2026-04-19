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

# Equal weights — the ML model captures stability vs. activity through DDG directly.
# Temperature ranking is handled via the trained model's features (ThermoMutDB data),
# not by manually shifting weights here.
STABILITY_WEIGHT = 0.5
ACTIVITY_WEIGHT = 0.5

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Optimal pH for IsPETase and most characterised PETase variants (literature consensus)
PETASE_OPTIMAL_PH = 8.0
PH_WIDTH = 2.5  # Gaussian width (std dev in pH units)

# Thermostability hotspot bonus — grounded in known PETase biology
def _get_hotspot_bonus(target_temp: float) -> float:
    """Bonus for mutations at known thermostability hotspot positions."""
    if target_temp <= 50:
        return 0.01
    elif target_temp <= 65:
        return 0.03
    else:
        return 0.05


def _ph_adjustment(ph: float) -> float:
    """Gaussian penalty for pH deviation from PETase optimum (pH 8.0).

    Derived from the pH vs. activity profile of IsPETase (Yoshida 2016)
    and LCC (Tournier 2020):  full activity at pH 7.5–9.0, ~70% at pH 6,
    ~50% below pH 5.  Models this as a Gaussian centred at pH 8.0.

    Returns a multiplier in (0, 1].
    """
    import math
    return math.exp(-0.5 * ((ph - PETASE_OPTIMAL_PH) / PH_WIDTH) ** 2)


def _compute_esm_robustness(
    mutations: list[str],
    sequence: str,
    confidence_cache: dict,
) -> tuple[float, str]:
    """Compute Chemical Robustness score for a candidate.

    Primary:  ESM-2 log-likelihood ratio (LLR) summed across all mutations,
              normalised to [0, 1] via sigmoid.  Higher LLR means the mutation
              is more evolutionarily tolerated → proxy for general chemical
              robustness (Meier et al. 2021; Notin et al. 2022).

    Fallback: average XGBoost classifier confidence when ESM-2 is unavailable
              (e.g. memory-constrained deployment).

    Returns (score, source_label).
    """
    try:
        from . import esm_engine
        # Skip ESM entirely if the model isn't already loaded in memory —
        # loading the 650M model blocks the request for minutes.
        # Fall back to classifier confidence immediately.
        if esm_engine._model is None:
            raise RuntimeError("ESM-2 not loaded — using confidence fallback")
        import math
        total_llr = 0.0
        count = 0
        for mut_str in mutations:
            if len(mut_str) < 3:
                continue
            wt_aa  = mut_str[0]
            mut_aa = mut_str[-1]
            position = int(mut_str[1:-1]) - 1
            llr = esm_engine.predict_mutation_effect(sequence, position, mut_aa)
            total_llr += llr
            count += 1
        if count == 0:
            raise ValueError("no valid mutations")
        avg_llr = total_llr / count
        # sigmoid(avg_llr * 1.5) maps typical LLR range to [0, 1]
        score = 1.0 / (1.0 + math.exp(-avg_llr * 1.5))
        return round(float(score), 4), "ESM-2 evolutionary fitness (UniRef50)"
    except Exception:
        # Fallback: XGBoost average confidence
        confs = [confidence_cache.get(m, 0.5) for m in mutations]
        score = float(sum(confs) / len(confs)) if confs else 0.5
        return round(score, 4), "classifier confidence (ESM-2 unavailable on this server)"


def _scan_beneficial_mutations(sequence: str, top_k: int = 50,
                               temperature: float = 60.0, ph: float = 8.0) -> list[dict]:
    """Scan single-point mutations using the trained classifier (vectorized).

    Uses raw numpy arrays to score all ~6000 mutations in a single batch,
    then only builds result dicts for the beneficial ones (~50).

    temperature: user-selected assay temperature (°C) — passed as ML feature 49
    ph: user-selected assay pH — passed as ML feature 50
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

    # Single batch scored at user's actual temperature and pH — these are now
    # real ML features (49 and 50) in the v7 model, so ranking changes with conditions
    all_ddg, all_prob = _clf.predict_mutations_batch_raw(
        mutation_tuples, sequence=sequence,
        temperature=temperature, ph=ph,
    )

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
    ph: float = 8.0,
    contamination_scenario: str = "lab",
) -> dict:
    """Run optimization to generate improved PETase candidates.

    Uses the trained XGBoost classifier and amino acid property analysis.
    Results are cached in memory so repeated requests for the same
    sequence/temperature return instantly.
    """
    # Check result cache (include ph so different pH runs aren't conflated)
    cache_key = f"{sequence}:{num_candidates}:{target_temp}:{ph}"
    if cache_key in _optimize_cache:
        return _optimize_cache[cache_key]

    # Step 1: Scan for beneficial single mutations scored at user's conditions
    beneficial = _scan_beneficial_mutations(
        sequence, top_k=optimization_steps,
        temperature=target_temp, ph=ph,
    )

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

    # Score candidates.
    # Equal 0.5/0.5 weights — the XGBoost model (trained on ThermoMutDB temperatures)
    # already encodes the temperature signal through DDG.  Manually shifting weights
    # based on temperature is not supported by any training data and was removed.
    ph_factor = _ph_adjustment(ph)
    hotspot_per_mut = _get_hotspot_bonus(target_temp)

    scored = []
    for cand in top_to_score:
        stability, activity = _score_candidate(cand["sequence"], sequence, score_cache=score_cache)
        combined = STABILITY_WEIGHT * stability + ACTIVITY_WEIGHT * activity

        # Hotspot bonus: positions known to affect thermostability in PETase literature
        hotspot_bonus = sum(
            hotspot_per_mut for m in cand["mutations"]
            if int(m[1:-1]) - 1 in THERMOSTABILITY_HOTSPOTS
        )
        combined += hotspot_bonus

        # Apply pH adjustment derived from published PETase activity profiles
        combined_ph_adjusted = combined * ph_factor

        scored.append({
            "sequence": cand["sequence"],
            "mutations": cand["mutations"],
            "predicted_stability_score": round(stability, 6),
            "predicted_activity_score": round(activity, 6),
            "combined_score": round(combined_ph_adjusted, 6),
        })

    # Step 5: Rank and return
    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    top_candidates = scored[:num_candidates]

    for i, cand in enumerate(top_candidates):
        cand["rank"] = i + 1

    # Step 6: Compute ΔTm predictions for all top candidates (single batch)
    from . import trained_classifier as _clf_dtm
    dtm_mutation_tuples_per_cand = []
    for cand in top_candidates:
        tuples = []
        for mut_str in cand["mutations"]:
            if len(mut_str) >= 3:
                wt_aa  = mut_str[0]
                mut_aa = mut_str[-1]
                try:
                    position = int(mut_str[1:-1])
                    tuples.append((wt_aa, position, mut_aa))
                except ValueError:
                    pass
        dtm_mutation_tuples_per_cand.append(tuples)

    # Flatten, predict, then slice back per candidate
    flat_tuples = [t for ts in dtm_mutation_tuples_per_cand for t in ts]
    if flat_tuples:
        dtm_flat = _clf_dtm.predict_dtm_batch(
            flat_tuples, sequence=sequence,
            temperature=target_temp, ph=ph,
        )
    else:
        dtm_flat = np.array([])

    offset = 0
    for cand, tuples in zip(top_candidates, dtm_mutation_tuples_per_cand):
        if len(tuples) > 0 and len(dtm_flat) > 0:
            slice_dtm = dtm_flat[offset: offset + len(tuples)]
            cand["predicted_dtm"] = round(float(np.mean(slice_dtm)), 4)
            offset += len(tuples)
        else:
            cand["predicted_dtm"] = None  # ΔTm model not yet available

    # Step 7: Add explainability, literature validation, and classifier predictions
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

        # Chemical robustness via ESM-2 LLR (falls back to classifier confidence)
        esm_score, esm_source = _compute_esm_robustness(
            cand["mutations"], sequence, confidence_cache
        )
        cand["esm_robustness"] = esm_score
        cand["esm_robustness_source"] = esm_source
        cand["ph_used"] = round(ph, 2)
        cand["ph_adjustment_factor"] = round(ph_factor, 4)

    # Latent space summary
    wt_combined = STABILITY_WEIGHT * 0.5 + ACTIVITY_WEIGHT * 0.5  # neutral baseline
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
