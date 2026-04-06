"""
PET Degradation Simulation Framework
=====================================

Models enzyme-catalyzed PET plastic degradation under realistic conditions:
  - Temperature effects (Arrhenius kinetics + thermal denaturation)
  - PET crystallinity (amorphous → highly crystalline accessibility)
  - Contaminant inhibition (dyes, coatings, additives)

Scientific basis:
  - Tournier et al. (2020) Nature: LCC engineering for PET recycling
  - Austin et al. (2018) PNAS: IsPETase characterization
  - Lu et al. (2022) Nature: FAST-PETase machine-learning design
  - Yoshida et al. (2016) Science: Original PETase discovery

Author: PET Lab — AVDS Hackathon 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional
import json
import os

# ═══════════════════════════════════════════════════════════
# 1. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

@dataclass
class EnzymeProfile:
    """
    Defines an enzyme's kinetic and thermal properties.

    Parameters are based on published experimental data where available,
    or reasonable estimates from homologous enzymes.
    """
    name: str

    # Michaelis-Menten kinetics (on amorphous PET film)
    kcat: float           # Turnover number (s⁻¹) — catalytic rate
    Km: float             # Michaelis constant (mM) — substrate affinity

    # Thermal profile
    T_opt: float          # Optimal temperature (°C)
    T_half: float         # Half-life temperature (°C) — where activity drops to 50%
    thermal_width: float  # Width of thermal activity curve (°C)
                          # Narrower = more sensitive to temperature changes

    # Stability
    half_life_hours: float  # Enzyme half-life at T_opt (hours)

    # Contaminant tolerance (0 = no tolerance, 1 = fully resistant)
    contaminant_tolerance: float = 0.5

    # Optional metadata
    pdb_id: str = ""
    source_organism: str = ""
    mutations: list = field(default_factory=list)
    reference: str = ""


@dataclass
class PETSubstrate:
    """
    Models a PET plastic sample with physical/chemical properties.

    Crystallinity is the key factor: amorphous PET (~7% crystallinity)
    is ~10-20x easier to degrade than bottle-grade PET (~30%).
    Above the glass transition temperature (Tg ≈ 70°C), PET chains
    become mobile, making even crystalline regions more accessible.
    """
    crystallinity: float       # Fraction crystalline (0.0 to 1.0)
    Tg: float = 70.0          # Glass transition temperature (°C)
    thickness_um: float = 250  # Film thickness (micrometers)
    surface_area_cm2: float = 1.0  # Available surface area


@dataclass
class PreProcessingConfig:
    """
    PET pre-processing protocol.

    Industrial approach (Tournier et al., 2020):
    1. Heat PET above glass transition (Tg ≈ 70°C) or near melting (~250°C)
    2. Rapidly cool (quench) to preserve amorphous state
    3. Then add enzyme at its optimal temperature

    This converts crystalline regions to amorphous, dramatically
    improving enzyme accessibility. Tournier achieved ~90% degradation
    of pretreated PET in 10 hours with LCC-ICCG.
    """
    enabled: bool = False
    preheat_temperature: float = 250.0   # °C — above Tg, below melting (260°C)
    preheat_duration_min: float = 10.0   # Minutes at preheat temperature
    cooling_rate_C_per_min: float = 50.0 # Fast quench rate
    # Slow cooling (< 5°C/min) allows recrystallization
    # Fast cooling (> 30°C/min) preserves amorphous state


def compute_preprocessed_crystallinity(
    original_crystallinity: float,
    config: PreProcessingConfig
) -> float:
    """
    Compute effective crystallinity after thermal pre-processing.

    Model based on polymer physics:
    - Heating above Tg mobilizes amorphous chains
    - Heating near melting temperature disrupts crystalline regions
    - Fast cooling (quenching) prevents recrystallization
    - Slow cooling allows chains to re-order

    Tournier et al. (2020): 10 min at 250°C + fast quench
    reduced crystallinity from ~30% to ~5%.
    """
    if not config.enabled:
        return original_crystallinity

    Tg = 70.0   # PET glass transition
    Tm = 260.0   # PET melting point

    # How much crystallinity is disrupted by heating
    # More disruption at higher temperatures (sigmoid around Tm)
    T = config.preheat_temperature
    if T < Tg:
        disruption_fraction = 0.0  # Below Tg: nothing happens
    elif T < 150:
        disruption_fraction = 0.2 * (T - Tg) / (150 - Tg)  # Mild
    elif T < 220:
        disruption_fraction = 0.2 + 0.4 * (T - 150) / (220 - 150)  # Moderate
    else:
        disruption_fraction = 0.6 + 0.35 * (1.0 / (1.0 + np.exp(-(T - Tm) / 8.0)))

    # Duration effect: longer heating = more disruption
    duration_factor = 1.0 - np.exp(-config.preheat_duration_min / 8.0)

    # Cooling rate effect: fast cooling preserves amorphous state
    # Below ~5°C/min, significant recrystallization occurs
    if config.cooling_rate_C_per_min > 30:
        recryst_fraction = 0.05  # Very fast: minimal recrystallization
    elif config.cooling_rate_C_per_min > 10:
        recryst_fraction = 0.15  # Fast: some recrystallization
    elif config.cooling_rate_C_per_min > 5:
        recryst_fraction = 0.40  # Moderate: significant recrystallization
    else:
        recryst_fraction = 0.70  # Slow: most crystallinity recovers

    # New crystallinity
    disrupted = original_crystallinity * disruption_fraction * duration_factor
    remaining = original_crystallinity - disrupted
    recrystallized = disrupted * recryst_fraction

    new_crystallinity = remaining + recrystallized

    # Physical minimum: even best processing leaves ~2% crystallinity
    return max(0.02, min(original_crystallinity, new_crystallinity))


@dataclass
class ReactionConditions:
    """
    Environmental conditions for the degradation reaction.
    """
    temperature: float         # Reaction temperature (°C)
    pH: float = 8.0            # Buffer pH (most PETases optimal at ~8)
    enzyme_conc_uM: float = 1.0   # Enzyme concentration (µM)
    contaminant_level: float = 0.0  # Contaminant fraction (0 = pure, 1 = heavily contaminated)
    contaminant_type: str = "none"  # Type: "none", "dye", "coating", "mixed"
    reaction_time_hours: float = 24.0  # Total reaction time


@dataclass
class DegradationResult:
    """
    Output of a single simulation run.
    """
    enzyme_name: str
    temperature: float
    crystallinity: float
    contaminant_level: float

    # Core metrics
    degradation_rate: float     # mg PET / hour / µmol enzyme
    total_degraded_mg: float    # Total PET degraded over reaction time
    percent_degraded: float     # % of initial PET mass degraded
    efficiency: float           # Catalytic efficiency (kcat/Km)

    # Time-resolved data
    time_points: np.ndarray = field(default_factory=lambda: np.array([]))
    degradation_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    # Component factors (for analysis)
    thermal_factor: float = 0.0
    crystallinity_factor: float = 0.0
    contaminant_factor: float = 0.0


# ═══════════════════════════════════════════════════════════
# 2. ENZYME DATABASE
#    Published kinetic parameters where available;
#    estimated values marked with (est.)
# ═══════════════════════════════════════════════════════════

ENZYME_DATABASE = {
    "IsPETase_WT": EnzymeProfile(
        name="IsPETase (Wild-Type)",
        kcat=0.8,            # Low activity, measured on PET film
        Km=0.3,              # Moderate affinity
        T_opt=30.0,          # Mesophilic — active at room temp
        T_half=40.0,         # Denatures above 40°C
        thermal_width=12.0,  # Narrow thermal window
        half_life_hours=24,
        contaminant_tolerance=0.3,
        pdb_id="5XJH",
        source_organism="Ideonella sakaiensis",
        reference="Yoshida et al., Science (2016)"
    ),

    "ThermoPETase": EnzymeProfile(
        name="ThermoPETase",
        kcat=1.6,            # ~2x improvement over WT
        Km=0.28,
        T_opt=60.0,          # Engineered for higher temp
        T_half=72.0,         # Stable up to 72°C
        thermal_width=18.0,  # Broader thermal tolerance
        half_life_hours=48,
        contaminant_tolerance=0.5,
        pdb_id="6IJ6",
        source_organism="Engineered",
        mutations=["S121E", "D186H", "R280A"],
        reference="Son et al., ACS Catal. (2019)"
    ),

    "FAST-PETase": EnzymeProfile(
        name="FAST-PETase",
        kcat=4.5,            # Major improvement
        Km=0.22,             # Better substrate binding
        T_opt=50.0,          # Works at moderate temperatures
        T_half=60.0,
        thermal_width=20.0,  # Wide operating range
        half_life_hours=72,
        contaminant_tolerance=0.6,
        pdb_id="N/A",
        source_organism="ML-Engineered",
        mutations=["S121E", "D186H", "R280A", "N233K", "S58E"],
        reference="Lu et al., Nature (2022)"
    ),

    "LCC_ICCG": EnzymeProfile(
        name="LCC-ICCG",
        kcat=6.0,            # Highest known activity on PET
        Km=0.18,             # Strong substrate binding
        T_opt=72.0,          # Near PET glass transition
        T_half=85.0,         # Very thermostable
        thermal_width=15.0,
        half_life_hours=96,
        contaminant_tolerance=0.7,
        pdb_id="4EB0",
        source_organism="Metagenome (engineered)",
        mutations=["F243I", "D238C", "S283C", "Y127G"],
        reference="Tournier et al., Nature (2020)"
    ),

    "PETase_AI_Candidate": EnzymeProfile(
        name="PET Lab AI Candidate",
        kcat=3.2,            # Predicted by our ML pipeline
        Km=0.20,
        T_opt=55.0,
        T_half=68.0,
        thermal_width=22.0,  # Predicted broad tolerance
        half_life_hours=60,
        contaminant_tolerance=0.55,
        pdb_id="5XJH",
        source_organism="AI-Designed",
        mutations=["N122Y", "L239F"],
        reference="PET Lab ML Pipeline (2026)"
    ),
}


# ═══════════════════════════════════════════════════════════
# 3. SCIENTIFIC MODELS
# ═══════════════════════════════════════════════════════════

def thermal_activity_factor(T: float, enzyme: EnzymeProfile) -> float:
    """
    Models enzyme activity as a function of temperature.

    Uses an asymmetric Gaussian: activity rises with temperature
    (Arrhenius-like) up to T_opt, then drops sharply due to
    thermal denaturation.

    The asymmetry reflects biology: enzymes speed up gradually
    with heat but unfold suddenly above their stability limit.

    Returns a factor between 0 and 1.

    Model:
        Below T_opt:  f(T) = exp(-((T - T_opt) / w_left)^2)
        Above T_opt:  f(T) = exp(-((T - T_opt) / w_right)^2)
        where w_right < w_left (sharper drop on hot side)
    """
    # Width of the rising side (broader — Arrhenius-like)
    w_left = enzyme.thermal_width * 1.2
    # Width of the falling side (narrower — denaturation is steep)
    w_right = enzyme.thermal_width * 0.6

    if T <= enzyme.T_opt:
        width = w_left
    else:
        width = w_right

    factor = np.exp(-((T - enzyme.T_opt) / width) ** 2)
    return float(np.clip(factor, 0.0, 1.0))


def crystallinity_accessibility_factor(
    crystallinity: float,
    temperature: float,
    Tg: float = 70.0
) -> float:
    """
    Models how PET crystallinity affects enzyme accessibility.

    Scientific basis:
    - Amorphous PET regions have mobile polymer chains that enzymes
      can attack. Crystalline regions are tightly packed and resistant.
    - Above the glass transition temperature (Tg ≈ 70°C), amorphous
      regions become rubbery and MORE accessible.
    - Crystalline regions remain resistant even above Tg.

    Model:
        accessibility = (1 - crystallinity)^alpha * Tg_boost

    where alpha controls the nonlinear difficulty of crystalline PET
    and Tg_boost gives a bonus when T > Tg (chains become mobile).

    Returns a factor between 0 and 1.
    """
    # Nonlinearity exponent: makes high crystallinity disproportionately hard
    alpha = 1.8

    # Base accessibility from amorphous fraction
    amorphous_fraction = 1.0 - crystallinity
    base_accessibility = amorphous_fraction ** alpha

    # Glass transition boost: above Tg, amorphous chains are more mobile
    # This is a sigmoid transition (not a sharp cutoff)
    if temperature > Tg - 10:
        Tg_boost = 1.0 + 0.5 * (1.0 / (1.0 + np.exp(-(temperature - Tg) / 3.0)))
    else:
        Tg_boost = 1.0

    # Even crystalline PET has defects and surfaces (~5% minimum access)
    factor = max(0.05, base_accessibility * Tg_boost)

    return float(np.clip(factor, 0.0, 1.5))  # Can exceed 1.0 above Tg


def contaminant_inhibition_factor(
    contaminant_level: float,
    contaminant_type: str,
    enzyme_tolerance: float
) -> float:
    """
    Models how contaminants reduce enzyme activity.

    Scientific basis:
    - Dyes (e.g., from colored bottles) can bind to enzyme active sites,
      acting as competitive inhibitors.
    - Coatings (e.g., labels, adhesives) physically block surface access.
    - Mixed waste has both effects plus pH/ionic interference.

    Model:
        inhibition = 1 - severity * contaminant_level * (1 - tolerance)

    Different contaminant types have different severity factors:
      - "none": no effect
      - "dye": moderate — competitive inhibition
      - "coating": strong — surface blocking
      - "mixed": severe — multiple mechanisms

    Returns a factor between 0 and 1.
    """
    severity = {
        "none": 0.0,
        "dye": 0.4,        # Dyes cause ~40% max inhibition at full contamination
        "coating": 0.7,     # Physical barriers are worse
        "mixed": 0.85,      # Real-world waste is harsh
    }.get(contaminant_type, 0.3)

    # Enzyme tolerance reduces the impact
    effective_inhibition = severity * contaminant_level * (1.0 - enzyme_tolerance)

    factor = 1.0 - effective_inhibition
    return float(np.clip(factor, 0.05, 1.0))


def enzyme_decay_factor(time_hours: float, half_life: float, temperature: float, T_opt: float) -> float:
    """
    Models enzyme deactivation over time.

    Enzymes lose activity due to thermal denaturation, surface
    adsorption, and product inhibition. The decay follows
    first-order kinetics, accelerated at temperatures above T_opt.

    Model:
        activity(t) = exp(-k_decay * t)
        k_decay = ln(2) / half_life * temperature_acceleration

    Returns a factor between 0 and 1.
    """
    # Accelerate decay when operating above optimal temperature
    if temperature > T_opt:
        T_excess = temperature - T_opt
        acceleration = 1.0 + 0.1 * T_excess  # 10% faster decay per °C above T_opt
    else:
        acceleration = 1.0

    k_decay = (np.log(2) / half_life) * acceleration
    factor = np.exp(-k_decay * time_hours)
    return float(np.clip(factor, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════
# 4. SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════

class PETDegradationSimulator:
    """
    Core simulation engine.

    Integrates all environmental factors to predict PET
    degradation over time for a given enzyme under specified
    conditions.
    """

    # PET monomer molecular weight for unit conversion
    PET_MONOMER_MW = 192.17  # g/mol (terephthalic acid + ethylene glycol)

    def __init__(self, time_resolution_minutes: float = 30.0):
        """
        Args:
            time_resolution_minutes: Simulation time step (default 30 min).
                Smaller = more accurate but slower.
        """
        self.dt_hours = time_resolution_minutes / 60.0

    def simulate(
        self,
        enzyme: EnzymeProfile,
        substrate: PETSubstrate,
        conditions: ReactionConditions,
        preprocessing: PreProcessingConfig = None,
    ) -> DegradationResult:
        """
        Run a single degradation simulation.

        The simulation integrates the instantaneous degradation rate
        over time, accounting for:
          1. Enzyme thermal activity (temperature-dependent)
          2. Substrate accessibility (crystallinity-dependent)
          3. Contaminant inhibition
          4. Enzyme decay (time-dependent deactivation)
          5. Substrate depletion (less PET available over time)

        Returns:
            DegradationResult with time-resolved degradation curve.
        """
        T = conditions.temperature

        # ── Pre-processing: reduce crystallinity before enzymatic step ──
        effective_crystallinity = substrate.crystallinity
        if preprocessing and preprocessing.enabled:
            effective_crystallinity = compute_preprocessed_crystallinity(
                substrate.crystallinity, preprocessing
            )

        # ── Static factors (constant over time) ──
        f_thermal = thermal_activity_factor(T, enzyme)
        f_crystal = crystallinity_accessibility_factor(
            effective_crystallinity, T, substrate.Tg
        )
        f_contam = contaminant_inhibition_factor(
            conditions.contaminant_level,
            conditions.contaminant_type,
            enzyme.contaminant_tolerance
        )

        # ── Catalytic efficiency ──
        # kcat/Km is the standard measure of enzyme efficiency (M⁻¹s⁻¹)
        catalytic_efficiency = enzyme.kcat / enzyme.Km

        # ── Initial PET mass estimate ──
        # Assume PET density ≈ 1.38 g/cm³, convert film to mass
        pet_volume_cm3 = (
            substrate.surface_area_cm2
            * substrate.thickness_um
            * 1e-4  # µm → cm
        )
        pet_density = 1.38  # g/cm³
        initial_mass_mg = pet_volume_cm3 * pet_density * 1000  # mg
        # Scale up to realistic industrial sample (1g PET chip)
        initial_mass_mg = max(initial_mass_mg, 1000.0)

        # ── Time integration ──
        n_steps = int(conditions.reaction_time_hours / self.dt_hours)
        time_points = np.linspace(0, conditions.reaction_time_hours, n_steps + 1)
        degraded_cumulative = np.zeros(n_steps + 1)

        remaining_mass_mg = initial_mass_mg

        for i in range(1, n_steps + 1):
            t = time_points[i]

            # Enzyme decay over time
            f_decay = enzyme_decay_factor(
                t, enzyme.half_life_hours, T, enzyme.T_opt
            )

            # Substrate depletion: less PET surface available
            f_substrate = remaining_mass_mg / initial_mass_mg

            # Instantaneous rate (mg/hour)
            # rate = kcat/Km * [E] * f_thermal * f_crystal * f_contam * f_decay * f_substrate
            # Convert enzyme concentration µM → effective rate via a scaling constant
            # Rate equation:
            # The base rate is kcat * [E], modulated by all factors.
            # We scale by PET_MONOMER_MW to convert turnovers to mass.
            # The 1e-6 converts µM enzyme to molar, and we adjust
            # for the heterogeneous (solid/liquid) reaction.
            rate_mg_per_hour = (
                enzyme.kcat                  # turnovers per second
                * conditions.enzyme_conc_uM  # µM enzyme
                * 1e-6                       # µM → M
                * 1e3                        # L → mL volume factor
                * f_thermal
                * f_crystal
                * f_contam
                * f_decay
                * f_substrate
                * self.PET_MONOMER_MW        # g/mol per turnover
                * 3600                       # seconds → hours
                * 0.1                        # heterogeneous reaction efficiency
            )

            # Integrate over time step
            degraded_this_step = rate_mg_per_hour * self.dt_hours
            degraded_this_step = min(degraded_this_step, remaining_mass_mg)

            remaining_mass_mg -= degraded_this_step
            degraded_cumulative[i] = degraded_cumulative[i - 1] + degraded_this_step

        total_degraded = degraded_cumulative[-1]
        percent_degraded = (total_degraded / initial_mass_mg) * 100

        # Initial rate (first hour) — better metric than average rate
        first_hour_idx = min(int(1.0 / self.dt_hours), n_steps)
        initial_rate = degraded_cumulative[first_hour_idx] / max(time_points[first_hour_idx], 0.01)

        return DegradationResult(
            enzyme_name=enzyme.name,
            temperature=T,
            crystallinity=substrate.crystallinity,
            contaminant_level=conditions.contaminant_level,
            degradation_rate=initial_rate,
            total_degraded_mg=total_degraded,
            percent_degraded=percent_degraded,
            efficiency=catalytic_efficiency,
            time_points=time_points,
            degradation_curve=degraded_cumulative,
            thermal_factor=f_thermal,
            crystallinity_factor=f_crystal,
            contaminant_factor=f_contam,
        )

    def parameter_sweep(
        self,
        enzyme: EnzymeProfile,
        temperatures: np.ndarray,
        crystallinities: np.ndarray,
        contaminant_levels: np.ndarray = np.array([0.0]),
        contaminant_type: str = "none",
        reaction_time_hours: float = 24.0,
    ) -> list[DegradationResult]:
        """
        Sweep across a grid of conditions for one enzyme.

        Returns a list of DegradationResult objects for every
        (temperature, crystallinity, contaminant) combination.
        """
        results = []
        for T in temperatures:
            for cr in crystallinities:
                for cl in contaminant_levels:
                    substrate = PETSubstrate(crystallinity=cr)
                    conditions = ReactionConditions(
                        temperature=T,
                        contaminant_level=cl,
                        contaminant_type=contaminant_type,
                        reaction_time_hours=reaction_time_hours,
                    )
                    result = self.simulate(enzyme, substrate, conditions)
                    results.append(result)
        return results

    def compare_enzymes(
        self,
        enzyme_names: list[str],
        temperatures: np.ndarray,
        crystallinities: np.ndarray,
        contaminant_levels: np.ndarray = np.array([0.0]),
        contaminant_type: str = "none",
    ) -> dict[str, list[DegradationResult]]:
        """
        Compare multiple enzymes across the same parameter space.
        """
        all_results = {}
        for name in enzyme_names:
            enzyme = ENZYME_DATABASE[name]
            all_results[name] = self.parameter_sweep(
                enzyme, temperatures, crystallinities,
                contaminant_levels, contaminant_type
            )
        return all_results


# ═══════════════════════════════════════════════════════════
# 5. ANALYSIS METRICS
# ═══════════════════════════════════════════════════════════

def compute_robustness_metrics(results: list[DegradationResult]) -> dict:
    """
    Compute performance metrics beyond simple "max degradation."

    These metrics evaluate how RELIABLY an enzyme performs across
    varying real-world conditions — critical for industrial use.

    Returns:
        dict with the following metrics:

        - peak_performance: Maximum degradation rate observed
        - mean_performance: Average across all conditions
        - robustness_score: Ratio of mean to peak (0-1)
          Higher = enzyme performs consistently across conditions
          An enzyme with 0.9 robustness loses only 10% on average
        - thermal_breadth: Temperature range where activity > 50% of peak
          Wider = more tolerant of temperature fluctuations
        - crystallinity_penalty: How much performance drops on crystalline PET
          Lower = better at degrading real-world waste
        - worst_case: Minimum performance across all conditions
          Critical for industrial reliability guarantees
        - coefficient_of_variation: Standard deviation / mean
          Lower = more predictable performance
    """
    rates = np.array([r.degradation_rate for r in results])
    temps = np.array([r.temperature for r in results])

    peak = np.max(rates)
    mean = np.mean(rates)
    worst = np.min(rates)

    # Robustness: how close is average performance to peak?
    robustness = mean / peak if peak > 0 else 0

    # Thermal breadth: temperature range with >50% of peak activity
    threshold = peak * 0.5
    active_temps = temps[rates >= threshold]
    thermal_breadth = (
        float(np.max(active_temps) - np.min(active_temps))
        if len(active_temps) > 1 else 0.0
    )

    # Crystallinity penalty: compare amorphous vs crystalline performance
    amorphous_results = [r for r in results if r.crystallinity < 0.15]
    crystalline_results = [r for r in results if r.crystallinity > 0.25]
    if amorphous_results and crystalline_results:
        mean_amorphous = np.mean([r.degradation_rate for r in amorphous_results])
        mean_crystalline = np.mean([r.degradation_rate for r in crystalline_results])
        crystallinity_penalty = (
            1.0 - (mean_crystalline / mean_amorphous)
            if mean_amorphous > 0 else 1.0
        )
    else:
        crystallinity_penalty = None

    # Coefficient of variation
    cv = float(np.std(rates) / mean) if mean > 0 else float('inf')

    return {
        "peak_performance": float(peak),
        "mean_performance": float(mean),
        "worst_case": float(worst),
        "robustness_score": float(robustness),
        "thermal_breadth_C": thermal_breadth,
        "crystallinity_penalty": float(crystallinity_penalty) if crystallinity_penalty is not None else None,
        "coefficient_of_variation": cv,
    }


# ═══════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════════════════════

# Use a clean, publication-ready style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Colorblind-friendly palette (Wong, 2011 — Nature Methods)
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']


def plot_thermal_profiles(enzymes: list[str], save_path: Optional[str] = None):
    """
    Plot temperature-activity curves for multiple enzymes.
    Shows how each enzyme's activity changes with temperature.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    temps = np.linspace(10, 95, 200)

    for i, name in enumerate(enzymes):
        enzyme = ENZYME_DATABASE[name]
        activities = [thermal_activity_factor(T, enzyme) for T in temps]
        ax.plot(temps, activities, color=COLORS[i % len(COLORS)],
                linewidth=2.2, label=enzyme.name)

        # Mark T_opt
        ax.axvline(enzyme.T_opt, color=COLORS[i % len(COLORS)],
                   linestyle=':', alpha=0.4, linewidth=1)

    # Mark PET glass transition
    ax.axvspan(65, 75, alpha=0.08, color='red',
               label='PET Tg region (65-75°C)')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Relative Activity')
    ax.set_title('Thermal Activity Profiles')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(10, 95)
    ax.set_ylim(-0.02, 1.08)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_degradation_heatmap(
    results: list[DegradationResult],
    enzyme_name: str,
    temperatures: np.ndarray,
    crystallinities: np.ndarray,
    metric: str = "percent_degraded",
    save_path: Optional[str] = None,
):
    """
    2D heatmap: temperature vs crystallinity colored by degradation.
    This is the core figure for showing enzyme performance space.
    """
    nT = len(temperatures)
    nC = len(crystallinities)
    grid = np.zeros((nC, nT))

    for r in results:
        if r.enzyme_name != enzyme_name.replace("_", " ").replace("WT", "(Wild-Type)"):
            # Match against the enzyme database name
            matched = False
            for key, enzyme in ENZYME_DATABASE.items():
                if enzyme.name == r.enzyme_name:
                    matched = True
                    break
            if not matched:
                continue

        ti = np.argmin(np.abs(temperatures - r.temperature))
        ci = np.argmin(np.abs(crystallinities - r.crystallinity))
        grid[ci, ti] = getattr(r, metric, r.percent_degraded)

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        grid, aspect='auto', origin='lower',
        cmap='YlOrRd',
        extent=[temperatures[0], temperatures[-1],
                crystallinities[0] * 100, crystallinities[-1] * 100],
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    metric_label = metric.replace('_', ' ').title()
    cbar.set_label(f'{metric_label} (%)')

    # Overlay contour lines
    ax.contour(
        grid, levels=5, colors='white', alpha=0.5, linewidths=0.8,
        extent=[temperatures[0], temperatures[-1],
                crystallinities[0] * 100, crystallinities[-1] * 100],
    )

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('PET Crystallinity (%)')

    # Find enzyme display name
    display_name = enzyme_name
    if enzyme_name in ENZYME_DATABASE:
        display_name = ENZYME_DATABASE[enzyme_name].name

    ax.set_title(f'PET Degradation — {display_name}')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_enzyme_comparison_radar(
    metrics_by_enzyme: dict[str, dict],
    save_path: Optional[str] = None,
):
    """
    Radar/spider chart comparing enzymes across multiple metrics.
    Excellent for publications — shows trade-offs at a glance.
    """
    categories = [
        'Peak\nPerformance', 'Robustness', 'Thermal\nBreadth',
        'Crystallinity\nTolerance', 'Consistency'
    ]
    N = len(categories)

    # Normalize each metric to 0-1 for the radar chart
    all_values = {}
    for name, metrics in metrics_by_enzyme.items():
        values = [
            metrics['peak_performance'],
            metrics['robustness_score'],
            metrics['thermal_breadth_C'],
            1.0 - (metrics['crystallinity_penalty'] or 0),  # Invert: higher = better
            1.0 / (1.0 + metrics['coefficient_of_variation']),  # Invert: lower CV = better
        ]
        all_values[name] = values

    # Normalize across all enzymes
    maxes = [max(v[i] for v in all_values.values()) for i in range(N)]
    for name in all_values:
        all_values[name] = [
            v / m if m > 0 else 0 for v, m in zip(all_values[name], maxes)
        ]

    # Plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, (name, values) in enumerate(all_values.items()):
        values_plot = values + values[:1]
        display_name = ENZYME_DATABASE[name].name if name in ENZYME_DATABASE else name
        ax.plot(angles, values_plot, 'o-', linewidth=2,
                color=COLORS[i % len(COLORS)], label=display_name, markersize=5)
        ax.fill(angles, values_plot, alpha=0.1, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8, color='grey')
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=9)
    ax.set_title('Multi-Metric Enzyme Comparison', pad=20)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_degradation_curves(
    results_by_enzyme: dict[str, DegradationResult],
    save_path: Optional[str] = None,
):
    """
    Time-course degradation curves for multiple enzymes
    under the same conditions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (name, result) in enumerate(results_by_enzyme.items()):
        display_name = ENZYME_DATABASE[name].name if name in ENZYME_DATABASE else name
        ax.plot(
            result.time_points, result.degradation_curve,
            color=COLORS[i % len(COLORS)], linewidth=2.2,
            label=f'{display_name} ({result.percent_degraded:.1f}%)'
        )

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cumulative PET Degraded (mg)')
    ax.set_title('Degradation Time Course')
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_contaminant_impact(
    enzyme_names: list[str],
    contaminant_types: list[str] = ["none", "dye", "coating", "mixed"],
    save_path: Optional[str] = None,
):
    """
    Bar chart showing how different contaminants affect each enzyme.
    """
    sim = PETDegradationSimulator()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(enzyme_names))
    width = 0.8 / len(contaminant_types)

    contam_colors = {
        'none': '#4CAF50', 'dye': '#FF9800',
        'coating': '#F44336', 'mixed': '#9C27B0'
    }

    for j, ctype in enumerate(contaminant_types):
        perfs = []
        for name in enzyme_names:
            enzyme = ENZYME_DATABASE[name]
            substrate = PETSubstrate(crystallinity=0.10)
            conditions = ReactionConditions(
                temperature=enzyme.T_opt,
                contaminant_level=0.5,
                contaminant_type=ctype,
            )
            result = sim.simulate(enzyme, substrate, conditions)
            perfs.append(result.percent_degraded)

        ax.bar(x + j * width, perfs, width, label=ctype.title(),
               color=contam_colors[ctype], alpha=0.85)

    ax.set_xticks(x + width * (len(contaminant_types) - 1) / 2)
    ax.set_xticklabels(
        [ENZYME_DATABASE[n].name for n in enzyme_names],
        fontsize=9, rotation=15, ha='right'
    )
    ax.set_ylabel('PET Degraded (%)')
    ax.set_title('Contaminant Impact on Enzyme Performance')
    ax.legend(title='Contaminant Type', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def generate_full_report(
    output_dir: str = "simulation_results",
    enzymes: Optional[list[str]] = None,
):
    """
    Generate a complete analysis report with all figures and metrics.

    Creates:
      1. Thermal profiles plot
      2. Degradation heatmap per enzyme
      3. Radar comparison chart
      4. Time-course curves
      5. Contaminant impact chart
      6. JSON metrics file
    """
    os.makedirs(output_dir, exist_ok=True)

    if enzymes is None:
        enzymes = list(ENZYME_DATABASE.keys())

    sim = PETDegradationSimulator()

    # Parameter grid
    temperatures = np.linspace(20, 90, 30)
    crystallinities = np.linspace(0.05, 0.45, 20)

    print("=" * 60)
    print("PET DEGRADATION SIMULATION REPORT")
    print("=" * 60)

    # ── 1. Thermal profiles ──
    print("\n[1/6] Generating thermal activity profiles...")
    plot_thermal_profiles(
        enzymes,
        save_path=os.path.join(output_dir, "thermal_profiles.png")
    )
    plt.close()

    # ── 2. Heatmaps + metrics per enzyme ──
    all_metrics = {}
    for name in enzymes:
        enzyme = ENZYME_DATABASE[name]
        print(f"\n[2/6] Simulating {enzyme.name}...")

        results = sim.parameter_sweep(
            enzyme, temperatures, crystallinities
        )

        plot_degradation_heatmap(
            results, name, temperatures, crystallinities,
            save_path=os.path.join(output_dir, f"heatmap_{name}.png")
        )
        plt.close()

        metrics = compute_robustness_metrics(results)
        all_metrics[name] = metrics

        print(f"      Peak: {metrics['peak_performance']:.4f} mg/hr/µmol")
        print(f"      Robustness: {metrics['robustness_score']:.3f}")
        print(f"      Thermal breadth: {metrics['thermal_breadth_C']:.1f}°C")

    # ── 3. Radar chart ──
    print("\n[3/6] Generating radar comparison...")
    plot_enzyme_comparison_radar(
        all_metrics,
        save_path=os.path.join(output_dir, "radar_comparison.png")
    )
    plt.close()

    # ── 4. Time-course at each enzyme's optimal temperature ──
    print("\n[4/6] Generating time-course curves...")
    time_results = {}
    for name in enzymes:
        enzyme = ENZYME_DATABASE[name]
        substrate = PETSubstrate(crystallinity=0.10)
        conditions = ReactionConditions(
            temperature=enzyme.T_opt,
            reaction_time_hours=48,
        )
        time_results[name] = sim.simulate(enzyme, substrate, conditions)

    plot_degradation_curves(
        time_results,
        save_path=os.path.join(output_dir, "time_courses.png")
    )
    plt.close()

    # ── 5. Contaminant impact ──
    print("\n[5/6] Simulating contaminant effects...")
    plot_contaminant_impact(
        enzymes,
        save_path=os.path.join(output_dir, "contaminant_impact.png")
    )
    plt.close()

    # ── 6. Save metrics ──
    print("\n[6/6] Saving metrics...")
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Enzyme':<24} {'Peak':>8} {'Robust':>8} {'Breadth':>8} {'CV':>8}")
    print("-" * 60)
    for name in enzymes:
        m = all_metrics[name]
        display = ENZYME_DATABASE[name].name
        if len(display) > 22:
            display = display[:22]
        print(
            f"{display:<24} "
            f"{m['peak_performance']:>7.4f} "
            f"{m['robustness_score']:>7.3f} "
            f"{m['thermal_breadth_C']:>6.1f}°C "
            f"{m['coefficient_of_variation']:>7.3f}"
        )

    print(f"\nResults saved to: {os.path.abspath(output_dir)}/")
    print("=" * 60)

    return all_metrics


# ═══════════════════════════════════════════════════════════
# 7. QUICK-START ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    generate_full_report(
        output_dir="simulation_results",
        enzymes=["IsPETase_WT", "ThermoPETase", "FAST-PETase", "LCC_ICCG", "PETase_AI_Candidate"],
    )
