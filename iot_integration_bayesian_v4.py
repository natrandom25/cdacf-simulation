# -*- coding: utf-8 -*-
"""
CDACF: Cross-Domain Adaptive Consensus Framework for Heterogeneous IoT
========================================================================
Extends ADACON [Bhore et al., 2025] to heterogeneous IoT environments.

Key change from prior version:
  - REMOVED:  LSTM temporal detector, Random Forest classifier, Autoencoder
  - REPLACED: Bayesian probabilistic threat detection (Beta distribution updating)
              directly embedded in the Stability-Aware Consensus Layer,
              matching the ADACON threat-indicator formulas (Equations 4-9).

Architecture (3 layers):
  1. Application Domain Layer  — traffic management / environmental monitoring
  2. Stability-Aware Consensus Layer  — Bayesian detector + hysteresis gate
  3. Device Capability Layer  — tier-aware feasibility constraints

Reference: Bhore, S. S., Natraj, N. A., & Hallur, G. G. (2026).
           Bayesian-driven autonomous defense adaptive consensus optimisation
           for blockchain networks. Scientific Reports, 16, 2158.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import random
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tabulate import tabulate
from scipy import stats

# ── Output directory ────────────────────────────────────────────────────────
try:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUTPUT_DIR = os.getcwd()
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _fig_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


# =============================================================================
# ENUMS
# =============================================================================

class ConsensusType(Enum):
    POW  = "Proof of Work"
    POS  = "Proof of Stake"
    PBFT = "Practical Byzantine Fault Tolerance"
    POA  = "Proof of Authority"
    DPOS = "Delegated Proof of Stake"


class IoTDomain(Enum):
    TRAFFIC     = "Traffic Management"
    ENVIRONMENT = "Environmental Monitoring"


class AttackType(Enum):
    SYBIL    = "sybil"
    DOS      = "dos"
    BYZANTINE= "byzantine"
    ECLIPSE  = "eclipse"
    MAJORITY = "majority"
    ROUTING  = "routing"


# ── Fair-comparison constants ────────────────────────────────────────────────
# Per-mechanism consensus costs applied IDENTICALLY to the static baselines and
# to whichever mechanism the adaptive framework is running at each time step.
# (Previously only the baselines paid these costs, which inflated CDACF's
# latency/throughput results by construction — corrected per Reviewer 2, R2.)
LATENCY_OVERHEAD = {
    ConsensusType.POW: 2.50, ConsensusType.POS: 1.60,
    ConsensusType.PBFT: 1.40, ConsensusType.POA: 1.55, ConsensusType.DPOS: 1.20,
}
THROUGHPUT_FACTOR = {
    ConsensusType.POW: 0.45, ConsensusType.POS: 0.80,
    ConsensusType.PBFT: 0.90, ConsensusType.POA: 0.92, ConsensusType.DPOS: 0.95,
}

# ADACON hysteresis-corrected switching baseline (Table 3, Bhore et al. 2025).
ADACON_HYSTERESIS_BASELINE = 287


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NetworkMetrics:
    latency:      float   # normalised 0-1
    throughput:   float   # normalised 0-1
    cpu_usage:    float   # 0-1
    memory_usage: float   # 0-1
    bandwidth:    float   # 0-1
    error_rate:   float   # 0-1
    packet_loss:  float   # 0-1


@dataclass
class NetworkState:
    active_nodes:      int
    total_nodes:       int
    metrics:           NetworkMetrics
    current_consensus: ConsensusType
    domain:            IoTDomain
    battery_soh:       float = 1.0
    emergency_mode:    bool  = False


@dataclass
class ConsensusPerformanceMetrics:
    consensus_type:    str
    security_score:    float
    latency:           float
    throughput:        float
    energy_efficiency: float
    decentralization:  float
    scalability:       float
    finality_time:     float
    attack_resistance: Dict[str, float]
    consensus_switches:int = 0
    domain:            str = ""


# =============================================================================
# BAYESIAN THREAT DETECTION MODULE  (Section 3.5 — ADACON Eq. 4-9)
# =============================================================================

class BayesianThreatDetector:
    """
    Extends ADACON's Bayesian threat detection [Bhore et al., 2025] to
    heterogeneous multi-domain IoT environments.

    For each of the six attack vectors a weighted metric indicator is computed
    from real-time network telemetry (Equations 4-9, ADACON paper).
    Each indicator value x feeds a Beta distribution updater:

        Beta(alpha + x,  beta + (1 - x))

    with prior Beta(alpha=2, beta=8) reflecting a low baseline threat for a
    healthy network.  The posterior mean P(Attack | Data) = alpha' / (alpha'+beta')
    forms the six-element probability vector that drives the scoring multipliers.

    Domain-specific prior calibration:
      - Traffic  (5G, low baseline noise):  alpha=2, beta=10  (tighter posterior)
      - Env. Mon (LoRaWAN, higher noise):   alpha=2, beta=8   (wider posterior)
    """

    # ADACON Eq. 4-9 indicator weights — domain-knowledge calibrated
    INDICATOR_WEIGHTS = {
        'sybil':    {'error_rate': 0.70, 'cpu_usage_inv': 0.30},
        'dos':      {'latency': 0.40, 'bandwidth': 0.40, 'error_rate': 0.20},
        'byzantine':{'error_rate': 0.80, 'latency': 0.20},
        'eclipse':  {'latency': 0.40, 'error_rate': 0.30, 'throughput_inv': 0.30},
        'majority': {'cpu_usage': 0.60, 'throughput': 0.20, 'error_rate': 0.20},
        'routing':  {'latency': 0.50, 'throughput_inv': 0.30, 'error_rate': 0.20},
    }

    # Domain-specific Beta priors
    DOMAIN_PRIORS = {
        IoTDomain.TRAFFIC:     {'alpha': 2.0, 'beta': 10.0},  # tighter: less noisy baseline
        IoTDomain.ENVIRONMENT: {'alpha': 2.0, 'beta': 8.0},   # wider: LoRaWAN natural variance
    }

    WINDOW = 20   # sliding window length for posterior estimation (steps)

    def __init__(self, domain: IoTDomain):
        self.domain = domain
        prior = self.DOMAIN_PRIORS[domain]
        # Maintain running alpha/beta per attack type (kept for reset_priors compatibility)
        self._alpha = {k: prior['alpha'] for k in self.INDICATOR_WEIGHTS}
        self._beta  = {k: prior['beta']  for k in self.INDICATOR_WEIGHTS}
        # Sliding window of raw indicator values (one list per attack type)
        self._window: Dict[str, list] = {k: [] for k in self.INDICATOR_WEIGHTS}

    def _compute_indicator(self, attack: str, m: NetworkMetrics) -> float:
        """Compute the raw threat indicator value x in [0,1] (ADACON Eq. 4-9)."""
        w   = self.INDICATOR_WEIGHTS[attack]
        val = 0.0
        if 'error_rate'     in w: val += w['error_rate']     * m.error_rate
        if 'latency'        in w: val += w['latency']        * m.latency
        if 'bandwidth'      in w: val += w['bandwidth']      * m.bandwidth
        if 'cpu_usage'      in w: val += w['cpu_usage']      * m.cpu_usage
        if 'cpu_usage_inv'  in w: val += w['cpu_usage_inv']  * (1.0 - m.cpu_usage)
        if 'throughput'     in w: val += w['throughput']     * m.throughput
        if 'throughput_inv' in w: val += w['throughput_inv'] * (1.0 - m.throughput)
        return float(np.clip(val, 0.0, 1.0))

    def update(self, metrics: NetworkMetrics) -> Dict[str, float]:
        """
        Update Beta posteriors and return six-element threat probability vector.

        SLIDING WINDOW FIX: The original unbounded Beta accumulation caused
        posteriors to plateau far below activation thresholds after ~50 steps
        (alpha/beta grow without bound so new x has negligible weight).
        Instead we maintain a rolling window of the last WINDOW indicators and
        compute the posterior mean as the windowed average, which stays
        responsive to current network conditions while smoothing noise.
        Uses conjugate Beta-Bernoulli update: Beta(alpha+x, beta+(1-x)) on the
        window mean rather than the infinite running sum.
        """
        posteriors = {}
        for attack in self.INDICATOR_WEIGHTS:
            x = self._compute_indicator(attack, metrics)
            # Store in rolling window
            self._window[attack].append(x)
            if len(self._window[attack]) > self.WINDOW:
                self._window[attack].pop(0)
            # Window mean as the effective indicator
            x_win = float(np.mean(self._window[attack]))
            # Conjugate update on *fresh* priors + windowed evidence
            prior  = self.DOMAIN_PRIORS[self.domain]
            a_post = prior['alpha'] + x_win * self.WINDOW
            b_post = prior['beta']  + (1.0 - x_win) * self.WINDOW
            posteriors[attack] = a_post / (a_post + b_post)
        return posteriors

    def reset_priors(self):
        """
        Partial decay toward domain priors (called after recovery).
        Clears the second half of the sliding window so the posterior
        re-sensitises for the next attack without discarding all history.
        """
        keep = self.WINDOW // 2
        for k in self.INDICATOR_WEIGHTS:
            self._window[k] = self._window[k][-keep:]  # keep only recent half


# =============================================================================
# DOMAIN-AWARE CONSENSUS SELECTION ENGINE  (Section 3.2)
# =============================================================================

class DomainAwareConsensusEngine:
    """
    Score(C) = Σ wᵢ · Aᵢ(C) · φ(threat)

    φ(threat) is now the Bayesian posterior probability vector from the
    BayesianThreatDetector, replacing the former LSTM-RF-Autoencoder ensemble.
    """

    # Base performance attributes (Table 7, ADACON paper)
    MECHANISM_ATTRIBUTES = {
        ConsensusType.POW: {
            'security': 0.90, 'scalability': 0.25,
            'energy_efficiency': 0.08, 'decentralization': 0.95, 'finality_speed': 0.20
        },
        ConsensusType.POS: {
            'security': 0.78, 'scalability': 0.72,
            'energy_efficiency': 0.88, 'decentralization': 0.82, 'finality_speed': 0.68
        },
        ConsensusType.PBFT: {
            'security': 0.96, 'scalability': 0.55,
            'energy_efficiency': 0.70, 'decentralization': 0.45, 'finality_speed': 0.97
        },
        ConsensusType.POA: {
            'security': 0.72, 'scalability': 0.88,
            'energy_efficiency': 0.93, 'decentralization': 0.35, 'finality_speed': 0.92
        },
        ConsensusType.DPOS: {
            'security': 0.80, 'scalability': 0.87,
            'energy_efficiency': 0.86, 'decentralization': 0.58, 'finality_speed': 0.84
        },
    }

    # Domain weight profiles (Section 3.2)
    WEIGHT_PROFILES = {
        IoTDomain.TRAFFIC: {
            'normal':    {'scalability': 0.32, 'finality_speed': 0.28, 'security': 0.18,
                          'decentralization': 0.12, 'energy_efficiency': 0.10},
            'high_threat':{'security': 0.45, 'finality_speed': 0.25, 'scalability': 0.15,
                           'decentralization': 0.10, 'energy_efficiency': 0.05},
            'emergency': {'security': 0.55, 'finality_speed': 0.30, 'scalability': 0.08,
                          'decentralization': 0.04, 'energy_efficiency': 0.03},
        },
        IoTDomain.ENVIRONMENT: {
            # Normal: energy-efficiency first (LoRaWAN battery-constrained sensors)
            'normal':          {'energy_efficiency': 0.38, 'security': 0.25,
                                'scalability': 0.15, 'decentralization': 0.15,
                                'finality_speed': 0.07},
            # High-threat: security rises significantly — enough to flip from PoS to PBFT/DPoS
            'high_threat':     {'security': 0.48, 'energy_efficiency': 0.22,
                                'scalability': 0.13, 'decentralization': 0.10,
                                'finality_speed': 0.07},
            # Battery critical: energy dominates everything
            'battery_critical':{'energy_efficiency': 0.65, 'security': 0.18,
                                'scalability': 0.08, 'decentralization': 0.05,
                                'finality_speed': 0.04},
        },
    }

    # Activation thresholds calibrated to actual posterior distributions produced
    # by the Traffic / Environmental simulators (measured empirically per attack):
    #   sybil    posterior_mean~0.245  baseline~0.175  → threshold 0.22
    #   dos      posterior_mean~0.408  baseline~0.135  → threshold 0.32
    #   byzantine posterior_mean~0.190 baseline~0.067  → threshold 0.17
    #   eclipse  posterior_mean~0.141  baseline~0.095  → threshold 0.13
    #   majority posterior_mean~0.501  baseline~0.190  → threshold 0.40
    #   routing  posterior_mean~0.189  baseline~0.065  → threshold 0.17
    # Each threshold sits midway between healthy-baseline and attack posterior.
    ACTIVATION_THRESHOLDS = {
        'sybil':    0.22,
        'dos':      0.32,
        'byzantine':0.17,
        'eclipse':  0.13,
        'majority': 0.40,
        'routing':  0.17,
    }

    def _get_weight_profile(self, state: NetworkState,
                             threat_probs: Dict[str, float]) -> Dict[str, float]:
        max_threat = max(threat_probs.values()) if threat_probs else 0.0
        profiles   = self.WEIGHT_PROFILES[state.domain]

        if state.emergency_mode:
            profile = profiles.get('emergency', profiles['normal'])
        elif state.domain == IoTDomain.ENVIRONMENT and state.battery_soh < 0.20:
            profile = profiles['battery_critical']
        elif max_threat > 0.35:
            profile = profiles['high_threat']
        else:
            profile = profiles['normal']

        total = sum(profile.values())
        return {k: v / total for k, v in profile.items()}

    def calculate_mechanism_score(self, mechanism: ConsensusType,
                                   state: NetworkState,
                                   threat_probs: Dict[str, float]) -> float:
        """
        Compute Score(C) = Σ wᵢ · Aᵢ(C) · φ(threat).

        φ(threat) is the Bayesian posterior probability that drives
        attack-specific multipliers — replacing the former AI ensemble.
        """
        weights = self._get_weight_profile(state, threat_probs)
        attrs   = self.MECHANISM_ATTRIBUTES[mechanism]
        score   = sum(weights.get(k, 0) * attrs.get(k, 0) for k in weights)

        # ── Attack-specific Bayesian posterior multipliers (φ(threat)) ─────────
        # Posteriors exceed activation threshold → shift weight toward best mechanism
        byzantine_p = threat_probs.get('byzantine', 0)
        dos_p       = threat_probs.get('dos',       0)
        sybil_p     = threat_probs.get('sybil',     0)
        eclipse_p   = threat_probs.get('eclipse',   0)
        majority_p  = threat_probs.get('majority',  0)
        routing_p   = threat_probs.get('routing',   0)
        max_threat  = max(threat_probs.values()) if threat_probs else 0.0

        # Byzantine → PBFT (only BFT-safe protocol)
        if byzantine_p > self.ACTIVATION_THRESHOLDS['byzantine']:
            if mechanism == ConsensusType.PBFT:
                score *= (1.0 + byzantine_p * 0.80)
            elif mechanism in [ConsensusType.POW, ConsensusType.DPOS]:
                score *= (1.0 - byzantine_p * 0.50)

        # DoS → PBFT (distributed leader) + DPoS (elected delegates)
        if dos_p > self.ACTIVATION_THRESHOLDS['dos']:
            if mechanism == ConsensusType.PBFT:
                score *= (1.0 + dos_p * 0.60)
            elif mechanism == ConsensusType.DPOS:
                score *= (1.0 + dos_p * 0.30)
            elif mechanism == ConsensusType.POA:
                score *= (1.0 - dos_p * 0.40)

        # Sybil → PoS (stake makes fake identities expensive)
        if sybil_p > self.ACTIVATION_THRESHOLDS['sybil']:
            if mechanism == ConsensusType.POS:
                score *= (1.0 + sybil_p * 1.10)
            elif mechanism == ConsensusType.POW:
                score *= (1.0 + sybil_p * 0.60)
            elif mechanism == ConsensusType.PBFT:
                score *= (1.0 - sybil_p * 0.40)
            elif mechanism == ConsensusType.DPOS:
                score *= (1.0 - sybil_p * 0.30)

        # Eclipse → PoS (diverse validators resist partitioning)
        if eclipse_p > self.ACTIVATION_THRESHOLDS['eclipse']:
            if mechanism == ConsensusType.POS:
                score *= (1.0 + eclipse_p * 1.00)
            elif mechanism == ConsensusType.PBFT:
                score *= (1.0 + eclipse_p * 0.40)
            elif mechanism in [ConsensusType.POA, ConsensusType.DPOS]:
                score *= (1.0 - eclipse_p * 0.50)

        # Majority → PoA (authority-based immune to hash-power concentration)
        if majority_p > self.ACTIVATION_THRESHOLDS['majority']:
            if mechanism == ConsensusType.POA:
                score *= (1.0 + majority_p * 1.20)
            elif mechanism == ConsensusType.PBFT:
                score *= (1.0 + majority_p * 0.60)
            elif mechanism == ConsensusType.POW:
                score *= (1.0 - majority_p * 0.80)
            elif mechanism == ConsensusType.DPOS:
                score *= (1.0 - majority_p * 0.40)

        # Routing → DPoS (elected delegates bypass manipulated paths)
        if routing_p > self.ACTIVATION_THRESHOLDS['routing']:
            if mechanism == ConsensusType.DPOS:
                score *= (1.0 + routing_p * 0.90)
            elif mechanism == ConsensusType.POS:
                score *= (1.0 + routing_p * 0.50)
            elif mechanism == ConsensusType.POW:
                score *= (1.0 - routing_p * 0.50)
            elif mechanism == ConsensusType.PBFT:
                score *= (1.0 - routing_p * 0.20)

        # Normal operation: domain-preferred mechanisms
        if max_threat < 0.15:
            if state.domain == IoTDomain.TRAFFIC:
                if mechanism == ConsensusType.POA:  score *= 1.25
                elif mechanism == ConsensusType.DPOS: score *= 1.10
                elif mechanism == ConsensusType.PBFT: score *= 0.88
            else:  # ENVIRONMENT
                if mechanism == ConsensusType.POS:  score *= 1.30
                elif mechanism == ConsensusType.POA: score *= 1.15
                elif mechanism == ConsensusType.PBFT: score *= 0.85

        # PoW always penalised in IoT (energy cost)
        if mechanism == ConsensusType.POW:
            score *= 0.60

        # Network condition modifiers
        if state.metrics.cpu_usage > 0.80:
            if mechanism in [ConsensusType.POA, ConsensusType.DPOS]:
                score *= 1.08
        if state.metrics.latency > 0.40:
            if mechanism == ConsensusType.PBFT:
                score *= 1.10
        if state.active_nodes < state.total_nodes * 0.70:
            if mechanism == ConsensusType.PBFT:
                score *= 1.12
        # Cascading multipliers can push score > 1.0, which is meaningless for
        # a normalised suitability measure.  Hard-clamp after all adjustments.
        return float(np.clip(score, 0.0, 1.0))

    def select_consensus(self, state: NetworkState,
                          threat_probs: Dict[str, float]) -> Tuple[ConsensusType, Dict]:
        scores = {
            m: self.calculate_mechanism_score(m, state, threat_probs)
            for m in ConsensusType
        }
        best = max(scores, key=scores.get)
        return best, scores


# =============================================================================
# STABILITY-AWARE CONSENSUS LAYER  (Section 3.1)
# =============================================================================

class StabilityAwareConsensusLayer:
    """
    Domain-calibrated hysteresis gate preventing unnecessary switching.

    Traffic:     min 10 steps dwell; 8% improvement threshold
    Environment: min 15 steps dwell; 8% improvement threshold
    No switching during emergency mode or battery-critical condition.
    DoS rapid-switch override bypasses dwell for high-confidence detection.

    Bayesian detector is embedded here — no separate AI module needed.
    """

    # ── CALIBRATED HYSTERESIS PARAMETERS ─────────────────────────────────────
    # Reduced from 20/35 → 10/15 steps to match ADACON's validated 10-step
    # optimum (Table 4, Bhore et al. 2025) while preserving IoT-domain stability.
    # Under active high-confidence threat, dwell is further halved (see update()).
    MIN_DWELL = {
        IoTDomain.TRAFFIC:     10,
        IoTDomain.ENVIRONMENT: 15,
    }
    # Threshold kept at 8% (slightly above ADACON's 15% raw absolute, but applied
    # against normalised domain scores which are narrower in range).
    IMPROVEMENT_THRESHOLD = 0.08

    def __init__(self, domain: IoTDomain):
        self.domain             = domain
        self.engine             = DomainAwareConsensusEngine()
        self.bayesian_detector  = BayesianThreatDetector(domain)  # ← Bayesian replaces AI ensemble
        self.current_mechanism  = ConsensusType.PBFT
        self.current_score      = 0.0
        self.steps_since_switch = 0
        self.mechanism_history: List[ConsensusType] = []
        self.score_history:     List[float]         = []
        self.switch_log:        List[Dict]          = []

    def update(self, state: NetworkState) -> Tuple[ConsensusType, Dict[str, float]]:
        """
        Process one time step.  Runs Bayesian detection internally,
        then applies the stability gate before returning the active mechanism.
        """
        # Step 1: Bayesian posterior update from raw network telemetry
        threat_probs = self.bayesian_detector.update(state.metrics)

        # Step 2: Score all candidate mechanisms
        best, scores = self.engine.select_consensus(state, threat_probs)
        best_score   = scores[best]
        self.steps_since_switch += 1

        # Step 3: Rapid-switch override when ANY threat is very high-confidence
        # (mirrors ADACON's DoS-specific override but generalised to all attacks)
        max_threat_now = max(threat_probs.values()) if threat_probs else 0.0
        dos_override = (
            self.domain == IoTDomain.TRAFFIC
            and threat_probs.get('dos', 0) > 0.55
            and best == ConsensusType.PBFT
            and best != self.current_mechanism
        )
        # High-confidence override: any posterior > 0.55 halves the dwell requirement
        high_conf_override = (
            max_threat_now > 0.55
            and best != self.current_mechanism
            and self.steps_since_switch >= max(3, self.MIN_DWELL[self.domain] // 2)
        )

        # Step 4: Stability gate
        dwell_ok   = self.steps_since_switch >= self.MIN_DWELL[self.domain]
        score_ok   = best_score >= self.current_score * (1 + self.IMPROVEMENT_THRESHOLD)
        no_emg     = not state.emergency_mode
        battery_ok = not (self.domain == IoTDomain.ENVIRONMENT and state.battery_soh < 0.20)

        if best != self.current_mechanism and (
                dos_override or high_conf_override or (dwell_ok and score_ok and no_emg and battery_ok)):
            reason = ('DoS override'        if dos_override      else
                      'High-conf override'  if high_conf_override else
                      'Score improvement')
            self.switch_log.append({
                'step':   len(self.mechanism_history),
                'from':   self.current_mechanism.value,
                'to':     best.value,
                'reason': reason,
            })
            self.current_mechanism  = best
            self.current_score      = best_score
            self.steps_since_switch = 0
        else:
            self.current_score = scores.get(self.current_mechanism, self.current_score)

        self.mechanism_history.append(self.current_mechanism)
        self.score_history.append(self.current_score)
        return self.current_mechanism, threat_probs


# =============================================================================
# NETWORK SIMULATORS
# =============================================================================

class TrafficNetworkSimulator:
    """
    5,000-node urban sensor network (5G backbone, 15ms latency, 0.5% packet loss).
    500 Tier-1 edge nodes / 1,500 Tier-2 roadside units / 2,000 Tier-3 detectors.
    """
    TOTAL_NODES      = 5000
    BASE_LATENCY     = 0.015
    BASE_PACKET_LOSS = 0.005

    def __init__(self):
        self.step           = 0
        self.active_nodes   = self.TOTAL_NODES
        self.malicious_nodes= 0
        self._attack:       Optional[str] = None
        self.emergency_mode = False

    def generate_network_state(self) -> NetworkState:
        self.step += 1
        t    = (self.step % 1440) / 60
        peak = (1.0 + 2.5 * np.exp(-0.5 * ((t - 8) / 2) ** 2)
                    + 2.0 * np.exp(-0.5 * ((t - 17) / 1.5) ** 2))
        tx_norm     = min(1.0, random.gauss(0.30, 0.05) * peak)
        latency     = self.BASE_LATENCY + random.gauss(0, 0.005)
        packet_loss = self.BASE_PACKET_LOSS + random.gauss(0, 0.001)
        cpu_usage   = 0.30 + tx_norm * 0.30 + random.gauss(0, 0.05)
        mem_usage   = random.uniform(0.25, 0.55)
        bandwidth   = tx_norm
        error_rate  = max(0, packet_loss + self.malicious_nodes / self.TOTAL_NODES * 0.10)

        if self._attack == 'dos':
            latency     += random.gauss(0.25, 0.05)
            packet_loss += random.gauss(0.08, 0.015)
            cpu_usage    = min(0.98, cpu_usage + 0.35)
            bandwidth    = min(1.0, bandwidth * 10)
        elif self._attack == 'byzantine':
            # Strengthened: Byzantine nodes produce many invalid messages
            error_rate  += random.gauss(0.35, 0.05)
            packet_loss += random.gauss(0.08, 0.015)
            latency     += random.gauss(0.10, 0.02)    # extra re-broadcast delay
            cpu_usage    = min(0.95, cpu_usage + 0.25)
        elif self._attack == 'eclipse':
            # Strengthened: isolated nodes see stale chain, high latency & errors
            latency     += random.gauss(0.35, 0.06)
            packet_loss += random.gauss(0.10, 0.02)
            error_rate  += random.gauss(0.12, 0.025)  # stale-state validation failures
        elif self._attack == 'sybil':
            # Strengthened: fake nodes flood with invalid transactions
            error_rate  += random.gauss(0.28, 0.05)
            cpu_usage    = min(0.95, cpu_usage + 0.10) # Sybil IDs stay lightweight
        elif self._attack == 'routing':
            # Strengthened: hijacked BGP paths → severe latency & throughput loss
            latency     += random.gauss(0.40, 0.07)
            packet_loss += random.gauss(0.08, 0.015)
            error_rate  += random.gauss(0.08, 0.015)  # mis-routed packets
        elif self._attack == 'majority':
            cpu_usage    = min(0.99, cpu_usage + 0.40)
            error_rate  += random.gauss(0.10, 0.02)

        metrics = NetworkMetrics(
            latency      = min(1.0, max(0.005, latency)),
            throughput   = max(0.10, 1.0 - tx_norm * 0.3 - error_rate),
            cpu_usage    = min(1.0, max(0.05, cpu_usage)),
            memory_usage = min(1.0, max(0.05, mem_usage)),
            bandwidth    = min(1.0, max(0.01, bandwidth)),
            error_rate   = min(1.0, max(0.0,  error_rate)),
            packet_loss  = min(1.0, max(0.0,  packet_loss)),
        )
        return NetworkState(
            active_nodes=self.active_nodes, total_nodes=self.TOTAL_NODES,
            metrics=metrics, current_consensus=ConsensusType.PBFT,
            domain=IoTDomain.TRAFFIC, emergency_mode=self.emergency_mode,
        )

    def simulate_attack(self, attack_type: str):
        self._attack = attack_type
        if attack_type == 'sybil':
            self.active_nodes    = int(self.TOTAL_NODES * 1.30)
            self.malicious_nodes = int(self.TOTAL_NODES * 0.30)
        elif attack_type == 'dos':
            self.active_nodes    = int(self.TOTAL_NODES * 0.60)
        elif attack_type == 'byzantine':
            self.malicious_nodes = int(self.TOTAL_NODES * 0.20)
        elif attack_type == 'eclipse':
            self.active_nodes    = int(self.TOTAL_NODES * 0.70)
        elif attack_type == 'majority':
            self.malicious_nodes = int(self.TOTAL_NODES * 0.51)
        elif attack_type == 'routing':
            self.active_nodes    = int(self.TOTAL_NODES * 0.90)

    def recover_from_attack(self, attack_type: str):
        self._attack         = None
        self.active_nodes    = self.TOTAL_NODES
        self.malicious_nodes = 0


class EnvironmentalNetworkSimulator:
    """
    2,000-node remote watershed sensor network (LoRaWAN, 2.5s latency, 5% packet loss).
    200 Tier-1 cellular gateways / 400 Tier-2 weather stations / 1,400 Tier-3 probes.
    """
    TOTAL_NODES      = 2000
    BASE_LATENCY     = 0.50
    BASE_PACKET_LOSS = 0.05

    def __init__(self):
        self.step           = 0
        self.active_nodes   = self.TOTAL_NODES
        self.malicious_nodes= 0
        self._attack:       Optional[str] = None
        self.battery_soh    = 1.0

    def generate_network_state(self) -> NetworkState:
        self.step += 1
        collection_active = (self.step % 60) < 5
        tx_norm = (random.uniform(0.10, 0.25) if collection_active
                   else random.uniform(0.01, 0.05))
        latency     = self.BASE_LATENCY + random.gauss(0, 0.05)
        packet_loss = self.BASE_PACKET_LOSS + random.gauss(0, 0.01)
        cpu_usage   = 0.05 + tx_norm * 0.20 + random.gauss(0, 0.02)
        mem_usage   = random.uniform(0.05, 0.20)
        bandwidth   = tx_norm
        error_rate  = max(0, packet_loss + self.malicious_nodes / self.TOTAL_NODES * 0.10)

        if self._attack == 'dos':
            latency     += random.gauss(0.40, 0.08)
            packet_loss += random.gauss(0.12, 0.025)
            bandwidth    = min(1.0, bandwidth * 8)       # bandwidth saturation
        elif self._attack == 'byzantine':
            # Strengthened: invalid consensus messages → high error + latency
            error_rate  += random.gauss(0.35, 0.06)
            packet_loss += random.gauss(0.08, 0.015)
            latency     += random.gauss(0.08, 0.02)
        elif self._attack == 'eclipse':
            latency     += random.gauss(0.50, 0.10)
            packet_loss += random.gauss(0.10, 0.02)
            error_rate  += random.gauss(0.10, 0.02)      # stale-state failures
        elif self._attack == 'sybil':
            # Strengthened: fake identity flood raises error + some cpu cost
            error_rate  += random.gauss(0.28, 0.05)      # was no error_rate delta
            cpu_usage    = min(0.85, cpu_usage + 0.20)
        elif self._attack == 'routing':
            latency     += random.gauss(0.60, 0.12)
            packet_loss += random.gauss(0.06, 0.012)
            error_rate  += random.gauss(0.08, 0.015)     # mis-delivery errors
        elif self._attack == 'majority':
            cpu_usage    = min(0.90, cpu_usage + 0.45)

        drain = 0.0001 if not collection_active else 0.0003
        self.battery_soh = max(0.0, self.battery_soh - drain)

        metrics = NetworkMetrics(
            latency      = min(1.0, max(0.05,  latency)),
            throughput   = max(0.05, 1.0 - tx_norm * 0.2 - error_rate),
            cpu_usage    = min(1.0, max(0.01,  cpu_usage)),
            memory_usage = min(1.0, max(0.01,  mem_usage)),
            bandwidth    = min(1.0, max(0.005, bandwidth)),
            error_rate   = min(1.0, max(0.0,   error_rate)),
            packet_loss  = min(1.0, max(0.0,   packet_loss)),
        )
        return NetworkState(
            active_nodes=self.active_nodes, total_nodes=self.TOTAL_NODES,
            metrics=metrics, current_consensus=ConsensusType.PBFT,
            domain=IoTDomain.ENVIRONMENT, battery_soh=self.battery_soh,
        )

    def simulate_attack(self, attack_type: str):
        self._attack = attack_type
        if attack_type == 'sybil':
            self.active_nodes    = int(self.TOTAL_NODES * 1.30)
            self.malicious_nodes = int(self.TOTAL_NODES * 0.30)
        elif attack_type == 'dos':
            self.active_nodes    = int(self.TOTAL_NODES * 0.60)
        elif attack_type == 'byzantine':
            self.malicious_nodes = int(self.TOTAL_NODES * 0.20)
        elif attack_type == 'eclipse':
            self.active_nodes    = int(self.TOTAL_NODES * 0.70)
        elif attack_type == 'majority':
            self.malicious_nodes = int(self.TOTAL_NODES * 0.51)
        elif attack_type == 'routing':
            self.active_nodes    = int(self.TOTAL_NODES * 0.90)

    def recover_from_attack(self, attack_type: str):
        self._attack         = None
        self.active_nodes    = self.TOTAL_NODES
        self.malicious_nodes = 0


# =============================================================================
# PERFORMANCE METRICS ANALYSER
# =============================================================================

class PerformanceMetricsAnalyzer:
    def __init__(self):
        self._records:       List[ConsensusPerformanceMetrics] = []
        self._attack_events: List[Dict] = []

    def add_metrics(self, m: ConsensusPerformanceMetrics):
        self._records.append(m)

    def add_attack_event(self, step: int, attack: str, recovery: Optional[int]):
        self._attack_events.append({'step': step, 'attack': attack, 'recovery': recovery})

    def generate_consensus_performance_table(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        rows = []
        for ct in ConsensusType:
            subset = [r for r in self._records if r.consensus_type == ct.value]
            if not subset:
                continue
            rows.append({
                'Consensus Type':    ct.value,
                'Avg Security Score':round(np.mean([r.security_score for r in subset]), 4),
                'Avg Latency':       round(np.mean([r.latency for r in subset]), 4),
                'Avg Throughput':    round(np.mean([r.throughput for r in subset]), 4),
                'Energy Efficiency': round(np.mean([r.energy_efficiency for r in subset]), 4),
                'Scalability':       round(np.mean([r.scalability for r in subset]), 4),
                'Time Steps Used':   len(subset),
            })
        return pd.DataFrame(rows)

    def generate_attack_response_table(self) -> pd.DataFrame:
        rows = []
        for evt in self._attack_events:
            s, e = evt['step'], evt.get('recovery', evt['step'] + 100)
            if e is None:
                e = evt['step'] + 100
            phase = [r for r in self._records
                     if s <= self._records.index(r) < e]
            if not phase:
                continue
            for ct in ConsensusType:
                sub = [r for r in phase if r.consensus_type == ct.value]
                if sub:
                    rows.append({
                        'Attack':      evt['attack'].capitalize(),
                        'Mechanism':   ct.value,
                        'Avg Security':round(np.mean([r.security_score for r in sub]), 4),
                        'Avg Latency': round(np.mean([r.latency for r in sub]), 4),
                        'Steps Used':  len(sub),
                    })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def generate_switching_analysis_table(self, switch_log: List[Dict]) -> pd.DataFrame:
        rows = [
            {'Metric': 'Total Switches',           'Value': len(switch_log)},
            {'Metric': 'Switches per 1,000 steps', 'Value': len(switch_log)},
            {'Metric': 'ADACON Baseline (hysteresis-corrected)',
             'Value': ADACON_HYSTERESIS_BASELINE},
            {'Metric': 'Reduction vs ADACON',
             'Value': f"{(1-len(switch_log)/ADACON_HYSTERESIS_BASELINE)*100:.1f}%"},
        ]
        if switch_log:
            rc: Dict[str, int] = {}
            for s in switch_log:
                rc[s['reason']] = rc.get(s['reason'], 0) + 1
            for r, c in rc.items():
                rows.append({'Metric': f'  Reason: {r}', 'Value': c})
        return pd.DataFrame(rows)

    def generate_domain_comparison_table(self, t_res: Dict, e_res: Dict) -> pd.DataFrame:
        rows = []
        for label, res in [('Traffic Management', t_res),
                            ('Environmental Monitoring', e_res)]:
            rows.append({
                'Domain':            label,
                'Avg Latency':       round(np.mean(res['latency']), 4),
                'Avg Throughput':    round(np.mean(res['throughput']), 4),
                'Posture Index': f"{round(np.mean(res['posture_index'])*100, 1)}%",
                'Consensus Switches':res['n_switches'],
                'Primary Mechanism': max(res['mechanism_counts'],
                                         key=res['mechanism_counts'].get),
            })
        return pd.DataFrame(rows)


# =============================================================================
# STATIC BASELINE RUNNER
# =============================================================================

def run_static_baseline(domain: IoTDomain, mechanism: ConsensusType,
                         time_steps: int = 1000, seed: int = 42) -> Dict:
    """
    Fixed-mechanism baseline — same simulator, same attack schedule, same seed.
    Threat detection uses Bayesian posteriors but cannot switch mechanism.
    """
    random.seed(seed)
    np.random.seed(seed)

    simulator      = (TrafficNetworkSimulator() if domain == IoTDomain.TRAFFIC
                      else EnvironmentalNetworkSimulator())
    bayes_detector = BayesianThreatDetector(domain)

    attack_schedule = {
        int(time_steps * 0.10): 'sybil',
        int(time_steps * 0.25): 'dos',
        int(time_steps * 0.40): 'byzantine',
        int(time_steps * 0.55): 'eclipse',
        int(time_steps * 0.70): 'majority',
        int(time_steps * 0.85): 'routing',
    }
    recovery_schedule = {
        int(time_steps * 0.20): 'sybil',
        int(time_steps * 0.35): 'dos',
        int(time_steps * 0.50): 'byzantine',
        int(time_steps * 0.65): 'eclipse',
        int(time_steps * 0.80): 'majority',
        int(time_steps * 0.95): 'routing',
    }

    attrs = DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[mechanism]
    lat_mult  = LATENCY_OVERHEAD[mechanism]
    tput_mult = THROUGHPUT_FACTOR[mechanism]
    latencies, throughputs, energies, postures = [], [], [], []
    active_attack = None

    for i in range(time_steps):
        if i in attack_schedule:
            active_attack = attack_schedule[i]
            simulator.simulate_attack(active_attack)
        if i in recovery_schedule:
            simulator.recover_from_attack(recovery_schedule[i])
            active_attack = None

        state        = simulator.generate_network_state()
        threat_probs = bayes_detector.update(state.metrics)

        latencies.append(min(1.0, state.metrics.latency * lat_mult))
        throughputs.append(max(0.05, state.metrics.throughput * tput_mult))
        energies.append(attrs['energy_efficiency'])

        # Static mechanisms cannot switch — the outcome is computed with the
        # SAME deterministic formulae as the adaptive framework (double-spend
        # risk, finality violation, liveness), replacing the former
        # analytic components so both sides of every comparison are
        if active_attack:
            appropriate = {
                'sybil':    [ConsensusType.POS, ConsensusType.PBFT, ConsensusType.DPOS],
                'dos':      [ConsensusType.PBFT, ConsensusType.DPOS],
                'byzantine':[ConsensusType.PBFT],
                'eclipse':  [ConsensusType.POS, ConsensusType.PBFT],
                'majority': [ConsensusType.POA, ConsensusType.PBFT],
                'routing':  [ConsensusType.DPOS, ConsensusType.POS],
            }
            good      = mechanism in appropriate.get(active_attack, [ConsensusType.PBFT])
            threshold = DomainAwareConsensusEngine.ACTIVATION_THRESHOLDS.get(active_attack, 0.30)
            detected  = threat_probs.get(active_attack, 0) > threshold
            mech_sec  = attrs['security']
            mal_ratio = getattr(simulator, 'malicious_nodes', 0) / max(state.total_nodes, 1)
            dsr       = float(np.clip(mal_ratio * 2.0 + state.metrics.error_rate * 0.5
                                      - mech_sec * 0.6, 0.0, 1.0))
            fv        = float(np.clip(state.metrics.error_rate * 0.7
                                      + (1.0 - attrs['finality_speed']) * 0.3, 0.0, 1.0))
            liveness  = state.active_nodes / max(state.total_nodes, 1)
            det_bonus = 1.10 if detected else 0.75
            suit      = 1.05 if good else 0.85
            postures.append(float(np.clip(
                liveness * mech_sec * det_bonus * suit - dsr * 0.4 - fv * 0.2,
                0.05, 0.99)))
        else:
            postures.append(float(np.clip(
                state.active_nodes / max(state.total_nodes, 1) * 0.98, 0.90, 0.99)))

    return {
        'mechanism':         mechanism.value,
        'latency':           np.mean(latencies),
        'latency_std':       np.std(latencies),
        'throughput':        np.mean(throughputs),
        'energy':            np.mean(energies),
        'posture_index': np.mean(postures),
    }


def run_all_static_baselines(domain: IoTDomain,
                              time_steps: int = 1000,
                              seed: int = 42) -> Dict[str, Dict]:
    short = {
        ConsensusType.POW: 'PoW', ConsensusType.POS: 'PoS',
        ConsensusType.PBFT: 'PBFT', ConsensusType.POA: 'PoA',
        ConsensusType.DPOS: 'DPoS',
    }
    return {short[m]: run_static_baseline(domain, m, time_steps, seed)
            for m in ConsensusType}


# =============================================================================
# MAIN SIMULATION RUNNER
# =============================================================================

def run_simulation(domain: IoTDomain, time_steps: int = 1000,
                   seed: int = 42, verbose: bool = True) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    print(f"\nStarting simulation — {domain.value} — {time_steps} steps...")

    simulator       = (TrafficNetworkSimulator() if domain == IoTDomain.TRAFFIC
                       else EnvironmentalNetworkSimulator())
    stability_layer = StabilityAwareConsensusLayer(domain)   # Bayesian embedded here
    metrics_analyzer= PerformanceMetricsAnalyzer()

    attack_schedule = {
        int(time_steps * 0.10): 'sybil',
        int(time_steps * 0.25): 'dos',
        int(time_steps * 0.40): 'byzantine',
        int(time_steps * 0.55): 'eclipse',
        int(time_steps * 0.70): 'majority',
        int(time_steps * 0.85): 'routing',
    }
    recovery_schedule = {
        int(time_steps * 0.20): 'sybil',
        int(time_steps * 0.35): 'dos',
        int(time_steps * 0.50): 'byzantine',
        int(time_steps * 0.65): 'eclipse',
        int(time_steps * 0.80): 'majority',
        int(time_steps * 0.95): 'routing',
    }
    emergency_range = range(int(time_steps * 0.31), int(time_steps * 0.37))

    for step, attack in attack_schedule.items():
        metrics_analyzer.add_attack_event(
            step, attack,
            next((s for s, a in recovery_schedule.items() if a == attack), None))

    history = {
        'latency': [], 'throughput': [], 'cpu_usage': [], 'memory_usage': [],
        'bandwidth': [], 'error_rate': [], 'active_nodes': [], 'malicious_nodes': [],
        'threat_probs': [], 'mechanism': [], 'score': [], 'posture_index': [],
        'double_spend_risk': [], 'finality_violation': [], 'liveness_score': [],
    }
    mechanism_counts: Dict[str, int] = {}
    active_attack: Optional[str] = None

    for i in range(time_steps):
        if i in attack_schedule:
            active_attack = attack_schedule[i]
            simulator.simulate_attack(active_attack)
            if verbose:
                print(f"  Step {i:4d}: → ATTACK  [{active_attack.upper()}]")
        if i in recovery_schedule:
            simulator.recover_from_attack(recovery_schedule[i])
            # Partial decay of Beta posteriors so detector re-sensitises for next attack
            stability_layer.bayesian_detector.reset_priors()
            if verbose:
                print(f"  Step {i:4d}: → RECOVER [{recovery_schedule[i].upper()}]")
            active_attack = None

        if domain == IoTDomain.TRAFFIC:
            simulator.emergency_mode = i in emergency_range

        state = simulator.generate_network_state()

        # Stability layer runs Bayesian detection + scoring + gate internally
        mechanism, threat_probs = stability_layer.update(state)
        mechanism_counts[mechanism.value] = mechanism_counts.get(mechanism.value, 0) + 1

        # ── Fair-comparison fix (Reviewer 2, R2) ─────────────────────────────
        # Apply the SAME per-mechanism consensus overheads used by the static
        # baselines to whichever mechanism CDACF is currently running.
        # Previously CDACF recorded raw telemetry with no consensus cost,
        # which inflated its latency/throughput results by construction.
        eff_latency    = min(1.0,  state.metrics.latency    * LATENCY_OVERHEAD[mechanism])
        eff_throughput = max(0.05, state.metrics.throughput * THROUGHPUT_FACTOR[mechanism])
        # Deterministic formulae grounded in
        # the network state, so posture is an independent observable rather
        # than a self-referential random sample.
        #
        # Three concrete outcomes are tracked per step:
        #   • double_spend_risk   — probability attacker can rewrite history
        #   • finality_violation  — fraction of blocks lacking finality guarantees
        #   • liveness_score      — fraction of nodes still able to participate
        #
        # These are combined into a single posture rate in [0,1].
        if active_attack:
            posterior  = threat_probs.get(active_attack, 0)
            threshold  = DomainAwareConsensusEngine.ACTIVATION_THRESHOLDS.get(active_attack, 0.30)
            detected   = posterior > threshold

            _appropriate_map = {
                'sybil':    [ConsensusType.POS, ConsensusType.PBFT, ConsensusType.DPOS],
                'dos':      [ConsensusType.PBFT, ConsensusType.DPOS],
                'byzantine':[ConsensusType.PBFT],
                'eclipse':  [ConsensusType.POS, ConsensusType.PBFT],
                'majority': [ConsensusType.POA, ConsensusType.PBFT],
                'routing':  [ConsensusType.DPOS, ConsensusType.POS],
            }
            good_for_atk = mechanism in _appropriate_map.get(active_attack, [ConsensusType.PBFT])

            # Mechanism's own security attribute [0,1]
            mech_sec = DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[mechanism]['security']

            # Double-spend risk: driven by error_rate and whether attacker
            # controls majority of hash/stake (majority attack is worst case)
            mal_ratio = getattr(simulator, 'malicious_nodes', 0) / max(state.total_nodes, 1)
            double_spend_risk = float(np.clip(
                mal_ratio * 2.0 + state.metrics.error_rate * 0.5 - mech_sec * 0.6, 0.0, 1.0))

            # Finality violation: PBFT gives instant finality (low violation),
            # PoW gives probabilistic finality (high violation under attack)
            finality_attr = DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[mechanism]['finality_speed']
            finality_violation = float(np.clip(
                state.metrics.error_rate * 0.7 + (1.0 - finality_attr) * 0.3, 0.0, 1.0))

            # Liveness: fraction of network still participating
            liveness_score = float(np.clip(
                state.active_nodes / max(state.total_nodes, 1), 0.0, 1.0))

            # Prevention = detection_bonus × mechanism_suitability × liveness
            #            − double_spend_risk − finality_violation_penalty
            detection_bonus = 1.10 if detected else 0.75
            suit_factor     = 1.05 if good_for_atk else 0.85
            raw_prevention  = (liveness_score * mech_sec * detection_bonus * suit_factor
                               - double_spend_risk * 0.4
                               - finality_violation * 0.2)
            posture = float(np.clip(raw_prevention, 0.05, 0.99))

            # Store individual outcome metrics for reporting
            history['double_spend_risk'].append(double_spend_risk)
            history['finality_violation'].append(finality_violation)
            history['liveness_score'].append(liveness_score)
        else:
            # No active attack — baseline healthy operation
            posture = float(np.clip(
                state.active_nodes / max(state.total_nodes, 1) * 0.98, 0.90, 0.99))
            history['double_spend_risk'].append(0.0)
            history['finality_violation'].append(0.0)
            history['liveness_score'].append(
                state.active_nodes / max(state.total_nodes, 1))

        attrs = DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[mechanism]
        perf  = ConsensusPerformanceMetrics(
            consensus_type    = mechanism.value,
            security_score    = stability_layer.current_score,
            latency           = eff_latency,
            throughput        = eff_throughput,
            energy_efficiency = attrs['energy_efficiency'],
            decentralization  = attrs['decentralization'],
            scalability       = attrs['scalability'],
            finality_time     = 1.0 - attrs['finality_speed'],
            attack_resistance = {k: 1.0 - v for k, v in threat_probs.items()},
            consensus_switches= len(stability_layer.switch_log),
            domain            = domain.value,
        )
        metrics_analyzer.add_metrics(perf)

        history['latency'].append(eff_latency)
        history['throughput'].append(eff_throughput)
        history['cpu_usage'].append(state.metrics.cpu_usage)
        history['memory_usage'].append(state.metrics.memory_usage)
        history['bandwidth'].append(state.metrics.bandwidth)
        history['error_rate'].append(state.metrics.error_rate)
        history['active_nodes'].append(state.active_nodes)
        history['malicious_nodes'].append(getattr(simulator, 'malicious_nodes', 0))
        history['threat_probs'].append(threat_probs)
        history['mechanism'].append(mechanism)
        history['score'].append(stability_layer.current_score)
        history['posture_index'].append(posture)

    n_switches = len(stability_layer.switch_log)
    print(f"  Complete. Consensus switches: {n_switches}")

    return {
        'history':            history,
        'mechanism_counts':   mechanism_counts,
        'stability_layer':    stability_layer,
        'metrics_analyzer':   metrics_analyzer,
        'n_switches':         n_switches,
        'latency':            history['latency'],
        'throughput':         history['throughput'],
        'posture_index':  history['posture_index'],
        'double_spend_risk':  history['double_spend_risk'],
        'finality_violation': history['finality_violation'],
        'liveness_score':     history['liveness_score'],
        'attack_schedule':    attack_schedule,
        'recovery_schedule':  recovery_schedule,
    }


# =============================================================================
# VISUALISATIONS  — bar graphs wherever comparison is made
# =============================================================================

COLORS = {
    'PoW': '#e74c3c', 'PoS': '#2ecc71', 'PBFT': '#3498db',
    'PoA': '#f39c12', 'DPoS': '#9b59b6', 'Adaptive': '#1abc9c',
}
MECH_ORDER = ['PoW', 'PoS', 'PBFT', 'PoA', 'DPoS', 'Adaptive']


def plot_results(traffic_results: Dict, env_results: Dict,
                  traffic_static: Dict, env_static: Dict):
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 11})

    _fig1_bayesian_posteriors(traffic_results, env_results)
    _fig2_consensus_timeline(traffic_results, env_results)
    _fig3_latency_bar(traffic_results, traffic_static)
    _fig4_throughput_bar(env_results, env_static)
    _fig5_energy_bar(traffic_results, env_results, traffic_static, env_static)
    _fig6_posture_index_bar(traffic_results, env_results, traffic_static, env_static)
    _fig7_static_comparison_full(traffic_results, env_results, traffic_static, env_static)
    _fig8_switching_bar(traffic_results, env_results)
    _fig9_mechanism_distribution(traffic_results, env_results)
    _fig10_network_metrics(traffic_results, env_results)
    # Figure 11 (threat heatmap) REMOVED: it plotted a hard-coded illustrative
    # matrix, not simulation output, and is indefensible under reproducibility
    # review. Remove the corresponding figure from the manuscript as well.


# ── Figure 1: Bayesian posterior probabilities over time ─────────────────────
def _fig1_bayesian_posteriors(traffic_results: Dict, env_results: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Figure 1 — Bayesian Posterior Threat Probabilities Over Time\n"
        "(ADACON Eq. 4-9 indicators + Beta(α=2,β=8/10) domain priors)",
        fontsize=13, fontweight='bold')

    threat_types = ['sybil', 'dos', 'byzantine', 'eclipse', 'majority', 'routing']
    cols         = ['#e74c3c','#f39c12','#2ecc71','#3498db','#9b59b6','#1abc9c']

    for ax, res, title in [
            (axes[0], traffic_results,
             f'Traffic Management — {TrafficNetworkSimulator.TOTAL_NODES:,} nodes (5G, Beta(2,10))'),
            (axes[1], env_results,
             f'Environmental Monitoring — {EnvironmentalNetworkSimulator.TOTAL_NODES:,} nodes (LoRaWAN, Beta(2,8))')]:
        threat_hist = res['history']['threat_probs']
        for tt, col in zip(threat_types, cols):
            vals = [t[tt] for t in threat_hist]
            ax.plot(vals, label=tt.capitalize(), color=col, linewidth=1.2, alpha=0.85)
        for step in res['attack_schedule']:
            ax.axvline(x=step, color='red', linestyle='--', alpha=0.4, linewidth=1.0,
                       label='Attack start' if step == list(res['attack_schedule'].keys())[0] else '')
        for step in res['recovery_schedule']:
            ax.axvline(x=step, color='green', linestyle=':', alpha=0.4, linewidth=1.0,
                       label='Recovery' if step == list(res['recovery_schedule'].keys())[0] else '')
        ax.set_xlabel('Time Step'); ax.set_ylabel('Posterior Probability P(Attack|Data)')
        ax.set_title(title, fontweight='bold'); ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(_fig_path('Fig1_Bayesian_Posteriors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig1_Bayesian_Posteriors.png")


# ── Figure 2: Consensus mechanism timeline ────────────────────────────────────
def _fig2_consensus_timeline(traffic_results: Dict, env_results: Dict):
    mech_idx = {ct.value: i for i, ct in enumerate(ConsensusType)}
    short = {ct.value: ct.value.split()[-1] for ct in ConsensusType}
    col_list = list(COLORS.values())[:5]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Figure 2 — Consensus Mechanism Selection Over Time (Adaptive CDACF)",
                 fontsize=13, fontweight='bold')

    for ax, res, title, n_nodes in [
            (axes[0], traffic_results, 'Traffic Management',     TrafficNetworkSimulator.TOTAL_NODES),
            (axes[1], env_results,     'Environmental Monitoring', EnvironmentalNetworkSimulator.TOTAL_NODES)]:
        mechs = res['history']['mechanism']
        y     = [mech_idx[m.value] for m in mechs]
        c     = [col_list[mech_idx[m.value]] for m in mechs]
        ax.scatter(range(len(y)), y, c=c, s=4, alpha=0.7, zorder=2)
        for step in res['attack_schedule']:
            ax.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=1.2)
        for step in res['recovery_schedule']:
            ax.axvline(x=step, color='green', linestyle=':', alpha=0.5, linewidth=1.2)
        ax.set_yticks(list(mech_idx.values()))
        ax.set_yticklabels([short[v] for v in mech_idx], fontsize=10)
        ax.set_ylabel('Mechanism')
        ax.set_title(f'{title}  ({n_nodes:,} nodes)', fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.3)

    axes[1].set_xlabel('Time Step')
    patches = [mpatches.Patch(color=col_list[i], label=list(mech_idx.keys())[i])
               for i in range(5)]
    patches += [mpatches.Patch(color='red', linestyle='--', label='Attack start'),
                mpatches.Patch(color='green', linestyle=':', label='Recovery')]
    fig.legend(handles=patches, loc='lower center', ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(_fig_path('Fig2_Consensus_Timeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig2_Consensus_Timeline.png")


# ── Figure 3: Latency bar — Traffic, adaptive vs static ──────────────────────
def _fig3_latency_bar(traffic_results: Dict, traffic_static: Dict):
    labels = MECH_ORDER
    vals   = [traffic_static[m]['latency'] if m in traffic_static
              else np.mean(traffic_results['latency']) for m in labels]
    cols   = [COLORS[m] for m in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, vals, color=cols, edgecolor='white', linewidth=1.5,
                  alpha=0.87, width=0.6)
    bars[-1].set_edgecolor('#16a085'); bars[-1].set_linewidth(3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    adaptive_val = np.mean(traffic_results['latency'])
    ax.axhline(y=adaptive_val, color='#16a085', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Adaptive: {adaptive_val:.3f}')
    ax.set_ylabel('Normalised Latency (lower = better)', fontsize=12)
    ax.set_title('Figure 3 — Normalised Latency: Adaptive vs Static Baselines\n'
                 'Traffic Management Domain (same attack schedule, same seed)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim(0, max(vals) * 1.18)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(_fig_path('Fig3_Latency_Bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig3_Latency_Bar.png")


# ── Figure 4: Throughput bar — Environmental, adaptive vs static ─────────────
def _fig4_throughput_bar(env_results: Dict, env_static: Dict):
    labels = MECH_ORDER
    vals   = [env_static[m]['throughput'] if m in env_static
              else np.mean(env_results['throughput']) for m in labels]
    cols   = [COLORS[m] for m in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, vals, color=cols, edgecolor='white', linewidth=1.5,
                  alpha=0.87, width=0.6)
    bars[-1].set_edgecolor('#16a085'); bars[-1].set_linewidth(3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    adaptive_val = np.mean(env_results['throughput'])
    ax.axhline(y=adaptive_val, color='#16a085', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Adaptive: {adaptive_val:.3f}')
    ax.set_ylabel('Normalised Throughput (higher = better)', fontsize=12)
    ax.set_title('Figure 4 — Normalised Throughput: Adaptive vs Static Baselines\n'
                 'Environmental Monitoring Domain (same attack schedule, same seed)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(_fig_path('Fig4_Throughput_Bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig4_Throughput_Bar.png")


# ── Figure 5: Energy efficiency bar — both domains ───────────────────────────
def _fig5_energy_bar(traffic_results, env_results, traffic_static, env_static):
    labels = MECH_ORDER
    t_adapt_energy = np.mean([
        DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[m]['energy_efficiency']
        for m in traffic_results['history']['mechanism']])
    e_adapt_energy = np.mean([
        DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[m]['energy_efficiency']
        for m in env_results['history']['mechanism']])
    t_vals = [traffic_static[m]['energy'] if m in traffic_static else t_adapt_energy for m in labels]
    e_vals = [env_static[m]['energy']     if m in env_static     else e_adapt_energy for m in labels]

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, t_vals, w, label='Traffic Management',
                color=[COLORS[m] for m in labels], alpha=0.75, edgecolor='white')
    b2 = ax.bar(x + w/2, e_vals, w, label='Environmental Monitoring',
                color=[COLORS[m] for m in labels], alpha=0.95, edgecolor='black',
                linewidth=1.2, hatch='//')

    for bar, v in zip(list(b1) + list(b2), t_vals + e_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Energy Efficiency (higher = better)', fontsize=12)
    ax.set_title('Figure 5 — Energy Efficiency: Adaptive vs Static, Both Domains',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(_fig_path('Fig5_Energy_Bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig5_Energy_Bar.png")


# ── Figure 6: Composite posture index bar — both domains ───────────────────────────
def _fig6_posture_index_bar(traffic_results, env_results, traffic_static, env_static):
    labels = MECH_ORDER
    t_adapt = np.mean(traffic_results['posture_index'])
    e_adapt = np.mean(env_results['posture_index'])
    t_vals  = [traffic_static[m]['posture_index'] if m in traffic_static else t_adapt for m in labels]
    e_vals  = [env_static[m]['posture_index']     if m in env_static     else e_adapt for m in labels]

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, t_vals, w, label='Traffic Management',
                color=[COLORS[m] for m in labels], alpha=0.75, edgecolor='white')
    b2 = ax.bar(x + w/2, e_vals, w, label='Environmental Monitoring',
                color=[COLORS[m] for m in labels], alpha=0.95, edgecolor='black',
                linewidth=1.2, hatch='//')

    for bar, v in zip(list(b1) + list(b2), t_vals + e_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v*100:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Simulated Mitigation-Response Score (higher = better)', fontsize=12)
    ax.set_title('Figure 6 — Simulated Mitigation-Response Score: Adaptive vs Static, Both Domains\n'
                 '(Bayesian posterior-driven mechanism selection)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0, 1.10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(_fig_path('Fig6_ThreatPrevention_Bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig6_ThreatPrevention_Bar.png")


# ── Figure 7: Full 4-metric comparison grid ───────────────────────────────────
def _fig7_static_comparison_full(traffic_results, env_results, traffic_static, env_static):
    labels = MECH_ORDER
    t_adapt_energy = np.mean([
        DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[m]['energy_efficiency']
        for m in traffic_results['history']['mechanism']])
    e_adapt_energy = np.mean([
        DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[m]['energy_efficiency']
        for m in env_results['history']['mechanism']])

    metrics = {
        'Latency — Traffic\n(lower = better)': {
            m: traffic_static[m]['latency'] if m in traffic_static
               else np.mean(traffic_results['latency']) for m in labels},
        'Throughput — Environmental\n(higher = better)': {
            m: env_static[m]['throughput'] if m in env_static
               else np.mean(env_results['throughput']) for m in labels},
        'Energy Efficiency — Environmental\n(higher = better)': {
            m: env_static[m]['energy'] if m in env_static else e_adapt_energy for m in labels},
        'Mitigation Response — Traffic\n(higher = better)': {
            m: traffic_static[m]['posture_index'] if m in traffic_static
               else np.mean(traffic_results['posture_index']) for m in labels},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Figure 7 — Comprehensive Performance Comparison: Adaptive (CDACF) vs Static Baselines',
                 fontsize=13, fontweight='bold')

    for ax, (title, data) in zip(axes.flat, metrics.items()):
        vals = [data[m] for m in labels]
        cols = [COLORS[m] for m in labels]
        bars = ax.bar(labels, vals, color=cols, edgecolor='white',
                      linewidth=1.5, alpha=0.87, width=0.6)
        bars[-1].set_edgecolor('#16a085'); bars[-1].set_linewidth(3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylim(0, max(vals) * 1.20)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(_fig_path('Fig7_Full_Comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig7_Full_Comparison.png")


# ── Figure 8: Consensus switching bar vs ADACON baseline ─────────────────────
def _fig8_switching_bar(traffic_results: Dict, env_results: Dict):
    base        = ADACON_HYSTERESIS_BASELINE
    categories  = [f'ADACON Baseline\n({base} switches)', 'CDACF — Traffic\nManagement',
                   'CDACF — Environmental\nMonitoring']
    switch_vals = [base, traffic_results['n_switches'], env_results['n_switches']]
    cols        = ['#e74c3c', '#3498db', '#2ecc71']
    reductions  = [0,
                   (1 - traffic_results['n_switches'] / base) * 100,
                   (1 - env_results['n_switches'] / base) * 100]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(categories, switch_vals, color=cols, edgecolor='white',
                  linewidth=2, alpha=0.87, width=0.5)

    for bar, v, red in zip(bars, switch_vals, reductions):
        label = f'{v} switches'
        if red > 0:
            label += f'\n(↓ {red:.1f}% vs ADACON)'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Consensus Switches per 1,000 Steps', fontsize=12)
    ax.set_title('Figure 8 — Consensus Switching Stability\n'
                 'Stability-Aware Layer vs ADACON Unconstrained Baseline',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, base * 1.30)
    ax.axhline(y=base, color='red', linestyle='--', linewidth=1.5,
               alpha=0.5, label='ADACON baseline (hysteresis-corrected)')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(_fig_path('Fig8_Switching_Bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig8_Switching_Bar.png")


# ── Figure 9: Mechanism distribution bar ─────────────────────────────────────
def _fig9_mechanism_distribution(traffic_results: Dict, env_results: Dict):
    all_mechs  = [ct.value for ct in ConsensusType]
    short_names= ['PoW', 'PoS', 'PBFT', 'PoA', 'DPoS']
    col_list   = [COLORS[n] for n in short_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Figure 9 — Consensus Mechanism Distribution\n'
                 '(Steps active per mechanism — validates domain-specific adaptation)',
                 fontsize=13, fontweight='bold')

    for ax, res, title in [(axes[0], traffic_results, 'Traffic Management'),
                            (axes[1], env_results,     'Environmental Monitoring')]:
        counts = res['mechanism_counts']
        total  = sum(counts.values())
        steps  = [counts.get(m, 0) for m in all_mechs]
        pcts   = [100 * s / max(total, 1) for s in steps]

        bars = ax.bar(short_names, steps, color=col_list, edgecolor='white',
                      linewidth=1.5, alpha=0.87, width=0.6)
        for bar, s, p in zip(bars, steps, pcts):
            if s > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                        f'{s}\n({p:.1f}%)', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        ax.set_ylabel('Time Steps Active', fontsize=11)
        ax.set_title(f'{title}\n(Total switches: {res["n_switches"]})', fontweight='bold')
        ax.set_ylim(0, total * 1.20)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(_fig_path('Fig9_Mechanism_Distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig9_Mechanism_Distribution.png")


# ── Figure 10: Network metrics over time ──────────────────────────────────────
def _fig10_network_metrics(traffic_results: Dict, env_results: Dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Figure 10 — Network Metrics Over Time: Both IoT Domains',
                 fontsize=14, fontweight='bold')
    keys   = ['latency','throughput','cpu_usage','memory_usage','error_rate','active_nodes']
    labels = ['Latency','Throughput','CPU Usage','Memory Usage','Error Rate','Active Nodes']

    for idx, (key, label) in enumerate(zip(keys, labels)):
        ax = axes[idx // 3][idx % 3]
        ax.plot(traffic_results['history'][key], color='#3498db',
                linewidth=1, alpha=0.75, label='Traffic')
        ax.plot(env_results['history'][key], color='#e74c3c',
                linewidth=1, alpha=0.75, label='Environmental')
        for step in traffic_results['attack_schedule']:
            ax.axvline(x=step, color='red', linestyle='--', alpha=0.2, linewidth=0.8)
        ax.set_xlabel('Time Step'); ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold'); ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(_fig_path('Fig10_Network_Metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig10_Network_Metrics.png")


# ── Figure 11: REMOVED ────────────────────────────────────────────────────────
# The former threat-posture heatmap plotted a hard-coded matrix described as
# "representative values" rather than measured simulation output. Publishing it
# as a result is indefensible; it has been deleted. If a per-attack × mechanism
# heatmap is wanted in future, compute it from the per-attack-phase
# mitigation-response records of the (patched) baseline and adaptive runs.


# =============================================================================
# REVIEWER FIX 4 — MULTI-SEED STATISTICAL VALIDATION  (5 independent seeds)
# Matches ADACON's validation protocol (seeds 5-9, Table 11, Bhore et al. 2025)
# =============================================================================

VALIDATION_SEEDS = [5, 6, 7, 8, 9]   # same seed set as ADACON for direct comparison

def run_multi_seed_validation(domain: IoTDomain, time_steps: int = 1000) -> Dict:
    """
    Run 5 independent simulations with different random seeds.
    Returns per-seed records and aggregate statistics (mean, std, 95% CI, CV%).
    """
    records = []
    print(f"\n  Multi-seed validation — {domain.value}  (seeds {VALIDATION_SEEDS})")
    for seed in VALIDATION_SEEDS:
        r = run_simulation(domain, time_steps, seed, verbose=False)
        records.append({
            'seed':              seed,
            'latency_mean':      float(np.mean(r['latency'])),
            'throughput_mean':   float(np.mean(r['throughput'])),
            'posture_index_mean':  float(np.mean(r['posture_index'])),
            'dsr_mean':          float(np.mean(r['double_spend_risk'])),
            'fv_mean':           float(np.mean(r['finality_violation'])),
            'liveness_mean':     float(np.mean(r['liveness_score'])),
            'n_switches':        r['n_switches'],
        })
        print(f"    seed={seed}: lat={records[-1]['latency_mean']:.4f}  "
              f"tput={records[-1]['throughput_mean']:.4f}  "
              f"prev={records[-1]['posture_index_mean']*100:.1f}%  "
              f"switches={records[-1]['n_switches']}")

    df = pd.DataFrame(records)
    agg = {}
    for col in ['latency_mean','throughput_mean','posture_index_mean',
                'dsr_mean','fv_mean','liveness_mean','n_switches']:
        vals  = df[col].values.astype(float)
        mean  = float(np.mean(vals))
        std   = float(np.std(vals, ddof=1))
        se    = std / np.sqrt(len(vals))
        ci_lo = mean - 1.96 * se
        ci_hi = mean + 1.96 * se
        cv    = (std / mean * 100) if mean != 0 else 0.0
        agg[col] = {'mean': mean, 'std': std,
                    'ci': (ci_lo, ci_hi), 'cv': cv}
    return {'records': df, 'agg': agg}


def _print_validation_table(agg: Dict, domain_label: str):
    """Print a reviewer-ready statistical validation table."""
    rows = []
    metric_labels = {
        'latency_mean':      'Avg Latency (normalised)',
        'throughput_mean':   'Avg Throughput (normalised)',
        'posture_index_mean':  'Threat Prevention Rate',
        'dsr_mean':          'Double-Spend Risk',
        'fv_mean':           'Finality Violation Rate',
        'liveness_mean':     'Network Liveness',
        'n_switches':        'Consensus Switches / 1,000 steps',
    }
    for key, label in metric_labels.items():
        a = agg[key]
        is_pct = key in ('posture_index_mean',)
        fmt = lambda v: f"{v*100:.2f}%" if is_pct else f"{v:.4f}"
        cv_flag = '⚠' if a['cv'] > 10 else '✓'
        rows.append({
            'Metric':   label,
            'Mean':     fmt(a['mean']),
            'Std Dev':  fmt(a['std']),
            '95% CI':   f"[{fmt(a['ci'][0])}, {fmt(a['ci'][1])}]",
            'CV (%)':   f"{a['cv']:.2f}%  {cv_flag}",
        })
    print(f"\n{'='*72}")
    print(f"  STATISTICAL VALIDATION — {domain_label}  (n=5 seeds)")
    print(f"{'='*72}")
    print(tabulate(pd.DataFrame(rows), headers='keys',
                   tablefmt='fancy_grid', showindex=False))


# =============================================================================
# REVIEWER FIX 5 — CONCURRENT MULTI-VECTOR ATTACK SCENARIO
# Adds a simultaneous Sybil + DoS attack window to stress-test the framework.
# =============================================================================

def run_concurrent_attack_scenario(domain: IoTDomain,
                                    time_steps: int = 500,
                                    seed: int = 42) -> Dict:
    """
    Runs a shorter simulation where Sybil and DoS attacks are injected
    simultaneously (steps 150–300) to test framework behaviour under
    concurrent multi-vector threats — a gap identified by the reviewer.
    """
    random.seed(seed); np.random.seed(seed)
    simulator       = (TrafficNetworkSimulator() if domain == IoTDomain.TRAFFIC
                       else EnvironmentalNetworkSimulator())
    stability_layer = StabilityAwareConsensusLayer(domain)

    concurrent_start, concurrent_end = 150, 300
    history_mech, history_threat, history_prev = [], [], []
    switches_log = []
    active_attacks: List[str] = []

    _appropriate_map = {
        'sybil':    [ConsensusType.POS, ConsensusType.PBFT, ConsensusType.DPOS],
        'dos':      [ConsensusType.PBFT, ConsensusType.DPOS],
        'byzantine':[ConsensusType.PBFT],
        'eclipse':  [ConsensusType.POS, ConsensusType.PBFT],
        'majority': [ConsensusType.POA, ConsensusType.PBFT],
        'routing':  [ConsensusType.DPOS, ConsensusType.POS],
    }

    for i in range(time_steps):
        if i == concurrent_start:
            simulator.simulate_attack('sybil')
            simulator.simulate_attack('dos')   # stack second attack on top
            active_attacks = ['sybil', 'dos']
        if i == concurrent_end:
            simulator.recover_from_attack('sybil')
            simulator.recover_from_attack('dos')
            stability_layer.bayesian_detector.reset_priors()
            active_attacks = []

        state = simulator.generate_network_state()
        mechanism, threat_probs = stability_layer.update(state)
        history_mech.append(mechanism)
        history_threat.append(threat_probs)

        # Prevention under concurrent attack: must be appropriate for BOTH
        if active_attacks:
            mech_sec  = DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[mechanism]['security']
            good_both = all(mechanism in _appropriate_map.get(a, []) for a in active_attacks)
            detected_any = any(
                threat_probs.get(a, 0) > DomainAwareConsensusEngine.ACTIVATION_THRESHOLDS.get(a, 0.3)
                for a in active_attacks)
            mal_ratio   = getattr(simulator, 'malicious_nodes', 0) / max(state.total_nodes, 1)
            dsr         = float(np.clip(mal_ratio * 2.0 + state.metrics.error_rate * 0.5
                                        - mech_sec * 0.6, 0.0, 1.0))
            fin_attr    = DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[mechanism]['finality_speed']
            fv          = float(np.clip(state.metrics.error_rate * 0.7
                                        + (1.0 - fin_attr) * 0.3, 0.0, 1.0))
            liveness    = state.active_nodes / max(state.total_nodes, 1)
            det_bonus   = 1.10 if detected_any else 0.75
            suit        = 1.05 if good_both else 0.80   # stricter: must cover BOTH
            prev        = float(np.clip(liveness * mech_sec * det_bonus * suit
                                        - dsr * 0.4 - fv * 0.2, 0.05, 0.99))
        else:
            prev = float(np.clip(state.active_nodes / max(state.total_nodes, 1) * 0.98,
                                 0.90, 0.99))
        history_prev.append(prev)

    n_sw = len(stability_layer.switch_log)
    avg_prev_attack   = float(np.mean(history_prev[concurrent_start:concurrent_end]))
    avg_prev_baseline = float(np.mean(
        history_prev[:concurrent_start] + history_prev[concurrent_end:]))

    print(f"\n  Concurrent Sybil+DoS scenario ({domain.value}):")
    print(f"    Switches during concurrent window : "
          f"{sum(1 for s in stability_layer.switch_log if concurrent_start <= s['step'] < concurrent_end)}")
    print(f"    Mean posture index (concurrent window): {avg_prev_attack*100:.1f}%")
    print(f"    Mean posture index (baseline periods): {avg_prev_baseline*100:.1f}%")

    return {
        'history_mech':  history_mech,
        'history_threat':history_threat,
        'history_prev':  history_prev,
        'n_switches':    n_sw,
        'switch_log':    stability_layer.switch_log,
        'concurrent_start': concurrent_start,
        'concurrent_end':   concurrent_end,
        'avg_prev_attack':  avg_prev_attack,
        'avg_prev_baseline':avg_prev_baseline,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    SEED       = 42
    TIME_STEPS = 1000
    random.seed(SEED); np.random.seed(SEED)

    # ADACON correct baseline: hysteresis-corrected value from Table 3
    # (287 switches), NOT the pre-hysteresis 699.  Using 699 was misleading.
    ADACON_HYSTERESIS_BASELINE = 287

    print("=" * 72)
    print("  CDACF — Cross-Domain Adaptive Consensus Framework  (v2 — peer-review)")
    print("  Extends ADACON [Bhore et al., 2025] to Heterogeneous IoT")
    print("  Threat Detection: Bayesian Beta Updating (ADACON Eq. 4-9)")
    print(f"  Traffic Management    : {TrafficNetworkSimulator.TOTAL_NODES:,} nodes  (5G, 15ms base latency)")
    print(f"  Environmental Monitor : {EnvironmentalNetworkSimulator.TOTAL_NODES:,} nodes  (LoRaWAN, 500ms base latency)")
    print("  metrics  [4] 5-seed validation  [5] concurrent attacks  [6] PBFT note")
    print("=" * 72)

    # ── Node configuration ────────────────────────────────────────────────────
    node_cfg = [
        {'Domain': 'Traffic Management',
         'Total Nodes': f"{TrafficNetworkSimulator.TOTAL_NODES:,}",
         'Network': '5G', 'Base Latency': '15 ms', 'Base Packet Loss': '0.5%',
         'Sybil Fake Nodes': f"{int(TrafficNetworkSimulator.TOTAL_NODES*0.30):,}",
         'Majority Threshold': f"{int(TrafficNetworkSimulator.TOTAL_NODES*0.51):,}"},
        {'Domain': 'Env. Monitoring',
         'Total Nodes': f"{EnvironmentalNetworkSimulator.TOTAL_NODES:,}",
         'Network': 'LoRaWAN', 'Base Latency': '500 ms', 'Base Packet Loss': '5.0%',
         'Sybil Fake Nodes': f"{int(EnvironmentalNetworkSimulator.TOTAL_NODES*0.30):,}",
         'Majority Threshold': f"{int(EnvironmentalNetworkSimulator.TOTAL_NODES*0.51):,}"},
    ]
    print("\nNETWORK CONFIGURATION")
    print(tabulate(pd.DataFrame(node_cfg), headers='keys',
                   tablefmt='fancy_grid', showindex=False))
    t_n = TrafficNetworkSimulator.TOTAL_NODES
    pbft_msgs = t_n * (t_n - 1)
    print(f"\n  ⚠  PBFT O(n²) note: at {t_n:,} nodes, PBFT requires ~{pbft_msgs:,} messages")
    print(f"     per consensus round.  In production, PBFT would be replaced by a")
    print(f"     scalable BFT variant (HotStuff, Tendermint) beyond ~100 nodes.")
    print(f"     Simulation scores are adjusted: PBFT scalability attribute = 0.55")
    print(f"     (already penalised vs DPoS=0.87, PoS=0.72) — see MECHANISM_ATTRIBUTES.")

    # ── Primary simulation (seed 42) ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  PRIMARY SIMULATION  (seed=42)")
    print(f"{'─'*55}")
    traffic_results = run_simulation(IoTDomain.TRAFFIC,     TIME_STEPS, SEED)
    env_results     = run_simulation(IoTDomain.ENVIRONMENT, TIME_STEPS, SEED)

    print("\nRunning static baselines — Traffic Management...")
    traffic_static = run_all_static_baselines(IoTDomain.TRAFFIC,     TIME_STEPS, SEED)
    print("Running static baselines — Environmental Monitoring...")
    env_static     = run_all_static_baselines(IoTDomain.ENVIRONMENT, TIME_STEPS, SEED)

    analyzer_t = traffic_results['metrics_analyzer']
    analyzer_e = env_results['metrics_analyzer']

    # ── Table 1 & 2: Consensus performance ───────────────────────────────────
    print("\n" + "="*65)
    print(f"TABLE 1: CONSENSUS PERFORMANCE — TRAFFIC MANAGEMENT ({TrafficNetworkSimulator.TOTAL_NODES:,} nodes)")
    print("="*65)
    t1 = analyzer_t.generate_consensus_performance_table()
    if not t1.empty:
        print(tabulate(t1, headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n" + "="*65)
    print(f"TABLE 2: CONSENSUS PERFORMANCE — ENVIRONMENTAL MONITORING ({EnvironmentalNetworkSimulator.TOTAL_NODES:,} nodes)")
    print("="*65)
    t2 = analyzer_e.generate_consensus_performance_table()
    if not t2.empty:
        print(tabulate(t2, headers='keys', tablefmt='fancy_grid', showindex=False))

    # ── Table 3: Switching analysis (correct ADACON baseline) ────────────────
    def _switching_table(switch_log, label):
        n = len(switch_log)
        rows = [
            {'Metric': 'Total Switches',                      'Value': n},
            {'Metric': 'Switches per 1,000 steps',            'Value': n},
            {'Metric': 'ADACON hysteresis-corrected baseline', 'Value': ADACON_HYSTERESIS_BASELINE},
            {'Metric': 'Reduction vs ADACON (hysteresis)',
             'Value': f"{(1 - n/ADACON_HYSTERESIS_BASELINE)*100:.1f}%"},
        ]
        if switch_log:
            rc: Dict[str, int] = {}
            for s in switch_log: rc[s['reason']] = rc.get(s['reason'], 0) + 1
            for r, c in rc.items(): rows.append({'Metric': f'  Reason: {r}', 'Value': c})
        print(f"\n{'='*65}")
        print(f"TABLE 3: SWITCHING ANALYSIS — {label}")
        print("="*65)
        print(tabulate(pd.DataFrame(rows), headers='keys',
                       tablefmt='fancy_grid', showindex=False))

    _switching_table(traffic_results['stability_layer'].switch_log, 'TRAFFIC MANAGEMENT')
    _switching_table(env_results['stability_layer'].switch_log,     'ENVIRONMENTAL MONITORING')

    # ── Table 4: Cross-domain comparison ─────────────────────────────────────
    print("\n" + "="*65)
    print("TABLE 4: CROSS-DOMAIN PERFORMANCE COMPARISON")
    print("="*65)
    t4 = analyzer_t.generate_domain_comparison_table(traffic_results, env_results)
    print(tabulate(t4, headers='keys', tablefmt='fancy_grid', showindex=False))

    # ── Table 5/6: Static vs Adaptive ────────────────────────────────────────
    for domain_label, static_res, adapt_res in [
            ('TRAFFIC MANAGEMENT',       traffic_static, traffic_results),
            ('ENVIRONMENTAL MONITORING', env_static,     env_results)]:
        print(f"\n{'='*65}")
        print(f"TABLE 5/6: STATIC BASELINE vs ADAPTIVE — {domain_label}")
        print("="*65)
        adapt_energy = np.mean([
            DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES[m]['energy_efficiency']
            for m in adapt_res['history']['mechanism']])
        rows = []
        for name, r in static_res.items():
            rows.append({
                'Mechanism':         name + ' (Static)',
                'Avg Latency':       round(r['latency'], 4),
                'Avg Throughput':    round(r['throughput'], 4),
                'Energy Efficiency': round(r['energy'], 4),
                'Posture Index': f"{r['posture_index']*100:.1f}%",
                'Switches':          0,
            })
        rows.append({
            'Mechanism':         '★ Adaptive (CDACF)',
            'Avg Latency':       round(np.mean(adapt_res['latency']), 4),
            'Avg Throughput':    round(np.mean(adapt_res['throughput']), 4),
            'Energy Efficiency': round(adapt_energy, 4),
            'Posture Index': f"{np.mean(adapt_res['posture_index'])*100:.1f}%",
            'Switches':          adapt_res['n_switches'],
        })
        print(tabulate(pd.DataFrame(rows), headers='keys',
                       tablefmt='fancy_grid', showindex=False))

    # ── Table 7 (new): Concrete attack outcome metrics ───────────────────────
    print(f"\n{'='*65}")
    print("TABLE 7 (NEW): CONCRETE ATTACK OUTCOME METRICS — seed=42")
    print("  Analytic outcome components: liveness, double-spend risk, finality violation")
    print("="*65)
    for label, res in [('Traffic Management', traffic_results),
                        ('Environmental Monitoring', env_results)]:
        dsr = np.mean([v for v in res['double_spend_risk'] if v > 0] or [0])
        fv  = np.mean([v for v in res['finality_violation'] if v > 0] or [0])
        lv  = np.mean(res['liveness_score'])
        tp  = np.mean(res['posture_index'])
        print(f"\n  {label}:")
        outcome_rows = [
            {'Metric': 'Avg Double-Spend Risk (during attacks)',   'Value': f"{dsr:.4f}"},
            {'Metric': 'Avg Finality Violation Rate (during atks)','Value': f"{fv:.4f}"},
            {'Metric': 'Avg Network Liveness (all steps)',         'Value': f"{lv:.4f}"},
            {'Metric': 'Composite Threat Prevention Rate',         'Value': f"{tp*100:.2f}%"},
        ]
        print(tabulate(pd.DataFrame(outcome_rows), headers='keys',
                       tablefmt='fancy_grid', showindex=False))
    print(f"\n{'='*65}")
    print("MULTI-SEED STATISTICAL VALIDATION  (5 independent seeds: 5-9)")
    print("  Matches ADACON validation protocol — Bhore et al. 2025, Table 11")
    print("="*65)
    t_val = run_multi_seed_validation(IoTDomain.TRAFFIC,     TIME_STEPS)
    e_val = run_multi_seed_validation(IoTDomain.ENVIRONMENT, TIME_STEPS)
    _print_validation_table(t_val['agg'], 'TRAFFIC MANAGEMENT (5,000 nodes)')
    _print_validation_table(e_val['agg'], 'ENVIRONMENTAL MONITORING (2,000 nodes)')
    print(f"\n{'='*65}")
    print("CONCURRENT MULTI-VECTOR ATTACK SCENARIO  (Sybil + DoS simultaneously)")
    print("  Addresses reviewer concern: real adversaries deploy concurrent attacks")
    print("="*65)
    t_conc = run_concurrent_attack_scenario(IoTDomain.TRAFFIC,     seed=SEED)
    e_conc = run_concurrent_attack_scenario(IoTDomain.ENVIRONMENT, seed=SEED)
    conc_rows = [
        {'Domain': 'Traffic Management',
         'Switches (concurrent window)': sum(
             1 for s in t_conc['switch_log']
             if t_conc['concurrent_start'] <= s['step'] < t_conc['concurrent_end']),
         'Prevention (concurrent)': f"{t_conc['avg_prev_attack']*100:.1f}%",
         'Prevention (baseline)':   f"{t_conc['avg_prev_baseline']*100:.1f}%"},
        {'Domain': 'Environmental Monitoring',
         'Switches (concurrent window)': sum(
             1 for s in e_conc['switch_log']
             if e_conc['concurrent_start'] <= s['step'] < e_conc['concurrent_end']),
         'Prevention (concurrent)': f"{e_conc['avg_prev_attack']*100:.1f}%",
         'Prevention (baseline)':   f"{e_conc['avg_prev_baseline']*100:.1f}%"},
    ]
    print(tabulate(pd.DataFrame(conc_rows), headers='keys',
                   tablefmt='fancy_grid', showindex=False))

    # ── Key result summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("KEY RESULT SUMMARY  (corrected per reviewer comments)")
    print("="*65)
    t_nodes = TrafficNetworkSimulator.TOTAL_NODES
    e_nodes = EnvironmentalNetworkSimulator.TOTAL_NODES
    t_agg   = t_val['agg']
    e_agg   = e_val['agg']
    summary = [
        ['Threat Detection',        'Bayesian sliding-window Beta (ADACON Eq. 4-9)'],
        ['ADACON comparison baseline', f'Hysteresis-corrected: {ADACON_HYSTERESIS_BASELINE} switches'],
        ['── TRAFFIC  ({:,} nodes) ──'.format(t_nodes), ''],
        ['  Latency mean ± std',    f"{t_agg['latency_mean']['mean']:.4f} ± {t_agg['latency_mean']['std']:.4f}  "
                                    f"(95% CI {t_agg['latency_mean']['ci'][0]:.4f}–{t_agg['latency_mean']['ci'][1]:.4f})"],
        ['  Throughput mean ± std', f"{t_agg['throughput_mean']['mean']:.4f} ± {t_agg['throughput_mean']['std']:.4f}"],
        ['  Composite posture index',     f"{t_agg['posture_index_mean']['mean']*100:.2f}% ± {t_agg['posture_index_mean']['std']*100:.2f}%"],
        ['  Double-spend risk',     f"{t_agg['dsr_mean']['mean']:.4f} (lower is better)"],
        ['  Finality violation',    f"{t_agg['fv_mean']['mean']:.4f} (lower is better)"],
        ['  Switches (mean)',       f"{t_agg['n_switches']['mean']:.1f}  (vs ADACON {ADACON_HYSTERESIS_BASELINE})"],
        ['── ENVIRONMENTAL  ({:,} nodes) ──'.format(e_nodes), ''],
        ['  Latency mean ± std',    f"{e_agg['latency_mean']['mean']:.4f} ± {e_agg['latency_mean']['std']:.4f}"],
        ['  Throughput mean ± std', f"{e_agg['throughput_mean']['mean']:.4f} ± {e_agg['throughput_mean']['std']:.4f}"],
        ['  Composite posture index',     f"{e_agg['posture_index_mean']['mean']*100:.2f}% ± {e_agg['posture_index_mean']['std']*100:.2f}%"],
        ['  Double-spend risk',     f"{e_agg['dsr_mean']['mean']:.4f}"],
        ['  Finality violation',    f"{e_agg['fv_mean']['mean']:.4f}"],
        ['  Switches (mean)',       f"{e_agg['n_switches']['mean']:.1f}"],
    ]
    print(tabulate(summary, headers=['Metric', 'Value'], tablefmt='fancy_grid'))

    # ── Statistical analysis ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("STATISTICAL ANALYSIS  (primary seed=42)")
    print("="*65)
    _run_anova(traffic_results, env_results)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_results(traffic_results, env_results, traffic_static, env_static)
    _fig_concurrent(t_conc, e_conc)
    _fig_multi_seed(t_val, e_val)
    _fig_outcome_metrics(traffic_results, env_results)

    print("\nSimulation complete — all tables and figures generated.")
    return {'traffic_results': traffic_results, 'env_results': env_results}


def _fig_concurrent(t_conc: Dict, e_conc: Dict):
    """Figure 12 — Concurrent Sybil+DoS attack: mechanism timeline & posture."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Figure 12 — Concurrent Multi-Vector Attack (Sybil + DoS simultaneously)\n"
                 "Addresses reviewer concern: real adversaries deploy concurrent threats",
                 fontsize=12, fontweight='bold')

    mech_idx  = {ct.value: i for i, ct in enumerate(ConsensusType)}
    col_list  = list(COLORS.values())[:5]

    for row, (conc, title) in enumerate([(t_conc, 'Traffic Management (5,000 nodes)'),
                                          (e_conc, 'Environmental Monitoring (2,000 nodes)')]):
        cs, ce = conc['concurrent_start'], conc['concurrent_end']
        steps  = range(len(conc['history_mech']))

        # Left: mechanism timeline
        ax = axes[row][0]
        y  = [mech_idx[m.value] for m in conc['history_mech']]
        c  = [col_list[mech_idx[m.value]] for m in conc['history_mech']]
        ax.scatter(steps, y, c=c, s=5, alpha=0.8)
        ax.axvspan(cs, ce, color='red', alpha=0.12, label='Concurrent Sybil+DoS')
        ax.set_yticks(list(mech_idx.values()))
        ax.set_yticklabels([ct.value.split()[-1] for ct in ConsensusType], fontsize=9)
        ax.set_title(f'{title} — Mechanism Timeline', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Step'); ax.legend(fontsize=8)

        # Right: posture rate
        ax2 = axes[row][1]
        ax2.plot(steps, conc['history_prev'], color='#1abc9c', linewidth=1.2, alpha=0.85)
        ax2.axvspan(cs, ce, color='red', alpha=0.12, label='Concurrent Sybil+DoS')
        ax2.axhline(conc['avg_prev_attack'],   color='red',   linestyle='--', linewidth=1.2,
                    label=f"Concurrent avg: {conc['avg_prev_attack']*100:.1f}%")
        ax2.axhline(conc['avg_prev_baseline'], color='green', linestyle='--', linewidth=1.2,
                    label=f"Baseline avg: {conc['avg_prev_baseline']*100:.1f}%")
        ax2.set_ylim(0, 1.05); ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Prevention Rate'); ax2.legend(fontsize=8)
        ax2.set_title(f'{title} — Prevention Rate', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(_fig_path('Fig12_Concurrent_Attack.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig12_Concurrent_Attack.png")


def _fig_multi_seed(t_val: Dict, e_val: Dict):
    """Figure 13 — Multi-seed validation: 95% CI bar charts."""
    metrics = ['latency_mean', 'throughput_mean', 'posture_index_mean',
               'dsr_mean', 'fv_mean', 'liveness_mean']
    labels  = ['Latency', 'Throughput', 'Threat\nPrevention',
               'Double-Spend\nRisk', 'Finality\nViolation', 'Liveness']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 13 — Multi-Seed Statistical Validation (n=5 seeds: 5–9)\n"
                 "Error bars = 95% Confidence Interval",
                 fontsize=12, fontweight='bold')

    for ax, val, title in [(axes[0], t_val, 'Traffic Management (5,000 nodes)'),
                            (axes[1], e_val, 'Environmental Monitoring (2,000 nodes)')]:
        means  = [val['agg'][m]['mean']           for m in metrics]
        ci_lo  = [val['agg'][m]['mean'] - val['agg'][m]['ci'][0] for m in metrics]
        ci_hi  = [val['agg'][m]['ci'][1] - val['agg'][m]['mean'] for m in metrics]
        colors = ['#3498db','#2ecc71','#1abc9c','#e74c3c','#f39c12','#9b59b6']
        bars   = ax.bar(labels, means, color=colors, edgecolor='white',
                        linewidth=1.5, alpha=0.88, width=0.6)
        ax.errorbar(labels, means, yerr=[ci_lo, ci_hi], fmt='none',
                    color='black', capsize=5, linewidth=1.5)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel('Normalised Value')
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(_fig_path('Fig13_MultiSeed_Validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig13_MultiSeed_Validation.png")


def _fig_outcome_metrics(traffic_results: Dict, env_results: Dict):
    """Figure 14 — Concrete attack outcome metrics over time."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Figure 14 — Concrete Attack Outcome Metrics Over Time\n"
                 "(analytic outcome components: liveness, double-spend risk, finality violation)",
                 fontsize=12, fontweight='bold')

    atk_cols = ['#e74c3c','#f39c12','#2ecc71','#3498db','#9b59b6','#1abc9c']
    metric_keys   = ['double_spend_risk', 'finality_violation', 'liveness_score']
    metric_labels = ['Double-Spend Risk', 'Finality Violation Rate', 'Network Liveness']

    for row, (res, title) in enumerate([
            (traffic_results, 'Traffic Management (5,000 nodes)'),
            (env_results,     'Environmental Monitoring (2,000 nodes)')]):
        for col, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row][col]
            ax.plot(res['history'][key], color=atk_cols[col], linewidth=1.0, alpha=0.85)
            for step in res['attack_schedule']:
                ax.axvline(x=step, color='red', linestyle='--', alpha=0.4, linewidth=1.0)
            for step in res['recovery_schedule']:
                ax.axvline(x=step, color='green', linestyle=':', alpha=0.4, linewidth=1.0)
            ax.set_ylim(0, 1.05)
            ax.set_title(f'{title}\n{label}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Time Step'); ax.set_ylabel(label)
            ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(_fig_path('Fig14_Outcome_Metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig14_Outcome_Metrics.png")


def _run_anova(traffic_results: Dict, env_results: Dict):
    history = traffic_results['history']
    groups  = {}
    for m, lat in zip(history['mechanism'], history['latency']):
        groups.setdefault(m.value, []).append(lat)
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups.values())
        sig = '→ Significant' if p_val < 0.05 else '→ Not significant'
        print(f"  Traffic ANOVA — latency across mechanisms: "
              f"F = {f_stat:.2f}, p = {p_val:.4f}  {sig}")

    e_history = env_results['history']
    e_groups  = {}
    for m, tput in zip(e_history['mechanism'], e_history['throughput']):
        e_groups.setdefault(m.value, []).append(tput)
    if len(e_groups) > 1:
        h_stat, p_kw = stats.kruskal(*e_groups.values())
        sig = '→ Significant' if p_kw < 0.05 else '→ Not significant'
        print(f"  Environmental Kruskal-Wallis — throughput: "
              f"H = {h_stat:.2f}, p = {p_kw:.4f}  {sig}")


if __name__ == "__main__":
    results = main()
