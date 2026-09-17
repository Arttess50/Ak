"""
Nova Electronics — AI-Enabled Decision Support System
MBA final-assignment prototype. Case-only model.

Run:
    pip install -r requirements.txt
    streamlit run nova_electronics_dss.py

All numerical inputs below are transcribed from the uploaded Nova Electronics
case PDF / accompanying workbook. No external data are used.
"""
import itertools
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog
import plotly.express as px

st.set_page_config(
    page_title="Nova Electronics | Sourcing DSS",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CASE DATA — ONLY FROM THE UPLOADED NOVA ELECTRONICS CASE
# ---------------------------------------------------------------------------
PLANTS = ["P1", "P2", "P3"]
PLANT_NAMES = {"P1": "Pune", "P2": "Chennai", "P3": "Noida"}
DEMAND_BASE = {"P1": 18000.0, "P2": 15000.0, "P3": 12000.0}
SHORTAGE_LOSS = {"P1": 1800.0, "P2": 1800.0, "P3": 2050.0}
CRITICALITY = {"P1": "High", "P2": "Medium", "P3": "High"}

SUPPLIERS = ["S1", "S2", "S3", "S4", "S5"]
SUPPLIER_NAMES = {
    "S1": "Alpha", "S2": "Beta", "S3": "Gamma",
    "S4": "Delta", "S5": "Epsilon"
}
CAPACITY = {"S1": 16000.0, "S2": 14000.0, "S3": 13000.0, "S4": 12000.0, "S5": 15000.0}
PURCHASE = {"S1": 1020.0, "S2": 980.0, "S3": 1050.0, "S4": 950.0, "S5": 1090.0}
QUAL_COST = {"S1": 250000.0, "S2": 300000.0, "S3": 200000.0, "S4": 280000.0, "S5": 180000.0}
DISRUPTION_BASE = {"S1": 0.06, "S2": 0.12, "S3": 0.04, "S4": 0.18, "S5": 0.03}
YIELD_BASE = {"S1": 0.988, "S2": 0.992, "S3": 0.995, "S4": 0.985, "S5": 0.994}
LEAD_TIME = {"S1": 5, "S2": 7, "S3": 6, "S4": 4, "S5": 8}
RECOVERY = {"S1": 0.60, "S2": 0.50, "S3": 0.70, "S4": 0.40, "S5": 0.75}

FREIGHT = {
    "S1": {"P1": 45, "P2": 70, "P3": 60},
    "S2": {"P1": 55, "P2": 40, "P3": 85},
    "S3": {"P1": 70, "P2": 50, "P3": 75},
    "S4": {"P1": 35, "P2": 65, "P3": 90},
    "S5": {"P1": 80, "P2": 75, "P3": 40},
}

SPOT_PRICE_BASE = 1550.0
EMERGENCY_CAP = 8000.0

DEMAND_CONDITIONS = [("Weak", 0.20, 0.90), ("Normal", 0.55, 1.00), ("Strong", 0.25, 1.15)]
SEVERITY = [("Minor", 0.35, 0.60), ("Major", 0.45, 0.30), ("Severe", 0.20, 0.00)]
FREIGHT_LOW = 0.85
FREIGHT_HIGH = 1.15
YIELD_VARIATION = 0.005
COMMON_SHOCK_PROB = 0.08
COMMON_SHOCK_FREIGHT = 0.25

QUALIFICATION_PREFERENCE = 900000.0
ENGINEERING_MAX_SUPPLIERS = 4
PROCUREMENT_CONTRACTED_SHARE = 0.70

# Validation targets reported in the uploaded MBA case analysis.
# They are NOT used as model inputs or assumptions.
VALIDATION_TARGETS = {
    "A": {"normal": 48_940_000, "emergency": 837, "shortage": 181, "short_prob": 0.0470},
    "B": {"normal": 49_110_000, "emergency": 577, "shortage": 127, "short_prob": 0.0270},
    "C": {"normal": 49_340_000, "emergency": 40, "shortage": 6, "short_prob": 0.0021},
}

STRATEGIES = {
    "A": {"name": "Cost-focused", "suppliers": ["S1", "S2", "S3", "S4"], "concentration": 0.60},
    "B": {"name": "Moderate resilience", "suppliers": ["S1", "S2", "S4", "S5"], "concentration": 0.50},
    "C": {"name": "High resilience", "suppliers": ["S1", "S2", "S3", "S4", "S5"], "concentration": 0.40},
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def money(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x/1e7:.3f} crore"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:.2f} lakh"
    return f"₹{x:,.0f}"

def pct(x: float) -> str:
    return f"{100*x:.2f}%"

def num(x: float) -> str:
    return f"{x:,.0f}"

def get_demands(multipliers: Dict[str, float]) -> Dict[str, float]:
    return {p: DEMAND_BASE[p] * multipliers[p] for p in PLANTS}

def effective_landed_cost(s: str, p: str, freight_mult: float = 1.0, yld: float | None = None) -> float:
    y = YIELD_BASE[s] if yld is None else yld
    return (PURCHASE[s] + FREIGHT[s][p] * freight_mult) / y

def case_strategy_normal_cost(strategy_key: str) -> float:
    active = STRATEGIES[strategy_key]["suppliers"]
    conc = STRATEGIES[strategy_key]["concentration"]
    sol = optimize(active, conc, {"P1": 1.0, "P2": 1.0, "P3": 1.0},
                   DISRUPTION_BASE, 1.0, YIELD_BASE, SPOT_PRICE_BASE, EMERGENCY_CAP)
    return float(sol["objective"]) if sol["success"] else math.nan

# ---------------------------------------------------------------------------
# LP ENGINE
#
# x[s,p] is supplier input quantity. Good output = yield[s] * x[s,p].
# This follows the case-analysis methodology:
# Effective landed cost = (Purchase + Freight) / Good-quality yield.
# ---------------------------------------------------------------------------
def optimize(
    active_suppliers: List[str],
    concentration: float,
    demand_multipliers: Dict[str, float],
    disruption_probs: Dict[str, float],
    capacity_multipliers: Dict[str, float] | None = None,
    freight_mult: float = 1.0,
    yields: Dict[str, float] | None = None,
    spot_price: float = SPOT_PRICE_BASE,
    emergency_cap: float = EMERGENCY_CAP,
    include_qualification_cost: bool = True,
) -> Dict:
    active = set(active_suppliers)
    D = get_demands(demand_multipliers)
    cap_mult = capacity_multipliers or {s: 1.0 for s in SUPPLIERS}
    yld = yields or YIELD_BASE

    # 15 supplier-plant x variables, 3 emergency variables, 3 shortage variables.
    n = 21
    c = np.zeros(n)

    for si, s in enumerate(SUPPLIERS):
        for pi, p in enumerate(PLANTS):
            c[si * 3 + pi] = effective_landed_cost(s, p, freight_mult, yld[s])
    c[15:18] = [spot_price] * 3
    c[18:21] = [SHORTAGE_LOSS[p] for p in PLANTS]

    bounds = []
    for s in SUPPLIERS:
        for _p in PLANTS:
            bounds.append((0.0, None if s in active else 0.0))
    bounds.extend([(0.0, None)] * 6)

    Aub, bub = [], []

    # Supplier capacity is input-quantity capacity.
    for s in SUPPLIERS:
        row = np.zeros(n)
        si = SUPPLIERS.index(s)
        row[si * 3: si * 3 + 3] = 1.0
        Aub.append(row)
        bub.append(CAPACITY[s] * cap_mult.get(s, 1.0) if s in active else 0.0)

    # Concentration applies to good output, per the case-analysis formulation.
    for s in SUPPLIERS:
        for p in PLANTS:
            row = np.zeros(n)
            si, pi = SUPPLIERS.index(s), PLANTS.index(p)
            row[si * 3 + pi] = yld[s]
            Aub.append(row)
            bub.append(concentration * D[p])

    # Emergency market cap.
    row = np.zeros(n)
    row[15:18] = 1.0
    Aub.append(row)
    bub.append(emergency_cap)

    # Plant balance: good supplier output + emergency + unresolved shortage = demand.
    Aeq, beq = [], []
    for p in PLANTS:
        pi = PLANTS.index(p)
        row = np.zeros(n)
        for s in SUPPLIERS:
            si = SUPPLIERS.index(s)
            row[si * 3 + pi] = yld[s]
        row[15 + pi] = 1.0
        row[18 + pi] = 1.0
        Aeq.append(row)
        beq.append(D[p])

    res = linprog(
        c,
        A_ub=np.asarray(Aub),
        b_ub=np.asarray(bub),
        A_eq=np.asarray(Aeq),
        b_eq=np.asarray(beq),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return {
            "success": False,
            "message": res.message,
            "objective": math.inf,
            "allocation": pd.DataFrame(),
            "emergency": {},
            "shortage": {},
            "demands": D,
        }

    x = res.x[:15].reshape(5, 3)
    emergency = {p: float(res.x[15 + i]) for i, p in enumerate(PLANTS)}
    shortage = {p: float(res.x[18 + i]) for i, p in enumerate(PLANTS)}

    alloc_rows = []
    for si, s in enumerate(SUPPLIERS):
        row = {"Supplier": s, "Name": SUPPLIER_NAMES[s], "Capacity": CAPACITY[s] * cap_mult.get(s, 1.0)}
        total_input = 0.0
        total_good = 0.0
        for pi, p in enumerate(PLANTS):
            inp = float(x[si, pi])
            good = inp * yld[s]
            row[p] = good
            total_input += inp
            total_good += good
        row["Input Used"] = total_input
        row["Good Units"] = total_good
        row["Utilization"] = total_input / row["Capacity"] if row["Capacity"] else 0.0
        row["Status"] = "Active" if s in active else "OFF"
        row["Disruption Prob."] = disruption_probs.get(s, DISRUPTION_BASE[s])
        alloc_rows.append(row)

    objective = float(res.fun)
    if include_qualification_cost:
        objective += sum(QUAL_COST[s] for s in active)

    return {
        "success": True,
        "message": res.message,
        "objective": objective,
        "sourcing_cost": float(res.fun) - sum(spot_price * emergency[p] + SHORTAGE_LOSS[p] * shortage[p] for p in PLANTS),
        "allocation": pd.DataFrame(alloc_rows),
        "emergency": emergency,
        "shortage": shortage,
        "demands": D,
        "yields": yld,
    }

# ---------------------------------------------------------------------------
# SIMULATION ENGINE
# ---------------------------------------------------------------------------
def _simulate_one(
    active_suppliers: Tuple[str, ...],
    concentration: float,
    demand_multipliers: Dict[str, float],
    disruption_probs: Dict[str, float],
    spot_price: float,
    emergency_cap: float,
    rng: np.random.Generator,
):
    # One network-wide demand condition, matching the case's stated demand-condition structure.
    r = rng.random()
    cumulative = 0.0
    dm = 1.0
    for _name, prob, mult in DEMAND_CONDITIONS:
        cumulative += prob
        if r <= cumulative:
            dm = mult
            break
    scenario_demand_mult = {p: demand_multipliers[p] * dm for p in PLANTS}

    # Independent supplier disruption draws because the case supplies individual probabilities
    # but no supplier-to-supplier correlation matrix.
    cap_mult = {s: 1.0 for s in SUPPLIERS}
    disrupted = []
    severity_used = {}
    for s in SUPPLIERS:
        if s not in active_suppliers:
            cap_mult[s] = 0.0
            continue
        if rng.random() < disruption_probs[s]:
            rr = rng.random()
            if rr <= 0.35:
                sev, cm = "Minor", 0.60
            elif rr <= 0.80:
                sev, cm = "Major", 0.30
            else:
                sev, cm = "Severe", 0.00
            cap_mult[s] = cm
            disrupted.append(s)
            severity_used[s] = sev

    freight_mult = rng.uniform(FREIGHT_LOW, FREIGHT_HIGH)
    if rng.random() < COMMON_SHOCK_PROB:
        freight_mult *= (1.0 + COMMON_SHOCK_FREIGHT)

    yld = {s: float(np.clip(YIELD_BASE[s] + rng.uniform(-YIELD_VARIATION, YIELD_VARIATION), 0.0, 1.0)) for s in SUPPLIERS}

    sol = optimize(
        list(active_suppliers),
        concentration,
        scenario_demand_mult,
        disruption_probs,
        cap_mult,
        freight_mult,
        yld,
        spot_price,
        emergency_cap,
        include_qualification_cost=False,
    )

    return {
        "emergency_units": sum(sol["emergency"].values()),
        "shortage_units": sum(sol["shortage"].values()),
        "shortage": sol["shortage"],
        "emergency": sol["emergency"],
        "total_cost_before_qualification": sol["objective"],
        "disrupted": disrupted,
        "severity": severity_used,
        "demand_mult": dm,
        "freight_mult": freight_mult,
    }

@st.cache_data(show_spinner=False)
def run_simulation(
    active_tuple: Tuple[str, ...],
    concentration: float,
    demand_tuple: Tuple[float, float, float],
    disruption_tuple: Tuple[float, float, float, float, float],
    spot_price: float,
    emergency_cap: float,
    trials: int,
    seed: int,
):
    active = tuple(active_tuple)
    demand_multipliers = dict(zip(PLANTS, demand_tuple))
    disruption_probs = dict(zip(SUPPLIERS, disruption_tuple))
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(trials):
        rows.append(_simulate_one(
            active, concentration, demand_multipliers, disruption_probs,
            spot_price, emergency_cap, rng
        ))

    emergency_by_plant = {p: np.mean([r["emergency"][p] for r in rows]) for p in PLANTS}
    shortage_by_plant = {p: np.mean([r["shortage"][p] for r in rows]) for p in PLANTS}
    plant_service = {
        p: float(np.mean([r["shortage"][p] <= 1e-6 for r in rows])) for p in PLANTS
    }
    network_shortage_prob = float(np.mean([r["shortage_units"] > 1e-6 for r in rows]))
    expected_emergency = float(np.mean([r["emergency_units"] for r in rows]))
    expected_shortage = float(np.mean([r["shortage_units"] for r in rows]))
    expected_emergency_cost = expected_emergency * spot_price
    expected_shortage_cost = sum(shortage_by_plant[p] * SHORTAGE_LOSS[p] for p in PLANTS)

    # Reconstruct sourcing cost from total scenario objective by removing emergency/shortage.
    expected_sourcing_cost = float(np.mean([
        r["total_cost_before_qualification"] - r["emergency_units"] * spot_price
        - sum(r["shortage"][p] * SHORTAGE_LOSS[p] for p in PLANTS)
        for r in rows
    ]))

    return {
        "network_shortage_prob": network_shortage_prob,
        "expected_emergency": expected_emergency,
        "expected_shortage": expected_shortage,
        "expected_emergency_cost": expected_emergency_cost,
        "expected_shortage_cost": expected_shortage_cost,
        "expected_disruption_cost": expected_emergency_cost + expected_shortage_cost,
        "expected_sourcing_cost": expected_sourcing_cost,
        "plant_service": plant_service,
        "emergency_by_plant": emergency_by_plant,
        "shortage_by_plant": shortage_by_plant,
        "rows": rows,
    }

# ---------------------------------------------------------------------------
# DYNAMIC VULNERABILITY + QUERY ENGINE
# ---------------------------------------------------------------------------
def supplier_vulnerability(active, current_alloc, disruption_probs, concentration, demand_mults):
    D = get_demands(demand_mults)
    scores = []
    for s in active:
        row = current_alloc[current_alloc["Supplier"] == s].iloc[0]
        score_good = sum(float(row[p]) for p in PLANTS)
        # Expected lost good supply before contingency response:
        # expected loss fraction conditional on failure = 0.35*0.40 + 0.45*0.70 + 0.20*1 = 0.655
        expected_lost = score_good * disruption_probs[s] * 0.655
        plant_exposure = max(
            float(row[p]) * disruption_probs[s] * 0.655 * SHORTAGE_LOSS[p] for p in PLANTS
        )
        scores.append({
            "Supplier": s,
            "Name": SUPPLIER_NAMES[s],
            "Allocation (good)": score_good,
            "Disruption probability": disruption_probs[s],
            "Expected lost units": expected_lost,
            "Max plant exposure": plant_exposure,
        })
    return pd.DataFrame(scores).sort_values(
        ["Max plant exposure", "Expected lost units"], ascending=False
    ).reset_index(drop=True)

def plant_vulnerability(active, current_alloc, disruption_probs, demand_mults):
    D = get_demands(demand_mults)
    rows = []
    for p in PLANTS:
        exposure = 0.0
        for s in active:
            alloc = float(current_alloc.loc[current_alloc["Supplier"] == s, p].iloc[0])
            exposure += alloc * disruption_probs[s] * 0.655
        rows.append({
            "Plant": p,
            "Name": PLANT_NAMES[p],
            "Demand": D[p],
            "Expected lost good units": exposure,
            "Loss/unit": SHORTAGE_LOSS[p],
            "Risk-weighted shortage exposure (₹)": exposure * SHORTAGE_LOSS[p],
        })
    return pd.DataFrame(rows).sort_values(
        "Risk-weighted shortage exposure (₹)", ascending=False
    ).reset_index(drop=True)

def recommendation(strategy_key, normal, sim, baseline_normal, concentration, active):
    premium = normal["objective"] - baseline_normal
    if strategy_key == "C" and len(active) == 5:
        return (
            f"Maintain the high-resilience configuration. Normal-state cost is {money(normal['objective'])}, "
            f"a {money(premium)} premium versus the cost-focused baseline. The simulation estimates "
            f"{num(sim['expected_shortage'])} unresolved shortage units/month and {num(sim['expected_emergency'])} "
            f"emergency units, indicating that the resilience premium is buying materially lower disruption exposure."
        )
    if "S4" not in active:
        return (
            f"S4 is currently OFF. Re-optimization has shifted volume across the remaining qualified suppliers. "
            f"Expected shortage is {num(sim['expected_shortage'])} units/month and emergency sourcing is "
            f"{num(sim['expected_emergency'])} units. Review P1 first because it is a high-criticality plant."
        )
    return (
        f"The current scenario produces a normal landed cost of {money(normal['objective'])} and expected "
        f"disruption cost of {money(sim['expected_disruption_cost'])}. The model recommends keeping the active "
        f"qualified set while monitoring concentration and the service probability of P1 and P3."
    )

def answer_query(query, strategy_key, normal, sim, baseline_normal, active, concentration, demand_mults, disruption_probs):
    q = query.lower().strip()
    alloc = normal["allocation"]
    vuln_s = supplier_vulnerability(active, alloc, disruption_probs, concentration, demand_mults)
    vuln_p = plant_vulnerability(active, alloc, disruption_probs, demand_mults)

    if "most critical" in q and "p1" in q:
        # Rank suppliers by P1 risk-weighted exposure, not probability alone.
        rows = []
        for s in active:
            a = float(alloc.loc[alloc["Supplier"] == s, "P1"].iloc[0])
            lost = a * disruption_probs[s] * 0.655
            rows.append((s, lost * SHORTAGE_LOSS["P1"], a, lost))
        rows.sort(reverse=True)
        s, cost_exp, a, lost = rows[0]
        return (
            f"{s} {SUPPLIER_NAMES[s]} is currently most critical to P1. "
            f"It supplies {num(a)} good units to P1; at the current disruption probability of "
            f"{pct(disruption_probs[s])}, the case severity distribution implies about {num(lost)} "
            f"expected lost P1 units before contingency response, with risk-weighted exposure of "
            f"{money(cost_exp)}. This ranking uses allocation exposure + disruption probability + the case "
            f"loss framework, rather than disruption probability alone."
        )

    if "s4" in q and "unavailable" in q:
        active2 = tuple(s for s in active if s != "S4")
        if len(active2) == 0:
            return "S4 cannot be disabled when it is the only remaining qualified supplier."
        alt = optimize(list(active2), concentration, demand_mults, disruption_probs,
                       {s: 1.0 for s in SUPPLIERS}, 1.0, YIELD_BASE,
                       SPOT_PRICE_BASE, EMERGENCY_CAP, include_qualification_cost=True)
        alt_sim = run_simulation(
            active2, concentration,
            tuple(demand_mults[p] for p in PLANTS),
            tuple(disruption_probs[s] for s in SUPPLIERS),
            spot_price, EMERGENCY_CAP, 10000, 20260914
        )
        return (
            f"With S4 unavailable, the re-optimized normal cost becomes {money(alt['objective'])}. "
            f"The 10,000-trial risk run estimates {num(alt_sim['expected_emergency'])} emergency units/month, "
            f"{num(alt_sim['expected_shortage'])} shortage units/month and a "
            f"{pct(alt_sim['network_shortage_prob'])} network shortage probability. "
            f"P1 service is {pct(alt_sim['plant_service']['P1'])}; P3 service is "
            f"{pct(alt_sim['plant_service']['P3'])}."
        )

    if "s4" in q and ("30%" in q or "30" in q) and ("probability" in q or "disruption" in q):
        probs2 = dict(disruption_probs)
        probs2["S4"] = 0.30
        sim2 = run_simulation(
            tuple(active), concentration,
            tuple(demand_mults[p] for p in PLANTS),
            tuple(probs2[s] for s in SUPPLIERS),
            spot_price, EMERGENCY_CAP, 10000, 20260914
        )
        return (
            f"At S4 disruption probability = 30%, the simulation estimates "
            f"{num(sim2['expected_shortage'])} shortage units/month, {num(sim2['expected_emergency'])} "
            f"emergency units/month and {pct(sim2['network_shortage_prob'])} network shortage probability. "
            f"P1 service is {pct(sim2['plant_service']['P1'])}. The model should be used to compare this "
            f"scenario with the current probability of {pct(disruption_probs['S4'])}."
        )

    if "40%" in q and "concentration" in q:
        alt = optimize(list(active), 0.40, demand_mults, disruption_probs,
                       {s: 1.0 for s in SUPPLIERS}, 1.0, YIELD_BASE,
                       SPOT_PRICE_BASE, EMERGENCY_CAP, include_qualification_cost=True)
        sim2 = run_simulation(
            tuple(active), 0.40,
            tuple(demand_mults[p] for p in PLANTS),
            tuple(disruption_probs[s] for s in SUPPLIERS),
            SPOT_PRICE_BASE, EMERGENCY_CAP, 10000, 20260914
        )
        return (
            f"At a 40% concentration cap, normal landed cost is {money(alt['objective'])}; "
            f"the simulation estimates {num(sim2['expected_shortage'])} shortage units/month, "
            f"{num(sim2['expected_emergency'])} emergency units/month and "
            f"{pct(sim2['network_shortage_prob'])} network shortage probability."
        )

    if "most critical overall" in q:
        r = vuln_s.iloc[0]
        return (
            f"{r['Supplier']} {r['Name']} is currently the most critical active supplier overall by the "
            f"model's risk-weighted exposure. Its current good-unit allocation is {num(r['Allocation (good)'])}, "
            f"disruption probability is {pct(r['Disruption probability'])}, and expected lost supply before "
            f"contingency response is about {num(r['Expected lost units'])} units."
        )

    if "most vulnerable plant" in q:
        r = vuln_p.iloc[0]
        return (
            f"{r['Plant']} {r['Name']} is currently the most vulnerable plant by risk-weighted exposure: "
            f"{num(r['Expected lost good units'])} expected lost good units before contingency response, "
            f"equivalent to {money(r['Risk-weighted shortage exposure (₹)'])} of shortage-loss exposure."
        )

    if "backup" in q:
        # Backup choice is based on low disruption probability + emergency recovery + network fit.
        rows = []
        for s in active:
            rows.append((
                disruption_probs[s], -RECOVERY[s], -QUAL_COST[s], s
            ))
        rows.sort()
        s = rows[0][3]
        return (
            f"{s} {SUPPLIER_NAMES[s]} is the strongest backup candidate among the current active suppliers "
            f"on the case inputs: disruption probability {pct(disruption_probs[s])}, emergency recovery "
            f"{pct(RECOVERY[s])} of capacity after one week, and qualification cost {money(QUAL_COST[s])}. "
            f"The allocation fit should still be checked under the current concentration cap."
        )

    if "resilience premium" in q or "justified" in q:
        premium = normal["objective"] - baseline_normal
        return (
            f"The current normal-state resilience premium is {money(premium)} "
            f"({pct(premium / baseline_normal)} of the cost-focused baseline). "
            f"The simulation estimates expected disruption cost of {money(sim['expected_disruption_cost'])}. "
            f"Compare the premium with the baseline disruption-cost component before making the decision."
        )

    if "cheapest" in q:
        vals = {k: case_strategy_normal_cost(k) for k in STRATEGIES}
        k = min(vals, key=vals.get)
        return f"Among the three case-defined strategies, {k} — {STRATEGIES[k]['name']} has the lowest modeled normal landed cost at {money(vals[k])}."

    if "most resilient" in q:
        return "Strategy C — High resilience is the most resilient of the three case-defined strategies because it qualifies all five suppliers and uses the lowest concentration cap (40%)."

    return (
        "I can answer model-based questions such as: Which supplier is most critical to P1? "
        "What if S4 is unavailable? What if S4 disruption probability rises to 30%? "
        "What if maximum concentration changes to 40%? Which plant is most vulnerable? "
        "Is the resilience premium justified?"
    )

# ---------------------------------------------------------------------------
# SIDEBAR — MANAGER CONTROLS
# ---------------------------------------------------------------------------
st.title("Nova Electronics — Cost–Resilience Sourcing DSS")
st.caption("Manager-facing prototype | Case-only numerical model | LP + Monte Carlo + model-based Q&A")

with st.sidebar:
    st.header("Scenario Controls")

    strategy_key = st.selectbox(
        "Sourcing philosophy",
        options=list(STRATEGIES.keys()),
        format_func=lambda k: f"{k} — {STRATEGIES[k]['name']}",
        index=2,
    )

    st.subheader("Supplier availability")
    default_active = STRATEGIES[strategy_key]["suppliers"]
    active = []
    for s in SUPPLIERS:
        on = st.checkbox(
            f"{s} — {SUPPLIER_NAMES[s]}",
            value=s in default_active,
            key=f"active_{s}",
        )
        if on:
            active.append(s)

    if len(active) < 2:
        st.warning("Fewer than two qualified suppliers remain. The case's high-criticality dual-source preference cannot be protected.")

    st.subheader("Disruption probability")
    disruption_probs = {}
    for s in SUPPLIERS:
        disruption_probs[s] = st.slider(
            f"{s} base {pct(DISRUPTION_BASE[s])}",
            0.0, 1.0, float(DISRUPTION_BASE[s]), 0.01,
            format="%.0f%%",
            key=f"prob_{s}",
        )

    st.subheader("Plant demand multiplier")
    demand_mults = {}
    for p in PLANTS:
        demand_mults[p] = st.number_input(
            f"{p} — {PLANT_NAMES[p]}",
            min_value=0.50, max_value=1.50, value=1.00, step=0.05,
            key=f"dm_{p}",
        )

    spot_price = st.number_input(
        "Emergency spot price (₹/unit)",
        min_value=0.0, value=float(SPOT_PRICE_BASE), step=50.0,
    )

    concentration = st.slider(
        "Maximum supplier concentration",
        min_value=0.20, max_value=1.00,
        value=float(STRATEGIES[strategy_key]["concentration"]),
        step=0.05,
        format="%.0f%%",
    )

    trials = st.number_input(
        "Monte Carlo trials",
        min_value=1000, max_value=10000, value=10000, step=1000,
        help="Default is 10,000 as required by the case assignment.",
    )

    seed = st.number_input("Simulation seed", min_value=1, max_value=999999999, value=20260914, step=1)

    if len(active) > ENGINEERING_MAX_SUPPLIERS:
        st.info(f"Engineering preference is ≤{ENGINEERING_MAX_SUPPLIERS} qualified suppliers unless resilience benefit is material.")
    qual_total = sum(QUAL_COST[s] for s in active)
    if qual_total > QUALIFICATION_PREFERENCE:
        st.warning(f"Qualification cost is {money(qual_total)} vs the case preference of {money(QUALIFICATION_PREFERENCE)}. The case says this is a preference, not a hard ceiling.")

# ---------------------------------------------------------------------------
# CURRENT NORMAL OPTIMIZATION
# ---------------------------------------------------------------------------
if not active:
    st.error("At least one supplier must be active.")
    st.stop()

normal = optimize(
    active, concentration, demand_mults, disruption_probs,
    {s: 1.0 for s in SUPPLIERS}, 1.0, YIELD_BASE,
    spot_price, EMERGENCY_CAP, include_qualification_cost=True
)
if not normal["success"]:
    st.error(f"No feasible solution under the current controls: {normal['message']}")
    st.stop()

baseline_normal = case_strategy_normal_cost("A")

sim = run_simulation(
    tuple(active), concentration,
    tuple(demand_mults[p] for p in PLANTS),
    tuple(disruption_probs[s] for s in SUPPLIERS),
    spot_price, EMERGENCY_CAP, int(trials), int(seed),
)

premium = normal["objective"] - baseline_normal
premium_pct = premium / baseline_normal if baseline_normal else math.nan

# ---------------------------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------------------------
st.markdown("### Management Dashboard")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Normal landed cost", money(normal["objective"]), f"{money(premium)} vs A")
k2.metric("Expected total cost", money(sim["expected_sourcing_cost"] + sim["expected_disruption_cost"] + qual_total))
k3.metric("Expected shortage", f"{num(sim['expected_shortage'])} units", f"{pct(sim['network_shortage_prob'])} shortage probability")
k4.metric("Expected emergency", f"{num(sim['expected_emergency'])} units", money(sim["expected_emergency_cost"]))

k5, k6, k7, k8 = st.columns(4)
k5.metric("P1 service probability", pct(sim["plant_service"]["P1"]))
k6.metric("P2 service probability", pct(sim["plant_service"]["P2"]))
k7.metric("P3 service probability", pct(sim["plant_service"]["P3"]))
k8.metric("Resilience premium", money(premium), f"{pct(premium_pct)} vs A")

# ---------------------------------------------------------------------------
# ALLOCATION + CONCENTRATION
# ---------------------------------------------------------------------------
left, right = st.columns([1.25, 1])

with left:
    st.markdown("### Optimal Allocation")
    alloc = normal["allocation"].copy()
    display = alloc[["Supplier", "Name", "Capacity", "P1", "P2", "P3", "Utilization", "Status", "Disruption Prob."]].copy()
    display["Capacity"] = display["Capacity"].map(lambda x: f"{x:,.0f}")
    for p in PLANTS:
        display[p] = display[p].map(lambda x: f"{x:,.0f}")
    display["Utilization"] = display["Utilization"].map(lambda x: f"{100*x:.1f}%")
    display["Disruption Prob."] = display["Disruption Prob."].map(lambda x: f"{100*x:.0f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)

    conc_rows = []
    for p in PLANTS:
        D = normal["demands"][p]
        for _, r in alloc.iterrows():
            share = float(r[p]) / D if D else 0.0
            if share > 1e-9:
                conc_rows.append({"Plant": p, "Supplier": r["Supplier"], "Share": share})
    conc_df = pd.DataFrame(conc_rows)
    if not conc_df.empty:
        fig = px.bar(
            conc_df, x="Plant", y="Share", color="Supplier",
            barmode="stack", text_auto=".0%",
            title="Supplier concentration by plant",
        )
        fig.add_hline(y=concentration, line_dash="dash", annotation_text=f"Cap {pct(concentration)}")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Service & Shortage by Plant")
    service_df = pd.DataFrame({
        "Plant": PLANTS,
        "Plant Name": [PLANT_NAMES[p] for p in PLANTS],
        "Service Probability": [sim["plant_service"][p] for p in PLANTS],
        "Expected Shortage": [sim["shortage_by_plant"][p] for p in PLANTS],
        "Expected Emergency": [sim["emergency_by_plant"][p] for p in PLANTS],
    })
    st.dataframe(
        service_df.assign(
            **{"Service Probability": service_df["Service Probability"].map(lambda x: f"{x:.2%}")},
            **{"Expected Shortage": service_df["Expected Shortage"].map(lambda x: f"{x:,.1f}")},
            **{"Expected Emergency": service_df["Expected Emergency"].map(lambda x: f"{x:,.1f}")},
        ),
        use_container_width=True, hide_index=True
    )

    fig2 = px.bar(
        service_df, x="Plant", y="Service Probability",
        text_auto=".2%", title="Probability each plant meets demand"
    )
    fig2.update_yaxes(tickformat=".0%", range=[0, 1.02])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Emergency & Shortage Cost")
    st.write(f"Emergency purchase cost: **{money(sim['expected_emergency_cost'])} / month**")
    st.write(f"Unresolved shortage cost: **{money(sim['expected_shortage_cost'])} / month**")
    st.write(f"Total disruption-cost component: **{money(sim['expected_disruption_cost'])} / month**")

# ---------------------------------------------------------------------------
# RISK ANALYSIS
# ---------------------------------------------------------------------------
st.markdown("### Network Vulnerability")
v1, v2 = st.columns(2)

with v1:
    sv = supplier_vulnerability(active, alloc, disruption_probs, concentration, demand_mults)
    st.dataframe(
        sv.assign(
            **{
                "Allocation (good)": sv["Allocation (good)"].map(lambda x: f"{x:,.0f}"),
                "Disruption probability": sv["Disruption probability"].map(lambda x: f"{x:.1%}"),
                "Expected lost units": sv["Expected lost units"].map(lambda x: f"{x:,.0f}"),
                "Max plant exposure": sv["Max plant exposure"].map(money),
            }
        ),
        use_container_width=True, hide_index=True
    )
    st.caption("Dynamic supplier vulnerability combines current allocation exposure, disruption probability and the case severity distribution.")

with v2:
    pv = plant_vulnerability(active, alloc, disruption_probs, demand_mults)
    st.dataframe(
        pv.assign(
            Demand=pv["Demand"].map(lambda x: f"{x:,.0f}"),
            **{
                "Expected lost good units": pv["Expected lost good units"].map(lambda x: f"{x:,.0f}"),
                "Risk-weighted shortage exposure (₹)": pv["Risk-weighted shortage exposure (₹)"].map(money),
            }
        ),
        use_container_width=True, hide_index=True
    )

# ---------------------------------------------------------------------------
# BASELINE COMPARISON
# ---------------------------------------------------------------------------
st.markdown("### Baseline Comparison — Cost-Focused Strategy A")
baseline_sim = run_simulation(
    tuple(STRATEGIES["A"]["suppliers"]), 0.60,
    (1.0, 1.0, 1.0),
    tuple(DISRUPTION_BASE[s] for s in SUPPLIERS),
    SPOT_PRICE_BASE, EMERGENCY_CAP, 10000, 20260914
)
comparison = pd.DataFrame([
    ["Normal landed cost", baseline_normal, normal["objective"], normal["objective"] - baseline_normal],
    ["Expected shortage", baseline_sim["expected_shortage"], sim["expected_shortage"], sim["expected_shortage"] - baseline_sim["expected_shortage"]],
    ["Emergency purchases", baseline_sim["expected_emergency"], sim["expected_emergency"], sim["expected_emergency"] - baseline_sim["expected_emergency"]],
    ["P1 service probability", baseline_sim["plant_service"]["P1"], sim["plant_service"]["P1"], sim["plant_service"]["P1"] - baseline_sim["plant_service"]["P1"]],
    ["P2 service probability", baseline_sim["plant_service"]["P2"], sim["plant_service"]["P2"], sim["plant_service"]["P2"] - baseline_sim["plant_service"]["P2"]],
    ["P3 service probability", baseline_sim["plant_service"]["P3"], sim["plant_service"]["P3"], sim["plant_service"]["P3"] - baseline_sim["plant_service"]["P3"]],
    ["Resilience premium", 0.0, premium, premium],
], columns=["Metric", "Baseline A", "Current Scenario", "Change"])

def format_metric_row(row):
    if "probability" in row["Metric"].lower():
        return [row["Metric"], pct(row["Baseline A"]), pct(row["Current Scenario"]), pct(row["Change"])]
    if "cost" in row["Metric"].lower() or "premium" in row["Metric"].lower():
        return [row["Metric"], money(row["Baseline A"]), money(row["Current Scenario"]), money(row["Change"])]
    return [row["Metric"], num(row["Baseline A"]), num(row["Current Scenario"]), num(row["Change"])]

formatted = pd.DataFrame([format_metric_row(r) for _, r in comparison.iterrows()],
                         columns=comparison.columns)
st.dataframe(formatted, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# STRATEGY BENCHMARK
# ---------------------------------------------------------------------------
st.markdown("### Case-Defined Strategy Comparison")
strategy_rows = []
for k, cfg in STRATEGIES.items():
    c = case_strategy_normal_cost(k)
    strategy_rows.append({
        "Strategy": f"{k} — {cfg['name']}",
        "Qualified": ", ".join(cfg["suppliers"]),
        "Concentration": cfg["concentration"],
        "Normal landed cost": c,
        "Qualification cost": sum(QUAL_COST[s] for s in cfg["suppliers"]),
        "Resilience premium vs A": c - baseline_normal,
    })
strategy_df = pd.DataFrame(strategy_rows)
st.dataframe(
    strategy_df.assign(
        Concentration=strategy_df["Concentration"].map(lambda x: f"{x:.0%}"),
        **{
            "Normal landed cost": strategy_df["Normal landed cost"].map(money),
            "Qualification cost": strategy_df["Qualification cost"].map(money),
            "Resilience premium vs A": strategy_df["Resilience premium vs A"].map(money),
        }
    ),
    use_container_width=True, hide_index=True
)

# ---------------------------------------------------------------------------
# AI-STYLE MANAGEMENT Q&A
# ---------------------------------------------------------------------------
st.markdown("### AI Management Query")
st.caption("Answers are generated from the current LP/simulation outputs and case data; this prototype does not invent external facts.")

query = st.text_input(
    "Ask the DSS",
    placeholder="e.g., Which supplier is most critical to P1?",
)
if query:
    answer = answer_query(
        query, strategy_key, normal, sim, baseline_normal, active,
        concentration, demand_mults, disruption_probs
    )
    st.info(answer)

# ---------------------------------------------------------------------------
# MANAGEMENT RECOMMENDATION
# ---------------------------------------------------------------------------
st.markdown("### Management Recommendation")
st.success(
    recommendation(strategy_key, normal, sim, baseline_normal, concentration, active)
)

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
with st.expander("Model validation against uploaded case-analysis targets"):
    st.write(
        "The targets below come from the uploaded MBA case analysis and are used only as validation references. "
        "They are not hard-coded into the optimization or simulation."
    )
    val_key = strategy_key
    target = VALIDATION_TARGETS[val_key]
    current_cost = case_strategy_normal_cost(val_key)
    st.table(pd.DataFrame({
        "Metric": ["Normal cost", "Expected emergency", "Expected shortage", "Network shortage probability"],
        "Case-analysis validation target": [
            money(target["normal"]), f"{target['emergency']:,.0f}", f"{target['shortage']:,.0f}", pct(target["short_prob"])
        ],
        "Current DSS model": [
            money(current_cost),
            f"{sim['expected_emergency']:,.0f}",
            f"{sim['expected_shortage']:,.0f}",
            pct(sim["network_shortage_prob"])
        ],
    }))
    st.warning(
        "Normal-cost reconciliation: the uploaded workbook explicitly notes that its formula-traceable "
        "landed-cost build differs from the report's headline figures by roughly ₹5 lakh because of an "
        "arithmetic/reconciliation difference. The DSS follows the stated effective-cost methodology and "
        "case inputs rather than hard-coding the report headline."
    )

# ---------------------------------------------------------------------------
# MODEL NOTES
# ---------------------------------------------------------------------------
with st.expander("Case-only modelling notes"):
    st.markdown(
        """
- Supplier disruptions are simulated independently because the case gives individual disruption probabilities but no supplier-correlation matrix.
- The case's 8% common logistics shock is modelled separately as a shared freight event.
- Demand is sampled from Weak/Normal/Strong using the case probabilities and multipliers.
- Disruption severity is sampled from Minor/Major/Severe using the case conditional probabilities.
- Freight varies from 0.85× to 1.15×; quality yield varies by ±0.5 percentage points.
- Emergency sourcing is capped at 8,000 units/month and used after qualified capacity/concentration constraints in the optimization.
- Safety stock is not included in the core optimization because the case places it outside the core model.
- Qualification cost is treated as a fixed monthly cost for active/qualified suppliers.
- If a requested item is not specified by the case, it is not modelled.
        """
    )

st.caption("Nova Electronics DSS • Case-only MBA prototype • No external web data used")
