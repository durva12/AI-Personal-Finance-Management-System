"""
Dynamic AI Recommendation Engine
==================================
Replaces the static percentage-based recommendation logic.

Core principles:
  - Only recommends changes when a category is genuinely problematic
  - Uses actual spending percentages vs adaptive thresholds (city-aware)
  - Analyses historical trends to detect spikes and drift
  - Clusters peer behaviour to contextualise "normal"
  - Never recommends negative values or unsafe cuts to protected categories
  - Generates rich explanations with priority + monthly impact
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from models.rl_finance_agent import (
    CATEGORIES, IDEAL_RATIOS, CITY_MULTIPLIERS,
    compute_health_score, score_label,
)

# ── Protected categories — never cut below these income ratios ────────────────
PROTECTED_MINIMUMS = {
    "Housing":        0.10,   # always need shelter
    "Food":           0.07,   # always need food
    "Healthcare":     0.02,   # always need basic care
    "Debt Repayment": 0.00,   # cannot cut someone else's obligation
}

# ── Thresholds for triggering a recommendation (% of income) ─────────────────
# A category is flagged only when current % exceeds IDEAL_RATIOS[cat][1] + tolerance
TOLERANCE = {
    "Housing":        0.05,   # city-adjusted later
    "Food":           0.03,
    "Transport":      0.03,
    "Entertainment":  0.02,
    "Healthcare":     0.02,   # only recommend reduce if really excessive
    "Savings":        0.00,   # always try to improve savings
    "Debt Repayment": 0.04,
    "Miscellaneous":  0.02,
}

# Minimum absolute difference (₹) before we bother recommending
MIN_IMPACT_INR = 200

# Savings target
SAVINGS_TARGET_RATIO = 0.20

# Category display metadata
CAT_META = {
    "Housing":        {"icon": "🏠", "tips": ["Consider roommates or co-living", "Negotiate lease renewal", "Claim HRA tax benefit"]},
    "Food":           {"icon": "🍽️", "tips": ["Cook at home 4+ days a week", "Reduce food delivery apps", "Buy groceries in bulk"]},
    "Transport":      {"icon": "🚗", "tips": ["Use metro/bus pass instead of cabs", "Carpool with colleagues", "Work-from-home days save commute cost"]},
    "Entertainment":  {"icon": "🎬", "tips": ["Set a monthly entertainment envelope", "Use free/shared streaming plans", "Prioritise experiences over impulse purchases"]},
    "Healthcare":     {"icon": "🏥", "tips": ["Use generic medicines", "Employer health insurance may cover more", "Annual checkups prevent expensive emergencies"]},
    "Savings":        {"icon": "💰", "tips": ["Automate SIP on salary day", "Start with ₹1,000 more than current", "Open a separate high-yield savings account"]},
    "Debt Repayment": {"icon": "💳", "tips": ["Avalanche method: pay highest-rate debt first", "Negotiate EMI restructuring", "Avoid new loans until ratio is under 15%"]},
    "Miscellaneous":  {"icon": "🛒", "tips": ["Track every rupee for 30 days", "Question every purchase over ₹500", "Cancel unused subscriptions"]},
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    category:         str
    icon:             str
    current:          float
    recommended:      float
    difference:       float          # positive = increase, negative = reduce
    current_pct:      float
    recommended_pct:  float
    reason:           str
    detail:           str
    tips:             list[str]
    priority:         str            # "High" | "Medium" | "Low"
    monthly_impact:   float          # absolute ₹ saved/gained per month
    action:           str            # "increase" | "decrease" | "maintain"
    is_fixed:         bool = False
    trend_flag:       str = ""       # "spike" | "rising" | "falling" | ""
    trend_detail:     str = ""

    def to_dict(self) -> dict:
        return {
            "category":        self.category,
            "icon":            self.icon,
            "current":         round(self.current,        2),
            "recommended":     round(self.recommended,    2),
            "difference":      round(abs(self.difference), 2),
            "current_pct":     round(self.current_pct,    1),
            "recommended_pct": round(self.recommended_pct, 1),
            "reason":          self.reason,
            "detail":          self.detail,
            "tips":            self.tips,
            "priority":        self.priority,
            "monthly_impact":  round(self.monthly_impact,  2),
            "action":          self.action,
            "is_fixed":        self.is_fixed,
            "trend_flag":      self.trend_flag,
            "trend_detail":    self.trend_detail,
            # legacy fields so existing budget table still works
            "explanation":     self.reason,
            "is_good":         self.action == "maintain",
        }


@dataclass
class HealthyCategory:
    """Category that needs no action — still shown for transparency."""
    category:     str
    icon:         str
    current:      float
    current_pct:  float
    status:       str   # "good" | "excellent"
    message:      str

    def to_dict(self) -> dict:
        return {
            "category":    self.category,
            "icon":        self.icon,
            "current":     round(self.current, 2),
            "current_pct": round(self.current_pct, 1),
            "status":      self.status,
            "message":     self.message,
        }


# ── Trend Analyser ────────────────────────────────────────────────────────────

def analyse_trends(history: list[dict], income: float) -> dict[str, dict]:
    """
    Given a list of past FinanceAnalysis records (newest first),
    return per-category trend info.

    Each record's `expenses` is a JSON dict {category: amount}.
    Returns: { category: { "trend": "spike"|"rising"|"falling"|"stable",
                            "avg_3m": float, "avg_6m": float,
                            "pct_change_3m": float, "detail": str } }
    """
    if not history:
        return {}

    results: dict[str, dict] = {}
    for cat in CATEGORIES:
        vals = []
        for rec in history:
            exp = rec.get("expenses") or {}
            if isinstance(exp, str):
                import json
                try:
                    exp = json.loads(exp)
                except Exception:
                    exp = {}
            v = float(exp.get(cat, 0))
            vals.append(v)

        if not vals:
            continue

        avg_all = sum(vals) / len(vals)
        avg_3m  = sum(vals[:3])  / min(3, len(vals))
        avg_6m  = sum(vals[:6])  / min(6, len(vals))

        pct_change_3m = ((avg_3m - avg_6m) / max(avg_6m, 1)) * 100 if len(vals) >= 3 else 0.0

        if pct_change_3m >= 40:
            trend  = "spike"
            detail = f"spending jumped {pct_change_3m:.0f}% vs your 6-month average"
        elif pct_change_3m >= 15:
            trend  = "rising"
            detail = f"spending up {pct_change_3m:.0f}% over the last 3 months"
        elif pct_change_3m <= -15:
            trend  = "falling"
            detail = f"spending down {abs(pct_change_3m):.0f}% — good progress"
        else:
            trend  = "stable"
            detail = "spending is consistent with your recent history"

        results[cat] = {
            "trend":          trend,
            "avg_3m":         round(avg_3m,  2),
            "avg_6m":         round(avg_6m,  2),
            "avg_all":        round(avg_all, 2),
            "pct_change_3m":  round(pct_change_3m, 1),
            "detail":         detail,
        }

    return results


# ── Main Engine ───────────────────────────────────────────────────────────────

class DynamicRecommendationEngine:
    """
    Generates smart, context-aware recommendations.
    Only flags categories that genuinely need attention.
    """

    def __init__(self):
        pass

    def _adj_hi(self, cat: str, city: str) -> float:
        """City-adjusted upper threshold."""
        lo, hi = IDEAL_RATIOS[cat]
        if cat in ("Housing", "Transport"):
            return hi * CITY_MULTIPLIERS.get(city, 1.0)
        return hi

    def _priority(self, overshoot_ratio: float, monthly_impact: float,
                  trend: str = "") -> str:
        """
        overshoot_ratio: how far above the limit as a fraction (0.5 = 50% over)
        monthly_impact:  ₹ difference
        trend:           "spike" boosts to High
        """
        if trend == "spike" or overshoot_ratio >= 0.5 or monthly_impact >= 3000:
            return "High"
        if overshoot_ratio >= 0.25 or monthly_impact >= 1000:
            return "Medium"
        return "Low"

    def generate(
        self,
        income:           float,
        expenses:         dict,
        city:             str             = "Other",
        fixed_categories: list            = None,
        history:          list[dict]      = None,
        cluster_label:    str             = "",
    ) -> dict:
        """
        Main entry point. Returns:
          {
            "recommendations": [ Recommendation.to_dict(), ... ],  # only problem areas
            "healthy":         [ HealthyCategory.to_dict(), ... ],  # all-clear categories
            "budget":          { cat: budget_detail_dict },          # for existing table
            "health_score":    float,
            "projected_score": float,
            "score_label":     str,
            "score_color":     str,
            "score_emoji":     str,
            "score_improvement": float,
            "total_expenses":  float,
            "monthly_surplus": float,
            "summary":         str,   # plain-english paragraph
          }
        """
        fixed   = set(fixed_categories or [])
        safe_inc = max(float(income), 1.0)
        trends   = analyse_trends(history or [], safe_inc)

        recommendations: list[Recommendation] = []
        healthy:         list[HealthyCategory] = []
        budget_detail:   dict                  = {}

        for cat in CATEGORIES:
            current  = float(expenses.get(cat, 0))
            curr_pct = current / safe_inc
            adj_hi   = self._adj_hi(cat, city)
            lo, _    = IDEAL_RATIOS[cat]
            tol      = TOLERANCE.get(cat, 0.03)
            meta     = CAT_META.get(cat, {"icon": "📋", "tips": []})
            t_info   = trends.get(cat, {})
            trend    = t_info.get("trend", "")

            # ── FIXED: never touch ────────────────────────────────────────────
            if cat in fixed:
                budget_detail[cat] = self._bdet(cat, current, current, safe_inc,
                                                 "Fixed expense — not adjusted",
                                                 is_fixed=True)
                healthy.append(HealthyCategory(
                    category=cat, icon=meta["icon"],
                    current=current, current_pct=round(curr_pct * 100, 1),
                    status="fixed", message="Marked as fixed — no recommendation generated"
                ))
                continue

            # ── SAVINGS: special upward target ───────────────────────────────
            if cat == "Savings":
                rec_amt, r_reason, r_detail, action = self._savings_rec(
                    current, curr_pct, safe_inc, t_info, cluster_label
                )
                diff = rec_amt - current
                impact = abs(diff)

                if action == "maintain":
                    budget_detail[cat] = self._bdet(cat, current, rec_amt, safe_inc,
                                                     r_reason)
                    healthy.append(HealthyCategory(
                        category=cat, icon=meta["icon"],
                        current=current, current_pct=round(curr_pct * 100, 1),
                        status="excellent" if curr_pct >= 0.25 else "good",
                        message=r_reason
                    ))
                else:
                    priority = ("High" if curr_pct < 0.05 else
                                "Medium" if curr_pct < 0.12 else "Low")
                    rec = Recommendation(
                        category=cat, icon=meta["icon"],
                        current=current, recommended=rec_amt,
                        difference=diff,
                        current_pct=round(curr_pct * 100, 1),
                        recommended_pct=round(rec_amt / safe_inc * 100, 1),
                        reason=r_reason, detail=r_detail,
                        tips=meta["tips"],
                        priority=priority, monthly_impact=impact,
                        action="increase",
                        trend_flag=trend,
                        trend_detail=t_info.get("detail", ""),
                    )
                    recommendations.append(rec)
                    budget_detail[cat] = self._bdet(cat, current, rec_amt, safe_inc,
                                                     r_reason)
                continue

            # ── DEBT REPAYMENT: nuanced ───────────────────────────────────────
            if cat == "Debt Repayment":
                rec_amt, r_reason, r_detail, action = self._debt_rec(
                    current, curr_pct, safe_inc, t_info
                )
                diff   = rec_amt - current
                impact = abs(diff)
                if action == "maintain" or impact < MIN_IMPACT_INR:
                    budget_detail[cat] = self._bdet(cat, current, current, safe_inc,
                                                     r_reason)
                    healthy.append(HealthyCategory(
                        category=cat, icon=meta["icon"],
                        current=current, current_pct=round(curr_pct * 100, 1),
                        status="good", message=r_reason
                    ))
                else:
                    overshoot = max(0, (curr_pct - adj_hi) / max(adj_hi, 0.01))
                    rec = Recommendation(
                        category=cat, icon=meta["icon"],
                        current=current, recommended=rec_amt,
                        difference=diff,
                        current_pct=round(curr_pct * 100, 1),
                        recommended_pct=round(rec_amt / safe_inc * 100, 1),
                        reason=r_reason, detail=r_detail,
                        tips=meta["tips"],
                        priority=self._priority(overshoot, impact, trend),
                        monthly_impact=impact,
                        action="decrease" if diff < 0 else "maintain",
                        trend_flag=trend,
                        trend_detail=t_info.get("detail", ""),
                    )
                    recommendations.append(rec)
                    budget_detail[cat] = self._bdet(cat, current, rec_amt, safe_inc,
                                                     r_reason)
                continue

            # ── EXPENSE CATEGORIES ────────────────────────────────────────────
            # Only flag if current > ideal_max + tolerance
            trigger_threshold = adj_hi + tol
            protected_min_amt = PROTECTED_MINIMUMS.get(cat, 0.0) * safe_inc

            # Compute "recommended" for budget table — even if no rec generated
            if curr_pct > trigger_threshold:
                # Target: reduce to adj_hi (the ideal max), but not below protected min
                raw_rec = safe_inc * adj_hi
                rec_amt = max(raw_rec, protected_min_amt)
                rec_amt = round(rec_amt, 2)
                diff    = rec_amt - current
                impact  = abs(diff)

                if impact < MIN_IMPACT_INR:
                    # Barely over threshold — don't nag
                    budget_detail[cat] = self._bdet(cat, current, current, safe_inc,
                                                     "Slightly over target — monitor but OK")
                    healthy.append(HealthyCategory(
                        category=cat, icon=meta["icon"],
                        current=current, current_pct=round(curr_pct * 100, 1),
                        status="good", message="Slightly over ideal — within acceptable range"
                    ))
                    continue

                overshoot = (curr_pct - adj_hi) / max(adj_hi, 0.01)

                # Build reason
                reason, detail = self._expense_reason(
                    cat, current, rec_amt, curr_pct, adj_hi, t_info, safe_inc
                )

                rec = Recommendation(
                    category=cat, icon=meta["icon"],
                    current=current, recommended=rec_amt,
                    difference=diff,
                    current_pct=round(curr_pct * 100, 1),
                    recommended_pct=round(rec_amt / safe_inc * 100, 1),
                    reason=reason, detail=detail,
                    tips=meta["tips"],
                    priority=self._priority(overshoot, impact, trend),
                    monthly_impact=impact,
                    action="decrease",
                    trend_flag=trend,
                    trend_detail=t_info.get("detail", ""),
                )
                recommendations.append(rec)
                budget_detail[cat] = self._bdet(cat, current, rec_amt, safe_inc, reason)

            else:
                # Healthy — no recommendation
                if curr_pct >= adj_hi * 0.85:
                    status  = "good"
                    message = f"{cat} is within the ideal range ({round(curr_pct*100,1)}% of income)"
                else:
                    status  = "excellent"
                    message = f"{cat} is well-managed — only {round(curr_pct*100,1)}% of income"

                budget_detail[cat] = self._bdet(cat, current, current, safe_inc, message)
                healthy.append(HealthyCategory(
                    category=cat, icon=meta["icon"],
                    current=current, current_pct=round(curr_pct * 100, 1),
                    status=status, message=message
                ))

        # ── Sort recommendations by priority then impact ──────────────────────
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 3), -r.monthly_impact)
        )

        # ── Compute projected budget (using recommendations) ──────────────────
        projected_expenses = dict(expenses)
        for r in recommendations:
            projected_expenses[r.category] = r.recommended

        health_score    = compute_health_score(income, expenses, city)
        projected_score = compute_health_score(income, projected_expenses, city)
        score_info      = score_label(health_score)

        total_exp     = sum(float(v) for v in expenses.values())
        total_savings = sum(r.monthly_impact for r in recommendations
                            if r.action in ("decrease", "increase"))

        summary = self._build_summary(
            income, total_exp, health_score, recommendations, healthy, cluster_label
        )

        return {
            "recommendations":  [r.to_dict() for r in recommendations],
            "healthy":          [h.to_dict() for h in healthy],
            "budget":           budget_detail,
            "health_score":     health_score,
            "projected_score":  projected_score,
            "score_label":      score_info["label"],
            "score_color":      score_info["color"],
            "score_emoji":      score_info["emoji"],
            "score_improvement": round(projected_score - health_score, 1),
            "total_expenses":   round(total_exp, 2),
            "monthly_surplus":  round(income - total_exp, 2),
            "total_potential_savings": round(total_savings, 2),
            "summary":          summary,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bdet(cat, current, recommended, income, explanation,
              is_fixed=False) -> dict:
        diff = recommended - current
        return {
            "current":          round(current,     2),
            "recommended":      round(recommended, 2),
            "difference":       round(diff,         2),
            "current_pct":      round(current     / income * 100, 1),
            "recommended_pct":  round(recommended / income * 100, 1),
            "action":  ("increase" if diff > 0 else ("decrease" if diff < 0 else "maintain")),
            "explanation":      explanation,
            "is_fixed":         is_fixed,
        }

    @staticmethod
    def _savings_rec(current, curr_pct, income, t_info, cluster_label):
        target_amt   = income * SAVINGS_TARGET_RATIO
        gap          = target_amt - current
        trend        = t_info.get("trend", "")

        if curr_pct >= SAVINGS_TARGET_RATIO:
            if curr_pct >= 0.30:
                return current, "Outstanding savings rate — keep it up!", \
                       f"You save {round(curr_pct*100,1)}% of income, well above the 20% target.", \
                       "maintain"
            return current, \
                   f"Savings on track at {round(curr_pct*100,1)}% of income.", \
                   "You're meeting the 20% savings target. Consider increasing to 25%+ when possible.", \
                   "maintain"

        if curr_pct < 0.05:
            reason = "Critical: savings rate is very low"
            detail = (f"You currently save only {round(curr_pct*100,1)}% of income (₹{current:,.0f}). "
                      f"The recommended minimum is 20% (₹{target_amt:,.0f}/month). "
                      f"Increasing savings by ₹{gap:,.0f}/month will build financial security.")
        else:
            reason = f"Savings below target ({round(curr_pct*100,1)}% vs 20% goal)"
            detail = (f"Increasing savings by ₹{gap:,.0f}/month will bring you to the "
                      f"20% target. With AI-optimised budget cuts in other categories, "
                      f"this target becomes achievable.")

        if trend == "falling":
            reason += " — and declining"
            detail += " Your savings have been decreasing recently, which needs attention."

        return round(target_amt, 2), reason, detail, "increase"

    @staticmethod
    def _debt_rec(current, curr_pct, income, t_info):
        if curr_pct == 0:
            return current, "Debt-free — excellent!", \
                   "No debt repayments. This frees up income for savings and investments.", \
                   "maintain"
        if curr_pct <= 0.15:
            return current, f"Debt repayment manageable at {round(curr_pct*100,1)}% of income", \
                   "Your debt-to-income ratio is within the healthy range (≤15%). Continue repaying consistently.", \
                   "maintain"
        if curr_pct <= 0.25:
            rec_amt = round(income * 0.20, 2)
            return rec_amt, \
                   f"Debt repayment elevated at {round(curr_pct*100,1)}% of income", \
                   (f"Your debt repayments consume {round(curr_pct*100,1)}% of income. "
                    f"Aim to bring this below 20% (₹{rec_amt:,.0f}/month) using the avalanche method. "
                    f"Focus extra payments on the highest-interest debt first."), \
                   "decrease"
        # > 25%
        rec_amt = round(income * 0.20, 2)
        return rec_amt, \
               f"High debt burden — {round(curr_pct*100,1)}% of income in repayments", \
               (f"Over a quarter of your income goes to debt. This severely limits savings "
                f"and emergency fund building. Target: reduce to 20% (₹{rec_amt:,.0f}/month) by "
                f"restructuring or paying off the smallest balance first (snowball method)."), \
               "decrease"

    @staticmethod
    def _expense_reason(cat, current, recommended, curr_pct, adj_hi,
                        t_info, income) -> tuple[str, str]:
        over_pct   = round((curr_pct - adj_hi) * 100, 1)
        saving_amt = round(current - recommended, 0)
        trend      = t_info.get("trend", "")
        trend_det  = t_info.get("detail", "")

        short_reason = (
            f"{cat} is {round(curr_pct*100,1)}% of income "
            f"(limit: {round(adj_hi*100,0):.0f}%) — "
            f"{over_pct}pp above target"
        )

        detail_parts = [
            f"Current {cat.lower()} spending: ₹{current:,.0f} ({round(curr_pct*100,1)}% of income).",
            f"Recommended: ₹{recommended:,.0f} ({round(adj_hi*100,0):.0f}% of income).",
            f"Reducing this frees up ₹{saving_amt:,.0f}/month that can go toward savings or debt.",
        ]

        if trend in ("spike", "rising"):
            detail_parts.append(f"⚠️ Trend alert: your {trend_det}.")

        return short_reason, " ".join(detail_parts)

    @staticmethod
    def _build_summary(income, total_exp, score, recommendations, healthy, cluster_label) -> str:
        n_recs    = len(recommendations)
        n_healthy = len(healthy)
        surplus   = income - total_exp
        hi_count  = sum(1 for r in recommendations if r.priority == "High")

        if n_recs == 0:
            return (f"Your finances are in great shape — all {n_healthy} spending categories "
                    f"are within healthy limits. Keep maintaining this discipline and consider "
                    f"investing any surplus for long-term wealth creation.")

        total_impact = sum(r.monthly_impact for r in recommendations)
        lines = []

        if hi_count:
            lines.append(f"{hi_count} high-priority area{'s' if hi_count>1 else ''} need immediate attention.")

        lines.append(
            f"By acting on {n_recs} recommendation{'s' if n_recs>1 else ''}, "
            f"you could free up ₹{total_impact:,.0f}/month — "
            f"₹{total_impact*12:,.0f}/year."
        )

        if surplus < 0:
            lines.append("⚠️ You are currently spending more than you earn. Prioritise expense cuts immediately.")
        elif surplus < income * 0.05:
            lines.append("Your monthly surplus is thin. Small changes can have a big impact.")

        if cluster_label:
            lines.append(f"Your spending profile: {cluster_label}.")

        return " ".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: DynamicRecommendationEngine | None = None


def get_engine() -> DynamicRecommendationEngine:
    global _engine
    if _engine is None:
        _engine = DynamicRecommendationEngine()
    return _engine


def generate_recommendations(
    income:           float,
    expenses:         dict,
    city:             str        = "Other",
    fixed_categories: list       = None,
    history:          list[dict] = None,
    cluster_label:    str        = "",
) -> dict:
    """Public API — drop-in replacement for agent.recommend_budget()."""
    return get_engine().generate(
        income=income,
        expenses=expenses,
        city=city,
        fixed_categories=fixed_categories,
        history=history,
        cluster_label=cluster_label,
    )
