"""
Credit scorecard app (Streamlit).

Reads scorecard.json (written by the notebook), so it uses the same cleaning rules,
bins and points as the notebook.

Run with:  streamlit run app.py
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Late 30-59 days",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "Late 90+ days",
    "NumberRealEstateLoansOrLines": "Real-estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Late 60-89 days",
    "NumberOfDependents": "Dependents",
}
PERCENT_VARS = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]

# Risk labels based on the estimated PD (just to make the result easier to read)
BANDS = [(0.02, "Low risk"), (0.05, "Moderate risk"), (0.10, "Elevated risk"), (1.01, "High risk")]


# ----------------------------------------------------------------------------
# Scoring logic (same rules as the notebook)
# ----------------------------------------------------------------------------
def load_scorecard(path=None):
    # scorecard.json sits next to app.py, wherever the app is started from
    if path is None:
        path = Path(__file__).parent / "scorecard.json"
    with open(path) as f:
        return json.load(f)


def clean_inputs(raw):
    clean = dict(raw)

    # past-due counts capped at 10
    for col in ["NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate",
                "NumberOfTime60-89DaysPastDueNotWorse"]:
        clean[col] = min(clean[col], 10)

    # utilization capped at 2 (200%)
    clean["RevolvingUtilizationOfUnsecuredLines"] = min(clean["RevolvingUtilizationOfUnsecuredLines"], 2)

    # a debt ratio above 5 is not a real ratio, so it is treated as missing
    if clean["DebtRatio"] is not None and clean["DebtRatio"] > 5:
        clean["DebtRatio"] = None

    return clean


def find_bin(value, edges, labels):
    # bins are (lower, upper]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Missing"
    position = int(np.searchsorted(edges, value, side="left"))
    return labels[position]


def score_to_pd(score, factor, offset):
    odds_good = math.exp((score - offset) / factor)
    return 1 / (1 + odds_good)


def format_value(var_name, value):
    if value is None:
        return "Missing"
    if var_name in PERCENT_VARS:
        return f"{value * 100:.0f}%"
    if var_name == "MonthlyIncome":
        return f"{value:,.0f}"
    return f"{value:g}"


def score_borrower(raw, scorecard):
    """Returns (score, PD, table with one row per variable)."""
    clean = clean_inputs(raw)
    rows = []
    total = 0
    for var in scorecard["variables"]:
        value = clean[var["name"]]
        bin_label = find_bin(value, var["edges"], var["labels"])
        points = var["points"][bin_label]
        total += points
        rows.append({
            "Variable": NAMES[var["name"]],
            "Value": format_value(var["name"], value),
            "Group": bin_label,
            "Points": points,
            "Best possible": max(var["points"].values()),
        })
    pd_estimate = score_to_pd(total, scorecard["factor"], scorecard["offset"])
    return total, pd_estimate, pd.DataFrame(rows)


def score_limits(scorecard):
    low = sum(min(v["points"].values()) for v in scorecard["variables"])
    high = sum(max(v["points"].values()) for v in scorecard["variables"])
    return low, high


def risk_label(pd_estimate):
    for limit, label in BANDS:
        if pd_estimate < limit:
            return label


# ----------------------------------------------------------------------------
# Small display helpers
# ----------------------------------------------------------------------------
def gauge_html(score, low, high):
    position = (score - low) / (high - low) * 100
    position = min(max(position, 0), 100)
    return f"""
    <div style="margin: 10px 0 4px 0;">
      <div style="position:relative; height:12px; border-radius:6px;
                  background:linear-gradient(90deg,#c4605b,#e0b64a,#5a9e6f);">
        <div style="position:absolute; left:{position}%; top:-5px; width:4px; height:22px;
                    background:#1c2430; border-radius:2px; transform:translateX(-50%);"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:12px; opacity:0.65; margin-top:5px;">
        <span>{low} (riskiest)</span><span>{high} (safest)</span>
      </div>
    </div>
    """


def points_chart(breakdown, compare=None, labels=("Current", "Scenario")):
    """Horizontal bars: points per variable. With `compare`, shows two borrowers side by side."""
    data = breakdown.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.barh(y, data["Best possible"], color="#e6e9ee", height=0.7, label="Best possible")
    if compare is None:
        ax.barh(y, data["Points"], color="#1f4e79", height=0.7, label="This borrower")
    else:
        other = compare.iloc[::-1].reset_index(drop=True)
        ax.barh(y + 0.18, data["Points"], color="#9aa7b8", height=0.34, label=labels[0])
        ax.barh(y - 0.18, other["Points"], color="#1f4e79", height=0.34, label=labels[1])

    ax.set_yticks(y)
    ax.set_yticklabels(data["Variable"], fontsize=9)
    ax.set_xlabel("Points", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def show_result(score, pd_estimate, breakdown, scorecard):
    low, high = score_limits(scorecard)
    average_pd = scorecard["train_default_rate"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", score)
    c2.metric("Estimated PD (2 years)", f"{pd_estimate:.2%}")
    c3.metric("Risk level", risk_label(pd_estimate))

    st.markdown(gauge_html(score, low, high), unsafe_allow_html=True)
    st.write(
        f"The average default rate in the training data is {average_pd:.2%}, "
        f"so this borrower is at {pd_estimate / average_pd:.1f} times the average."
    )

    # where the borrower loses the most points compared with the best group
    lost = breakdown.assign(Lost=breakdown["Best possible"] - breakdown["Points"])
    lost = lost[lost["Lost"] > 0].sort_values("Lost", ascending=False).head(3)
    st.markdown("**What lowers this score the most**")
    if lost.empty:
        st.write("Nothing in particular: this borrower is in the best group for every variable.")
    else:
        for _, row in lost.iterrows():
            st.write(f"- {row['Variable']} ({row['Value']}): {row['Lost']} points below the best group")

    fig = points_chart(breakdown)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Detailed points table"):
        st.dataframe(breakdown, hide_index=True, width="stretch")


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Credit scorecard", layout="wide")

    try:
        scorecard = load_scorecard()
    except FileNotFoundError:
        st.error("scorecard.json not found. Run credit_scorecard.ipynb first, it creates this file.")
        st.stop()

    if "borrower" not in st.session_state:
        st.session_state["borrower"] = None
        st.session_state["n_calc"] = 0

    st.title("Credit scorecard")
    st.caption(
        "Estimates the probability of default (PD) of a borrower over 2 years. "
        "Built on the Kaggle 'Give Me Some Credit' data, for learning purposes only."
    )

    tab_score, tab_whatif, tab_about = st.tabs(["Score a borrower", "What-if comparison", "How it works"])

    # ------------------------------------------------------------------ tab 1
    with tab_score:
        with st.form("borrower_form"):
            left, right = st.columns(2)

            with left:
                st.markdown("**Profile**")
                age = st.slider("Age", 18, 100, 40)
                dependents = st.slider("Number of dependents", 0, 10, 0)
                income = st.slider("Monthly income", 0, 20000, 5000, step=100,
                                   help="Anything above 10,000 falls in the same group.")
                income_unknown = st.checkbox("I don't know the income (the slider above is ignored)")
                debt_ratio = st.slider("Debt ratio (monthly debt payments as % of income)", 0, 500, 35, step=5)
                ratio_unknown = st.checkbox("I don't know the debt ratio (the slider above is ignored)")

            with right:
                st.markdown("**Credit use**")
                utilization = st.slider("Credit utilization (% of the credit limit used)", 0, 200, 30, step=5,
                                        help="Values above 200% are treated as 200%.")
                open_lines = st.slider("Open credit lines and loans", 0, 30, 6)
                real_estate = st.slider("Real-estate loans", 0, 10, 1)

                st.markdown("**Payment history**")
                late_30 = st.slider("Times 30-59 days late", 0, 10, 0, help="10 means 10 or more.")
                late_60 = st.slider("Times 60-89 days late", 0, 10, 0, help="10 means 10 or more.")
                late_90 = st.slider("Times 90+ days late", 0, 10, 0, help="10 means 10 or more.")

            submitted = st.form_submit_button("Calculate score", type="primary")

        if submitted:
            st.session_state["borrower"] = {
                "RevolvingUtilizationOfUnsecuredLines": utilization / 100,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30,
                "DebtRatio": None if ratio_unknown else debt_ratio / 100,
                "MonthlyIncome": None if income_unknown else float(income),
                "NumberOfOpenCreditLinesAndLoans": open_lines,
                "NumberOfTimes90DaysLate": late_90,
                "NumberRealEstateLoansOrLines": real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60,
                "NumberOfDependents": dependents,
            }
            st.session_state["n_calc"] += 1

        borrower = st.session_state["borrower"]
        if borrower is None:
            st.info("Set the borrower's details above and press Calculate score.")
        else:
            st.divider()
            score, pd_estimate, breakdown = score_borrower(borrower, scorecard)
            show_result(score, pd_estimate, breakdown, scorecard)

    # ------------------------------------------------------------------ tab 2
    with tab_whatif:
        borrower = st.session_state["borrower"]
        if borrower is None:
            st.info("Calculate a score first. Then come back here to see how changes would affect it.")
        else:
            n = st.session_state["n_calc"]   # new keys for each calculation, so the sliders reset
            st.write("Move the sliders to see how the score of the borrower you just scored would change.")

            scenario = dict(borrower)
            col_a, col_b = st.columns(2)
            with col_a:
                scenario["RevolvingUtilizationOfUnsecuredLines"] = st.slider(
                    "Credit utilization (%)", 0, 200,
                    int(round(borrower["RevolvingUtilizationOfUnsecuredLines"] * 100)),
                    step=5, key=f"wi_util_{n}") / 100
                scenario["NumberOfTimes90DaysLate"] = st.slider(
                    "Times 90+ days late", 0, 10, int(borrower["NumberOfTimes90DaysLate"]), key=f"wi_90_{n}")
                scenario["NumberOfTime30-59DaysPastDueNotWorse"] = st.slider(
                    "Times 30-59 days late", 0, 10, int(borrower["NumberOfTime30-59DaysPastDueNotWorse"]),
                    key=f"wi_30_{n}")
            with col_b:
                if borrower["MonthlyIncome"] is None:
                    st.caption("Income was not provided, so it is not part of this comparison.")
                else:
                    scenario["MonthlyIncome"] = float(st.slider(
                        "Monthly income", 0, 20000, int(min(borrower["MonthlyIncome"], 20000)),
                        step=100, key=f"wi_inc_{n}"))
                if borrower["DebtRatio"] is None:
                    st.caption("Debt ratio was not provided, so it is not part of this comparison.")
                else:
                    scenario["DebtRatio"] = st.slider(
                        "Debt ratio (%)", 0, 500, int(round(min(borrower["DebtRatio"], 5) * 100)),
                        step=5, key=f"wi_debt_{n}") / 100
                scenario["NumberOfOpenCreditLinesAndLoans"] = st.slider(
                    "Open credit lines", 0, 30, int(borrower["NumberOfOpenCreditLinesAndLoans"]),
                    key=f"wi_open_{n}")

            base_score, base_pd, base_table = score_borrower(borrower, scorecard)
            new_score, new_pd, new_table = score_borrower(scenario, scorecard)

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Score", new_score, delta=int(new_score - base_score),
                      help=f"Original score: {base_score}")
            m2.metric("Estimated PD", f"{new_pd:.2%}",
                      delta=f"{(new_pd - base_pd) * 100:+.2f} pts", delta_color="inverse",
                      help=f"Original PD: {base_pd:.2%}")
            m3.metric("Risk level", risk_label(new_pd), help=f"Original: {risk_label(base_pd)}")

            fig = points_chart(base_table, compare=new_table, labels=("Original", "Scenario"))
            st.pyplot(fig)
            plt.close(fig)

    # ------------------------------------------------------------------ tab 3
    with tab_about:
        st.markdown(
            f"""
            The score is the sum of points earned in each variable. A higher score means a lower
            probability of default.

            - A score of **{scorecard['base_score']}** corresponds to odds of **{scorecard['base_odds']}:1**
              (50 borrowers who repay for each one who defaults).
            - Every **+{scorecard['pdo']} points** doubles those odds.
            - The score is converted back to a PD with the same formula.
            """
        )

        m = scorecard["metrics_test"]
        a, b, c = st.columns(3)
        a.metric("AUC (test)", f"{m['AUC']:.3f}")
        b.metric("Gini (test)", f"{m['Gini']:.3f}")
        c.metric("KS (test)", f"{m['KS']:.3f}")

        st.markdown("**Points per group**")
        names = {NAMES[v["name"]]: v for v in scorecard["variables"]}
        chosen = st.selectbox("Variable", list(names.keys()))
        variable = names[chosen]
        order = variable["labels"] + ["Missing"]
        table = pd.DataFrame(
            [{"Group": g, "Points": variable["points"][g]} for g in order if g in variable["points"]]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Educational project, not a real lending decision tool.")


if __name__ == "__main__":
    main()
