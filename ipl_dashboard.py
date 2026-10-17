"""
IPL 2025 Analytics Dashboard
----------------------------
An interactive Streamlit dashboard for exploring IPL 2025 player and team data.

Run locally:
    pip install -r requirements.txt
    streamlit run ipl_dashboard.py

Data files (CSV) are looked up in a `data/` folder first, then next to this script:
    orange_cap.csv, purple_cap.csv, matches.csv, player_info.csv   (required)
    ipl_2025_allrounders.csv                                        (optional)
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.errors import StreamlitAPIException

# ========================= CONFIG =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIRS = (BASE_DIR / "data", BASE_DIR)

# Required files and the columns this app reads from each of them.
CORE_FILES = {
    "orange_cap.csv": ["Batsman", "Team", "Runs", "Average", "Strike_rate", "Sixes", "Fours"],
    "purple_cap.csv": ["Bowler", "Team", "Wickets", "Economy_rate", "Maidens"],
    "matches.csv": ["match_winner"],
    "player_info.csv": ["TEAM", "AGE", "COUNTRY", "SOLD_PRICE"],
}
ALLROUNDER_FILE = "ipl_2025_allrounders.csv"

# "Best player" scoring weights. These are a judgement call, not a standard
# formula: change them here and the dashboard updates. Each stat is min-max
# scaled (0-1) among the players who pass the minimum-qualification filter.
BAT_WEIGHTS = {"Runs": 0.35, "Average": 0.25, "Strike_rate": 0.20, "Sixes": 0.15, "Fours": 0.05}
BOWL_WEIGHTS = {"Wickets": 0.40, "Economy_rate": 0.30, "Maidens": 0.20, "Five_wicket_haul": 0.10}

# The source CSVs spell some columns differently, so we accept alternatives.
FOUR_WICKET_COLS = ["Four_wicket_haul", "Four_wicket_hall", "Four_wickets"]
FIVE_WICKET_COLS = ["Five_wicket_haul", "Five_wicket_hall", "Five_wickets"]
ROLE_COLS = ["Paying_Role", "Playing_Role", "Role"]
VENUE_COLS = ["venue", "stadium", "ground", "city"]
TOSS_WINNER_COLS = ["toss_winner"]
TOSS_DECISION_COLS = ["toss_decision"]

CSS = """
<style>
    .main, [data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #0f1419, #1a1f2e); color: #f0f0f0;}
    h1, h2, h3, h4 {color: #ffd700 !important; font-weight: 800 !important; text-align: center;}
    section[data-testid="stSidebar"] {background: #0d1117 !important; border-right: 3px solid #ffd700;}
    .best-player-card {
        background: linear-gradient(135deg, #ff8c00, #ffd700);
        padding: 2.6rem; border-radius: 28px; text-align: center;
        box-shadow: 0 20px 50px rgba(255,140,0,0.7); margin: 2rem 0;
        color: #000; font-weight: bold; border: 5px solid #ffffff;
    }
    .best-player-card h1, .best-player-card h2 {color: #000 !important;}
    .metric-card {
        background: rgba(13,17,23,0.95); padding: 1.8rem; border-radius: 20px;
        border: 3px solid #ffd700; text-align: center; color: #ffd700;
        font-size: 1.2rem; box-shadow: 0 8px 25px rgba(255,215,0,0.4);
    }
    .stTabs [data-baseweb="tab"] {font-size: 1.2rem; font-weight: 700; color: #ffd700; background: #161b22; border-radius: 15px 15px 0 0;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background: #ffd700; color: #000;}
</style>
"""


# ========================= HELPERS =========================
def find_file(name: str) -> Path | None:
    """Look for a data file in data/ first, then next to this script."""
    for folder in DATA_DIRS:
        path = folder / name
        if path.exists():
            return path
    return None


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = find_file(name)
    return pd.read_csv(path) if path is not None else None


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present in df (case-insensitive)."""
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to numbers; anything unparseable becomes 0."""
    for col in cols:
        actual = find_col(df, [col])
        if actual:
            df[actual] = pd.to_numeric(df[actual], errors="coerce").fillna(0)
    return df


def minmax(series: pd.Series) -> pd.Series:
    """Scale a series to 0-1. A constant series maps to 0."""
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / span


_SALARY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(crore|cr|lakhs|lakh|lac|l)?\b")


def parse_salary(value) -> float:
    """Turn strings like '1.5 Cr' or '50 Lakh' into rupees. Unparseable -> NaN.

    Replaces the earlier eval()-based parser, which would execute whatever text
    was in the CSV as code.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", "").replace("₹", "").strip()
    match = _SALARY_RE.match(text)
    if not match:
        return np.nan
    amount = float(match.group(1))
    unit = match.group(2)
    if unit in ("crore", "cr"):
        return amount * 1e7
    if unit in ("lakhs", "lakh", "lac", "l"):
        return amount * 1e5
    return amount


def inr(value) -> str:
    """Format rupees as '₹x.xx Cr' or '₹x.x Lakh'."""
    if pd.isna(value):
        return "N/A"
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    return f"₹{value / 1e5:.1f} Lakh"


def fmt_num(value, digits: int = 0) -> str:
    """Format a value as a number if possible, otherwise show it as text."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def score_batsmen(df: pd.DataFrame, min_runs: int) -> pd.DataFrame:
    """Rank qualifying batsmen by the weighted score defined in BAT_WEIGHTS."""
    qualified = df[df["Runs"] >= min_runs].copy()
    if qualified.empty:
        return qualified
    qualified["Score"] = sum(w * minmax(qualified[col]) for col, w in BAT_WEIGHTS.items())
    return qualified.sort_values("Score", ascending=False)


def score_bowlers(df: pd.DataFrame, min_wickets: int) -> pd.DataFrame:
    """Rank qualifying bowlers by the weighted score defined in BOWL_WEIGHTS."""
    qualified = df[df["Wickets"] >= min_wickets].copy()
    if qualified.empty:
        return qualified
    five_col = find_col(qualified, FIVE_WICKET_COLS)
    five = minmax(qualified[five_col]) if five_col else pd.Series(0.0, index=qualified.index)
    qualified["Score"] = (
        BOWL_WEIGHTS["Wickets"] * minmax(qualified["Wickets"])
        + BOWL_WEIGHTS["Economy_rate"] * (1 - minmax(qualified["Economy_rate"]))  # lower economy is better
        + BOWL_WEIGHTS["Maidens"] * minmax(qualified["Maidens"])
        + BOWL_WEIGHTS["Five_wicket_haul"] * five
    )
    return qualified.sort_values("Score", ascending=False)


def venue_stats(matches: pd.DataFrame) -> dict | None:
    """Venue and toss summaries from matches.csv. Returns None if no venue column exists."""
    venue_col = find_col(matches, VENUE_COLS)
    if venue_col is None:
        return None

    by_venue = matches[venue_col].dropna().value_counts().reset_index()
    by_venue.columns = ["Venue", "Matches"]
    stats = {"venue_col": venue_col, "by_venue": by_venue, "toss_decision": None, "toss_win_pct": None}

    decision_col = find_col(matches, TOSS_DECISION_COLS)
    if decision_col:
        decisions = matches[decision_col].dropna().value_counts().reset_index()
        decisions.columns = ["Decision", "Matches"]
        stats["toss_decision"] = decisions

    toss_col = find_col(matches, TOSS_WINNER_COLS)
    if toss_col and "match_winner" in matches.columns:
        valid = matches[[toss_col, "match_winner"]].dropna()
        if len(valid):
            stats["toss_win_pct"] = float((valid[toss_col] == valid["match_winner"]).mean() * 100)
    return stats


def wins_at_venue(matches: pd.DataFrame, venue_col: str, venue: str) -> pd.DataFrame:
    wins = matches.loc[matches[venue_col] == venue, "match_winner"].dropna().value_counts().reset_index()
    wins.columns = ["Team", "Wins"]
    return wins


def style_fig(fig):
    """Apply the dark dashboard theme to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f0f0",
        title_font_color="#ffd700",
    )
    return fig


def show_plot(fig):
    """Render a Plotly chart at full width across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except (TypeError, StreamlitAPIException):
        st.plotly_chart(fig, use_container_width=True)


def show_table(df):
    """Render a dataframe at full width across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch")
    except (TypeError, StreamlitAPIException):
        st.dataframe(df, use_container_width=True)


def metric_card(label: str, value) -> str:
    return f"<div class='metric-card'>{html.escape(str(label))}<br><h2>{html.escape(str(value))}</h2></div>"


def best_player_card(title: str, name: str, details: str) -> None:
    st.markdown(
        f"""
        <div class="best-player-card">
            <h2>{html.escape(title)}</h2>
            <h1>{html.escape(str(name))}</h1>
            <p style='font-size:1.5rem; margin:15px 0;'>{html.escape(details)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ========================= DATA LOADING =========================
def load_core_data() -> tuple[dict, list[str]]:
    """Load required CSVs and report any missing files or columns."""
    data, problems = {}, []
    for name, required_cols in CORE_FILES.items():
        df = load_csv(name)
        if df is None:
            problems.append(f"`{name}` was not found (looked in `data/` and the app folder).")
            continue
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            problems.append(f"`{name}` is missing column(s): {', '.join(missing)}")
        data[name] = df
    return data, problems


# ========================= SECTIONS =========================
def render_batsmen(orange: pd.DataFrame) -> None:
    st.markdown("### Orange Cap Race 2025")
    top10 = orange.sort_values("Runs", ascending=False).head(10).copy()
    top10.insert(0, "Rank", range(1, len(top10) + 1))

    c1, c2 = st.columns([4, 2])
    with c1:
        fig = px.bar(top10, x="Batsman", y="Runs", text="Runs", color="Runs",
                     color_continuous_scale="Oranges", height=620, title="Top 10 Run Scorers")
        fig.update_traces(textposition="outside", textfont_size=15)
        show_plot(style_fig(fig))
    with c2:
        cols = ["Batsman", "Team", "Runs", "Average", "Strike_rate", "Sixes", "Fours"]
        show_table(top10.set_index("Rank")[cols])

    st.markdown("### Best Batsman")
    max_runs = max(int(orange["Runs"].max()), 1)
    min_runs = st.slider("Minimum runs to qualify", 0, max_runs, min(150, max_runs), step=10, key="min_runs")
    ranked = score_batsmen(orange, min_runs)
    with st.expander("How is the best batsman chosen?"):
        weights = ", ".join(f"{col} {int(w * 100)}%" for col, w in BAT_WEIGHTS.items())
        st.write(
            f"Each stat is scaled 0-1 among players with at least {min_runs} runs, then combined: {weights}. "
            "The weights are a judgement call, not an official formula, and can be changed in `BAT_WEIGHTS` "
            "at the top of the script. The qualification filter stops a player with very few runs from "
            "winning on average or strike rate alone."
        )
    if ranked.empty:
        st.info("No batsman meets the minimum-runs filter. Lower the slider.")
        return
    best = ranked.iloc[0]
    best_player_card(
        "Best Batsman of IPL 2025", best["Batsman"],
        f"{int(best['Runs'])} runs • Avg {best['Average']:.1f} • SR {best['Strike_rate']:.1f} • "
        f"{int(best['Sixes'])} sixes • {int(best['Fours'])} fours",
    )
    st.markdown("#### Top 5 by score")
    show_table(ranked.head(5)[["Batsman", "Team", "Runs", "Average", "Strike_rate", "Score"]].round(3).reset_index(drop=True))


def render_bowlers(purple: pd.DataFrame) -> None:
    st.markdown("### Purple Cap Race 2025")
    top10 = purple.sort_values("Wickets", ascending=False).head(10).copy()
    top10.insert(0, "Rank", range(1, len(top10) + 1))

    c1, c2 = st.columns([4, 2])
    with c1:
        fig = px.bar(top10, x="Bowler", y="Wickets", text="Wickets", color="Wickets",
                     color_continuous_scale="Purples", height=620, title="Top 10 Wicket-Takers")
        fig.update_traces(textposition="outside", textfont_size=15)
        show_plot(style_fig(fig))
    with c2:
        four_col = find_col(purple, FOUR_WICKET_COLS)
        cols = ["Bowler", "Team", "Wickets", "Economy_rate", "Maidens"] + ([four_col] if four_col else [])
        show_table(top10.set_index("Rank")[cols])

    st.markdown("### Best Bowler")
    max_wkts = max(int(purple["Wickets"].max()), 1)
    min_wkts = st.slider("Minimum wickets to qualify", 0, max_wkts, min(10, max_wkts), key="min_wkts")
    ranked = score_bowlers(purple, min_wkts)
    with st.expander("How is the best bowler chosen?"):
        weights = ", ".join(f"{col.replace('_', ' ')} {int(w * 100)}%" for col, w in BOWL_WEIGHTS.items())
        st.write(
            f"Each stat is scaled 0-1 among bowlers with at least {min_wkts} wickets (lower economy scores higher), "
            f"then combined: {weights}. Weights are a judgement call and can be changed in `BOWL_WEIGHTS`."
        )
    if ranked.empty:
        st.info("No bowler meets the minimum-wickets filter. Lower the slider.")
        return
    best = ranked.iloc[0]
    best_player_card(
        "Best Bowler of IPL 2025", best["Bowler"],
        f"{int(best['Wickets'])} wickets • Economy {best['Economy_rate']:.2f} • {int(best['Maidens'])} maidens",
    )


def render_allrounders() -> None:
    st.markdown("### Top All-Rounders of IPL 2025")
    ar = load_csv(ALLROUNDER_FILE)
    if ar is None:
        st.info(f"Add `{ALLROUNDER_FILE}` to the `data/` folder to enable this section.")
        return

    needed = ["Player Name", "Runs Scored", "Wickets Taken", "Batting Average", "Eco Rate"]
    missing = [c for c in needed if c not in ar.columns]
    if missing:
        st.warning(f"`{ALLROUNDER_FILE}` is missing column(s): {', '.join(missing)}")
        return

    ar = to_numeric(ar, ["Runs Scored", "Wickets Taken", "Batting Average", "Eco Rate", "All_Rounder_Score"])
    sort_col = "All_Rounder_Score" if "All_Rounder_Score" in ar.columns else "Runs Scored"
    top_ar = ar.sort_values(sort_col, ascending=False).head(10).copy()
    top_ar.insert(0, "Rank", range(1, len(top_ar) + 1))
    top_ar = top_ar.set_index("Rank")

    numeric_cols = list(top_ar.select_dtypes("number").columns)
    try:
        show_table(top_ar.style.highlight_max(subset=numeric_cols))
    except (ImportError, ValueError, TypeError):
        show_table(top_ar)

    best = top_ar.iloc[0]
    best_player_card(
        "Best All-Rounder of IPL 2025", best["Player Name"],
        f"{fmt_num(best['Runs Scored'])} runs • {fmt_num(best['Wickets Taken'])} wickets • "
        f"Bat Avg {fmt_num(best['Batting Average'], 1)} • Eco {fmt_num(best['Eco Rate'], 2)}",
    )


def render_team_analysis(matches: pd.DataFrame, player_info: pd.DataFrame) -> None:
    st.markdown("### Team Performance & Squad Analysis")

    wins = matches["match_winner"].dropna().value_counts().reset_index()
    wins.columns = ["Team", "Wins"]
    fig = px.bar(wins, x="Team", y="Wins", text="Wins", color="Wins",
                 color_continuous_scale="Oranges", title="IPL 2025 Wins Leaderboard")
    fig.update_traces(textposition="outside")
    show_plot(style_fig(fig))

    teams = sorted(player_info["TEAM"].dropna().unique())
    selected_team = st.selectbox("Select Team for Deep Dive", teams)
    squad = player_info[player_info["TEAM"] == selected_team].copy()

    if not squad.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(metric_card("Squad Size", len(squad)), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("Average Age", f"{squad['AGE'].mean():.1f}"), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("Players with Salary Data",
                                    f"{squad['Salary'].notna().sum()} / {len(squad)}"), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(squad, x="AGE", nbins=10, title="Age Distribution",
                               color_discrete_sequence=["#ffd700"])
            show_plot(style_fig(fig))
        with col2:
            country_counts = squad["COUNTRY"].value_counts()
            fig = px.bar(x=country_counts.index, y=country_counts.values, text=country_counts.values,
                         labels={"x": "Country", "y": "Number of Players"},
                         title="Country-wise Player Distribution",
                         color=country_counts.values, color_continuous_scale="Oranges")
            fig.update_traces(textposition="outside", textfont_size=15)
            fig.update_layout(height=520)
            show_plot(style_fig(fig))

    st.markdown("### Auction Spending Analysis")
    team_salary = st.selectbox("Select Team for Salary Breakdown", teams, key="salary")
    salary_data = player_info[(player_info["TEAM"] == team_salary) & player_info["Salary"].notna()]

    if salary_data.empty:
        st.info("No salary data available for this team yet.")
        return

    role_col = find_col(salary_data, ROLE_COLS)
    if role_col:
        role_avg = salary_data.groupby(role_col)["Salary"].mean().reset_index()
        fig = px.bar(role_avg, x=role_col, y="Salary", text=role_avg["Salary"].apply(inr),
                     color=role_col, title=f"Average Salary by Role — {team_salary}", height=620)
        fig.update_traces(textposition="outside", textfont_size=16, textfont_color="#ffffff")
        fig.update_layout(yaxis=dict(range=[0, role_avg["Salary"].max() * 1.5]))
        show_plot(style_fig(fig))
    else:
        st.info("No player-role column found in `player_info.csv`, so the role breakdown is skipped.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(metric_card("Total Auction Spend", inr(salary_data["Salary"].sum())), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Average Salary", inr(salary_data["Salary"].mean())), unsafe_allow_html=True)


def render_ground_analysis(matches: pd.DataFrame) -> None:
    st.markdown("### Ground & Venue Analysis")
    stats = venue_stats(matches)
    if stats is None:
        st.info("No venue column found in `matches.csv`. Columns available: " + ", ".join(matches.columns))
        return

    by_venue = stats["by_venue"]
    fig = px.bar(by_venue.head(10), x="Venue", y="Matches", text="Matches", color="Matches",
                 color_continuous_scale="Oranges", title="Most-Used Venues (Top 10)")
    fig.update_traces(textposition="outside")
    show_plot(style_fig(fig))

    col1, col2 = st.columns(2)
    if stats["toss_win_pct"] is not None:
        with col1:
            st.markdown(metric_card("Toss winner also won the match", f"{stats['toss_win_pct']:.1f}%"),
                        unsafe_allow_html=True)
    if stats["toss_decision"] is not None:
        with col2:
            fig = px.pie(stats["toss_decision"], names="Decision", values="Matches",
                         title="Toss Decision Split", color_discrete_sequence=["#ffd700", "#ff8c00"])
            show_plot(style_fig(fig))

    venue = st.selectbox("Select Venue", list(by_venue["Venue"]), key="venue")
    venue_wins = wins_at_venue(matches, stats["venue_col"], venue)
    if venue_wins.empty:
        st.info("No completed matches recorded at this venue.")
        return
    fig = px.bar(venue_wins, x="Team", y="Wins", text="Wins", color="Wins",
                 color_continuous_scale="Oranges", title=f"Wins at {venue}")
    fig.update_traces(textposition="outside")
    show_plot(style_fig(fig))


# ========================= APP =========================
def main() -> None:
    st.set_page_config(page_title="IPL 2025 Analytics Dashboard", page_icon="🏏",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    data, problems = load_core_data()
    if problems:
        st.error("The dashboard can't start because of data problems:\n\n" + "\n".join(f"- {p}" for p in problems))
        st.stop()

    orange = to_numeric(data["orange_cap.csv"].copy(), ["Runs", "Average", "Strike_rate", "Sixes", "Fours"])
    purple = data["purple_cap.csv"].copy()
    purple = to_numeric(purple, ["Wickets", "Economy_rate", "Maidens"] + FOUR_WICKET_COLS + FIVE_WICKET_COLS)
    matches = data["matches.csv"]
    player_info = data["player_info.csv"].copy()
    player_info["AGE"] = pd.to_numeric(player_info["AGE"], errors="coerce")
    player_info["Salary"] = player_info["SOLD_PRICE"].apply(parse_salary)

    st.markdown("<h1 style='padding: 2rem 0 0.5rem 0;'>🏏 IPL 2025 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#b0b0b0; font-size:1.35rem; margin-bottom:2rem;'>"
                "Interactive player and team analytics</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Navigation")
        section = st.selectbox("Choose Section", ["Player Analysis", "Team Analysis", "Ground Analysis"],
                               label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### Current Season Leaders")
        if not orange.empty:
            oc = orange.loc[orange["Runs"].idxmax()]
            st.markdown(f"**Orange Cap** 🏆\n\n**{oc['Batsman']}**\n\n{int(oc['Runs'])} runs")
        if not purple.empty:
            pc = purple.loc[purple["Wickets"].idxmax()]
            st.markdown(f"**Purple Cap** 🏆\n\n**{pc['Bowler']}**\n\n{int(pc['Wickets'])} wickets")

    if section == "Player Analysis":
        tab1, tab2, tab3 = st.tabs(["Batsmen 🔥", "Bowlers 🏏", "All-Rounders ⚡"])
        with tab1:
            render_batsmen(orange)
        with tab2:
            render_bowlers(purple)
        with tab3:
            render_allrounders()
    elif section == "Team Analysis":
        render_team_analysis(matches, player_info)
    else:
        render_ground_analysis(matches)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#888; padding:30px; font-size:1.1rem;'>"
                "IPL 2025 Analytics Dashboard • Built with Streamlit, Pandas and Plotly</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
