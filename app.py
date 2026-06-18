import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(page_title="Copa 2026 Live Calibration", layout="wide")
st.title("🏆 Copa 2026 Match Predictor — Live Calibration Tracker")

# Paths
PREDICTIONS_PATH = Path("predictions/group_stage_predictions.csv")
RESULTS_PATH = Path("results/actual_outcomes.csv")
RESULTS_PATH.parent.mkdir(exist_ok=True)

# Load predictions
@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)

# Load or initialize results
def load_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    else:
        return pd.DataFrame(columns=['match_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'outcome', 'timestamp'])

# Save results
def save_results(df):
    df.to_csv(RESULTS_PATH, index=False)
    st.success("✓ Result saved!")

# Calculate Brier score
def brier_score(predictions_df, results_df):
    """Calculate Brier score for all completed matches"""
    if len(results_df) == 0:
        return None
    
    # Merge predictions with results
    merged = predictions_df.merge(results_df, on=['match_id', 'date'], how='inner')
    
    # Map outcomes to binary vectors
    p_win = merged['p_home_win'].values
    p_draw = merged['p_draw'].values
    p_loss = merged['p_away_win'].values
    
    outcomes = merged['outcome'].values
    o_win = (outcomes == 'home_win').astype(int)
    o_draw = (outcomes == 'draw').astype(int)
    o_loss = (outcomes == 'away_win').astype(int)
    
    bs = np.mean((p_win - o_win)**2 + (p_draw - o_draw)**2 + (p_loss - o_loss)**2)
    return bs

# Calculate log-loss
def log_loss(predictions_df, results_df):
    """Calculate log-loss for all completed matches"""
    if len(results_df) == 0:
        return None
    
    merged = predictions_df.merge(results_df, on=['match_id', 'date'], how='inner')
    
    p_win = np.clip(merged['p_home_win'].values, 1e-15, 1 - 1e-15)
    p_draw = np.clip(merged['p_draw'].values, 1e-15, 1 - 1e-15)
    p_loss = np.clip(merged['p_away_win'].values, 1e-15, 1 - 1e-15)
    
    outcomes = merged['outcome'].values
    o_win = (outcomes == 'home_win').astype(int)
    o_draw = (outcomes == 'draw').astype(int)
    o_loss = (outcomes == 'away_win').astype(int)
    
    ll = -np.mean(o_win * np.log(p_win) + o_draw * np.log(p_draw) + o_loss * np.log(p_loss))
    return ll

# Create reliability diagram
def reliability_diagram(predictions_df, results_df):
    """Create calibration reliability diagram"""
    if len(results_df) == 0:
        return None
    
    merged = predictions_df.merge(results_df, on=['match_id', 'date'], how='inner')
    
    p_win = merged['p_home_win'].values
    outcomes = merged['outcome'].values
    o_win = (outcomes == 'home_win').astype(int)
    
    # Bin predictions
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_true = []
    bin_count = []
    
    for i in range(len(bins) - 1):
        mask = (p_win >= bins[i]) & (p_win < bins[i+1])
        if np.sum(mask) > 0:
            bin_true.append(np.mean(o_win[mask]))
            bin_count.append(np.sum(mask))
        else:
            bin_true.append(None)
            bin_count.append(0)
    
    return bin_centers, bin_true, bin_count

# ============ MAIN DASHBOARD ============

predictions = load_predictions()
results = load_results()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Dashboard", "Enter Results", "Match History"])

if page == "Dashboard":
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(predictions))
    with col2:
        st.metric("Completed", len(results))
    with col3:
        remaining = len(predictions) - len(results)
        st.metric("Remaining", remaining)
    with col4:
        pct = (len(results) / len(predictions) * 100) if len(predictions) > 0 else 0
        st.metric("Progress", f"{pct:.1f}%")
    
    st.divider()
    
    # Calibration metrics
    if len(results) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            bs = brier_score(predictions, results)
            st.metric("Brier Score", f"{bs:.4f}", help="Lower is better. Perfect: 0.0, Random: 0.667")
        
        with col2:
            ll = log_loss(predictions, results)
            st.metric("Log-Loss", f"{ll:.4f}", help="Lower is better. Penalizes confident wrong predictions.")
    
    st.divider()
    
    # Reliability diagram
    if len(results) > 5:  # Need enough data for meaningful diagram
        st.subheader("📈 Calibration Reliability Diagram")
        
        bin_centers, bin_true, bin_count = reliability_diagram(predictions, results)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
        
        # Reliability diagram
        valid_idx = [i for i in range(len(bin_true)) if bin_true[i] is not None]
        if valid_idx:
            x = bin_centers[valid_idx]
            y = np.array(bin_true)[valid_idx]
            sizes = np.array(bin_count)[valid_idx] * 10
            
            ax.scatter(x, y, s=sizes, alpha=0.6, edgecolors='black', linewidth=2, color='steelblue')
        
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Frequency', fontsize=11)
        ax.set_title(f'Calibration Check (n={len(results)} matches)', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
    elif len(results) > 0:
        st.info(f"📊 Need at least 5 completed matches for meaningful reliability diagram. Current: {len(results)}")
    else:
        st.info("📊 Results will appear here as matches are completed.")
    
    st.divider()
    
    # Recent matches
    if len(results) > 0:
        st.subheader("📋 Recently Completed Matches")
        
        recent = predictions.merge(results, on=['match_id', 'date'], how='inner', suffixes=('', '_result')).sort_values('date', ascending=False).head(10)
        
        display_cols = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'p_home_win', 'p_draw', 'p_away_win', 'predicted_winner', 'outcome']
        display_df = recent[display_cols].copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        display_df['p_home_win'] = display_df['p_home_win'].apply(lambda x: f"{x:.1%}")
        display_df['p_draw'] = display_df['p_draw'].apply(lambda x: f"{x:.1%}")
        display_df['p_away_win'] = display_df['p_away_win'].apply(lambda x: f"{x:.1%}")
        display_df = display_df.rename(columns={
            'date': 'Date',
            'home_team': 'Home',
            'away_team': 'Away',
            'home_goals': 'HG',
            'away_goals': 'AG',
            'p_home_win': 'P(H)',
            'p_draw': 'P(D)',
            'p_away_win': 'P(A)',
            'outcome': 'Result'
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "Enter Results":
    st.subheader("⚽ Enter Match Result")
    
    # Get upcoming matches
    upcoming = predictions[~predictions['match_id'].isin(results['match_id'].values)].sort_values('date')
    
    if len(upcoming) == 0:
        st.success("✅ All matches completed!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # Match selector
            match_options = [f"{row['date']} | {row['home_team']} vs {row['away_team']} ({row['group']})" 
                           for _, row in upcoming.iterrows()]
            selected = st.selectbox("Select Match:", match_options)
            selected_idx = match_options.index(selected)
            selected_match = upcoming.iloc[selected_idx]
        
        with col2:
            st.write("")  # Spacing
        
        # Display match info
        st.info(f"**{selected_match['home_team']}** vs **{selected_match['away_team']}** | {selected_match['group']}")
        st.write(f"Predicted: {selected_match['predicted_winner']} to win | Top scoreline: {selected_match['top_scoreline']}")
        
        # Score input
        col1, col2 = st.columns(2)
        
        with col1:
            home_goals = st.number_input(f"{selected_match['home_team']} Goals", min_value=0, max_value=10, step=1)
        
        with col2:
            away_goals = st.number_input(f"{selected_match['away_team']} Goals", min_value=0, max_value=10, step=1)
        
        # Determine outcome
        if home_goals > away_goals:
            outcome = "home_win"
            result_text = f"✅ **{selected_match['home_team']}** wins **{home_goals}-{away_goals}**"
        elif away_goals > home_goals:
            outcome = "away_win"
            result_text = f"✅ **{selected_match['away_team']}** wins **{home_goals}-{away_goals}**"
        else:
            outcome = "draw"
            result_text = f"⚖️ **Draw** **{home_goals}-{home_goals}**"
        
        st.write(result_text)
        
        # Submit button
        if st.button("✓ Submit Result", type="primary"):
            new_result = pd.DataFrame({
                'match_id': [selected_match['match_id']],
                'date': [selected_match['date']],
                'home_team': [selected_match['home_team']],
                'away_team': [selected_match['away_team']],
                'home_goals': [home_goals],
                'away_goals': [away_goals],
                'outcome': [outcome],
                'timestamp': [datetime.now().isoformat()]
            })
            
            results = pd.concat([results, new_result], ignore_index=True)
            save_results(results)
            st.rerun()

elif page == "Match History":
    st.subheader("📚 Match History")
    
    if len(results) == 0:
        st.info("No matches completed yet.")
    else:
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            groups = ['All'] + sorted(predictions['group'].unique().tolist())
            selected_group = st.selectbox("Filter by Group:", groups)
        
        with col2:
            outcomes = ['All', 'home_win', 'draw', 'away_win']
            selected_outcome = st.selectbox("Filter by Outcome:", outcomes)
        
        # Apply filters
        filtered = predictions.merge(results, on=['match_id', 'date'], how='inner', suffixes=('', '_result'))
        
        if selected_group != 'All':
            filtered = filtered[filtered['group'] == selected_group]
        
        if selected_outcome != 'All':
            filtered = filtered[filtered['outcome'] == selected_outcome]
        
        filtered = filtered.sort_values('date')
        
        # Display
        display_cols = ['date', 'group', 'home_team', 'away_team', 'home_goals', 'away_goals', 'p_home_win', 'p_draw', 'p_away_win', 'predicted_winner', 'outcome']
        display_df = filtered[display_cols].copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        display_df['p_home_win'] = display_df['p_home_win'].apply(lambda x: f"{x:.1%}")
        display_df['p_draw'] = display_df['p_draw'].apply(lambda x: f"{x:.1%}")
        display_df['p_away_win'] = display_df['p_away_win'].apply(lambda x: f"{x:.1%}")
        display_df = display_df.rename(columns={
            'date': 'Date',
            'group': 'Group',
            'home_team': 'Home',
            'away_team': 'Away',
            'home_goals': 'HG',
            'away_goals': 'AG',
            'p_home_win': 'P(H)',
            'p_draw': 'P(D)',
            'p_away_win': 'P(A)',
            'predicted_winner': 'Predicted',
            'outcome': 'Actual'
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.write(f"\nTotal: {len(filtered)} matches")

st.divider()
st.markdown("**Last updated:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))