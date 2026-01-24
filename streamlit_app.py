"""
Streamlit app for interactive pitch recommendation analysis.

This app allows you to:
1. Select a pitcher and load their data
2. Choose a pitch count and first pitch type
3. Visualize the top 2 next pitch recommendations
4. Tune cost function weights and see results update
5. View data depth metrics
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import polars as pl
import logging
from utils import (
    get_pitches_all_sequences,
    return_top_sequences,
    cost_function_description_based,
    get_matching_sequences,
    analyze_by_situation,
    optimize_all_situations,
    get_pitch_recommendation
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Pitch Recommendation System",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ Pitch Recommendation System")
st.markdown("Interactive pitch tunneling optimization and visualization")

# Sidebar for inputs
st.sidebar.header("Pitcher Selection")

# Pitcher lookup
pitcher_last = st.sidebar.text_input("Pitcher Last Name", value="wheeler")
pitcher_first = st.sidebar.text_input("Pitcher First Name", value="zack")
start_date = st.sidebar.text_input("Start Date", value="2020-04-02")
end_date = st.sidebar.text_input("End Date", value="2025-10-09")

# Load data button
if st.sidebar.button("Load Pitcher Data"):
    with st.spinner(f"Loading data for {pitcher_first} {pitcher_last}..."):
        try:
            sequences_df = get_pitches_all_sequences(pitcher_last, pitcher_first, start_date, end_date)
            st.session_state['sequences_df'] = sequences_df
            st.sidebar.success(f"Loaded {len(sequences_df)} pitch sequences!")

            # Get top sequences
            top_sequences = return_top_sequences(sequences_df, top_n=10)
            st.session_state['top_sequences'] = top_sequences
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")
            logger.exception(e)

# Check if data is loaded
if 'sequences_df' not in st.session_state:
    st.info("👈 Please load pitcher data from the sidebar to begin")
    st.stop()

sequences_df = st.session_state['sequences_df']

# Main content area
st.sidebar.header("Game Situation")

# Count selection (after pitch 1)
balls = st.sidebar.selectbox("Balls (after pitch 1)", options=[0, 1, 2, 3], index=0)
strikes = st.sidebar.selectbox("Strikes (after pitch 1)", options=[0, 1, 2], index=0)
count_label = f"{balls}-{strikes}"

# Filter to this count
count_sequences = sequences_df.filter(
    pl.col('count_after_pitch1') == count_label
)

if len(count_sequences) == 0:
    st.warning(f"No data available for count {count_label}")
    st.stop()

# Get available first pitch types for this count
available_sequences = count_sequences.group_by('pitch_sequence').agg(
    pl.count().alias('count')
).sort('count', descending=True)

pitch_sequence_options = available_sequences['pitch_sequence'].to_list()

# First pitch selection
st.sidebar.header("Pitch Sequence")
selected_sequence = st.sidebar.selectbox(
    "Select Pitch Sequence (Pitch1-Pitch2)",
    options=pitch_sequence_options,
    help="Most common sequences listed first"
)

pitch1_type = selected_sequence.split('-')[0]
pitch2_type = selected_sequence.split('-')[1]

st.sidebar.markdown(f"**First Pitch:** {pitch1_type}")
st.sidebar.markdown(f"**Next Pitch:** {pitch2_type}")

# Cost function tuning
st.sidebar.header("Cost Function Weights")
st.sidebar.markdown("Adjust these to change optimization priorities")

weight_good = st.sidebar.slider(
    "Good Outcomes (strikes/whiffs)",
    min_value=-1.0,
    max_value=0.0,
    value=-0.5,
    step=0.1,
    help="Negative = reward. More negative = prioritize strikes/whiffs more"
)

weight_bad = st.sidebar.slider(
    "Bad Outcomes (balls)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1,
    help="Positive = penalty. Higher = avoid balls more"
)

weight_contact = st.sidebar.slider(
    "Contact Quality (xwOBA)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1,
    help="Positive = penalty. Higher = prioritize weak contact more"
)

# Main display area - two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"Strike Zone Visualization - Count: {count_label}")

    # Filter data for selected sequence and count
    sequence_data = count_sequences.filter(
        pl.col('pitch_sequence') == selected_sequence
    )

    if len(sequence_data) < 20:
        st.warning(f"Limited data for {selected_sequence} at count {count_label}: {len(sequence_data)} observations")

    # Calculate statistics for this sequence
    stats = sequence_data.select([
        pl.col('delta_plate_x').std().alias('std_x'),
        pl.col('delta_plate_z').std().alias('std_z'),
        pl.col('delta_velocity').std().alias('std_vel'),
        pl.col('delta_plate_x').mean().alias('mean_x'),
        pl.col('delta_plate_z').mean().alias('mean_z'),
        pl.col('delta_velocity').mean().alias('mean_vel'),
    ])

    # Set up tolerances
    tolerance_multiplier = 3.0
    tol_x = stats['std_x'][0] * tolerance_multiplier
    tol_z = stats['std_z'][0] * tolerance_multiplier
    tol_vel = stats['std_vel'][0] * tolerance_multiplier
    tolerances_list = [tol_x, tol_z, tol_vel]

    # Create a grid of points to evaluate cost function
    x_range = np.linspace(-2.5, 2.5, 40)  # Horizontal position
    z_range = np.linspace(1.0, 4.0, 40)   # Vertical position
    X, Z = np.meshgrid(x_range, z_range)

    # Calculate cost at each point
    costs = np.zeros_like(X)
    with st.spinner("Calculating cost function..."):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Use mean velocity delta for visualization
                params = [X[i, j], Z[i, j], stats['mean_vel'][0]]
                try:
                    cost = cost_function_description_based(
                        params,
                        sequence_data,
                        tolerances_list,
                        weight_good=weight_good,
                        weight_bad=weight_bad,
                        weight_contact=weight_contact
                    )
                    costs[i, j] = cost
                except:
                    costs[i, j] = np.nan

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot cost function heatmap
    contour = ax.contourf(X, Z, costs, levels=20, cmap='RdYlGn_r', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Cost (lower is better)')

    # Draw strike zone
    strike_zone = patches.Rectangle(
        (-17/24, 1.5),  # Bottom left corner (converted to feet)
        34/24,          # Width in feet
        2.0,            # Height (approximate)
        linewidth=2,
        edgecolor='black',
        facecolor='none',
        linestyle='--'
    )
    ax.add_patch(strike_zone)

    # Plot actual pitch locations for pitch 2
    pitch2_locs = sequence_data.select(['next_plate_x', 'next_plate_z'])
    ax.scatter(
        pitch2_locs['next_plate_x'],
        pitch2_locs['next_plate_z'],
        c='blue',
        alpha=0.3,
        s=30,
        label=f'{pitch2_type} locations (historical)'
    )

    # Find and plot optimal location
    min_cost_idx = np.unravel_index(np.nanargmin(costs), costs.shape)
    optimal_x = X[min_cost_idx]
    optimal_z = Z[min_cost_idx]
    optimal_cost = costs[min_cost_idx]

    ax.scatter(
        [optimal_x],
        [optimal_z],
        c='red',
        s=200,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label=f'Optimal location (cost={optimal_cost:.3f})',
        zorder=10
    )

    ax.set_xlabel('Horizontal Position (ft, catcher view)', fontsize=12)
    ax.set_ylabel('Vertical Position (ft)', fontsize=12)
    ax.set_title(f'{pitch1_type} → {pitch2_type} at {count_label} count', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(1.0, 4.0)

    st.pyplot(fig)

with col2:
    st.header("Analysis")

    # Data depth
    st.subheader("Data Depth")
    st.metric("Total Observations", len(sequence_data))

    # Count distribution
    count_dist = sequences_df.group_by('count_after_pitch1').agg(
        pl.count().alias('n')
    ).sort('n', descending=True)

    st.markdown("**Observations by Count:**")
    for row in count_dist.head(5).iter_rows(named=True):
        st.markdown(f"- {row['count_after_pitch1']}: {row['n']}")

    # Optimal recommendation
    st.subheader("Optimal Recommendation")
    st.markdown(f"**Next Pitch Location:**")
    st.markdown(f"- Horizontal: {optimal_x:.2f} ft")
    st.markdown(f"- Vertical: {optimal_z:.2f} ft")
    st.markdown(f"- Velocity Δ: {stats['mean_vel'][0]:.1f} mph")
    st.markdown(f"- Expected Cost: {optimal_cost:.3f}")

    # Outcome breakdown for optimal region
    matching = get_matching_sequences(
        [optimal_x, optimal_z, stats['mean_vel'][0]],
        sequence_data,
        tolerances_list
    )

    if len(matching) > 0:
        st.subheader("Outcome Distribution")
        st.markdown(f"*(Based on {len(matching)} similar pitches)*")

        # Calculate outcome percentages
        total = len(matching)

        good = matching.filter(pl.col('next_description').is_in([
            'swinging_strike', 'swinging_strike_blocked', 'called_strike', 'foul_tip'
        ]))
        bad = matching.filter(pl.col('next_description').is_in([
            'ball', 'blocked_ball', 'hit_by_pitch'
        ]))
        contact = matching.filter(pl.col('next_description') == 'hit_into_play')

        st.markdown(f"- Strikes/Whiffs: {len(good)/total*100:.1f}%")
        st.markdown(f"- Balls: {len(bad)/total*100:.1f}%")
        st.markdown(f"- Contact: {len(contact)/total*100:.1f}%")

        if len(contact) > 0:
            avg_xwoba = contact['next_xwoba'].mean()
            if avg_xwoba:
                st.markdown(f"- Avg xwOBA on contact: {avg_xwoba:.3f}")

# Bottom section - top recommendations
st.header("Top Pitch Recommendations for this Count")

# Get top 3 sequences for this count
top_for_count = count_sequences.group_by('pitch_sequence').agg(
    pl.count().alias('n_obs')
).sort('n_obs', descending=True).head(5)

st.markdown(f"**Most common pitch sequences at {count_label} count:**")

for row in top_for_count.iter_rows(named=True):
    seq = row['pitch_sequence']
    n = row['n_obs']

    # Calculate optimal cost for this sequence
    seq_data = count_sequences.filter(pl.col('pitch_sequence') == seq)

    if len(seq_data) >= 20:
        seq_stats = seq_data.select([
            pl.col('delta_plate_x').std().alias('std_x'),
            pl.col('delta_plate_z').std().alias('std_z'),
            pl.col('delta_velocity').std().alias('std_vel'),
            pl.col('delta_plate_x').mean().alias('mean_x'),
            pl.col('delta_plate_z').mean().alias('mean_z'),
            pl.col('delta_velocity').mean().alias('mean_vel'),
        ])

        seq_tol = [
            seq_stats['std_x'][0] * 3,
            seq_stats['std_z'][0] * 3,
            seq_stats['std_vel'][0] * 3
        ]

        # Calculate cost at mean location
        cost = cost_function_description_based(
            [seq_stats['mean_x'][0], seq_stats['mean_z'][0], seq_stats['mean_vel'][0]],
            seq_data,
            seq_tol,
            weight_good=weight_good,
            weight_bad=weight_bad,
            weight_contact=weight_contact
        )

        st.markdown(f"- **{seq}**: {n} obs, Cost = {cost:.3f}")
    else:
        st.markdown(f"- **{seq}**: {n} obs (insufficient data)")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data from Statcast via pybaseball")
