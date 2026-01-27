"""
Streamlit app for interactive pitch recommendation using Bayesian optimization.

This app allows you to:
1. Select a pitcher and load their data
2. Choose a pitch count and first pitch type
3. Run Bayesian optimization to find optimal next pitch locations
4. Tune cost function weights and see results update
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import polars as pl
import logging
from utils import (
    get_pitches_all_sequences,
    get_matching_sequences,
    optimize_sequence,
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
st.markdown("Bayesian optimization for pitch tunneling")

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
            sequences_df = get_pitches_all_sequences(
                pitcher_last, pitcher_first, start_date, end_date
            )
            st.session_state['sequences_df'] = sequences_df
            # Clear stale optimizer results
            st.session_state.pop('optimizer_results', None)
            st.sidebar.success(f"Loaded {len(sequences_df)} pitch sequences!")
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

# Count selection (after pitch 1) - default to 1-0 since 0-0 rarely has a previous pitch
balls = st.sidebar.selectbox("Balls (after pitch 1)", options=[0, 1, 2, 3], index=1)
strikes = st.sidebar.selectbox("Strikes (after pitch 1)", options=[0, 1, 2], index=0)
count_label = f"{balls}-{strikes}"

# Skip 0-0 count as it rarely has a previous pitch
if count_label == "0-0":
    st.warning("0-0 count is not available - there is rarely a previous pitch to analyze at this count.")
    st.stop()

# Filter to this count
count_sequences = sequences_df.filter(
    pl.col('count_after_pitch1') == count_label
)

if len(count_sequences) == 0:
    st.warning(f"No data available for count {count_label}")
    st.stop()

# First pitch type selection
st.sidebar.header("First Pitch")
first_pitch_types = count_sequences.group_by('pitch_type').agg(
    pl.count().alias('n')
).sort('n', descending=True)
pitch1_options = first_pitch_types['pitch_type'].to_list()

selected_pitch1 = st.sidebar.selectbox(
    "First Pitch Type",
    options=pitch1_options,
    help="What pitch was just thrown? Most common listed first."
)

# Cost function tuning
st.sidebar.header("Cost Function Weights")
st.sidebar.markdown("Adjust these to change optimization priorities")

weight_good = st.sidebar.slider(
    "Good Outcomes (strikes/whiffs)",
    min_value=-1.0, max_value=0.0, value=-0.5, step=0.1,
    help="Negative = reward. More negative = prioritize strikes/whiffs more"
)

weight_bad = st.sidebar.slider(
    "Bad Outcomes (balls)",
    min_value=0.0, max_value=1.0, value=0.3, step=0.1,
    help="Positive = penalty. Higher = avoid balls more"
)

weight_contact = st.sidebar.slider(
    "Contact Quality (xwOBA)",
    min_value=0.0, max_value=1.0, value=0.3, step=0.1,
    help="Positive = penalty. Higher = prioritize weak contact more"
)

# Display cost function equation
st.sidebar.markdown("---")
st.sidebar.markdown("**Cost Function:**")
st.sidebar.latex(
    r"\text{Cost} = w_g \cdot R_{\text{good}} + w_b \cdot R_{\text{bad}}"
    r" + w_c \cdot R_{\text{contact}} \cdot \frac{\text{xwOBA}}{0.320}"
)
st.sidebar.markdown("**Current Weights:**")
st.sidebar.markdown(f"- $w_g$ = {weight_good} (strikes/whiffs)")
st.sidebar.markdown(f"- $w_b$ = {weight_bad} (balls)")
st.sidebar.markdown(f"- $w_c$ = {weight_contact} (contact quality)")
st.sidebar.caption("Lower cost = better for pitcher. Negative $w_g$ rewards good outcomes.")

# ─── Main content area ───────────────────────────────────────────────────────

# Filter to sequences starting with selected first pitch type
pitch1_data = count_sequences.filter(pl.col('pitch_type') == selected_pitch1)

if len(pitch1_data) < 10:
    st.warning(f"Not enough data for {selected_pitch1} at count {count_label}")
    st.stop()

# Get top 3 sequences from this first pitch type
top_seqs = pitch1_data.group_by('pitch_sequence').agg(
    pl.count().alias('n_obs')
).sort('n_obs', descending=True).head(3)

seq_list = list(top_seqs.iter_rows(named=True))

# Average first pitch location (anchor point for the visualization)
first_pitch_loc = pitch1_data.select([
    pl.col('plate_x').mean().alias('avg_x'),
    pl.col('plate_z').mean().alias('avg_z'),
])
first_x = first_pitch_loc['avg_x'][0]
first_z = first_pitch_loc['avg_z'][0]

# Header
st.header(f"Bayesian Optimizer: {selected_pitch1} → ? at {count_label}")
st.markdown(
    f"Finding optimal next pitch locations for the top 3 sequences "
    f"starting with **{selected_pitch1}** when the count is **{count_label}**."
)
st.markdown(
    f"The **{selected_pitch1}** anchor point is its average location at this count "
    f"({first_x:.2f}, {first_z:.2f}). The optimizer finds optimal *deltas* "
    f"from pitch 1 to pitch 2, bounded by historically observed ranges."
)

# Show data availability
col_info = st.columns(len(seq_list))
for i, seq_info in enumerate(seq_list):
    with col_info[i]:
        p2 = seq_info['pitch_sequence'].split('-')[1]
        st.metric(
            f"{seq_info['pitch_sequence']}",
            f"{seq_info['n_obs']} obs",
            help=f"Next pitch: {p2}"
        )

# Run optimizer button
if st.button("Run Bayesian Optimizer", type="primary"):
    results = []
    progress_bar = st.progress(0, text="Starting optimization...")

    for i, seq_info in enumerate(seq_list):
        seq = seq_info['pitch_sequence']
        n_obs = seq_info['n_obs']

        progress_bar.progress(
            i / len(seq_list),
            text=f"Optimizing {seq} ({n_obs} obs)..."
        )

        seq_data = pitch1_data.filter(pl.col('pitch_sequence') == seq)

        opt_result = optimize_sequence(
            seq_data,
            weight_good=weight_good,
            weight_bad=weight_bad,
            weight_contact=weight_contact,
            n_calls=30
        )

        if opt_result is not None:
            pitch2_type = seq.split('-')[1]
            results.append({
                'sequence': seq,
                'pitch2': pitch2_type,
                'best_delta_x': opt_result['best_delta_x'],
                'best_delta_z': opt_result['best_delta_z'],
                'best_delta_vel': opt_result['best_delta_velocity'],
                'best_cost': opt_result['best_cost'],
                'opt_x': first_x + opt_result['best_delta_x'],
                'opt_z': first_z + opt_result['best_delta_z'],
                'n_obs': n_obs,
            })

    progress_bar.progress(1.0, text="Done!")

    # Sort by cost (lower is better)
    results = sorted(results, key=lambda x: x['best_cost'])

    # Store results + context so they persist across reruns
    st.session_state['optimizer_results'] = results
    st.session_state['opt_first_x'] = first_x
    st.session_state['opt_first_z'] = first_z
    st.session_state['opt_pitch1'] = selected_pitch1
    st.session_state['opt_count'] = count_label
    st.rerun()

# ─── Display results ─────────────────────────────────────────────────────────

if 'optimizer_results' not in st.session_state:
    st.info("Click **Run Bayesian Optimizer** above to generate recommendations.")
    st.stop()

results = st.session_state['optimizer_results']
opt_first_x = st.session_state['opt_first_x']
opt_first_z = st.session_state['opt_first_z']
opt_pitch1 = st.session_state['opt_pitch1']
opt_count = st.session_state['opt_count']

# Warn if inputs changed since last run
if opt_pitch1 != selected_pitch1 or opt_count != count_label:
    st.warning(
        "Inputs have changed since last optimization. "
        "Click **Run Bayesian Optimizer** to update."
    )

if not results:
    st.warning("Optimizer returned no results. Try a different count or pitch type.")
    st.stop()

# Color map for pitch types
PITCH_COLORS = {
    'FF': '#E41A1C',  # Red - Four-seam fastball
    'SI': '#377EB8',  # Blue - Sinker
    'FC': '#4DAF4A',  # Green - Cutter
    'SL': '#984EA3',  # Purple - Slider
    'CU': '#FF7F00',  # Orange - Curveball
    'CH': '#F0E130',  # Yellow - Changeup
    'FS': '#A65628',  # Brown - Splitter
    'KC': '#F781BF',  # Pink - Knuckle curve
    'ST': '#999999',  # Gray - Sweeper
}

# ─── Strike zone visualization ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))

# Strike zone rectangle
strike_zone = patches.Rectangle(
    (-17/24, 1.5), 34/24, 2.0,
    linewidth=2, edgecolor='black', facecolor='lightgray',
    alpha=0.2, linestyle='--'
)
ax.add_patch(strike_zone)

# First pitch (anchor)
ax.scatter(
    [opt_first_x], [opt_first_z],
    c=PITCH_COLORS.get(opt_pitch1, 'gray'),
    s=600, alpha=0.8, edgecolors='black', linewidths=3, zorder=5
)
ax.annotate(
    f'{opt_pitch1}\n(avg)',
    (opt_first_x, opt_first_z),
    ha='center', va='center',
    fontsize=10, fontweight='bold', color='white', zorder=6
)

# Optimized next pitches
for i, rec in enumerate(results[:3]):
    pitch2 = rec['pitch2']
    color = PITCH_COLORS.get(pitch2, 'gray')

    # Arrow from pitch 1 to optimized pitch 2
    ax.annotate(
        '',
        xy=(rec['opt_x'], rec['opt_z']),
        xytext=(opt_first_x, opt_first_z),
        arrowprops=dict(arrowstyle='-|>', color=color, lw=2.5, alpha=0.6),
        zorder=3
    )

    # Pitch 2 circle
    ax.scatter(
        [rec['opt_x']], [rec['opt_z']],
        c=color, s=400, alpha=0.9,
        edgecolors='black', linewidths=2, zorder=7
    )
    ax.annotate(
        pitch2,
        (rec['opt_x'], rec['opt_z']),
        ha='center', va='center',
        fontsize=11, fontweight='bold', color='white', zorder=8
    )

    # Cost label
    ax.annotate(
        f'#{i+1}: cost={rec["best_cost"]:.3f}',
        (rec['opt_x'], rec['opt_z'] - 0.22),
        ha='center', va='top',
        fontsize=9, color=color, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        zorder=9
    )

ax.set_xlabel('Horizontal Position (ft, catcher view)', fontsize=12)
ax.set_ylabel('Vertical Position (ft)', fontsize=12)
ax.set_title(
    f'Bayesian Optimizer: Best Next Pitches after {opt_pitch1} at {opt_count}',
    fontsize=14
)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0.5, 4.5)
ax.grid(True, alpha=0.3)

# Legend
legend_elements = [
    patches.Patch(
        facecolor=PITCH_COLORS.get(opt_pitch1, 'gray'),
        edgecolor='black',
        label=f'Pitch 1: {opt_pitch1} (avg location)'
    )
]
for rec in results[:3]:
    legend_elements.append(
        patches.Patch(
            facecolor=PITCH_COLORS.get(rec['pitch2'], 'gray'),
            edgecolor='black',
            label=f"{rec['pitch2']}: cost={rec['best_cost']:.3f} ({rec['n_obs']} obs)"
        )
    )
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

st.pyplot(fig)

# ─── Detailed results ────────────────────────────────────────────────────────

st.subheader("Optimization Details")

for i, rec in enumerate(results[:3]):
    with st.expander(
        f"#{i+1}: {rec['sequence']} — Cost: {rec['best_cost']:.3f}",
        expanded=(i == 0)
    ):
        col_delta, col_outcomes = st.columns(2)

        with col_delta:
            st.markdown("**Optimal Delta (from pitch 1):**")
            st.markdown(f"- Horizontal: {rec['best_delta_x']:.2f} ft")
            st.markdown(f"- Vertical: {rec['best_delta_z']:.2f} ft")
            st.markdown(f"- Velocity: {rec['best_delta_vel']:.1f} mph")
            st.markdown(f"- Observations: {rec['n_obs']}")

        with col_outcomes:
            # Show outcome distribution at the optimal delta
            seq_data = pitch1_data.filter(
                pl.col('pitch_sequence') == rec['sequence']
            )
            seq_stats = seq_data.select([
                pl.col('delta_plate_x').std().alias('std_x'),
                pl.col('delta_plate_z').std().alias('std_z'),
                pl.col('delta_velocity').std().alias('std_vel'),
            ])
            tolerances = [
                seq_stats['std_x'][0] * 3,
                seq_stats['std_z'][0] * 3,
                seq_stats['std_vel'][0] * 3,
            ]
            matching = get_matching_sequences(
                [rec['best_delta_x'], rec['best_delta_z'], rec['best_delta_vel']],
                seq_data,
                tolerances
            )

            if len(matching) > 0:
                total = len(matching)
                good = matching.filter(pl.col('next_description').is_in([
                    'swinging_strike', 'swinging_strike_blocked',
                    'called_strike', 'foul_tip'
                ]))
                bad = matching.filter(pl.col('next_description').is_in([
                    'ball', 'blocked_ball', 'hit_by_pitch'
                ]))
                contact = matching.filter(
                    pl.col('next_description') == 'hit_into_play'
                )

                st.markdown(f"**Outcomes ({total} similar pitches):**")
                st.markdown(f"- Strikes/Whiffs: {len(good)/total*100:.1f}%")
                st.markdown(f"- Balls: {len(bad)/total*100:.1f}%")
                st.markdown(f"- Contact: {len(contact)/total*100:.1f}%")

                if len(contact) > 0:
                    avg_xwoba = contact['next_xwoba'].mean()
                    if avg_xwoba:
                        st.markdown(f"- Avg xwOBA: {avg_xwoba:.3f}")

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit | Bayesian optimization via scikit-optimize "
    "| Data from Statcast via pybaseball"
)
