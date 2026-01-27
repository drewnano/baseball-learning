"""
Utility functions for optimizing pitch tunneling.

This module provides functions for:
1. Loading pitcher data from Statcast
2. Creating pitch sequences with deltas (location, velocity changes)
3. Cost functions for evaluating pitch sequences
4. Bayesian optimization for finding optimal pitch deltas
"""
import logging

from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
from skopt import gp_minimize
from skopt.space import Real
import polars as pl

# Set up logger
logger = logging.getLogger(__name__)

################################################################################
# DATA LOADING AND PREPARATION
################################################################################

def get_pitches_all_sequences(lastname, firstname, start_date, end_date):
    """
    Load pitcher data and create ALL consecutive two-pitch sequences.

    This approach looks at ALL consecutive pitches, not just
    at-bat ending ones. Provides much more data for optimization.

    Parameters:
    ----------
    lastname : str
        Pitcher's last name
    firstname : str
        Pitcher's first name
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format

    Returns:
    -------
    polars DataFrame
        All consecutive pitch sequences with delta columns, count info, etc.
    """
    # Find playerid lookup
    pitcher = playerid_lookup(lastname, firstname)
    pitcherid = pitcher.loc[0, "key_mlbam"]
    pitcherid = float(pitcherid)

    # Pull pitcher's pitch data from Statcast
    pitches = statcast_pitcher(start_date, end_date, pitcherid)

    # Convert from pandas to polars
    pitches = pl.from_pandas(pitches)

    # Create unique ID for each at-bat
    pitches = pitches.with_columns(
        pl.concat_str([
            pl.col("game_pk").cast(pl.Utf8),
            pl.lit("_"),
            pl.col("at_bat_number").cast(pl.Utf8)
        ]).alias("ab_id")
    )

    # Create sequences looking FORWARD to next pitch
    sequences = pitches.lazy().sort(['ab_id', 'pitch_number']).with_columns([
        # Get next pitch info (shift -1 to look forward)
        pl.col('pitch_type').shift(-1).over('ab_id').alias('next_pitch_type'),
        pl.col('plate_x').shift(-1).over('ab_id').alias('next_plate_x'),
        pl.col('plate_z').shift(-1).over('ab_id').alias('next_plate_z'),
        pl.col('release_speed').shift(-1).over('ab_id').alias('next_release_speed'),
        pl.col('description').shift(-1).over('ab_id').alias('next_description'),
        pl.col('estimated_woba_using_speedangle').shift(-1).over('ab_id').alias('next_xwoba'),
    ]).filter(
        # Only keep rows where there IS a next pitch (not last pitch of at-bat)
        pl.col('next_pitch_type').is_not_null()
    ).with_columns([
        # Calculate deltas FROM current pitch TO next pitch
        (pl.col('next_plate_x') - pl.col('plate_x')).alias('delta_plate_x'),
        (pl.col('next_plate_z') - pl.col('plate_z')).alias('delta_plate_z'),
        (pl.col('next_release_speed') - pl.col('release_speed')).alias('delta_velocity'),
    ])

    # Collect results
    sequences_df = sequences.collect()

    # Create pitch sequence label
    sequences_df = sequences_df.with_columns(
        pitch_sequence = pl.concat_str([
            pl.col('pitch_type'),
            pl.lit('-'),
            pl.col('next_pitch_type')
        ])
    )

    # Add count AFTER first pitch (for situation-based analysis)
    sequences_df = sequences_df.with_columns([
        # Calculate count after first pitch based on its outcome
        pl.when(pl.col('description').is_in(['ball', 'blocked_ball', 'hit_by_pitch']))
            .then(pl.col('balls') + 1)
            .otherwise(pl.col('balls'))
            .alias('balls_after_pitch1'),

        pl.when(pl.col('description').is_in(['called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul_tip']))
            .then(pl.col('strikes') + 1)
            .when(pl.col('description') == 'foul')
            .then(pl.when(pl.col('strikes') < 2).then(pl.col('strikes') + 1).otherwise(pl.col('strikes')))
            .otherwise(pl.col('strikes'))
            .alias('strikes_after_pitch1')
    ])

    # Create count label
    sequences_df = sequences_df.with_columns(
        count_after_pitch1 = pl.concat_str([
            pl.col('balls_after_pitch1').cast(pl.Utf8),
            pl.lit('-'),
            pl.col('strikes_after_pitch1').cast(pl.Utf8)
        ])
    )

    logger.info(f"Total pitch sequences: {len(sequences_df)}")
    logger.info(f"This includes ALL consecutive pitches (not just at-bat endings)")

    return sequences_df


################################################################################
# HELPER FUNCTIONS FOR OPTIMIZATION
################################################################################

def get_matching_sequences(params, sequences_df, tolerances):
    """
    Find matching sequences within tolerance of specified parameters.

    This function filters the sequences dataframe to find pitches that match
    the given delta parameters within the specified tolerances.

    Parameters:
    ----------
    params : list/tuple
        [delta_plate_x, delta_plate_z, delta_velocity]
    sequences_df : polars DataFrame
        DataFrame containing pitch sequences with delta columns
    tolerances : list/tuple
        [tol_x, tol_z, tol_velocity]

    Returns:
    -------
    polars DataFrame
        Filtered DataFrame with matching sequences
    """
    logger.info("Searching for matching sequences...")

    delta_x, delta_z, delta_velocity = params
    tol_x, tol_z, tol_velocity = tolerances
    logger.info(f"Parameters: delta_x={delta_x}, delta_z={delta_z}, delta_velocity={delta_velocity}")
    logger.info(f"Tolerances: tol_x={tol_x}, tol_z={tol_z}, tol_velocity={tol_velocity}")

    # Filter to sequences that are close enough to the parameters
    matching = sequences_df.filter(
        ((pl.col('delta_plate_x') - delta_x).abs() < tol_x) &
        ((pl.col('delta_plate_z') - delta_z).abs() < tol_z) &
        ((pl.col('delta_velocity') - delta_velocity).abs() < tol_velocity)
    )
    logger.info(f"Found {matching.shape[0]} matching sequences for params: {params}")
    return matching

################################################################################
# COST FUNCTIONS - TUNE THESE TO CHANGE OPTIMIZATION BEHAVIOR
################################################################################

def cost_function_description_based(params, sequences_df, tolerances,
                                     weight_good=-0.5, weight_bad=0.3, weight_contact=0.3):
    """
    Cost function based on pitch descriptions (outcomes).

    Evaluates pitch sequences based on the outcome of the NEXT pitch
    (pitch 2 in the sequence).

    Optimizes for good pitcher outcomes: whiffs, called strikes, weak contact.

    Parameters:
    ----------
    params : list/tuple
        [delta_plate_x, delta_plate_z, delta_velocity]
    sequences_df : polars DataFrame
        Your pitch sequences data (with next_description column)
    tolerances : list/tuple
        [tol_x, tol_z, tol_velocity]
    weight_good : float
        Weight for good outcomes (negative = reward). Default: -0.5
    weight_bad : float
        Weight for bad outcomes (positive = penalty). Default: 0.3
    weight_contact : float
        Weight for contact quality (positive = penalty). Default: 0.3

    Returns:
    -------
    float
        Cost value (lower is better for the pitcher)
    """
    # Get matching sequences
    matching = get_matching_sequences(params, sequences_df, tolerances)
    logger.info(f"Number of matching sequences used: {len(matching)}")

    if len(matching) < 15:
        return 0.5  # Penalty for sparse regions - need enough data for reliable rates

    # Categorize outcomes for the NEXT pitch (the second pitch in the sequence)
    # Good outcomes for pitcher (we want to maximize these)
    good_outcomes = matching.filter(pl.col('next_description').is_in([
        'swinging_strike',
        'swinging_strike_blocked',
        'called_strike',
        'foul_tip',  # Foul with 2 strikes = strikeout
    ]))

    # Neutral outcomes (contact but foul)
    neutral_outcomes = matching.filter(pl.col('next_description').is_in([
        'foul',
        'foul_bunt',
    ]))

    # Bad outcomes for pitcher
    bad_outcomes = matching.filter(pl.col('next_description').is_in([
        'ball',
        'blocked_ball',
        'hit_by_pitch',
    ]))

    # Contact in play - need to assess quality
    contact = matching.filter(pl.col('next_description') == 'hit_into_play')

    logger.info(f"Good outcomes: {len(good_outcomes)}, Neutral: {len(neutral_outcomes)}, "
                f"Bad: {len(bad_outcomes)}, Contact: {len(contact)}")

    # Calculate rates
    total = len(matching)
    good_rate = len(good_outcomes) / total if total > 0 else 0
    bad_rate = len(bad_outcomes) / total if total > 0 else 0
    contact_rate = len(contact) / total if total > 0 else 0

    # For contact, assess quality using xwOBA
    if len(contact) > 5:
        avg_xwoba = contact['next_xwoba'].mean()
        if avg_xwoba is None:
            avg_xwoba = 0.320  # League average if missing
    else:
        avg_xwoba = 0.320  # Assume league average

    # Build cost function
    # We want to MINIMIZE cost, so:
    # - Negative weight for good things (high good_rate lowers cost)
    # - Positive weight for bad things (high bad_rate raises cost)
    # - Penalize hard contact (high xwOBA)
    cost = (
        weight_good * good_rate +           # Reward strikes/whiffs
        weight_bad * bad_rate +             # Penalize balls
        weight_contact * contact_rate * (avg_xwoba / 0.320)  # Penalize hard contact
    )

    logger.info(f"Cost breakdown - good_rate: {good_rate:.3f}, bad_rate: {bad_rate:.3f}, "
                f"contact_rate: {contact_rate:.3f}, avg_xwoba: {avg_xwoba:.3f}, final_cost: {cost:.3f}")

    return cost


################################################################################
# BAYESIAN OPTIMIZATION
################################################################################

def optimize_sequence(seq_data, weight_good=-0.5, weight_bad=0.3,
                      weight_contact=0.3, n_calls=30):
    """
    Run Bayesian optimization for a single pitch sequence.

    Searches the delta space (delta_x, delta_z, delta_velocity) bounded by
    the min/max of historically observed deltas for this sequence type.
    Tolerances are set to 3x the standard deviation of each delta.

    Parameters:
    -----------
    seq_data : polars DataFrame
        Filtered data for one pitch sequence type (e.g. FF-SI at count 1-0)
    weight_good : float
        Cost weight for good outcomes (negative = reward)
    weight_bad : float
        Cost weight for bad outcomes (positive = penalty)
    weight_contact : float
        Cost weight for contact quality (positive = penalty)
    n_calls : int
        Number of Bayesian optimization iterations

    Returns:
    --------
    dict with best_delta_x, best_delta_z, best_delta_velocity, best_cost
    or None if insufficient data
    """
    if len(seq_data) < 20:
        logger.warning(f"Insufficient data for optimization ({len(seq_data)} < 20)")
        return None

    # Calculate stats for search bounds and tolerances
    stats = seq_data.select([
        pl.col('delta_plate_x').std().alias('std_x'),
        pl.col('delta_plate_z').std().alias('std_z'),
        pl.col('delta_velocity').std().alias('std_vel'),
        pl.col('delta_plate_x').mean().alias('mean_x'),
        pl.col('delta_plate_z').mean().alias('mean_z'),
        pl.col('delta_velocity').mean().alias('mean_vel'),
        pl.col('delta_plate_x').min().alias('min_x'),
        pl.col('delta_plate_z').min().alias('min_z'),
        pl.col('delta_velocity').min().alias('min_vel'),
        pl.col('delta_plate_x').max().alias('max_x'),
        pl.col('delta_plate_z').max().alias('max_z'),
        pl.col('delta_velocity').max().alias('max_vel'),
    ])

    # Tolerances = 3x standard deviation
    tolerances = [
        stats['std_x'][0] * 3,
        stats['std_z'][0] * 3,
        stats['std_vel'][0] * 3,
    ]

    # Objective function
    def objective(params):
        return cost_function_description_based(
            params, seq_data, tolerances,
            weight_good=weight_good,
            weight_bad=weight_bad,
            weight_contact=weight_contact
        )

    # Initial guess at the mean
    x0 = [stats['mean_x'][0], stats['mean_z'][0], stats['mean_vel'][0]]

    # Search space bounded by observed min/max
    dimensions = [
        Real(stats['min_x'][0], stats['max_x'][0], name='delta_plate_x'),
        Real(stats['min_z'][0], stats['max_z'][0], name='delta_plate_z'),
        Real(stats['min_vel'][0], stats['max_vel'][0], name='delta_velocity'),
    ]

    try:
        result = gp_minimize(
            objective,
            dimensions=dimensions,
            n_calls=n_calls,
            x0=x0,
            random_state=42
        )

        logger.info(f"Optimizer result: x={result.x}, cost={result.fun:.3f}")

        return {
            'best_delta_x': float(result.x[0]),
            'best_delta_z': float(result.x[1]),
            'best_delta_velocity': float(result.x[2]),
            'best_cost': float(result.fun),
        }
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return None
