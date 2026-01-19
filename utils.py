"""
Utility functions for optimizing pitch tunneling.

This module provides functions for:
1. Loading pitcher data from Statcast
2. Creating pitch sequences with deltas (location, velocity changes)
3. Cost functions for evaluating pitch sequences
4. Bayesian optimization for finding optimal pitch locations
5. Situation-based analysis (count-specific recommendations)
"""
#1. standard library imports
import os
import logging
import warnings
#2. third-party imports
from pybaseball import statcast
from pybaseball import statcast_pitcher
from pybaseball import statcast_batter
from pybaseball import playerid_lookup
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
import polars as pl

# Set up logger
logger = logging.getLogger(__name__)

################################################################################
# DATA LOADING AND PREPARATION
################################################################################

def get_pitches(lastname, firstname, start_date, end_date):
    """
    Load pitcher data from Statcast and prepare basic pitch sequences.

    This is the OLD approach that filters to only at-bat ending pitches.
    For more data, use get_pitches_all_sequences() instead.

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
        Pitch sequences with delta columns (backwards looking)
    """
    # Find playerid lookup of the player you specify
    pitcher = playerid_lookup(lastname, firstname)
    pitcherid = pitcher.loc[0, "key_mlbam"]
    pitcherid = float(pitcherid)

    # Pull pitcher's pitch data from Statcast
    pitches = statcast_pitcher(start_date, end_date, pitcherid)

    # Convert from pandas to polars dataframe
    pitches = pl.from_pandas(pitches)

    # Create unique ID for each at-bat
    pitches = pitches.with_columns(
        pl.concat_str([
            pl.col("game_pk").cast(pl.Utf8),
            pl.lit("_"),
            pl.col("at_bat_number").cast(pl.Utf8)
        ]).alias("ab_id")
    )

    # Process pitches using polars lazy version
    sequences = pitches.lazy().sort(['ab_id', 'pitch_number']).with_columns([
        # Get previous pitch info (shift 1 to look backward)
        pl.col('pitch_type').shift(1).over('ab_id').alias('first_pitch_type'),
        pl.col('plate_x').shift(1).over('ab_id').alias('first_plate_x'),
        pl.col('plate_z').shift(1).over('ab_id').alias('first_plate_z'),
        pl.col('release_speed').shift(1).over('ab_id').alias('first_release_speed'),
        pl.col('description').shift(1).over('ab_id').alias('first_description'),
        pl.col('events').shift(1).over('ab_id').alias('first_events'),
        pl.col('woba_value').shift(1).over('ab_id').alias('first_woba_value'),
    ]).filter(
        pl.col('events').is_not_null()  # Only keep at-bat ending pitches
    ).with_columns([
        (pl.col('plate_x') - pl.col('first_plate_x')).alias('delta_plate_x'),
        (pl.col('plate_z') - pl.col('first_plate_z')).alias('delta_plate_z'),
        (pl.col('release_speed') - pl.col('first_release_speed')).alias('delta_velocity'),
    ])

    # Collect to see results
    sequences_df = sequences.collect()

    # Create a pitch sequence label
    sequences_df = sequences_df.with_columns(
        pitch_sequence = pl.concat_str([
            pl.col('first_pitch_type'),
            pl.lit('-'),
            pl.col('pitch_type')
        ])
    )

    logger.info(f"Total pitch sequences: {len(sequences_df)}")
    return sequences_df


def get_pitches_all_sequences(lastname, firstname, start_date, end_date):
    """
    Load pitcher data and create ALL consecutive two-pitch sequences.

    This is the NEW approach that looks at ALL consecutive pitches, not just
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

def return_top_sequences(pitches, top_n=5):
    """
    Get the most common two-pitch sequences from raw pitch data.

    Parameters:
    ----------
    pitches : polars DataFrame
        Raw pitcher data with at least 'pitch_type', 'ab_id' columns
    top_n : int
        Number of top sequences to return

    Returns:
    -------
    polars DataFrame
        Top N sequences with counts
    """
    pitch_sequences = pitches.with_columns(
        next_pitch_type=pl.col("pitch_type").shift(-1).over("ab_id", order_by="pitch_number")
    )

    # Filter out the last pitch of each at-bat (where next_pitch_type is null)
    pitch_sequences = pitch_sequences.filter(pl.col('next_pitch_type').is_not_null())

    # Create a sequence label
    pitch_sequences = pitch_sequences.with_columns(
        sequence = pl.concat_str([
            pl.col('pitch_type'),
            pl.lit('-'),
            pl.col('next_pitch_type')
        ])
    )

    # Count each two-pitch sequence
    sequence_counts = pitch_sequences.group_by('sequence').agg(
        pl.count().alias('count')
    ).sort('count', descending=True)

    topnsequences = sequence_counts.head(top_n)
    return topnsequences



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
    NEW COST FUNCTION based on pitch descriptions (outcomes).

    This is the main cost function used in the notebook. It evaluates pitch sequences
    based on the outcome of the NEXT pitch (pitch 2 in the sequence).

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

    if len(matching) < 5:
        return 0.5  # Penalty for sparse regions

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


def cost_function_contact_quality(params, sequences_df, tolerances):
    """
    OLD COST FUNCTION: Two-part cost function combining whiff rate and contact quality.

    This was used in earlier versions. The new cost_function_description_based()
    is more comprehensive.

    Parameters:
    ----------
    params : list/tuple
        [delta_plate_x, delta_plate_z, delta_velocity]
    sequences_df : polars DataFrame
        Your pitch sequences data
    tolerances : list/tuple
        [tol_x, tol_z, tol_velocity]

    Returns:
    -------
    float
        Cost value (lower is better)
    """
    # Get matching sequences
    matching = get_matching_sequences(params, sequences_df, tolerances)
    logging.info(f"Number of matching sequences used: {len(matching)}")
    if len(matching) < 5:
        return 0.5  # Penalty for sparse regions
    
    # PART 1: Whiff rate (no contact at all)
    # Look at all swings (including fouls and contact)
    # Note: Using 'description' (not 'next_description') since you're looking at current pitch
    swings = matching.filter(pl.col('description').is_in([
        'swinging_strike',           # Miss
        'swinging_strike_blocked',   # Miss in dirt
        'foul',                      # Contact but foul
        'foul_tip',                  # Contact but foul
        'hit_into_play'              # Contact in play
    ]))
    #log number of swings found
    logger.info(f"Number of swings found: {len(swings)}")
    
    if len(swings) > 0:
        # What % of swings resulted in complete miss? (include the blocked swing because that is still a whiff)
        whiff_rate = (
            swings['description'].str.contains('swinging_strike') |
            swings['description'].str.contains('swinging_strike_blocked')
        ).sum() / len(swings)
    else:
        whiff_rate = 0
    
    # PART 2: Contact quality (ONLY when ball is put in play)
    contact = matching.filter(
        pl.col('description') == 'hit_into_play'
    )
    #log number of times it was put into play
    logger.info(f"Number of contact events found: {len(contact)}")
    if len(contact) > 5:
        # Average xwOBA on contact
        # High xwOBA = hard contact (bad for pitcher)
        # Low xwOBA = weak contact (good for pitcher)
        avg_xwoba = contact['estimated_woba_using_speedangle'].mean()
    else:
        avg_xwoba = 0.320  # Assume league average if not enough data
    
    # COMBINE: 
    # - Lower whiff_rate is bad (they're making contact)
    # - Higher xwOBA is bad (hard contact)
    cost = (0.6 * avg_xwoba) - (0.4 * whiff_rate)
    
    return cost

def get_tolerances(sequences_df, unique_sequences):
    """
    Calculate tolerance statistics for each pitch sequence.

    Parameters:
    ----------
    sequences_df : polars DataFrame
        DataFrame containing pitch sequences with delta columns.
    unique_sequences : list
        List of pitch sequences to calculate tolerances for.

    Returns:
    -------
    polars DataFrame
        DataFrame with tolerance statistics for each sequence.
    """
    # get some information that we can use for the tolerances
    tolerances_summary = sequences_df.select(['pitch_sequence', 'delta_plate_x', 'delta_plate_z', 'delta_velocity']).group_by(['pitch_sequence']).agg([
        pl.col('delta_plate_x').std().alias('std_delta_x'),
        pl.col('delta_plate_z').std().alias('std_delta_z'),
        pl.col('delta_velocity').std().alias('std_delta_velocity'),
        pl.col('delta_plate_x').mean().alias('mean_delta_x'),
        pl.col('delta_plate_z').mean().alias('mean_delta_z'),
        pl.col('delta_velocity').mean().alias('mean_delta_velocity'),
        pl.col('delta_plate_x').min().alias('min_delta_x'),
        pl.col('delta_plate_z').min().alias('min_delta_z'),
        pl.col('delta_velocity').min().alias('min_delta_velocity'),
        pl.col('delta_plate_x').max().alias('max_delta_x'),
        pl.col('delta_plate_z').max().alias('max_delta_z'),
        pl.col('delta_velocity').max().alias('max_delta_velocity'),
        pl.count().alias('count_sequence')])
    tolerances_summary = tolerances_summary.filter(pl.col('pitch_sequence').is_in(unique_sequences))
    return tolerances_summary

def run_optimizer(sequences_df, tolerances_row, n_calls=50):
    """
    Run Bayesian optimization for a single pitch sequence.

    Parameters:
    ----------
    sequences_df : polars DataFrame
        DataFrame containing pitch sequences with delta columns (for one sequence type).
    tolerances_row : polars DataFrame row
        Single row from tolerances DataFrame with min/max/std/mean values.
    n_calls : int
        Number of optimization iterations.

    Returns:
    -------
    skopt.OptimizeResult
        Optimization result object.
    """
    # for now, the tolerances is 3x the standard deviation of each delta
    tol_x = tolerances_row['std_delta_x'][0] * 3
    tol_z = tolerances_row['std_delta_z'][0] * 3
    tol_velocity = tolerances_row['std_delta_velocity'][0] * 3
    tolerances_list = [tol_x, tol_z, tol_velocity]

    # Define the search space using max and min from the tolerances dataframe
    search_space = [
        Real(tolerances_row['min_delta_x'][0], tolerances_row['max_delta_x'][0], name='delta_plate_x'),
        Real(tolerances_row['min_delta_z'][0], tolerances_row['max_delta_z'][0], name='delta_plate_z'),
        Real(tolerances_row['min_delta_velocity'][0], tolerances_row['max_delta_velocity'][0], name='delta_velocity'),
    ]

    # initial guess
    x0 = [
        tolerances_row['mean_delta_x'][0],
        tolerances_row['mean_delta_z'][0],
        tolerances_row['mean_delta_velocity'][0]
    ]

    # Objective function wrapper
    def objective(params):
        return cost_function_contact_quality(params, sequences_df, tolerances_list)

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        search_space,
        n_calls=n_calls,
        x0=x0
    )

    return result

def optimize_all_sequences(sequences_df, unique_sequences, n_calls=50):
    """
    Run optimizer for all top pitch sequences.

    Parameters:
    ----------
    sequences_df : polars DataFrame
        DataFrame containing all pitch sequences with delta columns.
    unique_sequences : list
        List of pitch sequences to optimize.
    n_calls : int
        Number of optimization iterations per sequence.

    Returns:
    -------
    polars DataFrame
        DataFrame with optimization results for each sequence.
    """
    # Get tolerances for all sequences
    tolerances_summary = get_tolerances(sequences_df, unique_sequences)

    # Initialize empty dataframe to store results
    resultsdf = pl.DataFrame({
        'pitch_sequence': pl.Series([], dtype=pl.Utf8),
        'best_delta_x': pl.Series([], dtype=pl.Float64),
        'best_delta_z': pl.Series([], dtype=pl.Float64),
        'best_delta_velocity': pl.Series([], dtype=pl.Float64),
        'best_cost': pl.Series([], dtype=pl.Float64)
    })

    for seq in unique_sequences:
        logger.info(f"Running optimizer for sequence: {seq}")
        try:
            # get tolerances for this sequence
            tolerances = tolerances_summary.filter(pl.col('pitch_sequence') == seq)
            # get sequences_df for this sequence
            sequences_df_seq = sequences_df.filter(pl.col('pitch_sequence') == seq)

            # run the optimizer
            result = run_optimizer(sequences_df_seq, tolerances, n_calls=n_calls)

            best_params = result.x
            best_cost = float(result.fun)

            logger.info(f"Best parameters for sequence {seq}: {best_params}")
            logger.info(f"Best cost for sequence {seq}: {best_cost}")

            # append to resultsdf
            row = pl.DataFrame({
                'pitch_sequence': [seq],
                'best_delta_x': [float(best_params[0])],
                'best_delta_z': [float(best_params[1])],
                'best_delta_velocity': [float(best_params[2])],
                'best_cost': [best_cost]
            })
            resultsdf = resultsdf.vstack(row)

        except Exception as e:
            logger.exception(f"Optimizer failed for sequence {seq}: {e}")
            continue

    return resultsdf


################################################################################
# SITUATION-BASED ANALYSIS (COUNT-SPECIFIC OPTIMIZATION)
################################################################################

def analyze_by_situation(situation_dict, sequences_df_full, top_n=3, tolerance_multiplier=3.0):
    """
    Analyze pitch sequences for a specific game situation (count).

    This answers: "I just threw pitch 1, count is now X-Y, what should pitch 2 be?"

    Parameters:
    -----------
    situation_dict : dict
        Dictionary with 'balls', 'strikes', 'name' keys (representing count AFTER pitch 1)
    sequences_df_full : polars DataFrame
        Full sequences dataframe with 'count_after_pitch1' column
    top_n : int
        How many top pitch sequences to analyze for this situation
    tolerance_multiplier : float
        Multiplier for standard deviation to set tolerances

    Returns:
    --------
    polars DataFrame
        Optimization results for this situation, or None if insufficient data
    """
    situation_name = situation_dict['name']
    logger.info(f"\n{'='*60}")
    logger.info(f"ANALYZING SITUATION: Count is now {situation_name} (after pitch 1)")
    logger.info(f"{'='*60}")

    # Filter sequences_df to only this count (AFTER pitch 1)
    situation_sequences = sequences_df_full.filter(
        pl.col('count_after_pitch1') == situation_name
    )

    logger.info(f"Total sequences where count became {situation_name} after pitch 1: {len(situation_sequences)}")

    if len(situation_sequences) < 50:
        logger.warning(f"Not enough data for {situation_name} (only {len(situation_sequences)} sequences)")
        return None

    # Find most common pitch sequences for THIS situation
    situation_pitch_counts = situation_sequences.group_by('pitch_sequence').agg(
        pl.count().alias('count')
    ).sort('count', descending=True).head(top_n)

    logger.info(f"Top {top_n} pitch sequences when count is {situation_name}:")
    for row in situation_pitch_counts.iter_rows(named=True):
        logger.info(f"  {row['pitch_sequence']}: {row['count']} occurrences")

    # Get list of sequences to optimize
    sequences_to_optimize = situation_pitch_counts['pitch_sequence'].to_list()

    # Results dataframe for this situation
    results = pl.DataFrame({
        'situation': pl.Series([], dtype=pl.Utf8),
        'pitch_sequence': pl.Series([], dtype=pl.Utf8),
        'n_observations': pl.Series([], dtype=pl.Int64),
        'best_delta_x': pl.Series([], dtype=pl.Float64),
        'best_delta_z': pl.Series([], dtype=pl.Float64),
        'best_delta_velocity': pl.Series([], dtype=pl.Float64),
        'best_cost': pl.Series([], dtype=pl.Float64)
    })

    for seq in sequences_to_optimize:
        logger.info(f"\n  Optimizing {seq} when count is {situation_name}...")

        # Filter to just this sequence in this situation
        seq_data = situation_sequences.filter(pl.col('pitch_sequence') == seq)
        n_obs = len(seq_data)

        logger.info(f"    {n_obs} observations for {seq}")

        if n_obs < 20:
            logger.warning(f"    Skipping {seq} - not enough data ({n_obs} < 20)")
            continue

        # Calculate stats for this sequence
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

        # Set up tolerances
        tol_x = stats['std_x'][0] * tolerance_multiplier
        tol_z = stats['std_z'][0] * tolerance_multiplier
        tol_vel = stats['std_vel'][0] * tolerance_multiplier
        tolerances_list = [tol_x, tol_z, tol_vel]

        # Objective function
        def objective(params):
            return cost_function_description_based(params, seq_data, tolerances_list)

        # Initial guess
        x0 = [stats['mean_x'][0], stats['mean_z'][0], stats['mean_vel'][0]]

        try:
            # Run optimizer
            result = gp_minimize(
                objective,
                dimensions=[
                    Real(stats['min_x'][0], stats['max_x'][0], name='delta_plate_x'),
                    Real(stats['min_z'][0], stats['max_z'][0], name='delta_plate_z'),
                    Real(stats['min_vel'][0], stats['max_vel'][0], name='delta_velocity'),
                ],
                n_calls=30,  # Fewer calls for speed
                x0=x0,
                random_state=42
            )

            best_params = result.x
            best_cost = float(result.fun)

            logger.info(f"    Best: x={best_params[0]:.2f}, z={best_params[1]:.2f}, "
                        f"vel={best_params[2]:.2f}, cost={best_cost:.3f}")

            # Add to results
            row = pl.DataFrame({
                'situation': [situation_name],
                'pitch_sequence': [seq],
                'n_observations': [n_obs],
                'best_delta_x': [float(best_params[0])],
                'best_delta_z': [float(best_params[1])],
                'best_delta_velocity': [float(best_params[2])],
                'best_cost': [best_cost]
            })
            results = results.vstack(row)

        except Exception as e:
            logger.error(f"    Optimization failed for {seq}: {e}")
            continue

    return results


def optimize_all_situations(sequences_df, top_n=3):
    """
    Run optimization for all 12 possible count situations.

    Parameters:
    -----------
    sequences_df : polars DataFrame
        Full sequences dataframe with 'count_after_pitch1' column
    top_n : int
        Number of top sequences to analyze per situation

    Returns:
    --------
    polars DataFrame
        All optimization results across all situations
    """
    # Define ALL possible count situations (what the count could be after pitch 1)
    all_counts = []
    for balls in range(4):  # 0, 1, 2, 3
        for strikes in range(3):  # 0, 1, 2
            all_counts.append({
                'balls': balls,
                'strikes': strikes,
                'name': f'{balls}-{strikes}'
            })

    # Initialize results dataframe
    all_results = pl.DataFrame({
        'situation': pl.Series([], dtype=pl.Utf8),
        'pitch_sequence': pl.Series([], dtype=pl.Utf8),
        'n_observations': pl.Series([], dtype=pl.Int64),
        'best_delta_x': pl.Series([], dtype=pl.Float64),
        'best_delta_z': pl.Series([], dtype=pl.Float64),
        'best_delta_velocity': pl.Series([], dtype=pl.Float64),
        'best_cost': pl.Series([], dtype=pl.Float64)
    })

    for situation in all_counts:
        logger.info(f"\nProcessing situation: {situation['name']}")
        result = analyze_by_situation(situation, sequences_df, top_n=top_n)

        if result is not None and len(result) > 0:
            all_results = all_results.vstack(result)
        else:
            logger.warning(f"No results for situation {situation['name']}")

    # Sort by situation and cost
    all_results = all_results.sort(['situation', 'best_cost'])

    return all_results


def get_pitch_recommendation(count_label, all_results_df):
    """
    Get pitch recommendation for a specific count.

    This answers: "I just threw pitch 1, count is now X-Y, what should I throw next?"

    Parameters:
    -----------
    count_label : str
        The count label AFTER pitch 1 (e.g., '0-2', '3-2')
    all_results_df : polars DataFrame
        Results from situation-based optimization

    Returns:
    --------
    dict
        Recommendation details including pitch types, deltas, cost, etc.
    """
    # Filter to this situation
    situation_results = all_results_df.filter(
        pl.col('situation') == count_label
    ).sort('best_cost')  # Lower cost is better

    if len(situation_results) == 0:
        return {
            'count': count_label,
            'recommendation': 'No data available',
            'pitch1_type': 'N/A',
            'pitch2_type': 'N/A',
            'delta_x': 0.0,
            'delta_z': 0.0,
            'delta_velocity': 0.0,
            'expected_cost': 0.0,
            'n_observations': 0,
            'has_data': False
        }

    # Best sequence is the one with lowest cost
    best = situation_results.row(0, named=True)

    pitch1_type = best['pitch_sequence'].split('-')[0]
    pitch2_type = best['pitch_sequence'].split('-')[1]

    return {
        'count': count_label,
        'recommended_sequence': best['pitch_sequence'],
        'pitch1_type': pitch1_type,
        'pitch2_type': pitch2_type,
        'delta_x': best['best_delta_x'],
        'delta_z': best['best_delta_z'],
        'delta_velocity': best['best_delta_velocity'],
        'expected_cost': best['best_cost'],
        'n_observations': best['n_observations'],
        'has_data': True,
        'interpretation': (
            f"Count is now {count_label}. Best next pitch: {pitch2_type} "
            f"(previous pitch was {pitch1_type}). "
            f"Location change: ({best['best_delta_x']:.2f}ft horizontal, "
            f"{best['best_delta_z']:.2f}ft vertical), "
            f"velocity change: {best['best_delta_velocity']:.1f} mph"
        )
    }
