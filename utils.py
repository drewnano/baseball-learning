"""
utility functions for optimizing pitch tunneling.
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
import logging

# utility functions for pitch tunneling analysis:
def get_pitches(lastname,firstname, start_date, end_date):
    #find playerid lookup of the player you specify
    pitcher= playerid_lookup(lastname,firstname)
    pitcherid = pitcher.loc[0,"key_mlbam"]
    pitcherid = float(pitcherid)
    #pull zack wheeler's pitch data from the 2020 to 2025 seasons
    #date is in format 'yyyy-mm-dd'
    pitches = statcast_pitcher(start_date, end_date, pitcherid)
    #convert pitches from pandas dataframe to polars dataframe
    pitches = pl.from_pandas(pitches)
    #create new column which is the unique ID of the at bat
    pitches = pitches.with_columns(
        pl.concat_str([pl.col("game_pk").cast(pl.Utf8), pl.lit("_"), pl.col("at_bat_number").cast(pl.Utf8)]).alias("ab_id")
    )
    #process pitches using polars lazy version
    sequences = pitches.lazy().sort(['ab_id', 'pitch_number']).with_columns([
        # Get next pitch info
        pl.col('pitch_type').shift(1).over('ab_id').alias('first_pitch_type'),
        pl.col('plate_x').shift(1).over('ab_id').alias('first_plate_x'),
        pl.col('plate_z').shift(1).over('ab_id').alias('first_plate_z'),
        pl.col('release_speed').shift(1).over('ab_id').alias('first_release_speed'),
        pl.col('description').shift(1).over('ab_id').alias('first_description'),
        pl.col('events').shift(1).over('ab_id').alias('first_events'),
        pl.col('woba_value').shift(1).over('ab_id').alias('first_woba_value'),
    ]).filter(
        pl.col('events').is_not_null()
    ).with_columns([
        (pl.col('plate_x') - pl.col('first_plate_x')).alias('delta_plate_x'),
        (pl.col('plate_z') - pl.col('first_plate_z')).alias('delta_plate_z'),
        (pl.col('release_speed') - pl.col('first_release_speed')).alias('delta_velocity'),
    ])

    # Collect to see results
    sequences_df = sequences.collect()

    logger.info(f"Total pitch sequences: {len(sequences_df)}")
    return pitches

def return_top_sequences(pitches, top_n=5):
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



def get_matching_sequences(params, sequences_df, tolerances):
    """
    Find matching sequences in your data for search.
    
    Parameters:
    ----------
    params : list/tuple
        params = [delta_plate_x, delta_plate_z, delta_velocity, pitch_sequence]
    sequences_df : polars DataFrame
        DataFrame containing pitch sequences with delta columns.
    tolerances : list/tuple
        Tolerances for each delta parameter.
        tolerances = [tol_x, tol_z, tol_velocity]
    
    Returns:
    -------
    polars DataFrame
        Filtered DataFrame with matching sequences.
    """
    logging.info("Searching for matching sequences...")

    delta_x, delta_z, delta_velocity = params
    tol_x, tol_z, tol_velocity = tolerances
    logging.info(f"Parameters: delta_x={delta_x}, delta_z={delta_z}, delta_velocity={delta_velocity}")
    logging.info(f"Tolerances: tol_x={tol_x}, tol_z={tol_z}, tol_velocity={tol_velocity}")
    # Filter to sequences that are close enough to the parameters
    # KEY FIX: Add parentheses around EACH comparison
    matching = sequences_df.filter(
        ((pl.col('delta_plate_x') - delta_x).abs() < tol_x) &
        ((pl.col('delta_plate_z') - delta_z).abs() < tol_z) &
        ((pl.col('delta_velocity') - delta_velocity).abs() < tol_velocity)
    )
    logger.info(f"Found {matching.shape[0]} matching sequences for params: {params}")
    return matching

def cost_function_contact_quality(params, sequences_df, tolerances):
    """
    Two-part cost function:
    combining swing and miss rate with contact quality (xwOBA on contact)
    
    Parameters:
    ----------
    params : list/tuple
        [delta_plate_x, delta_plate_z, delta_velocity, pitch_sequence]
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

def get_tolerances(pitches):
    # get some information that we can use for the tolerances 
    tolerances_summary = pitches.select([ 'pitch_sequence', 'delta_plate_x', 'delta_plate_z', 'delta_velocity']).group_by(['pitch_sequence']).agg([
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
    tolerances_summary = tolerances_summary.filter(pl.col('pitch_sequence').is_in(topnsequences['sequence'].to_list()))
    return tolerances_summary

def run_optimizer(sequences_df, tolerances, n_calls=50):
    # Define the search space
    # using max and min from the tolerances dataframe
    search_space = [
        Real(tolerances['min_delta_x'], tolerances['max_delta_x'], name='delta_plate_x'),
        Real(tolerances['min_delta_z'], tolerances['max_delta_z'], name='delta_plate_z'),
        Real(tolerances['min_delta_velocity'], tolerances['max_delta_velocity'], name='delta_velocity'),
    ]
    
    # Objective function wrapper
    def objective(params):
        return cost_function_contact_quality(params, sequences_df, tolerances)
    
    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        search_space,
        n_calls=n_calls,
        random_state=42
    )
    
    return result
