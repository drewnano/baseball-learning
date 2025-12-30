from pybaseball import statcast
from pybaseball import statcast_pitcher
from pybaseball import statcast_batter
from pybaseball import playerid_lookup
def get_pitches(lastname,firstname, start_date, end_date):
    #find playerid lookup of zack wheeler
    pitcher= playerid_lookup(lastname,firstname)
    pitcherid = pitcher.loc[0,"key_mlbam"]
    pitcherid = float(pitcherid)
    #pull zack wheeler's pitch data from the 2020 to 2025 seasons
    wheeler_pitches = statcast_pitcher(start_date, end_date, pitcherid)
    return wheeler_pitches

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

