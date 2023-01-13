import numpy as np
import nelpy as nel
from nelpy.analysis import replay
from ripple_heterogeneity.utils import functions, loading, compress_repeated_epochs
import os
import pandas as pd
from scipy import stats
import pickle
import copy
import warnings
import logging

warnings.filterwarnings("ignore")

logging.getLogger().setLevel(logging.ERROR)


def decode_and_score(bst, tc, pos):
    # access decoding accuracy on behavioral time scale
    posteriors, lengths, mode_pth, mean_pth = nel.decoding.decode1D(
        bst, tc, xmin=np.nanmin(pos.data), xmax=np.nanmax(pos.data)
    )

    actual_pos = np.interp(bst.bin_centers, pos.abscissa_vals, pos.data[0])

    bad_idx = np.isnan(actual_pos) | np.isnan(mode_pth)
    actual_pos = actual_pos[~bad_idx]
    mode_pth = mode_pth[~bad_idx]
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(actual_pos, mode_pth)
    median_error = np.nanmedian(np.abs(actual_pos - mode_pth))

    return rvalue, median_error


def pooled_incoherent_shuffle_bst(bst):
    out = copy.deepcopy(bst)
    data = out._data

    for uu in range(bst.n_units):
        segment = np.atleast_1d(np.squeeze(data[uu, :]))
        segment = np.roll(segment, np.random.randint(len(segment)))
        data[uu, :] = segment
    return out


def decode_and_shuff(bst, tc, pos, n_shuffles=500):
    """ """
    rvalue, median_error = decode_and_score(bst, tc, pos)

    scores = np.zeros(bst.n_epochs)
    if n_shuffles > 0:
        rvalue_time_swap = np.zeros((n_shuffles, 1))
        median_error_time_swap = np.zeros((n_shuffles, 1))

    for shflidx in range(n_shuffles):
        bst_shuff = pooled_incoherent_shuffle_bst(bst)
        rvalue_time_swap[shflidx], median_error_time_swap[shflidx] = decode_and_score(
            bst_shuff, tc, pos
        )

    return rvalue, median_error, rvalue_time_swap, median_error_time_swap


def get_features(bst_placecells, posteriors, bdries, mode_pth, pos, dp=3):
    """
    Using the posterior probability matrix, calculate several features on spatial trajectory
    and detects if the trajectory is foward or reverse depending on the rat's current position
    """

    place_bin_edges = np.arange(np.nanmin(pos.data[0]), np.nanmax(pos.data[0]) + dp, dp)
    place_bin_centers = place_bin_edges[:-1] + np.diff(place_bin_edges) / 2

    traj_dist = []
    traj_speed = []
    traj_step = []
    replay_type = []
    dist_rat_start = []
    dist_rat_end = []
    position = []

    # find direction of movement from position
    x_slope = []
    for p in pos:
        good_idx = ~np.isnan(p.data[0])
        cur_x = p.data[0][good_idx]
        if len(cur_x) == 0:
            x_slope.append(np.nan)
        else:
            b1, _, _, _, _ = stats.linregress(np.arange(len(cur_x)), cur_x)
            x_slope.append(b1)
    # if the majority (>.5) of laps have x coords that increase (positive slopes)
    if np.mean(np.array(x_slope) > 0) > 0.5:
        outbound = True
    else:
        outbound = False

    for idx in range(bst_placecells.n_epochs):

        x = bst_placecells[idx].bin_centers

        y = mode_pth[bdries[idx] : bdries[idx + 1]]
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        velocity, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
        y = x * velocity + intercept
        position.append(y)

        # get spatial difference between bins
        dy = np.abs(np.diff(y))
        # get cumulative distance
        traj_dist.append(np.nansum(dy))
        # calculate avg speed of trajectory (dist(cm) / time(sec))
        traj_speed.append(np.nansum(dy) / (np.nanmax(x) - np.nanmin(x)))
        # get mean step size
        traj_step.append(np.nanmean(dy))

        if velocity > 0:
            replay_type.append("forward")
        elif velocity < 0:
            replay_type.append("reverse")
        else:
            replay_type.append("unknown")

    return traj_dist, traj_speed, traj_step, replay_type, position


def flip_pos_within_epoch(pos, dir_epoch):
    """
    flip_pos_within_epoch: flips x coordinate within epoch
        Made to reverse x coordinate within nelpy array for replay analysis

    Input:
        pos: nelpy analog array with single dim
        dir_epoch: epoch to flip
    Output:
        pos: original pos, but fliped by epoch
    """

    def flip_x(x):
        return (x * -1) - np.nanmin(x * -1)

    # make pos df
    pos_df = pd.DataFrame()
    pos_df["ts"] = pos.abscissa_vals
    pos_df["x"] = pos.data.T
    pos_df["dir"] = False

    # make index within df of epoch
    for ep in dir_epoch:
        pos_df.loc[pos_df["ts"].between(ep.starts[0], ep.stops[0]), "dir"] = True

    # flip x within epoch
    pos_df.loc[pos_df.dir == True, "x"] = flip_x(pos_df[pos_df.dir == True].x)

    # add position back to input pos
    pos._data = np.expand_dims(pos_df.x.values, axis=0)

    return pos


def dissociate_laps_by_states(states, dir_epoch, states_of_interest=[1, 2]):
    # unique_states = np.unique(states.data[~np.isnan(states.data)])
    lap_id = []
    for ep in dir_epoch:
        state_count = []
        for us in states_of_interest:
            state_count.append(np.nansum(states[ep].data == us))
        lap_id.append(states_of_interest[np.argmax(state_count)])
    return np.array(lap_id).astype(int)


def get_pos(basepath, epoch_df, beh_epochs, task_idx):
    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")

    if position_df_no_nan.shape[0] == 0:
        return None, None, None, None

    if "linearized" not in position_df_no_nan.columns:
        return None, None, None, None

    if "states" not in position_df_no_nan.columns:
        return None, None, None, None

    pos = nel.PositionArray(
        data=position_df_no_nan["linearized"].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )
    # make min pos 2
    pos._data = (pos.data - np.nanmin(pos.data)) + 2

    pos = pos[beh_epochs[task_idx]]

    states = nel.AnalogSignalArray(
        data=position_df_no_nan["states"].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )
    states = states[beh_epochs[task_idx]]

    # get outbound and inbound epochs
    outbound_laps, inbound_laps = functions.get_linear_track_lap_epochs(
        pos.abscissa_vals, pos.data[0], newLapThreshold=20
    )

    inbound_laps = functions.find_good_lap_epochs(pos, inbound_laps)
    outbound_laps = functions.find_good_lap_epochs(pos, outbound_laps)

    if not inbound_laps.isempty:
        raise TypeError("inbound_laps should be empty for tmaze")

    # flip x coord
    if not outbound_laps.isempty:
        pos = flip_pos_within_epoch(pos, outbound_laps)

    # make min pos 2
    pos._data = (pos.data - np.nanmin(pos.data)) + 2

    # locate laps with the majority in state 1 or 2
    lap_id = dissociate_laps_by_states(states, outbound_laps, states_of_interest=[1, 2])

    right_epochs = nel.EpochArray(data=outbound_laps.data[lap_id == 1, :])
    left_epochs = nel.EpochArray(data=outbound_laps.data[lap_id == 2, :])

    position_df_no_nan = position_df_no_nan[
        position_df_no_nan["time"].between(
            epoch_df.iloc[task_idx].startTime, epoch_df.iloc[task_idx].stopTime
        )
    ]
    return pos, right_epochs, left_epochs, states, position_df_no_nan


def get_tuning_curves(
    pos, st_all, dir_epoch, speed_thres, ds_50ms, s_binsize, tuning_curve_sigma
):
    # compute and smooth speed
    speed1 = nel.utils.ddt_asa(pos[dir_epoch], smooth=True, sigma=0.1, norm=True)

    # find epochs where the animal ran > 4cm/sec
    run_epochs = nel.utils.get_run_epochs(speed1, v1=speed_thres, v2=speed_thres)

    # restrict spike trains to those epochs during which the animal was running
    st_run = st_all[dir_epoch][run_epochs]

    # smooth and re-bin:
    bst_run = st_run.bin(ds=ds_50ms)

    ext_xmin, ext_xmax = (
        np.floor(pos[dir_epoch].min() / 10) * 10,
        np.ceil(pos[dir_epoch].max()),
    )
    n_bins = int((ext_xmax - ext_xmin) / s_binsize)

    tc = nel.TuningCurve1D(
        bst=bst_run,
        extern=pos[dir_epoch][run_epochs],
        n_extern=n_bins,
        extmin=ext_xmin,
        extmax=ext_xmax,
        sigma=tuning_curve_sigma,
        min_duration=0.1,
    )
    return tc, st_run, bst_run


def restrict_to_place_cells(
    tc,
    st_run,
    bst_run,
    st_all,
    cell_metrics,
    place_cell_min_spks,
    place_cell_min_rate,
    place_cell_peak_mean_ratio,
):
    # locate pyr cells with >= 100 spikes, peak rate >= 1 Hz, peak/mean ratio >=1.5
    peak_firing_rates = tc.max(axis=1)
    mean_firing_rates = tc.mean(axis=1)
    ratio = peak_firing_rates / mean_firing_rates

    idx = (
        (st_run.n_events >= place_cell_min_spks)
        & (tc.ratemap.max(axis=1) >= place_cell_min_rate)
        & (ratio >= place_cell_peak_mean_ratio)
    )
    unit_ids_to_keep = (np.where(idx)[0] + 1).squeeze().tolist()

    sta_placecells = st_all._unit_subset(unit_ids_to_keep)
    tc = tc._unit_subset(unit_ids_to_keep)
    total_units = sta_placecells.n_active
    bst_run = bst_run.loc[:, unit_ids_to_keep]

    # restrict cell_metrics to place cells
    cell_metrics_ = cell_metrics[idx]

    return sta_placecells, tc, bst_run, cell_metrics_, total_units


def handle_canidate_events(
    basepath, ripples, expand_canidate_by_mua, min_rip_dur, expand_ripple=0.05
):
    """
    This function takes a list of ripple events and expands them by MUA events
        if expand_canidate_by_mua is True. It also removes events that are too short.
    Input:
        basepath: path to the folder containing the mua events
        ripples: pd.DataFrame of ripple events
        expand_canidate_by_mua: boolean, if True, will expand ripple events by MUA events
        min_rip_dur: minimum duration of ripple events to keep
    Output:
        ripples: pd.DataFrame of ripple events
        ripple_epochs: nel.Epochs object of ripple events
    """
    if expand_canidate_by_mua:
        # put ripples into epoch array
        ripple_epochs = nel.EpochArray(np.array([ripples.start, ripples.stop]).T)
        # get mua
        mua_df = loading.load_mua_events(basepath)
        # add mua to epoch array
        mua_epoch = nel.EpochArray(np.array([mua_df.start, mua_df.stop]).T)
        # find overlap between ripple and mua epochs
        # also expand ripples by 50ms to allow more overlap
        ripple_epochs, idx = functions.overlap_intersect(
            mua_epoch, ripple_epochs.expand(expand_ripple)
        )
        ripples = ripples.loc[idx]
    else:
        # restrict to events at least xx s long if not using mua
        ripples = ripples[ripples.duration >= min_rip_dur]

        # expand ripples that remain
        ripples.start = ripples.start - expand_ripple
        ripples.stop = ripples.stop + expand_ripple
        ripples.duration = ripples.stop - ripples.start

        ripple_epochs = nel.EpochArray(np.array([ripples.start, ripples.stop]).T)

    ripple_epochs = ripple_epochs.merge()

    # reassign ripple epochs to ripples dataframe
    ripples = pd.DataFrame()
    ripples["start"] = ripple_epochs.starts
    ripples["stop"] = ripple_epochs.stops
    ripples["duration"] = ripples.stop - ripples.start

    return ripples, ripple_epochs


def run(
    basepath,  # basepath to session
    traj_shuff=1500,  # number of shuffles to determine sig replay
    behav_shuff=250,  # number of shuffles to determine sig decoding
    ds_50ms=0.05,  # bin width to bin st for tuning curve
    s_binsize=3,  # spatial bins in tuning curve
    speed_thres=3,  # running threshold to determine tuning curves
    min_rip_dur=0.08,  # min ripple duration for replay
    place_cell_min_rate=1,  # min peak rate of tuning curve
    place_cell_min_spks=100,  # at least 100 spikes while running above speed_thres
    place_cell_peak_mean_ratio=1.5,  # peak firing rate / mean firing rate
    replay_binsize=0.02,  # bin size to decode replay
    tuning_curve_sigma=3,  # 3 cm sd of smoothing on tuning curve
    expand_canidate_by_mua=False,  # whether to expand candidate units by mua (note: will only take rips with mua)
    restrict_manipulation=True,  # whether to restrict manipulation epochs
    shuffle_parallel=True,  # whether to shuffle in parallel
    ds_beh_decode=0.2,  # bin width to bin st for decoding behavior
    expand_ripple=0.05,
):
    """
    Main function that conducts the replay analysis
    Inputs:
        basepath: path to session
        traj_shuff: number of shuffles to determine sig replay
        behav_shuff: number of shuffles to determine sig decoding
        ds_50ms: bin width to bin st for tuning curve
        s_binsize: spatial bins in tuning curve
        speed_thres: running threshold to determine tuning curves
        min_rip_dur: min ripple duration for replay
        place_cell_min_rate: min peak rate of tuning curve
        place_cell_min_spks: at least 100 spikes while running above speed_thres
        place_cell_peak_mean_ratio: peak firing rate / mean firing rate
        replay_binsize: bin size to decode replay
        tuning_curve_sigma: 3 cm sd of smoothing on tuning curve
        expand_canidate_by_mua: whether to expand candidate units by mua (note: will only take rips with mua)
        restrict_manipulation: whether to restrict manipulation epochs
    Outputs:
        results: dictionary of results
    """

    cell_metrics, data, ripples, fs_dat = loading.load_basic_data(basepath)

    restrict_idx = (
        (cell_metrics.putativeCellType == "Pyramidal Cell")
        & (cell_metrics.brainRegion.str.contains("CA1"))
        & (cell_metrics.bad_unit == False)
        & (cell_metrics.total >= 100)
    )
    # restrict cell metrics
    cell_metrics = cell_metrics[restrict_idx]

    if cell_metrics.shape[0] == 0:
        return

    # load session epoch data
    epoch_df = loading.load_epoch(basepath)
    # get session bounds to provide support
    session_bounds = nel.EpochArray(
        [epoch_df.startTime.iloc[0], epoch_df.stopTime.iloc[-1]]
    )
    # compress repeated sleep sessions
    epoch_df = compress_repeated_epochs.main(epoch_df)
    # put into nel format
    beh_epochs = nel.EpochArray([np.array([epoch_df.startTime, epoch_df.stopTime]).T])

    try:
        st_all = nel.SpikeTrainArray(
            timestamps=np.array(data["spikes"], dtype=object)[restrict_idx], fs=fs_dat
        )
    except:
        st_all = nel.SpikeTrainArray(
            timestamps=np.array(data["spikes"], dtype=object)[restrict_idx][0],
            fs=fs_dat,
        )

    # skip if less than 5 cells
    if st_all.n_active < 5:
        return

    position_df = loading.load_animal_behavior(basepath)

    # find tmaze task
    name = (
        position_df[~position_df.linearized.isnull()]
        .epochs.value_counts()
        .sort_values(ascending=False)
        .index[0]
    )
    task_idx = int(np.where(epoch_df.name == name)[0][0])

    pos, right_epochs, left_epochs, states, position_df_no_nan = get_pos(
        basepath, epoch_df, beh_epochs, task_idx
    )

    if pos is None:
        return

    # here we will only take ripples with high mua,
    #   plus the bounds of the candidate events will be defined by mua
    ripples, ripple_epochs = handle_canidate_events(
        basepath, ripples, expand_canidate_by_mua, min_rip_dur, expand_ripple
    )

    # iter through both running directions
    results = {}
    results["right_epochs"] = {}
    results["left_epochs"] = {}

    direction_str = ["right_epochs", "left_epochs"]
    for dir_i, dir_epoch in enumerate([right_epochs, left_epochs]):
        if dir_epoch.isempty:
            continue
        # construct tuning curves
        tc, st_run, bst_run = get_tuning_curves(
            pos, st_all, dir_epoch, speed_thres, ds_50ms, s_binsize, tuning_curve_sigma
        )

        # locate pyr cells with >= 100 spikes, peak rate >= 1 Hz, peak/mean ratio >=1.5
        (
            sta_placecells,
            tc,
            bst_run,
            cell_metrics_,
            total_units,
        ) = restrict_to_place_cells(
            tc,
            st_run,
            bst_run,
            st_all,
            cell_metrics,
            place_cell_min_spks,
            place_cell_min_rate,
            place_cell_peak_mean_ratio,
        )
        if tc.isempty:
            continue

        # access decoding accuracy on behavioral time scale
        bst_run_beh = sta_placecells.bin(ds=ds_beh_decode)[dir_epoch]
        decoding_r2, median_error, decoding_r2_shuff, _ = decode_and_shuff(
            bst_run_beh, tc, pos[dir_epoch], n_shuffles=behav_shuff
        )
        # check decoding quality against chance distribution
        _, decoding_r2_pval, _ = functions.get_significant_events(
            decoding_r2, decoding_r2_shuff
        )

        # get ready to decode replay
        # bin data for replay (20ms default)
        bst_placecells = sta_placecells.bin(ds=replay_binsize)[ripple_epochs]

        # count units per event
        n_active = [bst.n_active for bst in bst_placecells]
        n_active = np.array(n_active)
        # also count the proportion of bins in each event with 0 activity
        inactive_bin_prop = [
            sum(bst.n_active_per_bin == 0) / bst.lengths[0] for bst in bst_placecells
        ]
        inactive_bin_prop = np.array(inactive_bin_prop)
        # restrict bst to instances with >= 5 active units and < 50% inactive bins
        idx = (n_active >= 5) & (inactive_bin_prop < 0.5)
        bst_placecells = bst_placecells[np.where(idx)[0]]
        # restrict to instances with >= 5 active units
        n_active = n_active[idx]
        inactive_bin_prop = inactive_bin_prop[idx]

        if bst_placecells.isempty:
            continue

        current_ripples = pd.DataFrame()
        current_ripples["start"] = bst_placecells.support.data[:, 0]
        current_ripples["stop"] = bst_placecells.support.data[:, 1]
        current_ripples["duration"] = current_ripples["stop"] - current_ripples["start"]

        # decode each ripple event
        posteriors, bdries, mode_pth, mean_pth = nel.decoding.decode1D(
            bst_placecells,
            tc,
            xmin=np.nanmin(pos[dir_epoch].data),
            xmax=np.nanmax(pos[dir_epoch].data),
        )

        # score each event using trajectory_score_bst (sums the posterior probability in a range (w) from the LS line)
        scores, scores_time_swap, scores_col_cycle = replay.trajectory_score_bst(
            bst_placecells, tc, w=3, n_shuffles=traj_shuff, normalize=True
        )

        # find sig events using time and column shuffle distributions
        _, score_pval_time_swap, score_z_time_swap = functions.get_significant_events(
            scores, scores_time_swap
        )
        _, score_pval_col_cycle, score_z_col_cycle = functions.get_significant_events(
            scores, scores_col_cycle
        )

        (traj_dist, traj_speed, traj_step, replay_type, position) = get_features(
            bst_placecells, posteriors, bdries, mode_pth, pos[dir_epoch], dp=s_binsize
        )

        slope, intercept, r2values = replay.linregress_bst(bst_placecells, tc)

        # package data into results dictionary
        results[direction_str[dir_i]]["cell_metrics"] = cell_metrics_

        results[direction_str[dir_i]]["sta_placecells"] = sta_placecells
        results[direction_str[dir_i]]["bst_placecells"] = bst_placecells
        results[direction_str[dir_i]]["bst_run"] = bst_run
        results[direction_str[dir_i]]["bst_run_beh"] = bst_run_beh
        results[direction_str[dir_i]]["pos"] = pos[dir_epoch]
        results[direction_str[dir_i]]["tc"] = tc
        results[direction_str[dir_i]]["posteriors"] = posteriors
        results[direction_str[dir_i]]["bdries"] = bdries
        results[direction_str[dir_i]]["mode_pth"] = mode_pth
        results[direction_str[dir_i]]["position"] = position

        temp_df = current_ripples.copy()
        # add event by event metrics to df
        temp_df["n_active"] = n_active
        temp_df["inactive_bin_prop"] = inactive_bin_prop
        temp_df["trajectory_score"] = scores
        temp_df["r_squared"] = r2values
        temp_df["slope"] = slope
        temp_df["intercept"] = intercept
        temp_df["score_pval_time_swap"] = score_pval_time_swap
        temp_df["score_pval_col_cycle"] = score_pval_col_cycle
        temp_df["score_z_time_swap"] = score_z_time_swap
        temp_df["score_z_col_cycle"] = score_z_col_cycle
        temp_df["traj_dist"] = traj_dist
        temp_df["traj_speed"] = traj_speed
        temp_df["traj_step"] = traj_step
        temp_df["replay_type"] = replay_type
        results[direction_str[dir_i]]["df"] = temp_df

        results[direction_str[dir_i]]["session"] = basepath
        results[direction_str[dir_i]]["decoding_r2"] = decoding_r2
        results[direction_str[dir_i]]["decoding_r2_pval"] = decoding_r2_pval
        results[direction_str[dir_i]]["decoding_median_error"] = median_error
        results[direction_str[dir_i]]["total_units"] = total_units

    return results


def load_results(save_path, pre_task_post=False, verbose=False):
    """
    load_results: load results from a directory

    """
    import glob

    sessions = glob.glob(save_path + os.sep + "*.pkl")
    df = pd.DataFrame()
    for session in sessions:
        if verbose:
            print(session)
        with open(session, "rb") as f:
            results = pickle.load(f)
        if results is None:
            continue
        if (results["right_epochs"] == {}) & (results["right_epochs"] == {}):
            continue
        try:
            basepath = results["right_epochs"]["session"]
            epoch_df = loading.load_epoch(basepath)
        except:
            basepath = results["left_epochs"]["session"]
            epoch_df = loading.load_epoch(basepath)

        if pre_task_post:

            pattern_idx, _ = functions.find_epoch_pattern(
                epoch_df.environment, ["sleep", "tmaze", "sleep"]
            )
            if pattern_idx is None:
                continue

        for key_ in results.keys():
            try:
                results[key_]["sta_placecells"]
            except:
                continue
            # calc and add ripple participation
            st = results[key_]["sta_placecells"]
            bst = results[key_]["bst_placecells"]
            if len(bst.support.starts) == 0:
                results[key_]["df"]["pop_partic"] = 0
            else:
                particip_mat = functions.get_participation(
                    st.data, bst.support.starts, bst.support.stops
                )
                results[key_]["df"]["pop_partic"] = particip_mat.mean(axis=0)

            # add behavioral decoding quality
            results[key_]["df"]["decoding_r2"] = float(results[key_]["decoding_r2"])
            results[key_]["df"]["decoding_r2_pval"] = float(
                results[key_]["decoding_r2_pval"]
            )
            results[key_]["df"]["decoding_median_error"] = float(
                results[key_]["decoding_median_error"]
            )
            results[key_]["df"]["total_units"] = float(results[key_]["total_units"])
            results[key_]["df"]["direction"] = key_

            # add epoch
            if pre_task_post:
                if len(results[key_]["df"]) > 0:
                    results[key_]["df"]["epoch"] = "unknown"
                    epoch_df = loading.load_epoch(results[key_]["session"])
                    pattern_idx, _ = functions.find_epoch_pattern(
                        epoch_df.environment, ["sleep", "tmaze", "sleep"]
                    )
                    epoch_df = epoch_df[pattern_idx]
                    results[key_]["df"].loc[
                        results[key_]["df"].start.between(
                            epoch_df.startTime.iloc[0], epoch_df.stopTime.iloc[0]
                        ),
                        "epoch",
                    ] = "pre_sleep"
                    results[key_]["df"].loc[
                        results[key_]["df"].start.between(
                            epoch_df.startTime.iloc[1], epoch_df.stopTime.iloc[1]
                        ),
                        "epoch",
                    ] = "tmaze"
                    results[key_]["df"].loc[
                        results[key_]["df"].start.between(
                            epoch_df.startTime.iloc[2], epoch_df.stopTime.iloc[2]
                        ),
                        "epoch",
                    ] = "post_sleep"

            results[key_]["df"]["basepath"] = basepath
            df = pd.concat([df, results[key_]["df"]], ignore_index=True)
    return df
