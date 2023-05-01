import glob
import os
import pickle
import numpy as np
import pandas as pd
from ripple_heterogeneity.utils import functions, loading, add_new_deep_sup
import nelpy as nel
from ripple_heterogeneity.utils import compress_repeated_epochs
import itertools
import logging

logging.getLogger().setLevel(logging.ERROR)


def get_corrcoef(st, epoch, state_epoch, bin_size=0.05, single_bin_per_epoch=True):
    """
    Calculate correlation coefficient for epoch
    input:
        st: nel.SpikeTrain object
        epoch: nel.Epoch object
        state_epoch: nel.Epoch object
        bin_size: bin size in seconds (default: 0.05)
        single_bin_per_epoch: if True, each epoch is binned into a single bin, otherwise each epoch is binned into multiple bins
    output:
        corrcoef_r: correlation matrix
    """
    if single_bin_per_epoch:
        current_epoch = state_epoch[epoch]

        # return if no intervals left
        if current_epoch.isempty:
            return None, None

        n_intervals = current_epoch.n_intervals
        bst = functions.get_participation(
            st.data,
            current_epoch.starts,
            current_epoch.stops,
            par_type="firing_rate",
        )
    else:
        spk_train_list = st[state_epoch][epoch]
        if spk_train_list.isempty:
            return None
        bst = spk_train_list.bin(ds=bin_size).data
        n_intervals = bst.shape[1]

    return np.corrcoef(bst), n_intervals


def get_explained_var(
    st,
    beh_epochs,
    cell_metrics,
    state_epoch,
    task_binsize=0.125,
    restrict_task=False,
    theta_epochs=None,
    single_bin_per_epoch=True,
    shrink_post=None,
):
    """
    Calculate explained variance
    input:
        st: nelpy.SpikeTrain object
        beh_epochs: nel.EpochArray object with 3 epochs: sleep, task, sleep
        cell_metrics: pandas dataframe with cell metrics
        state_epoch: nel.Epoch object
        task_binsize: bin size in seconds for task epoch
        restrict_task: restrict to task epochs
        theta_epochs: nel.Epoch object with theta epochs
        single_bin_per_epoch: if True, make one bin per state_epoch
        shrink_post: if not None, shrink post-task epochs to this amount
    output:
        EV: explained variance
        rEV: reverse explained variance
    """

    # get correlation matrix per epoch

    # restrict spike times to epoch (could be many things)
    # st_restrict = st[state_epoch]

    # pre task
    corrcoef_r_pre, n_pre_events = get_corrcoef(
        st,
        beh_epochs[0],
        state_epoch,
        single_bin_per_epoch=single_bin_per_epoch,
    )

    # task
    corrcoef_r_beh, n_task_events = get_corrcoef(
        st,
        beh_epochs[1],
        beh_epochs[1],
        bin_size=task_binsize,
        single_bin_per_epoch=False,
    )

    # post task
    if shrink_post is not None:
        post_epoch = beh_epochs[2].shrink(
            beh_epochs[2].length - shrink_post * 60, direction="stop"
        )
    else:
        post_epoch = beh_epochs[2]

    corrcoef_r_post, n_post_events = get_corrcoef(
        st,
        post_epoch,
        state_epoch,
        single_bin_per_epoch=single_bin_per_epoch,
    )

    # get uids for ref and target cells
    c = np.array(list(itertools.product(cell_metrics.UID.values, repeat=2)))
    ref_uid = c[:, 0]
    target_uid = c[:, 1]

    if corrcoef_r_pre is None or corrcoef_r_beh is None or corrcoef_r_post is None:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # remove upper triangle correlations
    corrcoef_r_pre[np.tril_indices(corrcoef_r_pre.shape[0], 1)] = np.nan
    corrcoef_r_beh[np.tril_indices(corrcoef_r_beh.shape[0], 1)] = np.nan
    corrcoef_r_post[np.tril_indices(corrcoef_r_post.shape[0], 1)] = np.nan

    # flatten and calculate cross-correlation between epochs
    corr_df = pd.DataFrame(
        {
            "r_pre": corrcoef_r_pre.flatten(),
            "r_beh": corrcoef_r_beh.flatten(),
            "r_post": corrcoef_r_post.flatten(),
        }
    ).corr()
    # pull out specific between epoch correlations
    beh_pos = corr_df.loc["r_beh", "r_post"]
    beh_pre = corr_df.loc["r_beh", "r_pre"]
    pre_pos = corr_df.loc["r_pre", "r_post"]

    # calculate explained variance
    EV = (
        (beh_pos - beh_pre * pre_pos) / np.sqrt((1 - beh_pre**2) * (1 - pre_pos**2))
    ) ** 2
    # calculate reverse explained variance
    rEV = (
        (beh_pre - beh_pos * pre_pos) / np.sqrt((1 - beh_pos**2) * (1 - pre_pos**2))
    ) ** 2

    return (
        EV,
        rEV,
        corrcoef_r_pre.flatten(),
        corrcoef_r_beh.flatten(),
        corrcoef_r_post.flatten(),
        ref_uid,
        target_uid,
        n_pre_events,
        n_task_events,
        n_post_events,
    )

def get_mutually_exclusive_epochs(
    ied_epochs, pre_interval=[-20, -10], post_interval=[0, 10]
):
    ied_obs_pre_ = nel.EpochArray(
        np.array(
            [ied_epochs.start + pre_interval[0], ied_epochs.start + pre_interval[1]]
        ).T
    )
    ied_obs_post_ = nel.EpochArray(
        np.array(
            [ied_epochs.stop + post_interval[0], ied_epochs.stop + post_interval[1]]
        ).T
    )

    ied_obs_pre = ied_obs_pre_[~ied_obs_post_]
    ied_obs_post = ied_obs_post_[~ied_obs_pre_]

    return ied_obs_pre, ied_obs_post

def run(
    basepath,  # path to data folder
    brainRegions="CA1",  # reference region
    putativeCellType="Pyr",  # cell type
    min_cells=6,  # minimum number of cells per region
    restrict_task=False,  # restrict restriction_type to task epochs (ex. ripples in task (True) vs. all task (False))
    restriction_type="ripples",  # "ripples" or "NREMstate"
    task_binsize=0.125,  # in seconds, bin size for task epochs
    restrict_task_to_theta=True,  # restrict task to theta epochs
    single_bin_per_epoch=True,  # use single bin per restriction_type epoch for pre and post (ex. each ripple is a bin)
    rip_exp=0,  # ripple expansion start, in seconds, how much to expand ripples
    shrink_post=None,  # in minutes, how much time to shrink post task epoch to (ex. 30 minutes)
    env_to_compress=["sleep", "box", "open_field", "tmaze", "ymaze", "social_plusmaze"],
    min_barrage_time=0.3,
):
    # locate epochs
    ep_df = loading.load_epoch(basepath)
    # compress back to back epochs
    for val in env_to_compress:
        ep_df = compress_repeated_epochs.main(ep_df, epoch_name=val)

    # locate pre task post structure
    idx, _ = functions.find_pre_task_post(ep_df.environment)
    if idx is None:
        return None
    ep_df = ep_df[idx]
    # needs exactly 3 epochs for analysis
    if ep_df.shape[0] != 3:
        return None
    beh_epochs = nel.EpochArray(np.array([ep_df.startTime, ep_df.stopTime]).T)

    ripples = loading.load_ripples_events(basepath, return_epoch_array=True)

    # expand ripple epochs for analysis
    ripples = ripples.expand(rip_exp)

    ied_epochs = loading.load_ied_events(basepath)
    if ied_epochs.empty:
        return None
    
    ied_obs_pre, ied_obs_post = get_mutually_exclusive_epochs(ied_epochs)

    ripples_pre_ied = ripples[ied_obs_pre]
    ripples_post_ied = ripples[ied_obs_post]

    # initialize output
    evs = []
    revs = []
    regions = []
    n_cells = []
    events = []
    pairwise_corr = []
    pairwise_corr_epoch = []
    pairwise_corr_event = []
    pairwise_corr_ref_uid = []
    pairwise_corr_target_uid = []
    n_pre_events = []
    n_task_events = []
    n_post_events = []

    # load cells
    st, cell_metrics = loading.load_spikes(
        basepath, brainRegion=brainRegions, putativeCellType=putativeCellType
    )

    # check if enough cells
    if st.isempty | (st.n_active < min_cells):
        return
    
    for pre_post_restrict_epoch, restrict_label in zip(
        [ripples_pre_ied, ripples_post_ied],
        ["ripples_pre_ied", "ripples_post_ied"],
    ):

        # main calculation of explained variance
        (
            ev,
            rev,
            cor_pre,
            cor_beh,
            cor_post,
            ref_uid,
            target_uid,
            n_pre_events_,
            n_task_events_,
            n_post_events_,
        ) = get_explained_var(
            st,
            beh_epochs,
            cell_metrics,
            pre_post_restrict_epoch,
            task_binsize=task_binsize,
            restrict_task=restrict_task,
            single_bin_per_epoch=single_bin_per_epoch,
            shrink_post=shrink_post,
        )

        if np.isnan(ev):
            return None

        # store output
        evs.append(ev)
        revs.append(rev)
        events.append(restrict_label)
        n_cells.append(cell_metrics.shape[0])
        n_pre_events.append(n_pre_events_)
        n_task_events.append(n_task_events_)
        n_post_events.append(n_post_events_)

        # store pairwise correlations
        pairwise_corr.append(np.hstack([cor_pre, cor_beh, cor_post]))
        pairwise_corr_ref_uid.append(np.hstack([ref_uid, ref_uid, ref_uid]))
        pairwise_corr_target_uid.append(np.hstack([target_uid, target_uid, target_uid]))
        pairwise_corr_epoch.append(
            np.hstack(
                [
                    ["pre"] * len(cor_pre),
                    ["task"] * len(cor_beh),
                    ["post"] * len(cor_post),
                ]
            )
        )

        pairwise_corr_event.append(
            np.hstack(
                [
                    [restrict_label] * len(cor_pre),
                    [restrict_label] * len(cor_beh),
                    [restrict_label] * len(cor_post),
                ]
            )
        )
    # package output into dataframe
    ev_df = pd.DataFrame()
    ev_df["region"] = regions
    ev_df["events"] = events
    ev_df["ev"] = evs
    ev_df["rev"] = revs
    ev_df["n_cells"] = n_cells
    ev_df["n_pre_events"] = n_pre_events
    ev_df["n_task_events"] = n_task_events
    ev_df["n_post_events"] = n_post_events
    ev_df["basepath"] = basepath

    pairwise_corr_df = pd.DataFrame()
    if len(pairwise_corr) == 0:
        pairwise_corr_df["corrcoef"] = pairwise_corr
        pairwise_corr_df["epoch"] = pairwise_corr_epoch
        pairwise_corr_df["event"] = pairwise_corr_event
        pairwise_corr_df["ref_uid"] = pairwise_corr_ref_uid
        pairwise_corr_df["target_uid"] = pairwise_corr_target_uid
        pairwise_corr_df["basepath"] = basepath
    else:
        pairwise_corr_df["corrcoef"] = np.hstack(pairwise_corr)
        pairwise_corr_df["epoch"] = np.hstack(pairwise_corr_epoch)
        pairwise_corr_df["event"] = np.hstack(pairwise_corr_event)
        pairwise_corr_df["ref_uid"] = np.hstack(pairwise_corr_ref_uid)
        pairwise_corr_df["target_uid"] = np.hstack(pairwise_corr_target_uid)
        pairwise_corr_df["basepath"] = basepath

    # nest dataframes into dictionary and return
    results = {"ev_df": ev_df, "pairwise_corr_df": pairwise_corr_df}
    return results


def load_results(save_path, verbose=False):
    """
    load_results: load results from a directory
    """
    sessions = glob.glob(save_path + os.sep + "*.pkl")
    ev_df = pd.DataFrame()
    pairwise_corr_df = pd.DataFrame()
    for session in sessions:
        if verbose:
            print(session)
        with open(session, "rb") as f:
            results = pickle.load(f)
        if results is None:
            continue
        ev_df = pd.concat([ev_df, results["ev_df"]], ignore_index=True)
        pairwise_corr_df = pd.concat(
            [pairwise_corr_df, results["pairwise_corr_df"]], ignore_index=True
        )
    return ev_df, pairwise_corr_df
