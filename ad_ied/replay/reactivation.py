from ripple_heterogeneity.assembly import assembly_reactivation
from ripple_heterogeneity.utils import functions, loading, add_new_deep_sup
import numpy as np
import nelpy as nel
import pandas as pd
import os
import pickle
import glob


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


def run(basepath: str, restrict_to_nrem=False, restrict_pre_post_time=None):

    # initiate model
    assembly_react = assembly_reactivation.AssemblyReact(basepath=basepath)
    assembly_react.load_data()

    if assembly_react.isempty:
        return None

    # if this fails, there is no pre task post structure in session
    try:
        assembly_react.restrict_epochs_to_pre_task_post()
    except:
        return None

    # load ieds
    ied_epochs = loading.load_ied_events(basepath)
    ied_epoch_array = nel.EpochArray(ied_epochs[["start", "stop"]].values)
    if ied_epochs.empty:
        return None

    # get mutually exclusive epochs of pre and post ieds
    ied_obs_pre, ied_obs_post = get_mutually_exclusive_epochs(ied_epochs)

    state_dict = loading.load_SleepState_states(basepath)
    try:
        theta_epochs = nel.EpochArray(state_dict["THETA"])
    except:
        theta_epochs = nel.EpochArray(state_dict["WAKEtheta"])
    nrem_epochs = nel.EpochArray(state_dict["NREMstate"])

    # define pre / task / post epochs
    #   restrict to first hour of sleep
    #   restrict task to theta epochs and sleep to nrem
    if restrict_pre_post_time is not None:
        pre_start = assembly_react.epochs[0].start
        pre = assembly_react.epochs[0][nel.EpochArray([pre_start, pre_start + 3600])]
    else:
        pre = assembly_react.epochs[0]

    if restrict_to_nrem:
        pre = pre[nrem_epochs]

    task = assembly_react.epochs[1][theta_epochs]

    if restrict_pre_post_time is not None:
        post_start = assembly_react.epochs[2].start
        post = assembly_react.epochs[2][nel.EpochArray([post_start, post_start + 3600])]
    else:
        post = assembly_react.epochs[2]

    if restrict_to_nrem:
        post = post[nrem_epochs]

    if pre.isempty | task.isempty | post.isempty:
        return None

    response_df = pd.DataFrame()
    peth = pd.DataFrame()

    # detect assembies in task theta epoch
    assembly_react.get_weights(epoch=task)

    if assembly_react.n_assemblies() == 0:
        return

    assembly_act = assembly_react.get_assembly_act()

    # first get peth for assemblies around pre and post IEDs

    peth_ied_avg_pre, time_lags = functions.event_triggered_average(
        assembly_act.abscissa_vals,
        assembly_act.data.T,
        ied_epoch_array[pre].starts,
        sampling_rate=assembly_act.fs,
        window=[-0.5, 0.5],
    )

    peth_ied_avg_post, time_lags = functions.event_triggered_average(
        assembly_act.abscissa_vals,
        assembly_act.data.T,
        ied_epoch_array[post].starts,
        sampling_rate=assembly_act.fs,
        window=[-0.5, 0.5],
    )

    # gather some reponse metrics
    response_df_ied = pd.DataFrame()
    # get mean of peth 0 to 100ms
    response_df_ied["response"] = np.hstack(
        [
            np.nanmean(
                peth_ied_avg_pre[(time_lags > 0) & (time_lags < 0.1), :], axis=0
            ),
            np.nanmean(
                peth_ied_avg_post[(time_lags > 0) & (time_lags < 0.1), :], axis=0
            ),
        ]
    )
    response_df_ied["assembly_n"] = np.hstack(
        [np.arange(peth_ied_avg_pre.shape[1]), np.arange(peth_ied_avg_pre.shape[1])]
    )

    # add labels for epochs
    response_df_ied["epoch"] = np.hstack(
        [["pre"] * peth_ied_avg_pre.shape[1], ["post"] * peth_ied_avg_pre.shape[1]]
    )

    response_df_ied["event"] = "ied"
    response_df_ied["n_cells"] = assembly_react.st.n_active

    response_df_ied["basepath"] = basepath

    peth_ied_avg_pre = pd.DataFrame(
        index=time_lags, columns=np.arange(peth_ied_avg_pre.shape[1]), data=peth_ied_avg_pre
    )
    peth_ied_avg_post = pd.DataFrame(
        index=time_lags,
        columns=np.arange(peth_ied_avg_post.shape[1]),
        data=peth_ied_avg_post,
    )

    peth = pd.concat(
        [peth, pd.concat([peth_ied_avg_pre, peth_ied_avg_post], axis=1, ignore_index=True)],
        axis=1,
        ignore_index=True,
    )
    response_df = pd.concat([response_df, response_df_ied], ignore_index=True)

    # Now get peth for assemblies around ripples that occured during pre and post IEDs
    for pre_post_restrict_epoch, restrict_label in zip(
        [ied_obs_pre, ied_obs_post],
        ["ripples_pre_ied", "ripples_post_ied"],
    ):

        peth_avg_pre, time_lags = functions.event_triggered_average(
            assembly_act.abscissa_vals,
            assembly_act.data.T,
            assembly_react.ripples[pre & pre_post_restrict_epoch].starts,
            sampling_rate=assembly_act.fs,
            window=[-0.5, 0.5],
        )
        peth_avg_post, time_lags = functions.event_triggered_average(
            assembly_act.abscissa_vals,
            assembly_act.data.T,
            assembly_react.ripples[post & pre_post_restrict_epoch].starts,
            sampling_rate=assembly_act.fs,
            window=[-0.5, 0.5],
        )

        response_df_temp = pd.DataFrame()
        # get mean of peth 0 to 100ms
        response_df_temp["response"] = np.hstack(
            [
                np.nanmean(
                    peth_avg_pre[(time_lags > 0) & (time_lags < 0.1), :], axis=0
                ),
                np.nanmean(
                    peth_avg_post[(time_lags > 0) & (time_lags < 0.1), :], axis=0
                ),
            ]
        )
        response_df_temp["assembly_n"] = np.hstack(
            [np.arange(peth_avg_pre.shape[1]), np.arange(peth_avg_pre.shape[1])]
        )

        # add labels for epochs
        response_df_temp["epoch"] = np.hstack(
            [["pre"] * peth_avg_pre.shape[1], ["post"] * peth_avg_pre.shape[1]]
        )

        response_df_temp["event"] = restrict_label
        response_df_temp["n_cells"] = assembly_react.st.n_active

        response_df = pd.concat([response_df, response_df_temp], ignore_index=True)

        peth_avg_pre = pd.DataFrame(
            index=time_lags, columns=np.arange(peth_avg_pre.shape[1]), data=peth_avg_pre
        )
        peth_avg_post = pd.DataFrame(
            index=time_lags,
            columns=np.arange(peth_avg_post.shape[1]),
            data=peth_avg_post,
        )

        peth = pd.concat(
            [peth, pd.concat([peth_avg_pre, peth_avg_post], axis=1, ignore_index=True)],
            axis=1,
            ignore_index=True,
        )

    response_df["basepath"] = basepath

    results = {
        "results_df": response_df,
        "peth": peth
    }

    return results


def load_results(save_path: str, verbose: bool = False):
    """
    load_results: load results (pandas dataframe) from a pickle file

    This is the most basic results loader and
        **will only work if your output was a pandas dataframe (long format)**

    This will have to be adapted if your output was more complicated, but you can
        use this function as an example.
    """

    if not os.path.exists(save_path):
        raise ValueError(f"folder {save_path} does not exist")

    sessions = glob.glob(save_path + os.sep + "*.pkl")

    results = pd.DataFrame()
    peth = pd.DataFrame()

    for session in sessions:
        if verbose:
            print(session)
        with open(session, "rb") as f:
            results_ = pickle.load(f)
        if results_ is None:
            continue
        if results_["results_df"].shape[0] == 0:
            continue
        if results_["peth"].shape[0] == 501:
            test = 0
        results = pd.concat([results, results_["results_df"]], ignore_index=True)
        peth = pd.concat([peth, results_["peth"]], axis=1)

    return results, peth
