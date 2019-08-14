#!/usr/bin/env python

import datetime
import re

verbose = False

class CASALogInfo(object):
    """
    General run information, timings and other statistics extracted from a CASA run log,
    with a focus on timing of tasks.
    """
    def __init__(self):
        self._orig_log_filename = None
        self._run_machine = 'unknown'
        self._casa_version = 'unknown'
        self._first_tstamp = '---'
        self._last_tstamp = '---'
        self._mous = 'unknown'
        self._project_tstamp = None
        self._first_eb_uid = 'unknown'
        self._mpi_servers = 0
        self._task_time_accum = None
        # counters for CASA tasks. A dictionary with one entry for every different task
        self._casa_tasks_counter = None
        # counters for pipeline tasks, heuristics, etc. blocks
        self._pipe_tasks_counter = None
        # counters for pipeline stages
        self._pipe_stages_counter = None
        # counters for 'other' tasks, such as findCont
        self._special_tasks_counter = None
        # List of details about individual calls to tasks (especially for tclean)
        self._tasks_details_params = None
        self._total_time = 0
        self._total_time_casa_tasks = 0

    def __init__(self, log_fname, run_machine, casa_version, first_tstamp, last_tstamp,
                 project_tstamp, mous, first_eb_uid, mpi_server_cnt, casa_tasks_counter,
                 pipe_tasks_counter, pipe_stages_counter,
                 special_tasks_counter, tasks_details_params,
                 total_time, total_time_casa_tasks):
        self._orig_log_filename = log_fname
        self._run_machine = run_machine
        self._casa_version = casa_version
        self._first_tstamp = first_tstamp
        self._last_tstamp = last_tstamp
        self._project_tstamp = project_tstamp
        self._mous = mous
        self._first_eb_uid = first_eb_uid
        self._mpi_servers = mpi_server_cnt
        self._total_time = total_time
        self._total_time_casa_tasks = total_time_casa_tasks
        self._casa_tasks_counter = casa_tasks_counter
        self._pipe_tasks_counter = pipe_tasks_counter
        self._pipe_stages_counter = pipe_stages_counter
        self._tasks_details_params = tasks_details_params
        self._special_tasks_counter = special_tasks_counter

    FILENAME_TSTAMP_STRFTIME = '%Y%m%d_%H%M%S'

    def build_filename_extended_tags(self):
        """
        Produces a very long string to append to file names, consisting of:
        MOUS uid, # MPI servers, machine, CASA version, init timestamp.
        This is needed to avoid file name clashes when for example running
        repetitions of a same test at different times, or with different number
        of cores/MPI servers.
        """
        short_dataset_id = ''
        try:
            from casa_logs_mous_props import mous_short_names
            try:
                short_dataset_id = mous_short_names[self._mous]
            except KeyError:
                short_dataset_id = 'no_short_name_' + str(self._project_tstamp)
        except ImportError:
            pass

        return ('{0}_MOUS_{1}_mpi_{2}_host_{3}_casa_{4}_tstamp_{5}'.
                format(short_dataset_id,
                       self.build_file_mous_tag_name(),
                       self._mpi_servers, self._run_machine, self._casa_version,
                       self._first_tstamp.strftime(self.FILENAME_TSTAMP_STRFTIME)))

    def build_file_mous_tag_name(self):
        """
        Use to produce a MOUS or first EB uid tag for the file names of the
        tables and plots.

        For runs that in addition have the "Project+Timestamp" ID, like
        E2E5.1.00006.S_2017_09_12T19_54_03.778,
        this is also appended.
        """

        result = ''
        if 'unknown' == self._mous:
            result =  'unknown_first_eb_' +  self._first_eb_uid
        else:
            result = self._mous

        if self._project_tstamp:
            result += '_proj_' + self._project_tstamp

        return result


class PipeStageCounter(object):
    """
    Info about a pipeline stage extracted from a CASA log.
    """
    def __init__(self, name, equiv_call, start_t):
        self._name = name  # this is a number: stage 1, 2, etc.
        self._equiv_call = equiv_call
        self._start_t = start_t
        self._cnt = 1
        self._end_t = None
        self._taccum = 0 #datetime.timedelta(0)

    def ends(self, tstamp):
        self._end_t = tstamp
        self._taccum = (tstamp - self._start_t).total_seconds()

PIPE_FIRST_IMAGING_STAGE = 23
PIPE_LAST_STAGE = 36
class CASATaskAggLogInfo(object):
    """
    Aggregated info for a CASA task extracted from a CASA log file.
    This accumulates info for all the calls to a given casa task, 
    identified by its name, for example tclean, flagdata, etc.
    """
    def __init__(self, name):
        self._name = name
        self._cnt = 0
        self._taccum = 0 #datetime.timedelta(0)
        self._taccum_calib_1_22 = 0
        self._taccum_imaging_23_ = 0
        self._taccum_pipe_stages = {}
        self._block_found_open = 0

    def add_runtime_in_this_stage(self, task_runtime, pipe_stages_current):
        """
        :param task_runtime: runtime of an execution of this task, expected
        in seconds as usual
        :param pipe_stages_current: name of the current pipeline stage, string
        with an integer
        """
        self._taccum += task_runtime
        if pipe_stages_current not in self._taccum_pipe_stages:
            self._taccum_pipe_stages[pipe_stages_current] = task_runtime
        else:
            self._taccum_pipe_stages[pipe_stages_current] += task_runtime
        stage_idx = int(pipe_stages_current)
        if stage_idx >= 23:
            self._taccum_imaging_23_ += task_runtime
        else:
            self._taccum_calib_1_22 += task_runtime

class CommonBeamInfo(object):
    def __init__(self, major, minor, pa):
        self._major = major
        self._minor = minor
        self._pa = pa

class CASATaskDetails(object):
    """
    For tasks for which detailed called info is parsed, for example
    the parameters for individual tclean calls.

    """
    def __init__(self, name, tstamp, pipe_stage_seqno, pipe_stage_name, params):
        self._name = name
        self._call_time = tstamp
        self._pipe_stage_seqno = pipe_stage_seqno
        self._pipe_stage_name = pipe_stage_name
        self._runtime = 0
        self._params = params
        self._further_info = {}

def version_equal_or_after(vers_str, major, minor, patch):
    """" 
    usage: version_equal_or_after(5,4,0) to know if it was >= 5.4.0
    """
    vers = vers_str.split('.')
    if len(vers) != 3:
        raise RuntimeError('Cannot parse CASA version: {0}'.format(vers_str))

    return (vers[0] > major or 
            vers[0] == major and vers[1] > minor or
            vers[0] == major and vers[1] == minor and vers[2] >= patch)

def get_ranked_dict_by_taccum(dict_taccum):
    """
    Produces a list, where the sort order is the ranking by runtime

    :param dict_taccum: a dictionary of objects that have the attribute '_taccum' (with
    run time).
    """
    import operator
    ranked = sorted(dict_taccum.values(), key=operator.attrgetter('_taccum'), reverse=True)
    return ranked

def print_ranked_pipe_tasks_counter(ptc):
    sorted_ptc = get_ranked_dict_by_taccum(ptc)
    print('---------------------------')
    total = sum([item._taccum for item in sorted_ptc])
    print('# task_name, num_calls, time_sec, time_percentage_all_tasks')
    for val in sorted_ptc:
        # The pseudo-tasks/pipeline things/other often accumulate just 0s
        if 0 == total:
            pc_accum = 0
        else:
            pc_accum = val._taccum / total * 100.0
        print('{0}, {1}, {2}, {3:.2f}'.format(val._name, val._cnt, val._taccum,
                                              pc_accum))
    print('---------------------------')

def print_ranked_pipe_stages_counter(psc):
    sorted_psc = get_ranked_dict_by_taccum(psc)
    print('---------------------------')
    for val in sorted_psc:
        print('{0}, {1}, {2}'.format(val._name, val._equiv_call, val._taccum))
    print('---------------------------')

    
def format_tbl_elapsed(elapsed):
    days = elapsed.days
    rem_seconds = elapsed.seconds
    SEC_PER_HOUR = 3600
    hours = int(rem_seconds / SEC_PER_HOUR)
    rem_seconds -= hours * SEC_PER_HOUR
    SEC_PER_MIN = 60
    minutes = int(rem_seconds / SEC_PER_MIN)
    rem_seconds -= minutes * SEC_PER_MIN
    subday_str = ('{0}h{1}m{2}s'.
                  format(hours, minutes, rem_seconds))
    if days <= 0:
        return subday_str
    else:
        return ('{0}d {1}'.
                format(days, subday_str))

def get_ranked_list_from_dict(ind, reverse_order=True, rank_by_key=False):
    import operator
    if rank_by_key:
        item_idx = 0
    else:
        item_idx = 1
    sorted_d = sorted(ind.items(), key=operator.itemgetter(item_idx),
                      reverse=reverse_order)
    return sorted_d

def print_ranked_dict(ind):
    sorted_d = get_ranked_list_from_dict(ind)   
    totals_dict = sum(ind.values())

    for k,v in sorted_d:
        print("{0}{1}: {2} ({3:.4g}%)".format('   ', v, k, v/totals_dict*100.0))

    return sorted_d


def plot_tasks_pie_chart(axis, ranking):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    labels = [task[0] for task in ranking]
    sizes = [task[1] for task in ranking]
    cmap = cm.get_cmap('Set1')
    colors = [cmap(1.*idx/len(labels)) for idx in range(len(labels))]
    axis.pie(sizes, labels=labels, autopct='%1.1f%%',
             shadow=True, colors=colors) # explode=explode, startangle=90
    axis.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

def plot_tasks_bar(axis, ranking):
    import matplotlib.cm as cm

    ranking.reverse()   # Put bigger ones at the bottom
    labels = [task[0] for task in ranking]
    sizes = [task[1] for task in ranking]    
    axis.bar(0, sizes[0], label=labels[0], bottom=0) #, color=colors[0])
    #axis.bar(1, sizes[0]) #, color=colors[0])
    axis.bar(1, 0)
    axis.bar(2, 0)
    cmap = cm.get_cmap('Set1') # Set2,3
    bottom = 0
    for idx in xrange(1, len(sizes)):
        bottom += sizes[idx-1]
        seg_col = cmap(1-float(idx+1)/len(sizes))
        axis.bar(0, sizes[idx], bottom=bottom,
                 color=seg_col,
                 label=labels[idx]) # color=colors[j],
        #axis.bar(1, sizes[idx], bottom=sizes[idx-1],
        #         color=seg_col) # color=colors[j],

    # axis.legend(loc='best')
    # reverse up-down order of the legend labels
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1], title='Tasks', loc='best')
    axis.set_ylabel('time (s)')

def plot_timing_things(run_infos):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # TODO
    ranking = get_ranked_list_from_dict(run_infos[0]._task_time_accum)

    fig = plt.figure(figsize=(8,16))
    # ax_top = fig.add_subplot(211)
    ax_top = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    plot_tasks_pie_chart(ax_top, ranking)
    # ax_bottom = fig.add_subplot(212)
    ax_bottom = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    plot_tasks_bar(ax_bottom, ranking)
    plt.tight_layout()

    
    fig_fname = ('casa_logs_tasks_timing_barplot_MOUS_{0}.png'.
                 format(run_infos[-1].build_filename_extended_tags()))
    fig.savefig(fig_fname, transparent=False)
    print(' * Produced bar plot: {0}'.format(fig_fname))

    show_plots = False
    if show_plots:
        plt.show()

def prepare_colors_rows(rows, len_cols):
    colors_rows = []
    colors_cells = []
    for idx, row in enumerate(rows):
        if 0 == (idx % 2):
            colors_rows.append('white')
            colors_cells.append( ['white'] * len_cols)
        else:
            colors_rows.append('lightgrey')
            colors_cells.append( ['lightgrey'] * len_cols)

    return (colors_rows, colors_cells)

def prepare_title_etc(run_infos):
    """
    :param run_info: needed to add general info about the runs in the title
    :param ax: axes object where to show the title
    """

    title = ('Pipeline test run, MOUS: {0}, proj: {1}, first EB: {2}'.
             format(run_infos[0]._mous, run_infos[0]._project_tstamp,
                    run_infos[0]._first_eb_uid))

    for idx, rinfo in enumerate(run_infos):
        run_type = get_run_type_str(rinfo)

        title += ('\n- {0}: run on {1}, started {2}, finished: {3}. CASA {4}'.
                  format(run_type, rinfo._run_machine, rinfo._first_tstamp,
                         rinfo._last_tstamp, rinfo._casa_version))
        time_others = rinfo._total_time - rinfo._total_time_casa_tasks
        
    
    # This goes in the general CASA tasks table. Not sure if also useful here.
    # x_loc_info = -0.08
    # y_loc_info = -0.055
    # info_text = ('Run type:\ntotal time:\ntime CASA tasks:\ntime other:')
    # plt.text(x_loc_info, y_loc_info, info_text,
    #          horizontalalignment='left', verticalalignment='center',
    #          transform = ax.transAxes, fontsize=11)
    return title

def make_casa_tasks_table_file(out_fname, run_infos, rows, columns, cell_texts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    colors_rows, colors_cells = prepare_colors_rows(rows, len(columns))

    # Make figure
    fig = plt.figure(figsize=(11.69, 8.27), dpi=300)
    fig.set_size_inches(11.69, 8.27)
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_visible(False)
    ax.axis('off')
    title = ('pipeline test run, MOUS: {0}, proj: {1}, first EB: {2}'.
             format(run_infos[0]._mous, run_infos[0]._project_tstamp,
                    run_infos[0]._first_eb_uid))

    x_loc_info = -0.08
    y_loc_info = -0.055
    info_text = ('Run type:\ntotal time:\ntime CASA tasks:\ntime other:')
    plt.text(x_loc_info, y_loc_info, info_text,
             horizontalalignment='left', verticalalignment='center',
             transform = ax.transAxes, fontsize=11)

    time_others_ref = datetime.timedelta(seconds=(run_infos[0]._total_time -
                                                  run_infos[0]._total_time_casa_tasks))
    for idx, rinfo in enumerate(run_infos):
        if rinfo._mpi_servers > 0:
            run_type = 'parallel ({0})'.format(rinfo._mpi_servers + 1)
        else:
            run_type = 'serial'       

        title += ('\n- {0}: run on {1}, started {2}, finished: {3}. CASA {4}'.
                  format(run_type, rinfo._run_machine, rinfo._first_tstamp,
                         rinfo._last_tstamp, rinfo._casa_version))

        time_others = datetime.timedelta(seconds=
                                         rinfo._total_time - rinfo._total_time_casa_tasks)
        if 'serial' is run_type or idx < 1:
            info_text = ('{0}\n{1}\n{2}\n{3}'.format(run_type,
                                                     datetime.timedelta(
                                                         seconds=rinfo._total_time),
                                                     datetime.timedelta(
                                                         seconds=rinfo._total_time_casa_tasks),
                                                     time_others))
        else:
            info_text = ('{0} - {1}\n{2} - {3:.3f}\n{4} - {5:.3f}\n{6} - {7:.3f}'.
                         format(run_type, 'ratio',
                                datetime.timedelta(seconds=rinfo._total_time),
                                float(rinfo._total_time) / run_infos[0]._total_time,
                                datetime.timedelta(seconds=rinfo._total_time_casa_tasks),
                                float(rinfo._total_time_casa_tasks) /
                                run_infos[0]._total_time_casa_tasks,
                                time_others,
                                float(time_others.total_seconds()) /
                                time_others_ref.total_seconds()))
        
        plt.text(x_loc_info+0.3*(idx+1), y_loc_info, info_text,
                 horizontalalignment='left', verticalalignment='center',
                 transform = ax.transAxes, fontsize=11)

    fig.suptitle(title, fontsize=11, x=0.08, horizontalalignment='left')

    
    tbl = ax.table(cellText=cell_texts, rowLabels=rows, colLabels=columns,
                   rowColours=colors_rows, cellColours=colors_cells,
                   bbox=[0.05, 0, 1, 1],
                   cellLoc='left',
                   loc='top')

    # Manipulate widths
    cellDict = tbl.get_celld()
    for col_idx in range(0, cell_texts.shape[1]):
        # even index rows are narrower (number of calls, speedup ratio)
        if 0 != col_idx % 2:
            continue
        for row_idx in range(0, cell_texts.shape[0] + 1):
            cellDict[(row_idx, col_idx)].set_width(0.12)

    
    tbl.set_fontsize(11)

    # plt.show()
    fig.savefig(out_fname, transparent=True)
    # bbox_inches='tight'


def make_table_file(out_fname, run_infos, rows, columns, cell_texts,
                    fontsize=10, portrait=False):
    """
    Writes a table as (normally) a pdf file.
    This uses matplotlib's table for broader compatibility/support even though
    it is far from the nicest way of producing a table in a pdf file.

    :param out_fname: filename for the output table
    :param run_infos: Used to produce title with additional info about the runs
    :param rows: labels for the rows
    :param columns: labels for the columns
    :cell_texts: the values for the cells of the table, as text
    """
    # Note the 'Agg' backend
    # http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors_rows, colors_cells = prepare_colors_rows(rows, len(columns))

    title = prepare_title_etc(run_infos)

    if not portrait:
        # TODO: PDF table size
        #fig = plt.figure(figsize=(116.9, 82.7), dpi=300)
        #fig.set_size_inches(116.9, 82.7)
        fig = plt.figure(figsize=(11.69, 4*8.27), dpi=300)
        fig.set_size_inches(11.69, 4*8.27)
    else:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
        fig.set_size_inches(8.27, 11.69)
        
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_visible(False)
    ax.axis('off')

    fig.suptitle(title, fontsize=fontsize, x=0.05, horizontalalignment='left')

    if not portrait:
        tbl_bbox = [0.5, 0, 1, 1]
    else:
        # left x needs space for lengthy names
        tbl_bbox = [0.33, 0, 0.85, 1]

    tbl = ax.table(cellText=cell_texts, rowLabels=rows, colLabels=columns,
                   rowColours=colors_rows, cellColours=colors_cells,
                   bbox=tbl_bbox,
                   cellLoc='left',
                   loc='top')

    tbl.set_fontsize(fontsize)

    # If wanted to show on the fly
    # plt.show()
    fig.savefig(out_fname, transparent=True)

def generate_comparison_table(run_infos):

    generate_comparison_table_CASA_tasks(run_infos)
    generate_comparison_table_pipeline_stages(run_infos)
    generate_comparison_table_pipeline_tasks_etc(run_infos)

def get_run_type_str(run_info):
    if run_info._mpi_servers > 0:
        run_type = 'parallel {0}'.format(run_info._mpi_servers + 1)
    else:
        run_type = 'serial'
    return run_type
    
def generate_comparison_table_pipeline_stages(run_infos):
    """
    Print: stage#, stage_call; then for every run: time
    where stage_call is hif_makeimages, hif_findcont, hifa_exportdata, etc.
    """
    import numpy as np

    # build the name with the info of the last run which is usually the parallel one
    table_fname = ('casa_logs_pipe_stages_timing_table_MOUS_{0}.pdf'.
                   format(run_infos[-1].build_filename_extended_tags()))

    def produce_pipe_stages_cols(run_infos):
        cols = []
        for rinfo in run_infos:
            run_type = get_run_type_str(rinfo)
            cols.append('{0}, time (%)'.format(run_type))
        return cols

    cols = produce_pipe_stages_cols(run_infos)

    # Produces a list that has, for every different run, a ranked list of stages
    ranked_stages = [get_ranked_dict_by_taccum(rif._pipe_stages_counter) for rif in run_infos]

    # For row names, use the ranking of the first run
    rows = ['{0}, {1}'.format(stg._name, stg._equiv_call) for stg in ranked_stages[0]]

    values = np.ndarray((len(rows), len(cols)), dtype='object')
    for col_idx in range(0, values.shape[1]):
        ranking = ranked_stages[col_idx]
        sum_time = sum([stg._taccum for stg in ranking])
        sum_time = datetime.timedelta(seconds=sum_time)

        stages_cnt = run_infos[col_idx]._pipe_stages_counter
        for row_idx in range(0, values.shape[0]):
            try:
                # name of the row_idx-th stage for the first run compared in the table
                first_row_key = ranked_stages[0][row_idx]._name
            except KeyError as key_err:
                print(' *** Could not find this stage in ranking from log: {0}.'
                      ' Are all the logs given for the same test dataset?'
                      ' Error: {1}'.format(row_idx, key_err))
            this_time = datetime.timedelta(seconds=stages_cnt[first_row_key]._taccum)
            values[row_idx, col_idx] = '{0} ({1:.3g})'.format(this_time,
                                                              100 *
                                                              this_time.total_seconds() /
                                                              sum_time.total_seconds())
    
    make_table_file(table_fname, run_infos, rows, cols, values, portrait=False)
    print(' * Produced table of pipeline stages: {0}'.format(table_fname))
    

def generate_comparison_table_pipeline_tasks_etc(run_infos):
    """
    For every pipeline, etc. (where etc. can be heuristics, qa, etc.), produce:
    name, number of log lines where it is seen, aggregated time

    Example:
    qa.scorecalculator, 122, 0:03:52
    infrastructure.displays.sky, 577, 0:01:34
    hif.heuristics.imageparams_base, 330, 0:01:11
    """
    import numpy as np

    table_fname = ('casa_logs_pipe_tasks_heuristics_etc_timing_table_MOUS_{0}.pdf'.
                   format(run_infos[-1].build_filename_extended_tags()))

    MAX_ROWS = 55

    #rows = range(0, len(run_infos[0]._pipe_tasks_counter))
    rows = range(0, MAX_ROWS)

    def produce_pipe_tasks_cols(run_infos):
        cols = []
        for rinfo in run_infos:
            run_type = get_run_type_str(rinfo)
            # #log_lines not so informative
            # cols.append('{0} # log lines'.format(run_type))
            cols.append('{0}, time (%)'.format(run_type))
        return cols

    cols = produce_pipe_tasks_cols(run_infos)

    # Produces a list that has, for every different run, a ranked list of tasks/heuristics
    ranked_tasks = [get_ranked_dict_by_taccum(rif._pipe_tasks_counter) for rif in run_infos]

    if MAX_ROWS > 0:
        for idx, rts in enumerate(ranked_tasks):
            ranked_tasks[idx] = ranked_tasks[idx][0:MAX_ROWS]
    # Base row names on the ranking of the first run
    rows = ['{0}'.format(task._name) for task in ranked_tasks[0]]
    
    values = np.ndarray((len(rows), len(cols)), dtype='object')
    for col_idx in range(0, values.shape[1]):
        ranking = ranked_tasks[col_idx]
        sum_time = sum([task._taccum for task in ranking])
        sum_time = datetime.timedelta(seconds=sum_time)

        tasks_cnt = run_infos[col_idx]._pipe_tasks_counter
        for row_idx in range(0, values.shape[0]):
            first_row_key = ranked_tasks[0][row_idx]._name

            try:
                this_time = datetime.timedelta(seconds=tasks_cnt[first_row_key]._taccum)
            except KeyError as exc:
                msg = ('Could not find {0}. It is present in the first log but '
                       'missing from the others. Are all the logs from pipeline '
                       'runs, and do they use the same pipeline version? Error '
                       'details: {1}'.format(first_row_key, exc))
                # raise RuntimeError(msg)
                print(' * WARNING: {0}'.format(msg))
            values[row_idx, col_idx] = '{0} ({1:.3g})'.format(this_time,
                                                              100 *
                                                              this_time.total_seconds() /
                                                              sum_time.total_seconds())

    make_table_file(table_fname, run_infos, rows, cols, values, fontsize=6, portrait=True)
    print(' * Produced table of pipeline tasks, heuristics, etc.: {0}'.format(table_fname))

    
def generate_comparison_table_CASA_tasks(run_infos_orig):
    """
    Needs cleaning after too many changes to tables.
    """
    import numpy as np

    # horrible, if only one log given, duplicate it.
    if 1==len(run_infos_orig):
        run_infos = [run_infos_orig[0], run_infos_orig[0]]
    else:
        run_infos = run_infos_orig

    tasks_counter = run_infos[0]._casa_tasks_counter
    tasks_counter_other = run_infos[1]._casa_tasks_counter
    taccum_ranked = get_ranked_dict_by_taccum(run_infos[0]._casa_tasks_counter)
    counter_other = run_infos[1]._casa_tasks_counter

    rows = [item._name for item in taccum_ranked]
    mpi_level = run_infos[1]._mpi_servers

    cols = []
    for rinfo in run_infos:
        if rinfo._mpi_servers > 0:
            run_type = 'parallel ({0})'.format(rinfo._mpi_servers + 1)
        else:
            run_type = 'serial'
        cols.append('{0}\ncalls'.format(run_type))
        cols.append('{0}\ntime(s) (%)'.format(run_type))
    cols.append('speedup\nratio')

    values = np.ndarray((len(rows), len(cols)), dtype='object')

    # Get totals in seconds
    tot_taccum_serial = sum([obj._taccum for obj in taccum_ranked])
    tot_taccum_par = sum([obj._taccum for key, obj in counter_other.items()])
    for row_idx in range(0, len(rows)):
        task_name = taccum_ranked[row_idx]._name
        taccum_serial = taccum_ranked[row_idx]._taccum
        try:
            taccum_par = counter_other[task_name]._taccum
        except KeyError as key_err:
            print(' *** Could not find this task in log: {0}. '
                  'Are all the logs given for the same test?'
                  ' Error: {1}'.format(task_name, key_err))
            taccum_par = 0
        tcalls_serial = tasks_counter[task_name]._cnt
        try:
            tcalls_par = counter_other[task_name]._cnt
        except KeyError as key_err:
            print(' *** Could not find this task in log: {0}. '
                  'Are all the logs given for the same test?'
                  ' Error: {1}'.format(task_name, key_err))
            tcalls_par = 0

        if 0 == tot_taccum_serial:
            serial_col = 0
        else:
            serial_col = 100.0 * taccum_serial / tot_taccum_serial
        par_col = 100.0 * taccum_par / tot_taccum_par
        if 0 == taccum_par:
            ratio_col = 0
        else:
            ratio_col = float(taccum_serial) / float(taccum_par)
        # Time converted to int, as the logs have s precision.    
        values[row_idx, :] = np.array([str(tcalls_serial),
                                       "{:d} ({:.3g})".format(int(taccum_serial),
                                                                serial_col),
                                       str(tcalls_par),
                                       "{:d} ({:.3g})".format(int(taccum_par),
                                                                par_col),
                                       "{:.3g}".format(ratio_col)])
    
    table_fname = ('casa_logs_CASA_tasks_timing_table_MOUS_{0}.pdf'.
                   format(run_infos[-1].build_filename_extended_tags()))
    make_casa_tasks_table_file(table_fname, run_infos, rows, cols, values)
    print(' * Produced table: {0}'.format(table_fname))


CASA_TSTAMP_STRFTIME = '%Y-%m-%d %H:%M:%S'

def correct_casa_log_datetime(dt_str):
    """
    Casa log lines sometimes show date-time strings like:
    2017-08-22 24:00:00
    Standard datetime chokes on the 24 hours. This function replaces
    the non-standard 24 as in this example:
    datetime.replace('2017-08-22 24:00:00', '2017-08-23 00:00:00')

    :param dt_str: input datetime string
    :returns: datetime string, with 24 hours issue corrected
    """
    import re
    import datetime

    if dt_str[11:13] != '24':
        norm_dt_str = dt_str
    else:
        date_t = datetime.datetime.strptime(dt_str[0:10], '%Y-%m-%d')
        next_day_00 = date_t + datetime.timedelta(days=1)
        norm_dt_str = next_day_00.strftime(CASA_TSTAMP_STRFTIME)

    return norm_dt_str

def get_timestamp_from_casa_log_line(line):
    log_date_time_re = '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'

    match = re.search(log_date_time_re, line)
    if not match:
        return None

    # The timestamp starts always from the first char, but sometimes we
    # get weird characters in the middle. So we really need the regex.
    # line[0:19] will not work.
    time_str = match.group(0)
    corr_time_str = correct_casa_log_datetime(time_str)
    tstamp = datetime.datetime.strptime(corr_time_str, CASA_TSTAMP_STRFTIME)
    return tstamp


def go_through_log_lines(logf):
    import os
    import sys

    verbose_excess = False

    parse_tclean = True
    if parse_tclean:
        print(' * Note: parsing tclean arguments, the info dump files will be bigger.')

    first_found = False
    first_tstamp = None
    last_tstamp = None

    first_importdata_found = False
    first_eb_uid = 'unknown'
    first_id_or_importasdm_found = False
    proj_tstamp = None
    mous_dir = 'unknown'
    run_machine = 'unknown'
    casa_version = 'unknown'
    mpi_server_cnt = 0

    pipe_tier_tasks = ['tclean', 'plotms']
    pipe_tier_nestedness = 0

    all_cnt = 0
    all_taccum = 0

    pipe_cnt = 0
    pipe_taccum = 0
    pipe_line_found = False

    # For the pipeline tasks (like qa.scorecalculator, hif.heuristics.imageparams_base,
    # hif.tasks.findcont.findcont, etc.)
    pipe_tasks_counter = {}

    special_task_patterns = {}
    sp_patterns = ['findContinuum.py_v']
    for pattern in sp_patterns:
        special_task_patterns[pattern] = CASATaskAggLogInfo(pattern)

    # For the pipeline stages. Stage 1 hifa_importdata, stage 33 hif_makeimages, etc.
    pipe_stages_counter = {}
    pipe_stages_current = None
    pipe_stages_current_equiv_call = None

    # For the CASA tasks: importdata, bandpass, applycal, flagdata, mstransform, tclean, etc.
    casa_tasks_counter = {}
    task_start_tstamp = {}
    inside_task = None
    task_seq = []
    task_seq_runtimes = []
    task_seq_starttstamp = []

    tasks_details_params = []
    # Name of the task for which the details (all params, etc.) are being tracked
    task_details_task_name = None

    stop_counting_after_pipe_end = True
    stop_counting_minimum_pipe_stage = 33

    # Before this, the 'equiv call: ...' line used to be printed before the 
    # stage xx start line. With >= 5.4.0, it is printed after.
    casa_version_equiv_call_change = [5, 4, 0]

    # Example log line:
    # 2017-07-18 17:13:26	INFO	::casa::MPIServer-1	CASA Version 5.1.0-1
    for line in logf:
        all_cnt += 1

        tstamp = get_timestamp_from_casa_log_line(line)
        if not tstamp:
            continue

        if first_found:
            diff = tstamp - last_tstamp
            if verbose_excess:
                print("tstamp: {0}, last: {1}, diff: {2}, ", tstamp, last_tstamp, diff)

            diff_secs = diff.total_seconds()
            all_taccum += diff_secs
            if pipe_line_found:
                if diff.days >= 0:
                    pipe_taccum += diff_secs
                pipe_line_found = False

            for key, entry in pipe_tasks_counter.items():
                if entry._block_found_open:
                    if diff.days >= 0:
                        entry._taccum += diff_secs
                    entry._block_found_open = False
                    pipe_tasks_counter.update({key: entry})

            for task_pattern, entry in special_task_patterns.items():
                if entry._block_found_open:
                    if diff.days >= 0:
                        entry._taccum += diff_secs
                    entry._block_found_open = False
                    special_task_patterns.update({task_pattern: entry})

        else:
            first_found = True
            first_tstamp = tstamp

        last_tstamp = tstamp


        def parse_task_params_into_dict(params_str,
                                        par_names=None):
            """
            Simplistic parser for tclean task calls.
            It uses the '=' trick to a) keep this simple, and b) avoid further dependencies
            on parser libraries that might not be available on test machines and for example
            the standard Python distribution that comes with CASA.

            The simplest safe trick to split the parameters is is by '=' 
            Note that ',' is both separator and content of some parameter values, and those
            parameter values can be enclosed in either quotes or square brackets, or both!

            This function will most likely fail if '=' is used in any of the argument values.
            """
            too_verbose = False
            if too_verbose:
                print(' * Found tclean command: {0}'.format(params_str))
            params = {}

            # find keyword parameters, on the left of '=' characters
            par_names = []
            for found in re.finditer('([^\s]+)\s*=', params_str):
                par_names.append(found.group(1))
            
            for lhs in par_names:
                # Match the name on the lhs up to a comma followed by another 'keyword=' (or
                # end of line, for the last argument)
                par_re = '{0}\s*=\s*(.+?)(,\s*[^\s]+=|$)'.format(lhs)
                try: 
                    par_match = re.search(par_re, params_str)
                except re.error as exc:
                    print(' ***** Error when parsing task parameters. Setting empty '
                          'parameters for this call. Error: {0}. Parameters: {1}'.
                          format(exc, tclean_all_pars))

                if par_match:
                    params[lhs.strip()] = par_match.group(1).strip().strip('\'')
                else:
                    print(' * ERROR. Failed to find task parameter: {0} ({1})'.
                          format(lhs, par_re))

            if too_verbose:
                print('This is params: {0}'.format(params))
            return params

        # Note this catches 'Executing xxxtaskxxx(...' log lines from the pipeline
        # This will not catch anything from logs from non-pipeline runs of CASA
        details_task_target = 'tclean'
        details_pattern = 'Executing {0}'.format(details_task_target)
        if parse_tclean and 'pipeline.' in line and details_pattern in line:
            tclean_pars_re = 'Executing\s+tclean\((.+)\)'
            tclean_pars_match = re.search(tclean_pars_re, line)
            if tclean_pars_match:
                tclean_all_pars = tclean_pars_match.group(1)
                task_params = parse_task_params_into_dict(tclean_all_pars)
                details_params = CASATaskDetails('tclean', tstamp, pipe_stages_current,
                                                 pipe_stages_current_equiv_call,
                                                 task_params)
                tasks_details_params.append(details_params)
                # to get its runtime later, when 'End Task' found
                task_details_task_name = details_task_target
        
        if 'pipeline.' in line:
            pipe_line_found = True
            pipe_cnt += 1

            pipe_task_re = '.pipeline.([^:]+)::'
            pipe_task_match = re.search(pipe_task_re, line)
            if pipe_task_match:
                name = pipe_task_match.group(1)
                if name in pipe_tasks_counter:
                    tlb = pipe_tasks_counter[name]
                    tlb._cnt += 1
                else:
                    tlb = CASATaskAggLogInfo(name)
                    tlb._cnt = 1
                tlb._block_found_open = True

                pipe_tasks_counter.update({name: tlb})

            # To get inside tools, etc. C++ level - use with care - very verbose output
            count_specials = False
            if count_specials:
                pipe_c_specials = ['pipeline.hif.heuristics.imageparams_base::Imager::open()',
                                   'pipeline.hif.tasks.tclean.tclean::imager',
                                   'setDataOnThisMS()',
                                   'advise()'
                ]

                #                for csp in pipe_c_specials:
                #                    if csp in line:
                pipe_long_task_re = '.pipeline.([^\s]+)'
                pipe_long_task_match = re.search(pipe_long_task_re, line)
                if pipe_long_task_match:
                    csp = pipe_long_task_match.group(1)

                    # TODO: refactor horror
                    if csp in pipe_tasks_counter:
                        tlb = pipe_tasks_counter[csp]
                        tlb._cnt += 1
                    else:
                        tlb = CASATaskAggLogInfo(csp)
                        tlb._cnt = 1
                    tlb._block_found_open = True

                    pipe_tasks_counter.update({csp: tlb})
                

        for task_pattern in special_task_patterns:
            if task_pattern in line:
                entry = special_task_patterns[task_pattern]
                entry._cnt += 1
                entry._block_found_open = True
                special_task_patterns.update({task_pattern: entry})



        # Parse specific log lines from inside tclean
        # Beam and common beam
        # Beam. In serial it looks like:
        # task_tclean::SIImageStore::printBeamSet         Beam : 0.54814 arcsec, 0.369634 arcsec, 89.7918 deg
        # In parallel:
        # SIImageStore::restore   Common Beam : 0.54814 arcsec, 0.369634 arcsec, 89.7918 deg
        if (task_details_task_name == 'tclean' and
            ('Common Beam :' in line and '::SIImageStore::restore' in line
             or
             ('Beam :' in line and '::SIImageStore::printBeamSet' in line))):
            details = tasks_details_params[-1]
            if 'tclean' == details._name and (0 == mpi_server_cnt or
                                              'True' == details._params['parallel']):
                cb_values_re = '([^\s]+) arcsec, ([^\s]+) arcsec, ([^\s]+) deg'
                # Common Beam for chan : 0 : 0.027519 arcsec, 0.0220263 arcsec, -56.0013 deg
                cb0_hdr = 'Beam for chan : 0 :'
                if cb0_hdr in line:
                    cb_chan0_re = '{0} {1}'.format(cb0_hdr, cb_values_re)
                    re_match = re.search(cb_chan0_re, line)
                    if re_match:
                        cbeam_chan0 = CommonBeamInfo(re_match.group(1), re_match.group(2),
                                                     re_match.group(3))
                        details._further_info['common_beam_chan0'] = cbeam_chan0
                # Common Beam : 0.027519 arcsec, 0.0220263 arcsec, -56.0013 deg
                else:
                    cb_re = 'Beam : {0}'.format(cb_values_re)
                    re_match = re.search(cb_re, line)
                    if re_match:
                        cbeam = CommonBeamInfo(re_match.group(1), re_match.group(2),
                                               re_match.group(3))
                        details._further_info['common_beam'] = cbeam

                tasks_details_params[-1] = details
        # Completed 157 iterations.
        # ...
        # ------ Run Major Cycle 1 ------
        # (also lines like:)
        # ------ Run (Last) Major Cycle 1 ------
        major_cycle = 'Major Cycle '
        if (task_details_task_name == 'tclean' and
            major_cycle in line):
            details = tasks_details_params[-1]
            cycle_re = '{0} (\d+)'.format(major_cycle)
            re_match = re.search(cycle_re, line)
            if re_match:
                idx = re_match.group(1)
                details._further_info['major_cycle_max'] = idx
                tasks_details_params[-1] = details
                
        # def identify_first_eb
        # Look for a line that contains something like:
        # hifa_importdata(vis=['uid___A002_Xb8e961_Xb0d', 'uid___A002_Xb8f857_X1176', 'uid___A002_Xb91513_X1936'], session=['default', 'default', 'default'])
        importdata_re = "hifa_importdata\s*\(\s*vis\s*=\s*\[\s*'(\w+)'"
        if not first_importdata_found and 'hifa_importdata' in line:
            id_match = re.search(importdata_re, line)
            if id_match:
                first_importdata_found = True
                first_eb_uid = id_match.group(1)
                print(' * Found first execution block uid: {0}'.
                       format(first_eb_uid))

        # When running from EPPR,
        # Example Project+Timestamp ID: E2E5.1.00006.S_2017_09_12T19_54_03.778
        eppr_rawdir_re = "INFO\s+::casa\s+Working directory:.+/(.+)/SOUS_.+/GOUS_.+/MOUS_([a-zA-Z0-9_]+)/"
        if 'Working directory:' in line:
            print (' * Found EPPR working dir line ' + line)
            rawdir_match = eppr_rawdir_re = re.search(eppr_rawdir_re, line)
            if rawdir_match:
                proj_tstamp = rawdir_match.group(1)
                mous_dir = rawdir_match.group(2)
                first_id_or_importasdm_found = True
                print(' * Found project+timestamp ID from EPPR info: {0}, '
                      'and MOUS: {1}'.format(proj_tstamp, mous_dir))

        # example: importasdm(asdm="/lustre/naasc/users/scastro/pipeline/cycle5_testing/uid___A001_X879_X6d1/rawdata/uid___A002_Xb8e961_Xb0d"
        # Restricted to "standard" tests location
        # root_tests_dir = '/lustre/naasc/users/scastro/pipeline/cycle5_testing'
        # importasdm_re = 'importasdm\(asdm="{0}/([a-zA-Z0-9_]+)/rawdata/'.format(root_tests_dir)
        # Should work with directory trees using "normal" characters
        importasdm_re = 'importasdm\(asdm="[a-zA-Z0-9_\-/]+/([a-zA-Z0-9_]+)/rawdata/'
        if not first_id_or_importasdm_found and 'importasdm(asdm="' in line:
            iasdm_match = re.search(importasdm_re, line)
            if iasdm_match:
                first_id_or_importasdm_found = True
                mous_dir = iasdm_match.group(1)
                print(' * Found MOUS dir: {0}'.format(mous_dir))

        # example: pipeline::pipeline::casa        Pipeline version 40738 (trunk) running on zuul03
        machine_re = 'running on\s+(\w+)'
        if 'Pipeline version' in line and (
                not 'MPIServer' in line and 'running on' in line):
           machine_match = re.search(machine_re, line)
           if machine_match:
               this_machine = machine_match.group(1)
               if this_machine != run_machine:
                   run_machine = this_machine
                   print(' * Found machine: {0}'.format(run_machine))

        version_re = 'CASA Version\s+([0-9A-Za-z\t ._\-]+)'
        if 'CASA Version' in line and 'MPIServer' not in line:
            version_match = re.search(version_re, line)
            if version_match:
                casa_version = version_match.group(1).strip()                    

        # check mpi
        mpi_server_str = '::casa::MPIServer-'
        if mpi_server_str in line and 'CASA' in line:
            mpi_server_re = 'INFO\s+::casa::MPIServer-(\d+)\s+CASA\s+Version\s+'
            mpi_match = re.search(mpi_server_re, line)
            if mpi_match:
                server_idx = int(mpi_match.group(1))
                if server_idx > mpi_server_cnt:
                    mpi_server_cnt = server_idx


        # Handle begin/end task
        begin_task_re = '#\s+Begin Task:\s+(\w+)\s+#'
        end_task_re = '#\s+End Task:\s+(\w+)\s+#'

        bt_match = re.search(begin_task_re, line)
        no_pipe_tier = False
        if bt_match:
            task_name = bt_match.group(1)
            no_pipe_tier = '::MPIServer-' in line and task_name in pipe_tier_tasks
        et_match = re.search(end_task_re, line)
        if et_match:
            task_name = et_match.group(1)
            no_pipe_tier = '::MPIServer-' in line and task_name in pipe_tier_tasks

        consider_be_line = no_pipe_tier or (not '::MPIServer-' in line)
        if '## Begin Task: ' in line and consider_be_line:
            bt_match = re.search(begin_task_re, line)
            if bt_match and (not inside_task or no_pipe_tier):
                task_name = bt_match.group(1)
                if verbose:
                    sys.stdout.write(" +task: {}".format(task_name))

                #if task_name == 'tclean' and 'MPIServer' in line:
                #    print('******** MPIServer tclean')
                if not task_name in casa_tasks_counter:
                    cti = CASATaskAggLogInfo(task_name)
                    cti._cnt = 1
                else:
                    cti = casa_tasks_counter[task_name]
                    cti._cnt += 1

                casa_tasks_counter.update({task_name: cti})

                inside_task = task_name
                task_seq.append(task_name)
                if 0 == pipe_tier_nestedness:
                    task_start_tstamp[task_name] = tstamp

                if no_pipe_tier:
                    pipe_tier_nestedness += 1

        elif '## End Task: ' in line and consider_be_line:
            et_match = re.search(end_task_re, line)
            if et_match:
                if verbose:
                    sys.stdout.write(" -task: {}".format(et_match.group(1)))

                # example: partition inside importasdm
                if task_name == inside_task:
                    if pipe_tier_nestedness <= 1:
                        pipe_tier_nestedness = 0
                        inside_task = None
                        # if task_name not in task_init_tstamp
                        start_tstamp = task_start_tstamp[task_name]
                        task_runtime = (tstamp - start_tstamp).total_seconds()
                        task_seq_runtimes.append(task_runtime)
                        task_seq_starttstamp.append(start_tstamp)
                        if task_name not in casa_tasks_counter:
                            # Found new task
                            print(' * *** ERROR: this should not happen. Found END block '
                                  'for new task: {0}'.format(task_name))
                            cti = CASATaskAggLogInfo(task_name)
                            cti._cnt = 1
                        else:
                            cti = casa_tasks_counter[task_name]

                        cti.add_runtime_in_this_stage(task_runtime, pipe_stages_current)
                        casa_tasks_counter.update({task_name: cti})

                        if task_name == task_details_task_name:
                            tasks_details_params[-1]._runtime = task_runtime
                            task_details_task_name = None
                    else:
                        pipe_tier_nestedness -= 1


        # To know what's the current 'equivalent CASA call'
        # Looking for example for:
        # 2017-10-05 12:51:50     INFO    hifa_importdata::pipeline.infrastructure.basetask::@cvpost065:MPIClient Equivalent CASA call: hifa_importdata(vis=['uid___A002_Xc3412f_X31e1'], session=['session_1'])
        stage_equiv_call_str = 'Equivalent CASA call:'
        stage_equiv_call_re = stage_equiv_call_str + '\s+([a-zA-Z_]+)\('
        if stage_equiv_call_str in line and 'pipeline.infrastructure.basetask' in line:
            equiv_match = re.search(stage_equiv_call_re, line)
            if equiv_match:
                equiv_call_str = equiv_match.group(1)
                if not version_equal_or_after(casa_version,
                                              *casa_version_equiv_call_change):
                    pipe_stages_current_equiv_call = equiv_call_str
                else:
                    stage_cnt = pipe_stages_counter[pipe_stages_current]
                    stage_cnt._equiv_call = equiv_call_str
                    pipe_stages_counter.update({pipe_stages_current: stage_cnt})


        # ********************** TODO
        # TODO: Before CASA 5.4, the 'Equivalent CASA call' comes before 'Starting execution for stage'. After CASA 5.4 it comes after!
        # if casa_5_4 -> the equiv_match needs to be assigned to the last stage added (pipe_stages_curren.... and below, it should not be taken from pipe_stages_current_equiv_call

        # Handle start/end of pipeline stages
        begin_stage_str = 'Starting execution for stage'
        begin_stage_re = begin_stage_str + '\s+([0-9]+)'
        if begin_stage_str in line:
            bst_match = re.search(begin_stage_re, line)
            if bst_match:
                stage_name = bst_match.group(1)
                # The start message is printed two times...
                if stage_name != pipe_stages_current:
                    if pipe_stages_current:
                        cnt = pipe_stages_counter[pipe_stages_current]
                        cnt.ends(tstamp)
                        pipe_stages_counter.update({pipe_stages_current: cnt})

                    pipe_stages_current = stage_name
                    equiv_call = "not_yet_known"
                    if not version_equal_or_after(casa_version, 
                                                  *casa_version_equiv_call_change):
                        equiv_call = pipe_stages_current_equiv_call
                    new_stage_cnt = PipeStageCounter(stage_name, equiv_call, tstamp)
                    pipe_stages_counter[stage_name] = new_stage_cnt

        if stop_counting_after_pipe_end and\
           'Terminating procedure execution' in line and\
           pipe_stages_current and\
           int(pipe_stages_current) >= stop_counting_minimum_pipe_stage:
            # The 'stage #' > 1 is to not stop after the initial importasdm done by
            # the script calibPipeIF-NA.py and its variants.
            # Remember to end the last stage (there is no 'end stage' messages, only
            # starting stage messages.
            pipe_stages_counter[pipe_stages_current].ends(tstamp)
            print(' * Found end of pipeline procedure: \'{0}\''
                  ' --- Not counting log lines after this.'.format(line.strip()) )
            break


    if verbose:
        print(" - Sequence of tasks: {}".format(task_seq))

    all_casa_tasks_accum = sum([obj._taccum for key, obj in casa_tasks_counter.items()])

    print(" * Counted from first timestamp: {0} to last: {1}".
              format(first_tstamp, last_tstamp))
    elapsed_secs = (last_tstamp - first_tstamp).total_seconds()
    logi = CASALogInfo(os.path.realpath(logf.name), run_machine, casa_version,
                       first_tstamp, last_tstamp, proj_tstamp, mous_dir, first_eb_uid,
                       mpi_server_cnt, casa_tasks_counter, pipe_tasks_counter,
                       pipe_stages_counter, special_task_patterns,
                       tasks_details_params, elapsed_secs,
                       all_casa_tasks_accum)

    return (all_cnt, all_taccum, logi)

def casa_log_file_print_info(all_cnt, all_taccum, log_info):
    pipe_tasks_counter = log_info._pipe_tasks_counter

    def sum_counts_pipe_tasks(pattern):
        """
        Sum line counts of all the pipeline tasks/heuristics/etc. with names containing
        a pattern, for example 'infrastructure.' or 'hifa.'

        :param pattern: pattern to filter pipeline tasks
        """
        cnts = [obj._cnt for key, obj in
                pipe_tasks_counter.items() if pattern in key]
        return sum(cnts)

    pipe_cnt = sum([obj._cnt for key, obj in pipe_tasks_counter.items()])
    pipe_infra_cnt = sum_counts_pipe_tasks('infrastructure.')
    pipe_h_cnt = sum_counts_pipe_tasks('h.')
    pipe_hif_cnt = sum_counts_pipe_tasks('hif.')
    pipe_hifa_cnt = sum_counts_pipe_tasks('hifa.')
    pipe_recipereducer_cnt = sum_counts_pipe_tasks('recipereducer.')

    print(' * Number of mpi servers: {0}'.format(log_info._mpi_servers))
    print(" - Found total lines: {0}, pipeline lines: {1}, pipe.infra lines: {2}, "
          "pipe.h: {3}, pipe.hif: {4}, pipe.hifa: {5}, pipe.recipereducer: {6}".
          format(all_cnt, pipe_cnt, pipe_infra_cnt, pipe_h_cnt, pipe_hif_cnt,
                 pipe_hifa_cnt, pipe_recipereducer_cnt))

    print(" - Tasks call counts: {}".format([[key, obj._cnt] for key, obj in
                                             log_info._casa_tasks_counter.items()]))
    if verbose:
        print(" - Dict of task accumulated runtimes: {}".
              format(log_info._pipe_tasks_counter))
    all_tasks_accum = sum([obj._taccum for key, obj in
                           log_info._casa_tasks_counter.items()])
    print(" - All tasks accumulated runtime: {}".format(all_tasks_accum))
    print(" *** Ranking of tasks taccum:")
    taccum_ranked = print_ranked_pipe_tasks_counter(log_info._casa_tasks_counter)

    # print("Pipe tasks counter: {}", pipe_tasks_counter)
    print('\n')
    print(' *** Pipeline tasks with source identification ({0} different lines):'.
          format(len(pipe_tasks_counter)))
    print(' # line_origin_info, count, time (s)')
    print_ranked_pipe_tasks_counter(pipe_tasks_counter)

    print('\n')
    print(' *** Pipeline, special pseudo-tasks and others')
    print_ranked_pipe_tasks_counter(log_info._special_tasks_counter)

    print('\n')
    print(' *** Pipeline stages (found: {0} stages)'.
          format(len(log_info._pipe_stages_counter)))
    # This would need a better customized print function
    print_ranked_pipe_stages_counter(log_info._pipe_stages_counter)
    sum_stages = sum([obj._taccum for key, obj in
                      log_info._pipe_stages_counter.items()])
    print('Sum of all stage times: {0}\n'.format(sum_stages))

    print("First: {0}, last: {1}".format(log_info._first_tstamp,
                                         log_info._last_tstamp))

    def sum_runtimes_pipe_tasks(pattern):
        """
        Sum times of all the pipeline tasks/heuristics/etc. with names containing a 
        pattern, for example 'infrastructure.' or 'hifa.'

        :param pattern: pattern to filter pipeline tasks
        """
        runtimes = [obj._taccum for key, obj in
                    pipe_tasks_counter.items() if pattern in key]
        return sum(runtimes)

    pipe_taccum = sum([obj._taccum for key, obj in pipe_tasks_counter.items()])
    pipe_infra_taccum = sum_runtimes_pipe_tasks('infrastructure.')
    pipe_h_taccum = sum_runtimes_pipe_tasks('h.')
    pipe_hif_taccum = sum_runtimes_pipe_tasks('hif.')
    pipe_hifa_taccum = sum_runtimes_pipe_tasks('hifa.')
    pipe_recipereducer_taccum = sum_runtimes_pipe_tasks('recipereducer.')
    pipe_qa_taccum = sum_runtimes_pipe_tasks('qa.')
    print('Accum time.  All: {0}, \n\tpipe.: {1}, pipe.infra: {2}, '
          'pipe.h: {3}, pipe.hif: {4}, pipe.hifa: {5}, pipe.recipereducer: {6}, '
          'pipe.qa: {7}'.
          format(all_taccum, pipe_taccum, pipe_infra_taccum, pipe_h_taccum,
                 pipe_hif_taccum, pipe_hifa_taccum, pipe_recipereducer_taccum,
                 pipe_qa_taccum))

    task_imageparams = 'hif.heuristics.imageparams_base'
    try:
        cnt = pipe_tasks_counter[task_imageparams]._taccum
        print("Accum time, pipeline.{0}: {1}".
              format(task_imageparams, cnt))
    except KeyError:
        pass

    total_tdelta = datetime.timedelta(seconds=log_info._total_time)
    elapsed_tbl_format = format_tbl_elapsed(total_tdelta)
    print(" - Total elapsed time: {0} = {1} (in seconds: {2})".
          format(elapsed_tbl_format, total_tdelta,
                 log_info._total_time))
    print(" - Total time in CASA tasks: {0} (in seconds: {1})".
          format(datetime.timedelta(seconds=log_info._total_time_casa_tasks),
                 log_info._total_time_casa_tasks))        
    print(" - Explicit pipeline time: {} (in seconds: {})".
          format(datetime.timedelta(seconds=pipe_taccum), pipe_taccum))

    no_casa = log_info._total_time - log_info._total_time_casa_tasks
    print(' - Total elapsed time - all accumulated CASA tasks time: {0} '
          '(in seconds: {1})'.format(datetime.timedelta(seconds=no_casa), no_casa))

def casa_log_file_dump_to_json(filename, log_info):
    import json

    def json_default(obj):
        import datetime
        if isinstance(obj, datetime.datetime):
            return '{0}'.format(obj)
        elif isinstance (obj, datetime.timedelta):
            return '{0}'.format(obj.total_seconds())
        else:
            return obj.__dict__

    with open(filename, 'w') as jsonf:
        json.dump(log_info, jsonf, indent=2, sort_keys=True, default=json_default)


def casa_log_file_dump_to_pickle(filename, log_info):
    import pickle

    with open(filename, 'wb') as pklf:
        pickle.dump(log_info, pklf, protocol=pickle.HIGHEST_PROTOCOL)

def casa_log_file_dump_info(log_info, json=True, pickle=True):
    """
    For now dump to pickle and json files
    """

    filename = 'casa_logs_perf_' + log_info.build_filename_extended_tags()

    if pickle:
        filename_pkl = filename + '.pickle'
        print(' * Dumping log info object into pickle file: {0}'.format(filename_pkl))
        casa_log_file_dump_to_pickle(filename_pkl, log_info)

    if json:
        filename_json = filename + '.json'
        print(' * Dumping log info object into JSON file: {0}'.format(filename_json))
        casa_log_file_dump_to_json(filename_json, log_info)

    
def parse_casa_log_file_print_info(fname, print_info=True):
    with open(fname) as logf:        
        (all_cnt, all_taccum, log_info) = go_through_log_lines(logf)

        casa_log_file_print_info(all_cnt, all_taccum, log_info)

        casa_log_file_dump_info(log_info, pickle=False)

        return log_info
        
def process_casa_logs(log_fnames, make_plots=False, make_tables=False):
    """
    Gets an info object from a casa log file and passes the information on to
    functions that generate plots, tables, etc. for comparative analysis
    of test runs.
    """

    import time
    time_start = time.time()

    print(' * ===========================================')
    print(' * Log files: {0}'.format(log_fnames))
    print(' * Produce tables? {0}'.format(make_tables))
    print(' * Produce plots? {0}'.format(make_plots))
    print(' * ===========================================')
    print('\n')

    run_infos = []
    for fname in log_fnames:
        print(' * ===========================================')
        print(' * Processing log file: {0}'.format(fname))
        log_info = parse_casa_log_file_print_info(fname, print_info=True)
        run_infos.append(log_info)

    if make_plots:
        plot_timing_things(run_infos)

    if make_tables:
        generate_comparison_table(run_infos)

    time_end = time.time()
    print(" * Processed log(s) in seconds: {0:.3f}".format(time_end - time_start))
    print ('')

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parse a CASA log file and extract '
                                     'information such as version, machine, etc. and '
                                     'especially runtimes for CASA tasks, and pipeline '
                                     'tasks and stages ')
    parser.add_argument('--make-tables', action='store_true')
    parser.add_argument('--make-plots', action='store_true')
    parser.add_argument('log_files', nargs='+', type=str, 
                        help='names of the log files to process')
    args = parser.parse_args()
        
    process_casa_logs(args.log_files, args.make_plots, args.make_tables)


if __name__ == '__main__':
    main()
