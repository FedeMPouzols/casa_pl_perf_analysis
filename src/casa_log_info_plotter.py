#!/usr/bin/env python

import datetime

from casa_logs_mous_props import mous_sizes, mous_short_names, get_asdms_size, ebs_cnt
import casa_logs_mous_props

too_verbose = False

FONTSIZE_LEGEND = 16
FONTSIZE_AXES = 16
FONTSIZE_TITLE = 18

SECS_TO_HOURS = 3600.0
SECS_TO_DAYS = SECS_TO_HOURS*24

# TODO: get rid of this old stuff
# For CASA 5.2, 5.3
LAST_CALIB_STAGE = 22
# For CASA 5.4
#LAST_CALIB_STAGE = 23


class Struct:
    """
    Utility class for when you want to convert a dict to an object with
    an attribute for every entry in the dict.

    Usage: say you load mydict from a json file, then do
    myobj = Struct(**mydict)

    Similar to collections.namedtuple.
    This is non recursive.
    An alternative is the package bunch.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


class obj_hook(object):
    """
    Using this as object_hook parameter to json.load you could use:

    run_dict._casa_tasks_counter.mstransform._cnt
    instead of the dict alternative:
    run_dict['_casa_tasks_counter']['mstransform']['_cnt']
    """
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def load_run_info_json(fname):
    import json

    with open(fname, 'r') as fjs:
        run_dict = json.load(fjs) #, object_hook=obj_hook)

    return run_dict

def load_run_info(run_file):
    return load_run_info_json(run_file)

def find_info_files(subdir):
    import glob
    import os
    glob_pattern = os.path.join(subdir,'*uid__*.json')
    return [ifile for ifile in glob.glob(glob_pattern)]

def format_pl_runtime(time_secs):
    import datetime
    SECS_PER_MIN = float(60.)
    SECS_PER_HOUR = SECS_PER_MIN * 60.
    SECS_PER_DAY = SECS_PER_HOUR * 24.
    days = int(time_secs / SECS_PER_DAY)
    remainder = time_secs
    if days > 0:
        remainder -= days * SECS_PER_DAY

    hours = int(remainder / SECS_PER_HOUR)
    if hours > 0:
        remainder -= hours * SECS_PER_HOUR

    minutes = int(remainder / SECS_PER_MIN)

    res = ''
    if days > 0:
        res += '{0}d '.format(days)
    # res += '{0:.2}h'.format(hours)
    if hours > 0:
        res += ' {0}h '.format(hours)
    if minutes > 0:
        res += ' {0}m '.format(minutes)
    #print('days: {}, hours: {}'.format(days,hours))
    return res.strip()

def make_output_plot_name_base(file_base_prefix, more_prefix, what_plot):
    name_detail = what_plot.replace(' ', '_')
    name_detail = name_detail.replace('/', '_')
    return '{0}_{1}_{2}'.format(file_base_prefix, more_prefix, name_detail)

def gen_mpi_multicolor_plot(serial_x, serial_y, parallel_x_by_mpi, parallel_y_by_mpi,
                            what_plot='Total', time_units='hours', show_plot=False,
                            file_prefix='plot_runtime_colorized_by_mpi_servers_',
                            more_prefix='casa_task', x_axis='time',):
    import matplotlib.pyplot as plt

    print(' * gen_mpi_multicolor_plot')
    fig = plt.figure(figsize=(12,8))
    #fig.savefig()
    if 'time' == x_axis:
        plt.xlabel('{0} run time in serial mode ({1})'.format(what_plot, time_units),
                   fontsize=FONTSIZE_AXES)
    elif 'mous_size' == x_axis:
        plt.xlabel('Size of input MOUS (GB)', fontsize=FONTSIZE_AXES)
    plt.ylabel('{0} run time, serial and parallel ({1})'.format(what_plot, time_units),
               fontsize=FONTSIZE_AXES)
    plt.title('{0} run time, pipeline runs in serial (black) and parallel (colors) mode'.
              format(what_plot), fontsize=FONTSIZE_TITLE)
    if too_verbose:
        print(serial_x)
        print(serial_y)
    #plt.plot(serial_x, serial_y, color='k', marker='o', label='serial')
    plt.scatter(serial_x, serial_y, color="black", marker='^',
                label="serial")

    mpi_color_idx = 0
    mpi_colors = [ 'lightblue', 'blue', 'darkblue', 'blueviolet', 'cyan', 'lightgreen', 'green', 'darkgreen']
    # Colorize by key, where key is the mpi #servers
    for mpi, obj in parallel_x_by_mpi.items():
        plt.scatter(parallel_x_by_mpi[mpi], parallel_y_by_mpi[mpi],
                    color=mpi_colors[mpi_color_idx], marker='o', 
                    label="parallel ({0})".format(mpi))
        mpi_color_idx += 1

    # x=y dashed line
    plt.plot([0, 1000], [0, 1000], color='black', marker='', linestyle=':', label='')
    x_tol = (max(serial_x)-min(serial_x)) * 0.02
    y_tol = (max(serial_y)-min(serial_y)) * 0.02
    plt.xlim(0, max(serial_x) + x_tol)
    plt.ylim(0, max(max(parallel_y),max(serial_y)) + y_tol)
    plt.grid('on')
    leg = plt.legend(loc='upper left', prop={'size': FONTSIZE_LEGEND})
    leg.get_frame().set_edgecolor('k')

    name_base = make_output_plot_name_base(file_base_prefix, more_prefix, what_plot)
    fig.savefig('{0}.png'.format(name_base))
    if show_plot:
        plt.show()

    plt.close()
        
def gen_dual_plot(serial_x, serial_y, parallel_x, parallel_y,
                  what_plot='Total', time_units='hours', show_plot=False,
                  file_base_prefix='plot_runtime_serial_vs_parallel_',
                  more_prefix='casa_task', x_axis='time',
                  title=None, ylabel=None, xlabel=None, legend_loc=None):
    # TODO: this is getting too similar to the colorized plot
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(12,8))
    #fig.savefig()
    if not xlabel:
        if 'time' == x_axis:
            xlabel = '{0} run time in serial mode ({1})'.format(what_plot, time_units)
        elif 'mous_size' == x_axis:
            xlabel = 'Size of input MOUS (GB)'
    plt.xlabel(xlabel, fontsize=FONTSIZE_AXES)

    if not ylabel:
        ylabel = '{0} run time, serial and parallel ({1})'.format(what_plot, time_units)
    plt.ylabel(ylabel, fontsize=FONTSIZE_AXES)
    if not title:
        title = '{0} run time, pipeline runs, serial (black) and parallel (colors) mode'.format(what_plot)
    plt.title(title,fontsize=FONTSIZE_TITLE)

    if too_verbose:
        print(serial_x)
        print(serial_y)

    if 0 == len(parallel_x) or 0 == len(parallel_y):
        print(" *** ERROR, not producing this plot: len parallel_x: {0}, len parallel_y: {1}"
              ", what_plot: {2}, file_base_prefix: {3}, more_prefix: {4}".
              format(len(parallel_x), len(parallel_y), what_plot, file_base_prefix,
                     more_prefix))
        return

    #plt.plot(serial_x, serial_y, color='k', marker='o', label='serial')
    plt.scatter(serial_x, serial_y, color="black", marker='s',
                label="serial")
    plt.scatter(parallel_x, parallel_y, color="blue", marker='^',
                label="parallel")
    x_tol = (max(serial_x)-min(serial_x)) * 0.02
    y_tol = (max(serial_y)-min(serial_y)) * 0.02
    # x=y dashed line
    plt.plot([0, 1000], [0, 1000], color='black', marker='', linestyle=':', label='')
    plt.xlim(0, max(serial_x) + x_tol)
    plt.ylim(0, max(max(parallel_y),max(serial_y)) + y_tol)
    plt.yticks(fontsize=FONTSIZE_AXES)
    plt.xticks(fontsize=FONTSIZE_AXES)
    plt.grid('on')
    if not legend_loc:
        legend_loc = 'upper left'
    leg = plt.legend(loc=legend_loc, prop={'size': FONTSIZE_LEGEND}) # 'best' 'upper right'
    leg.get_frame().set_edgecolor('k')

    name_base = make_output_plot_name_base(file_base_prefix, more_prefix, what_plot)
    fig.savefig('{0}.png'.format(name_base))
    if show_plot:
        plt.show()

    plt.close()

def gen_plot_data_as_csv(serial_x, serial_y, parallel_x, parallel_y,
                         mpi_servers, point_tags_serial, point_tags_parallel, what_plot,
                         more_prefix, mous_sizes, x_axis='time', ylabel=None):
    """
    """
    import csv

    filename = '{0}.csv'.format(
        make_output_plot_name_base('plot_data_runtime_serial_vs_parallel_',
                                   more_prefix, what_plot))

    if 'time' == x_axis:
        second_col = 'x (serial time)'
    elif 'mous_size' == x_axis:
        second_col = 'MOUS size'

    if not ylabel:
        ylabel = 'y (serial or parallel time)'
    with open(filename, 'wb') as csvf:
        csvf.write('# mpi_servers, {0}, {1}, '
                   'MOUS, input_MOUS_size\n'.format(second_col, ylabel))

        writer = csv.writer(csvf, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)

        for idx, val in enumerate(serial_x):
            size = mous_sizes[point_tags_serial[idx]]
            writer.writerow([0, serial_x[idx], serial_y[idx], point_tags_serial[idx], size])

        for idx, val in enumerate(parallel_x):
            size = mous_sizes[point_tags_parallel[idx]]
            writer.writerow([mpi_servers[idx], parallel_x[idx], parallel_y[idx],
                             point_tags_parallel[idx], size])

def do_serial_parallel_plot_all_tasks(serial_infos, parallel_infos):
    print(' * Producing plots for runtimes of CASA tasks:')
    # Take all tasks from one of the runs
    # first_obj = serial_infos.itervalues().next()
    first_obj = parallel_infos.itervalues().next()
    task_names = []
    for key, obj in first_obj['_casa_tasks_counter'].items():
        print('Found task: {0}'.format(key))
        task_names.append(key)

    # examples:
    #metric_name = '_total_time_casa_tasks'
    #metric_name = 'flagdata'
    #metric_name = 'mstransform'
    #metric_name = 'applycal'
    #metric_name = 'tclean'
    for metric_name in task_names:
        do_serial_parallel_plot(metric_name, serial_infos, parallel_infos)

def do_serial_parallel_plot_totals(serial_infos, parallel_infos):
    print(' * Producing plots for total runtimes')
    first_obj = serial_infos.itervalues().next()
    task_names = []
    for key, obj in first_obj['_casa_tasks_counter'].items():
        task_names.append(key)

    do_serial_parallel_plot('_total_time', serial_infos, parallel_infos,
                            counter_type='totals')
    do_serial_parallel_plot('_total_time_casa_tasks', serial_infos, parallel_infos,
                            counter_type='totals')

def do_serial_parallel_plot_pipe_stages(serial_infos, parallel_infos):
    print(' * Producing plots for runtimes of pipeline stages:')
    first_obj = serial_infos.itervalues().next()
    stages_names = []
    for key, obj in first_obj['_pipe_stages_counter'].items():
        print('Found pipeline Stage: {0} - {1}'.format(key,
                                                       obj['_equiv_call']))
        stages_names.append(key)

    for metric_name in stages_names:
        do_serial_parallel_plot(metric_name, serial_infos, parallel_infos,
                                counter_type='pipe_stage')

    # TODO TODO TODO
    for stg in stages_names:
        time_stage = lambda info: info['_pipe_stages_counter'][stg]['_taccum']
        # for key, task in first_obj['_casa_tasks_counter'].items():
        #    print "Task key: ",key
        #    print "_taccum_pipe_stages:",task['_taccum_pipe_stages']
        time_tasks_in_stage = lambda info: sum([task_obj['_taccum_pipe_stages'].get(stg, 0) for key, task_obj in info['_casa_tasks_counter'].items()])
        metric_time_stage_outside_tasks = lambda info: time_stage(info) - time_tasks_in_stage(info)
        equiv_call = first_obj['_pipe_stages_counter'][stg]['_equiv_call']
        xlabel = 'Pipeline stage {0}-{1} (-tasks)run time in serial mode'.format(stg, equiv_call)
        title = 'Stage {0}-{1} (-tasks) run time, serial (black), parallel (blue) mode'.format(stg, equiv_call)
        filename = ('plot_runtime_serial_vs_parallel___Pipeline_wo_tasks_time_stage_{0}-{1}.png'.
                    format(stg, equiv_call))
        TODO_NEXT_serial_parallel_plot_stages(serial_infos, parallel_infos,
                                              metric_time_stage_outside_tasks,
                                              filename=filename,
                                              xlabel=xlabel,
                                              ylabel='Pipeline run time (hours)',
                                              title=title)

    for special_task in ['tclean']:
        stg = '33'
        filename = ('plot_runtime_serial_vs_parallel__casa_task_inside_pipe_stage_{0}_{1}'.
                    format(stg, special_task))
        xlabel = ('{0} (inside stage {1}) runtime in serial mode '.
                  format(special_task, stg))
        xlabel = '{0} run time in serial and parallel '.format(special_task)
        title = '{0} in pipeline stage {1}'.format(special_task, stg)
        metric_tclean_stg = lambda x: x['_casa_tasks_counter'][special_task]['_taccum_pipe_stages'][stg]
        TODO_NEXT_serial_parallel_plot_stages(serial_infos, parallel_infos,
                                              metric_tclean_stg,
                                              filename=filename,
                                              xlabel=xlabel,
                                              ylabel='Pipeline run time (hours)',
                                              title=title)
        

def do_serial_parallel_plot_pipe_tasks_functions(serial_infos, parallel_infos):
    print(' * Producing plots for runtimes of pipeline functions (tasks, heuristics, etc.)')
    
    functions_on = ['hif.heuristics.imageparams_base', 'infrastructure.displays.sky',
                    'qa.scorecalculator', 'hif.tasks.tclean.tclean',
                    'infrastructure.tablereader', 'hif.tasks.tclean.tclean',
                    'h.tasks.exportdata.exportdata', 'infrastructure.basetask',
                    'hif.heuristics.cleanbox', 'hif.tasks.tclean.renderer',
                    'infrastructure.basetask', 'infrastructure.jobrequest']

    print('* Selecting the following pipeline tasks/heuristics/etc. for plotting: {0}'.
          format(functions_on))
    function_names = []
    first_obj = serial_infos.itervalues().next()
    for key, obj in first_obj['_pipe_tasks_counter'].items():
        if key not in functions_on:
            continue

        print('Found pipeline task/function: {0} - {1}'.format(key,
                                                               obj['_name']))
        function_names.append(key)

    for metric_name in function_names:
        do_serial_parallel_plot(metric_name, serial_infos, parallel_infos,
                                counter_type='pipe_task')
        
def do_serial_parallel_plot(metric_name, serial_infos, parallel_infos,
                            counter_type='task', 
                            plot_type='dual',
                            produce_csv=True):
    """
    Plots that use serial run time as x axis

    :param counter_type: 'task', 'totals', 'pipe_stage', 'pipe_task'
    """
    print(' ** Producing serial/parallel plot. Type of counter: {0}. Metric: {1}'.
          format(counter_type, metric_name))
    serial_x = []
    serial_y = []

    parallel_x = []
    parallel_y = []

    point_tags_serial = []
    point_tags_parallel = []
    mpi_servers = []

    time_div = SECS_TO_HOURS
    
    serial_by_mous = {}
    parallel_x_by_mpi = {}
    parallel_y_by_mpi = {}
    for key, obj in serial_infos.items():
        mous = obj['_mous']
        serial_by_mous[mous] = obj

        # metric_val = float(obj[metric_name])
        try:
            if counter_type == 'task':
                metric_val = float(obj['_casa_tasks_counter'][metric_name]['_taccum'])
            elif 'totals' == counter_type:
                metric_val = float(obj[metric_name])
            elif 'pipe_stage' == counter_type:
                metric_val = float(obj['_pipe_stages_counter'][metric_name]['_taccum'])
            elif 'pipe_task' == counter_type:
                metric_val = float(obj['_pipe_tasks_counter'][metric_name]['_taccum'])
        except KeyError as exc:
            print(' * Error, could not find this key (type: {0}) in serial run: {1}.'
                  ' For MOUS: {2}. NOT? generating plot. Error details: {3}'.
                  format(counter_type, metric_name, mous, exc))
            # there is one example where imstat is not used in the pipeline (from findcont):
            # serial mode, # E2E5.1.00028.S_2017_10_06T12_40_20.706
            # So let's not leave that plot out, even if some points are missing.
            # return
            continue

        point_tags_serial.append(mous)
        serial_x.append(metric_val / time_div)
        serial_y.append(metric_val / time_div)

    for key, obj in parallel_infos.items():
        # metric_val = obj[metric_name]
        mous = obj['_mous']
        try:
            if counter_type == 'task':
                metric_val = obj['_casa_tasks_counter'][metric_name]['_taccum']
            elif 'totals' == counter_type:
                metric_val = obj[metric_name]
            elif 'pipe_stage' == counter_type:
                metric_val = obj['_pipe_stages_counter'][metric_name]['_taccum']
            elif 'pipe_task' == counter_type:
                metric_val = obj['_pipe_tasks_counter'][metric_name]['_taccum']
        except KeyError as exc:
            print(' * Error, could not find this key (type: {0}) in parallel run: {1}.'
                  ' For MOUS: {2}. NOT? generating plot. Error details: {3}'.
                  format(counter_type, metric_name, mous, exc))
            # there is one example where imstat is not used in the pipeline (from findcont):
            # serial mode, # E2E5.1.00028.S_2017_10_06T12_40_20.706
            # So let's not leave that plot out, even if some points are missing.
            # return
            continue

        try:
            # serial_metric_val = serial_by_mous[mous][metric_name]
            if counter_type == 'task':
                serial_metric_val = serial_by_mous[mous]['_casa_tasks_counter'][metric_name]['_taccum']
            elif 'totals' == counter_type:
                serial_metric_val = serial_by_mous[mous][metric_name]
            elif 'pipe_stage' == counter_type:
                serial_metric_val = serial_by_mous[mous]['_pipe_stages_counter'][metric_name]['_taccum']
            elif 'pipe_task' == counter_type:
                serial_metric_val = serial_by_mous[mous]['_pipe_tasks_counter'][metric_name]['_taccum']
        except KeyError as exc:
            # For example for 2015.1.01163.S_2017_10_13T20_24_02.441
            # - with errors in pipeline
            # imstat is used in parallel but not in serial
            print(' * Error, could not find this key (type: {0}) in serial run, even if it '
                  'is used in the parallel run: {1}.'
                  ' For MOUS: {2}. NOT? generating plot. Error details: {3}'.
                  format(counter_type, metric_name, mous, exc))

        point_tags_parallel.append(mous)
        par_val_y = float(metric_val) / time_div
        parallel_y.append(par_val_y)

        par_val_x = float(serial_metric_val) / time_div
        parallel_x.append(par_val_x)

        mpi = obj['_mpi_servers']
        mpi_servers.append(mpi)
        if mpi not in parallel_x_by_mpi:
            parallel_y_by_mpi[mpi] = [ par_val_y ]
            parallel_x_by_mpi[mpi] = [ par_val_x ]
        else:
            parallel_y_by_mpi[mpi].append(par_val_y)
            parallel_x_by_mpi[mpi].append(par_val_x)

    if 'task' == counter_type:
        more_prefix = 'casa_task'
        what_plot = '{0}'.format(metric_name)
    elif 'totals' == counter_type:
        more_prefix = 'totals'
        if '_total_time' == metric_name:
            what_plot = 'Total pipeline'
        elif '_total_time_casa_tasks' == metric_name:
            what_plot = 'CASA tasks total'
    elif 'pipe_stage' == counter_type:
        more_prefix = '' # to avoid repetition with the 'what_plot' below
        what_plot = 'Pipeline stage {0}-{1}'.format(metric_name,
                                                    obj['_pipe_stages_counter'][metric_name]['_equiv_call'])
    elif 'pipe_task' == counter_type:
        more_prefix = '' # to avoid repetition with the 'what_plot' below
        what_plot = 'Pipe task/etc {0}'.format(metric_name)
                                                                  

    gen_plot_data_as_csv(serial_x, serial_y, parallel_x, parallel_y,
                         mpi_servers, point_tags_serial, point_tags_parallel,
                         what_plot=what_plot,
                         more_prefix=more_prefix, mous_sizes=mous_sizes)
        
    if 'dual' == plot_type:
        gen_dual_plot(serial_x, serial_y, parallel_x, parallel_y,
                      what_plot=what_plot, more_prefix=more_prefix)
    else:
        gen_mpi_multicolor_plot(serial_x, serial_y, parallel_x_by_mpi, parallel_y_by_mpi,
                                what_plot=what_plot, more_prefix=more_prefix)

# TODO: enable 'produce_csv' - call gen_plot_data_as_csv
# TODO: this will be for tasks, etc. also - any lambda
"""
Prepare data and produce a plot and data (csv) file.

This is for plots of a cloud of points (tests), some of them
corresponding to serial mode runs, and some others for parallel
mode runs.

:param metric_lambda: function to plot
"""
def TODO_NEXT_serial_parallel_plot_stages(serial_infos, parallel_infos,
                                          metric_lambda,
                                          filename=None,
                                          xlabel=None,
                                          ylabel=None,
                                          title=None,
                                          produce_csv=True):
    import numpy as np
    import matplotlib.pyplot as plt

    print(' ** Producing new serial/parallel plot. stages w/ and w/o task time')
        
    serial_x = []
    serial_y = []

    parallel_x = []
    parallel_y = []

    point_tags_serial = []
    point_tags_parallel = []
    mpi_servers = []

    time_div = SECS_TO_HOURS

    serial_by_mous = {}
    parallel_x_by_mpi = {}
    parallel_y_by_mpi = {}
    for key, obj in serial_infos.items():
        mous = obj['_mous']
        serial_by_mous[mous] = obj
        metric_val = metric_lambda(obj)
        point_tags_serial.append(mous)
        serial_x.append(metric_val / time_div)
        serial_y.append(metric_val / time_div)

    for key, obj in parallel_infos.items():
        # metric_val = obj[metric_name]
        mous = obj['_mous']
        metric_val = metric_lambda(obj)
        serial_metric_val = metric_lambda(serial_by_mous[mous])

        point_tags_parallel.append(mous)
        par_val_y = float(metric_val) / time_div
        parallel_y.append(par_val_y)
        par_val_x = float(serial_metric_val) / time_div
        parallel_x.append(par_val_x)

        mpi = obj['_mpi_servers']
        mpi_servers.append(mpi)
        if mpi not in parallel_x_by_mpi:
            parallel_y_by_mpi[mpi] = [ par_val_y ]
            parallel_x_by_mpi[mpi] = [ par_val_x ]
        else:
            parallel_y_by_mpi[mpi].append(par_val_y)
            parallel_x_by_mpi[mpi].append(par_val_x)


    # This is like gen_dual_plot - no support for multi-mpi
    # make the serial_x, serial_y, parallel_x, parallel_y
    fig = plt.figure(figsize=(12,8))
    plt.scatter(serial_x, serial_y, color="black", marker='s',
                label="serial")
    plt.scatter(parallel_x, parallel_y, color="blue", marker='^',
                label="parallel")

    plt.xlabel(xlabel, fontsize=FONTSIZE_TITLE)
    ylabel = 'Serial and parallel runtime (hours)'
    plt.ylabel(ylabel, fontsize=FONTSIZE_TITLE)
    plt.plot([0, 1000], [0, 1000], color='black', marker='', linestyle=':', label='')
    x_tol = (max(serial_x)-min(serial_x)) * 0.05
    y_tol = (max(serial_y)-min(serial_y)) * 0.05
    plt.xlim(0, max(serial_x))
    plt.ylim(0, max(max(parallel_y),max(serial_y)))
    plt.grid('on')
    legend_loc = 'upper left'
    leg = plt.legend(loc=legend_loc, prop={'size': FONTSIZE_LEGEND}) # 'best' 'upper right'
    if leg:
        leg.get_frame().set_edgecolor('k')
    plt.title(title,fontsize=FONTSIZE_TITLE)
    fig.savefig(filename)
    plt.close()

        
def do_calib_imaging_time_plots(serial_infos, parallel_infos, plot_type='dual'):
    """
    This produces separate plots for the calibration and imaging
    phases of the pipeline.

    Exploits the fact that calibration happens within a known range of stages. At
    the time this doc was written, calibration goes from stage 1 up to 22 (included)
    in CASA 5.2. In CASA 5.4 the hifa_exportdata stage is 23.
    After that, the next stages are considered as imaging stuff.
    """
    do_pipe_stages_ranges_plot(serial_infos, parallel_infos,
                              [(1,LAST_CALIB_STAGE),
                               (LAST_CALIB_STAGE+1, 36)], ['Calibration', 'Imaging'],
                              plot_type)

    
def do_pipe_stages_ranges_plot(serial_infos, parallel_infos,
                               stages_min_max, range_names,
                               plot_type='dual'):
    """
    Do plots for specific ranges of pipeline stages, for two example useful ranges are:
    (1, 22) - calibration (pipeline used for cycle 5, at the time of CASA 5.2/5.3)
    (23, 36) - imaging (pipeline used for cycle 5, at the time of CASA 5.2/5.3)
    """
    for idx, range_name in enumerate(range_names):
        # with x-axis time 
        do_pipe_stages_one_range_plot(serial_infos, parallel_infos,
                                      stages_min_max[idx], range_names[idx], plot_type)
        # with x-axis mous size 
        do_pipe_stages_one_range_plot(serial_infos, parallel_infos,
                                      stages_min_max[idx], range_names[idx],
                                      plot_type, x_axis='mous_size',
)

def do_pipe_stages_one_range_plot(serial_infos, parallel_infos,
                                  stages_min_max, range_name,
                                  plot_type='dual',
                                  x_axis='time'):

    serial_x = []
    serial_y = []

    parallel_x = []
    parallel_y = []

    time_div = SECS_TO_HOURS

    point_tags_serial = []
    point_tags_parallel = []
    mpi_servers = []
    
    serial_by_mous = {}
    parallel_x_by_mpi = {}
    parallel_y_by_mpi = {}
    for key, obj in serial_infos.items():
        mous = obj['_mous']
        serial_by_mous[mous] = obj

        # metric_val = float(obj[metric_name])
        try:
            metric_val = 0
            for key, counter in obj['_pipe_stages_counter'].items():
                if key >= str(stages_min_max[0]) and key <= str(stages_min_max[1]):
                    metric_val += float(counter['_taccum'])
        except KeyError as exc:
            print(' * Error 1, could not find this key in serial run: {0}.'
                  ' For MOUS: {1}. NOT generating plot. Error details: {2}'.
                  format(key, mous, exc))
            # there is one example where imstat is not used in the pipeline (from findcont):
            # serial mode, # E2E5.1.00028.S_2017_10_06T12_40_20.706
            # So let's not leave that plot out, even if some points are missing.
            # return
            continue

        point_tags_serial.append(mous)
        if 'time' == x_axis:
            serial_x.append(metric_val / time_div)
        elif 'mous_size' == x_axis:
            serial_x.append(mous_sizes[mous])
        serial_y.append(metric_val / time_div)

    for key, obj in parallel_infos.items():
        # metric_val = obj[metric_name]
        mous = obj['_mous']
        try:
            metric_val = 0
            for key, counter in obj['_pipe_stages_counter'].items():
                if key >= str(stages_min_max[0]) and key <= str(stages_min_max[1]):
                    metric_val += float(counter['_taccum'])
        except KeyError as exc:
            print(' * Error 2, could not find this key in parallel run: {0}.'
                  ' For MOUS: {1}. NOT? generating plot. Error details: {2}'.
                  format(key, mous, exc))
            # return
            continue

        par_val_y = float(metric_val) / time_div
        parallel_y.append(par_val_y)

        # This counts a second time - should be refactored
        serial_metric_val = 0
        metric_val = 0
        for key, counter in obj['_pipe_stages_counter'].items():
            if key >= str(stages_min_max[0]) and key <= str(stages_min_max[1]):
                serial_metric_val += float(serial_by_mous[mous]['_pipe_stages_counter'][key]['_taccum'])

        point_tags_parallel.append(mous)
        if 'time' == x_axis:
            par_val_x = float(serial_metric_val) / time_div
        elif 'mous_size' == x_axis:
            par_val_x = mous_sizes[mous]
        parallel_x.append(par_val_x)

        mpi = obj['_mpi_servers']
        mpi_servers.append(mpi)
        if mpi not in parallel_x_by_mpi:
            parallel_y_by_mpi[mpi] = [ par_val_y ]
            parallel_x_by_mpi[mpi] = [ par_val_x ]
        else:
            parallel_y_by_mpi[mpi].append(par_val_y)
            parallel_x_by_mpi[mpi].append(par_val_x)

    if 'time' == x_axis:
        more_prefix = 'section_'.format(range_name)
        what_plot = '{0} pipeline'.format(range_name)
    elif 'mous_size' == x_axis:
        file_base_prefix='plot_runtime_serial_vs_parallel_',
        more_prefix = 'section_by_MOUS_size_'.format(range_name)
        what_plot = '{0} pipeline'.format(range_name)

    gen_plot_data_as_csv(serial_x, serial_y, parallel_x, parallel_y,
                         mpi_servers, point_tags_serial, point_tags_parallel,
                         what_plot=what_plot,
                         more_prefix=more_prefix, x_axis=x_axis, mous_sizes=mous_sizes)
        
    if 'dual' == plot_type:
        gen_dual_plot(serial_x, serial_y, parallel_x, parallel_y,
                      what_plot=what_plot, more_prefix=more_prefix, x_axis=x_axis)
    else:
        gen_mpi_multicolor_plot(serial_x, serial_y, parallel_x_by_mpi, parallel_y_by_mpi,
                                what_plot=what_plot, more_prefix=more_prefix, x_axis=x_axis)

def _TODO_reshape_this_do_other_time_plot(metric_name, serial_infos, parallel_infos,
                                          counter_type='task', 
                                          plot_type='dual'):
    """
    This is specific for the 'Total' Plots that use serial run time as x axis
    """
    serial_x = []
    serial_y = []

    parallel_x = []
    parallel_y = []

    time_div = SECS_TO_HOURS
    
    serial_by_mous = {}
    parallel_x_by_mpi = {}
    parallel_y_by_mpi = {}
    for key, obj in serial_infos.items():
        mous = obj['_mous']
        serial_by_mous[mous] = obj

        # metric_val = float(obj[metric_name])
        try:
            if counter_type == 'task':
                metric_val = float(obj['_casa_tasks_counter'][metric_name]['_taccum'])
            elif 'totals' == counter_type:
                metric_val = float(obj[metric_name])
        except KeyError as exc:
            print(' * Error, could not find this key (type: {0}) in serial run: {1}.'
                  ' For MOUS: {2}. NOT using this run info for plots. Error details: {3}'.
                  format(counter_type, metric_name, mous, exc))
            return

        serial_x.append(metric_val / time_div)
        serial_y.append(metric_val / time_div)

    for key, obj in parallel_infos.items():
        # metric_val = obj[metric_name]
        mous = obj['_mous']
        try:
            if counter_type == 'task':
                metric_val = obj['_casa_tasks_counter'][metric_name]['_taccum']
            elif 'totals' == counter_type:
                metric_val = obj[metric_name]
        except KeyError as exc:
            print(' * Error, could not find this key (type: {0}) in parallel run: {1}.'
                  ' For MOUS: {2}. NOT using this run info for plots. Error details: {3}'.
                  format(counter_type, metric_name, mous, exc))
            return

        par_val_y = float(metric_val) / time_div
        parallel_y.append(par_val_y)

        # serial_metric_val = serial_by_mous[mous][metric_name]
        if counter_type == 'task':
            serial_metric_val = serial_by_mous[mous]['_casa_tasks_counter'][metric_name]['_taccum']
        elif 'totals' == counter_type:
            serial_metric_val = serial_by_mous[mous][metric_name]

        par_val_x = float(serial_metric_val) / time_div
        parallel_x.append(par_val_x)

        mpi = obj['_mpi_servers']
        if mpi not in parallel_x_by_mpi:
            parallel_y_by_mpi[mpi] = [ par_val_y ]
            parallel_x_by_mpi[mpi] = [ par_val_x ]
        else:
            parallel_y_by_mpi[mpi].append(par_val_y)
            parallel_x_by_mpi[mpi].append(par_val_x)

    if 'task' == counter_type:
        more_prefix = 'casa_task'
        what_plot = '{0}'.format(metric_name)
    elif 'totals' == counter_type:
        more_prefix = 'totals'
        if '_total_time' == metric_name:
            what_plot = 'Total pipeline'
        elif '_total_time_casa_tasks' == metric_name:
            what_plot = 'CASA tasks total'
            
    if 'dual' == plot_type:
        gen_dual_plot(serial_x, serial_y, parallel_x, parallel_y,
                      what_plot=what_plot, more_prefix=more_prefix)
    else:
        gen_mpi_multicolor_plot(serial_x, serial_y, parallel_x_by_mpi, parallel_y_by_mpi,
                                what_plot=what_plot, more_prefix=more_prefix)

def parse_info_files(run_info_files, multi_as_dict=False):
    """
    :return: two dictionaries. One for serial runs, and a second one for parallel
    runs, both with MOUS uid_... codes as keys.
    """
    serial_infos = {}
    parallel_infos = {}
    multi_parallel_infos = {}
    for run_file in run_info_files:
        print(' Digesting file: {0}'.format(run_file))
        run_info = load_run_info(run_file)
        #run_infos[run_file] = run_dict

        # print run_info['_casa_tasks_counter']['tclean']['_cnt']
        # # looking for run_info['_casa_tasks_counter']['mstransform']['_taccum']))
        # for key, obj in run_info['_casa_tasks_counter'].items():
        #     print('{0}: {1}'.
        #           format(key, obj['_taccum']))

        mpi = int(run_info['_mpi_servers'])
        mous = run_info['_mous']
        if 0 == mpi:
            serial_infos[mous] = run_info
        else:
            parallel_infos[mous] = run_info

        # TODO: settle on this
        if multi_as_dict:
            if mous not in multi_parallel_infos:
                multi_parallel_infos[mous] = {mpi: run_info}
            else:
                multi_parallel_infos[mous][mpi] = run_info
        else:
            if mous in multi_parallel_infos:
                multi_parallel_infos[mous].append(run_info)
            else:
                multi_parallel_infos[mous] = [run_info]

    return serial_infos, parallel_infos, multi_parallel_infos


def gen_tclean_plot(serial_x, serial_y, parallel_x, parallel_y,
                  what_plot='Total', time_units='hours', show_plot=False,
                  file_base_prefix='plot_runtime_serial_vs_parallel_',
                  more_prefix='casa_task', x_axis='time'):
    # TODO: this is getting too similar to the colorized plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,8))
    #fig.savefig()
    if 'time' == x_axis:
        plt.xlabel('{0} run time in serial mode ({1})'.format(what_plot, time_units),
                   fontsize=FONTSIZE_AXES)
    elif 'cube_size' == x_axis:
        plt.xlabel('Size of cube (nchan * imsize[0] * imsize[1]) (pixels)',
                   fontsize=FONTSIZE_AXES)
    plt.ylabel('{0} run time, serial and parallel ({1})'.format(what_plot, time_units),
               fontsize=FONTSIZE_AXES)
    plt.title('{0} run time, pipeline runs, serial (black) and parallel (colors) mode'.
              format(what_plot),
              fontsize=FONTSIZE_TITLE)

    if too_verbose:
        print(serial_x)
        print(serial_y)
    #plt.plot(serial_x, serial_y, color='k', marker='o', label='serial')
    plt.scatter(serial_x, serial_y, color="black", marker='s',
                label="serial")
    plt.scatter(parallel_x, parallel_y, color="blue", marker='^',
                label="parallel")
    # x=y dashed line
    plt.plot([0, 1000], [0, 1000], color='black', marker='', linestyle=':', label='')
    x_tol = (max(serial_x)-min(serial_x)) * 0.02
    y_tol = (max(serial_y)-min(serial_y)) * 0.02
    plt.xlim(0, max(serial_x) + x_tol)
    plt.ylim(0, max(max(parallel_y),max(serial_y)) + y_tol)
    plt.yticks(fontsize=FONTSIZE_AXES)
    plt.xticks(fontsize=FONTSIZE_AXES)
    plt.grid('on')
    leg = plt.legend(loc='upper left', prop={'size': FONTSIZE_LEGEND}) # 'best' 'upper right'
    leg.get_frame().set_edgecolor('k')
    
    name_base = make_output_plot_name_base(file_base_prefix, more_prefix, what_plot)
    fig.savefig('{0}.png'.format(name_base))
    if show_plot:
        plt.show()
    plt.close()

# TODO: replace the 'continue' statements with:
# skip_lambda = lambda tcall: tcall['_params']['gridder'] or tcall['_params']['deconvolver'] or float(task_call['_params']['niter']) > 0
# (also for example: '29' != tcall['_pipe_stage_seqno']
def do_tclean_experimental_plots(serial_infos, parallel_infos, x_axis='time'):
    """
    TODO: very WIP
    """
    import ast
    group = {}
    serial_x = []
    serial_y = []
    parallel_x = []
    parallel_y = []
    serial_tclean_by_imagename = {}
    time_div = SECS_TO_HOURS

    interest_par_names = ['specmode', 'gridder', 'deconvolver', 'imsize', 'nchan',
                          'niter', 'threshold', 'imagename']
    interest_pars = {}
    for name in interest_par_names:
        interest_pars[name] = []
    point_tags_serial = []
    point_tags_parallel = []


    no_mosaics = lambda task_call: 'mosaic' == task_call['_params']['gridder']
    no_mtmfs = lambda task_call: 'mosaic' == task_call['_params']['deconvolver']
    filter_expr = [no_mosaics, no_mtmfs]
    
    mpi_servers = []
    pipe_stages = []
    for key, run_info in serial_infos.items():
        print('Going for serial {0}'.format(run_info['_mous']))
        for task_call in run_info['_tasks_details_params']:
            #if 'cube' != task_call['_params']['specmode']:
            #    continue

            if 'mosaic' == task_call['_params']['gridder']:
                continue
            if 'mtmfs' == task_call['_params']['deconvolver']:
                continue


            # if '29' != task_call['_pipe_stage_seqno']:
            #     continue
            # print('ns')
            # if 'True' == task_call['_params']['restoration']:
            #     continue

            # if 'cube_size' == x_axis and float(task_call['_params']['nchan']) < 0:
            #     continue
            # if float(task_call['_params']['nchan']) > 0:
            #     continue


            if float(task_call['_params']['niter']) > 0:
                continue

            
            if 'tclean' == task_call['_name'] and float(task_call['_runtime']) > 0:
                if 'True' == task_call['_params']['parallel']:
                    print('Detail: {0}'.format(task_call))
                else:
                    print('***: {0}'.format(task_call['_params']['parallel']))
                    #continue

                val_y = float(task_call['_runtime'])
                val_y /= time_div
                if 'time' == x_axis:
                    val_x = val_y
                elif 'cube_size' == x_axis:
                    nchan = float(task_call['_params']['nchan'])
                    imsize = ast.literal_eval(task_call['_params']['imsize'])
                    print(' * imsize: {0}'.format(imsize))
                    val_x = nchan * imsize[0] * imsize[1]
                    # x MOUS size
                    # val_x /= mous_sizes[run_info['_mous']]
                    # val_x *= float(task_call['_params']['niter'])

                serial_tclean_by_imagename[task_call['_params']['imagename']] = val_x
                serial_x.append(val_x)
                serial_y.append(val_y)

                mous = run_info['_mous']                
                point_tags_serial.append(mous)
                pipe_stages.append(task_call['_pipe_stage_seqno'])
                for name in interest_par_names:
                    interest_pars[name].append(task_call['_params'][name])


    for key, run_info in parallel_infos.items():
        print('Going for par {0}'.format(run_info['_mous']))
        for task_call in run_info['_tasks_details_params']:
            #if 'cube' != task_call['_params']['specmode']:
            #    continue

            if 'mosaic' == task_call['_params']['gridder']:
                continue
            if 'mtmfs' == task_call['_params']['deconvolver']:
                continue

            # if '29' != task_call['_pipe_stage_seqno']:
            #     continue
            # print('ns')
            # if 'True' == task_call['_params']['restoration']:
            #     continue

            # if 'cube_size' == x_axis and float(task_call['_params']['nchan']) < 0:
            #     continue

            if float(task_call['_params']['niter']) > 0:
                continue

            if 'tclean' == task_call['_name'] and float(task_call['_runtime']) > 0:
                if 'True' == task_call['_params']['parallel']:
                    print('Detail: {0}'.format(task_call))
                    #parallel_points.append()
                else:
                    print('***: {0}'.format(task_call['_params']['parallel']))
                    #continue

                try:
                    val_serial_x = serial_tclean_by_imagename[task_call['_params']['imagename']]
                except KeyError as exc:
                    print('\n *** ERROR, did not find this image in serial results: {0}\n\n'.
                          format(exc))
                    continue

                mpi = run_info['_mpi_servers']
                mpi_servers.append(mpi)

                val_y = float(task_call['_runtime'])
                val_y /= time_div
                # if 'time' == x_axis:
                #     val_serial_x = serial_tclean_by_imagename[task_call['_params']['imagename']]
                # elif 'cube_size' == x_axis:
                #     # ???
                #     nchan = float(task_call['_params']['nchan'])
                #     imsize = ast.literal_eval(task_call['_params']['imsize'])
                #     print(' * imsize: {0}'.format(imsize))
                #     val_x = nchan * imsize[0] * imsize[1]
                #     # x MOUS size
                #     val_x /= mous_sizes[run_info['_mous']]
                   
                val_x = val_serial_x
                parallel_x.append(val_x)
                parallel_y.append(val_y)

                mous = run_info['_mous']                
                point_tags_parallel.append(mous)
                pipe_stages.append(task_call['_pipe_stage_seqno'])
                for name in interest_par_names:
                    interest_pars[name].append(task_call['_params'][name])


    print('Generating tclean plot, with {0} call points in serial mode, and '
          '{1} call points in parallel mode'.format(len(serial_x), len(parallel_x)))


    more_prefix='casa_tclean'
    if 'time' == x_axis:
        more_prefix += '_by_time'
    elif 'cube_size' == x_axis:
        more_prefix += '_by_size'

    if True and 'time' == x_axis:
        gen_tclean_csv_exp(serial_x, serial_y, parallel_x, parallel_y,
                           interest_par_names, interest_pars,
                           mpi_servers, pipe_stages, point_tags_serial, point_tags_parallel,
                           mous_sizes)

    gen_tclean_plot(serial_x, serial_y, parallel_x, parallel_y,
                    what_plot='Individual tclean calls', time_units='hours',
                    show_plot=False,
                    file_base_prefix='plot_runtime_indiv_vs_parallel_',
                    more_prefix=more_prefix, x_axis=x_axis)

def gen_tclean_csv_exp(serial_x, serial_y, parallel_x, parallel_y,
                       interest_par_names, interest_pars,
                       mpi_servers, pipe_stages,
                       point_tags_serial, point_tags_parallel, mous_sizes,
                       file_base_prefix='plot_runtime_indiv_vs_parallel__',
                       more_prefix='casa_tclean', what_plot='misc',
                       x_axis='time'):
    """
    Should write: specmode, gridder, deconvolver, imsize, niter, imagename, stage, mous
    (and MOUS_size)
    """
    import csv

    filename = '{0}.csv'.format(
        make_output_plot_name_base(file_base_prefix, more_prefix, what_plot))

    if 'time' == x_axis:
        second_col = 'x (serial time)'
    elif 'mous_size' == x_axis:
        second_col = 'MOUS size'

    header = '# mpi_servers, {0}, y (serial or parallel time)'.format(second_col)
    for name in interest_par_names:
        header += ', {0}'.format(name)
    header+= ', stage, MOUS, input_MOUS_size'.format(second_col)
        
    with open(filename, 'wb') as csvf:
        csvf.write('{0}\n'.format(header))

        writer = csv.writer(csvf, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)

        for idx, val in enumerate(serial_x):
            size = mous_sizes[point_tags_serial[idx]]
            row_vals = [0, serial_x[idx], serial_y[idx]]
            for name in interest_par_names:
                row_vals.append(interest_pars[name][idx])
            row_vals.extend([pipe_stages[idx], point_tags_serial[idx], size])
            writer.writerow(row_vals)

        idx0 = len(serial_x)
        for idx, val in enumerate(parallel_x):
            size = mous_sizes[point_tags_parallel[idx]]
            row_vals = [mpi_servers[idx], parallel_x[idx], parallel_y[idx]]
            for name in interest_par_names:
                row_vals.append(interest_pars[name][idx0 + idx])
            row_vals.extend([pipe_stages[idx0 + idx], point_tags_parallel[idx], size])
            writer.writerow(row_vals)


def do_casa_tasks_percentage_serial_parallel_plot(serial_infos, parallel_infos,
                                                  counter_type='totals', 
                                                  plot_type='dual',
                                                  x_axis='time',
                                                  produce_csv=True):
    """
    Plots that use serial run time as x axis

    :param counter_type: 'task', 'totals', 'pipe_stage'
    """

    # metric is: 100.0 * _total_time_casa_tasks / _total_time
    serial_x = []
    serial_y = []

    parallel_x = []
    parallel_y = []

    point_tags_serial = []
    point_tags_parallel = []
    mpi_servers = []

    time_div = SECS_TO_HOURS

    serial_by_mous = {}
    parallel_x_by_mpi = {}
    parallel_y_by_mpi = {}
    for key, obj in serial_infos.items():
        mous = obj['_mous']
        serial_by_mous[mous] = obj

        # metric_val = float(obj[metric_name])
        try:
            metric_val = 100.0 * float(obj['_total_time_casa_tasks']) / float(obj['_total_time'])
        except KeyError as exc:
            print(' * Error, could not find this key (type: {0}) in serial run: {1}.'
                  ' For MOUS: {2}. NOT? generating plot. Error details: {3}'.
                  format(counter_type, metric_name, mous, exc))
            # there is one example where imstat is not used in the pipeline (from findcont):
            # serial mode, # E2E5.1.00028.S_2017_10_06T12_40_20.706
            # So let's not leave that plot out, even if some points are missing.
            # return
            continue

        point_tags_serial.append(mous)
        if 'time' == x_axis:
            serial_x.append(float(obj['_total_time']) / time_div)
        elif 'mous_size' == x_axis:
            serial_x.append(mous_sizes[mous])
        serial_y.append(metric_val)

    for key, obj in parallel_infos.items():
        # metric_val = obj[metric_name]
        mous = obj['_mous']
        try:            
            metric_val = 100.0 * float(obj['_total_time_casa_tasks']) / float(obj['_total_time'])
        except KeyError as exc:
            print(' * Error, could not find this key (type: {0}) in parallel run: {1}.'
                  ' For MOUS: {2}. NOT? generating plot. Error details: {3}'.
                  format(counter_type, metric_name, mous, exc))
            # return
            continue

        point_tags_parallel.append(mous)
        par_val_y = float(metric_val)
        parallel_y.append(par_val_y)

        # serial_metric_val = serial_by_mous[mous][metric_name]
        if 'time' == x_axis:
            serial_metric_val = serial_by_mous[mous]['_total_time'] / time_div
        elif 'mous_size' == x_axis:
            serial_metric_val = mous_sizes[mous]

        par_val_x = float(serial_metric_val)
        parallel_x.append(par_val_x)

        mpi = obj['_mpi_servers']
        mpi_servers.append(mpi)
        if mpi not in parallel_x_by_mpi:
            parallel_y_by_mpi[mpi] = [ par_val_y ]
            parallel_x_by_mpi[mpi] = [ par_val_x ]
        else:
            parallel_y_by_mpi[mpi].append(par_val_y)
            parallel_x_by_mpi[mpi].append(par_val_x)

    more_prefix = 'totals'
    title = 'Percentage of pipeline runtime used for CASA tasks'
    ylabel = 'CASA tasks (% of total pipeline runtime)'

    # x_axis = 'mous_size'
    if 'time' == x_axis:
        what_plot = 'Percentage casa tasks vs time'
        xlabel = 'Total pipeline run time (hours)'
        gen_plot_data_as_csv(serial_x, serial_y, parallel_x, parallel_y,
                             mpi_servers, point_tags_serial, point_tags_parallel,
                             mous_sizes=mous_sizes,
                             what_plot=what_plot,
                             more_prefix=more_prefix, 
                             ylabel='percentage of pipeline runtime in CASA tasks')

        gen_dual_plot(serial_x, serial_y, parallel_x, parallel_y,
                      what_plot=what_plot,
                      more_prefix=more_prefix, title=title,
                      xlabel=xlabel, ylabel=ylabel, legend_loc='lower right', x_axis=x_axis)
    elif 'mous_size' == x_axis:
        what_plot = 'Percentage casa tasks vs MOUS size'
        xlabel = 'MOUS size'
        gen_dual_plot(serial_x, serial_y, parallel_x, parallel_y,
                      what_plot=what_plot,
                      more_prefix=more_prefix, title=title,
                      xlabel=xlabel, ylabel=ylabel, legend_loc='lower right',
                      x_axis='mous_size')

def do_per_pl_stage_barplots_multicore_infos(infos):
    print(' * do_per_pl_stage_barplots')
    for key, obj in infos.items():
        mous = obj[0]['_mous']
        print('* Producing PL stages barplots for MOUS: {0}'.format(mous))
        for info in obj:
            do_per_pl_stage_barplots(info)

def do_per_dataset_casa_tasks_barplots(infos):
    print(' * do_per_dataset_casa_tasks_barplots')
    for key, obj in infos.items():
        mous = obj[0]['_mous']
        print('* Producing CASA tasks barplots for MOUS: {0}'.format(mous))
        for info in obj:
            do_casa_tasks_barplot(info)

def do_per_pl_stage_barplots(info):
    stages = sorted(list((map(int, info['_pipe_stages_counter'].keys()))))
    print(' stages: {0}'.format(stages))

    metric_total = lambda x, stg: x['_pipe_stages_counter'][str(stg)]['_taccum']
    #metric_casa_tasks = lambda x: x['_total_time_casa_tasks']
    def metric_all_casa_tasks(x, stg):
        total = 0
        for _key, task in x['_casa_tasks_counter'].items():
            total += task['_taccum_pipe_stages'].get(str(stg), 0)
        return total


    # TODO: Turn these horrors into a single def!!!

    metric_outside = lambda x, stg: metric_total(x) - metric_all_casa_tasks(x)
    # Not a single tclean? Eh?
    # metric_tclean = lambda x, stg: x['_casa_tasks_counter']['tclean']['_taccum_pipe_stages'].get(str(stg), 0)
    def metric_tclean(x, stg):
        if 'tclean' in x['_casa_tasks_counter']:
            return x['_casa_tasks_counter']['tclean']['_taccum_pipe_stages'].get(str(stg), 0)
        else:
            return 0

    metric_flagdata = lambda x,stg: x['_casa_tasks_counter']['flagdata']['_taccum_pipe_stages'].get(str(stg), 0)
    metric_plotms = lambda x,stg: x['_casa_tasks_counter']['plotms']['_taccum_pipe_stages'].get(str(stg), 0)
    metric_applycal = lambda x, stg: x['_casa_tasks_counter']['applycal']['_taccum_pipe_stages'].get(str(stg), 0)
    # Not a single gaincal? Eh?
    # metric_gaincal = lambda x, stg: x['_casa_tasks_counter']['gaincal']['_taccum_pipe_stages'].get(str(stg), 0)
    def metric_gaincal(x, stg):
        if 'gaincal' in x['_casa_tasks_counter']:
            return x['_casa_tasks_counter']['gaincal']['_taccum_pipe_stages'].get(str(stg), 0)
        else:
            return 0
    # Not a single plotbandpass? Eh?
    # metric_plotbandpass = lambda x, stg: x['_casa_tasks_counter']['plotbandpass']['_taccum_pipe_stages'].get(str(stg), 0)
    def metric_plotbandpass(x, stg):
        if 'plotbandpass' in x['_casa_tasks_counter']:
            return x['_casa_tasks_counter']['plotbandpass']['_taccum_pipe_stages'].get(str(stg), 0)
        else:
            return 0

    # Not a single importasdm? Eh?
    # metric_importasdm = lambda x, stg: x['_casa_tasks_counter']['importasdm']['_taccum_pipe_stages'].get(str(stg), 0)
    def metric_importasdm(x, stg):
        if 'importasdm' in x['_casa_tasks_counter']:
            return x['_casa_tasks_counter']['importasdm']['_taccum_pipe_stages'].get(str(stg), 0)
        else:
            return 0

    metric_included_tasks = lambda x, stg: (metric_tclean(x,stg)+metric_flagdata(x,stg)+
                                            metric_plotms(x,stg)+
                                            metric_applycal(x,stg)+
                                            metric_gaincal(x,stg)+
                                            metric_plotbandpass(x,stg)+
                                            metric_importasdm(x,stg))
    metric_other_tasks = lambda x, stg: metric_all_casa_tasks(x, stg)-metric_included_tasks(x, stg)
    metric_total_minus_tasks = lambda x,stg: metric_total(x,stg)-metric_included_tasks(x,stg)-metric_other_tasks(x,stg)


    stg_names = []
    for stg in stages:
        stg_names.append('{0}-{1}'.format(stg, info['_pipe_stages_counter'][str(stg)]['_equiv_call']))

    print(' * stages: {0}'.format(stg_names))
    plot_pl_stages_barplots(info, stages, stg_names,
                            [metric_other_tasks,
                             metric_importasdm, metric_plotbandpass, metric_gaincal,
                             metric_applycal, metric_plotms, metric_flagdata, metric_tclean,
                             metric_total_minus_tasks],
                            legends=['all other CASA tasks',
                                     'importasdm', 'plotbandpass',
                                     'gaincal', 'appycal', 'plotms', 'flagdata', 'tclean',
                                     'Outside CASA tasks']
    )
    
# gen_pl_stages_barplots
def plot_pl_stages_barplots(run_info, stages, stage_names,
                            metric_lambdas,
                            ylabel='Pipeline stages run time (hours)',
                            legends=None,
                            figname_base='plot_pl_stages_tasks_other_barplot'):
    import numpy as np
    import matplotlib.pyplot as plt

    print('* gen_pl_stages_barplots. Stages: {0}...'.format(stages))

    fig = plt.figure(figsize=(16,8))
    plt.xlabel('Pipeline stage', fontsize=FONTSIZE_TITLE)
    plt.ylabel(ylabel, fontsize=FONTSIZE_TITLE)

    bars = []
    colors = ['orangered', 'darkgreen', 'darkblue', 'darkorange', 'lightblue',
              'lightgreen', 'olive', 'lightyellow', 'plum',  'lightseagreen',
              'k', 'cyan', 'gold', 'magenta', 'darkred', 'red', 'blue', 'orange']

    prev_val = None
    sum_val = np.zeros((len(stages)))
    stages_axis = stages
    width = 0.5
    for idx, metric in enumerate(metric_lambdas):
        val_bar = np.array([metric(run_info, stg) for stg in stages]) / SECS_TO_HOURS
        color_idx = -idx -1 + len(metric_lambdas)
        if 0 == idx:
            bar = plt.bar(stages_axis, val_bar, width, color=colors[color_idx],
                          align='edge')
        else:
            bar = plt.bar(stages_axis, val_bar, width, bottom=sum_val,
                          color=colors[color_idx], align='edge')
        bars.append(bar)
        sum_val += val_bar

    if legends:
        get_lines = lambda bars: [b[0] for b in bars]
        # reversed because the first is the lowest in the stacked bars,
        # reverse will put then the first at the bottom of the legend lines
        leg = plt.legend(reversed(get_lines(bars)), reversed(legends), loc='upper center', 
                         prop={'size': FONTSIZE_TITLE})
        leg.get_frame().set_edgecolor('k')


    plt.xlim(1, len(stages)+1) #max(servers_axis)+0.7)
    plt.xticks(range(1, len(stage_names)+1), stage_names, rotation=90, fontsize=FONTSIZE_TITLE)
    plt.yticks(fontsize=8)#FONTSIZE_TITLE)

    mous = run_info['_mous']
    mpi = run_info['_mpi_servers']

    # TODO: make a function
    # short_name = ''
    # mous_size = ''
    # try:
    #     short_name = mous_short_names[mous]
    # except KeyError:
    #     short_name = run_info['_project_tstamp'].split('_')[0]
    # TODO: nope, forget old short names
    short_name = run_info['_project_tstamp'].split('_')[0]

    mous_size = get_asdms_size(mous)

    fig.suptitle('{0}, MOUS: {1}, ASDM size: {2:.1f} GB'.
                 format(short_name, mous, float(mous_size)),
                 fontsize=FONTSIZE_TITLE, fontweight='bold')
    fig.savefig('{0}_{1}_MOUS_{2}_mpi_{3}.png'.
                format(figname_base, short_name, mous, mpi),
                bbox_inches='tight')
    plt.close()


def do_casa_tasks_barplot(info, name_suffix=''):

    verbose = False
    if verbose:
        print(' ** Producing plots of overall CASA tasks runtime per pipeline execution. '
              'Name suffix: {}')

    tasks_bars = {}
    for task, task_counter in info['_casa_tasks_counter'].items():
        tasks_bars[task] = task_counter['_taccum']

    ctasks_time = format_pl_runtime(info['_total_time_casa_tasks'])
    title_long = 'Runtime of CASA tasks (total within CASA tasks: {})'.format(ctasks_time)
    proj = info['_project_tstamp'].split('_')[0]
    mous = info['_mous']
    parallel = info['_mpi_servers']
    suffix = 'proj_{0}_MOUS_{1}_{2}'.format(proj, mous, parallel)
    plot_bar_plot_casa_tasks(tasks_bars, figname_suffix=suffix,
                             title_txt_long=title_long, rotation=75)

def plot_bar_plot_casa_tasks(tasks_barplot, figname_suffix,
                             figname_base='plot_per_dataset_casa_tasks_barplot',
                             ylabel_txt='Runtime (hours)',
                             title_txt_long=None,
                             rotation=75, max_tasks=30):
    import numpy as np
    import matplotlib.pyplot as plt

    verbose = False

    data_ll = [val for _key, val in tasks_barplot.items()]
    task_names = [key for key in tasks_barplot]
    if verbose:
        print(' * CASA task names: {}'.format(task_names))

    sort_indices = list(reversed(np.argsort(data_ll)))
    if max_tasks:
        sort_indices = sort_indices[:max_tasks]
    
    data_ll = [data_ll[idx]/SECS_TO_HOURS for idx in sort_indices]
    task_names = [task_names[idx] for idx in sort_indices]
    print(' * sorted CASA task names: {}'.format(task_names))

    # bar plot
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1)
    x_axis = range(len(task_names))
    width = 0.5
    plt.bar(x_axis, data_ll, width, color='#7777DD', align='edge')
    plt.xticks(x_axis, task_names, rotation=rotation, fontsize=FONTSIZE_TITLE)
    plt.yticks(fontsize=FONTSIZE_TITLE)
    plt.ylabel(ylabel_txt, fontsize=FONTSIZE_TITLE)

    plt.grid(True)
    fig.subplots_adjust(bottom=0.3)

    if not title_txt_long:
        title_txt_long = 'Runtime'
    fig.suptitle(title_txt_long, fontsize=FONTSIZE_TITLE)
    fig.savefig('{0}_{1}.png'.format(figname_base, figname_suffix))
    plt.close()

    
def do_summed_runtime_plots(infos, name_suffix):
    print(' ** Producing plots of overall runtime (runtime summed up) per CASA'
          'task and pipeline stage. Name suffix {0}'.format(name_suffix))

    tasks_full_time = dict()
    tasks_calib_time = dict()
    tasks_imaging_time = dict()
    total_runtime = 0
    total_runtime_calib = 0
    total_runtime_img = 0
    for _key, run_info in infos.items():
        obj = run_info['_casa_tasks_counter']
        total_runtime += run_info['_total_time']
        len_stg = len(run_info['_pipe_stages_counter'])
        # print('Len of stages: {0}, MOUS: {1}'.format(len_stg, run_info['_mous']))
        if 36 == len_stg:
            max_calib = 22
            max_img = 36
        elif 38 == len_stg:
            max_calib = 24
            max_img = 38
        # Bad approx guesses start here. This needs to be smarter
        # elif 14 == len_stg:
        #     # Only imaging? Or broken/incomplete run?
        #     max_calib = 0
        #     max_img = len_stg
        elif 34 > len_stg:
            # This is calibration only? Or broken/incomplete runs?
            max_calib = len_stg
            max_img = 0

        
        # print('do_summed_runtime_plots(), mous: {0}'.format(run_info['_mous']))
        try:
            calib_stg_times = [run_info['_pipe_stages_counter'][str(stg_idx)]['_taccum'] for stg_idx in range(1, max_calib+1)]
        except KeyError as exc:
            print(' - BAD  SIGN - KeyError ({}), with run_info[_pipe_stages_counter]: {}.\n'
                  ' run_info proj: {}'.
                  format(exc, run_info['_pipe_stages_counter'],
                         run_info['_project_tstamp']))
        total_runtime_calib += sum(calib_stg_times)
        img_stg_times = [run_info['_pipe_stages_counter'][str(stg_idx)]['_taccum'] for stg_idx in range(max_calib+1, max_img)]
        total_runtime_img += sum(img_stg_times)

        #total =  run_info['_total_time_casa_tasks']
        for _task, task_info in obj.items():
            tasks_full_time.setdefault(_task, 0)
            tasks_calib_time.setdefault(_task, 0)
            tasks_imaging_time.setdefault(_task, 0)
            tasks_full_time[_task] += task_info['_taccum']
            tasks_calib_time[_task] += task_info['_taccum_calib_1_22']
            tasks_imaging_time[_task] += task_info['_taccum_imaging_23_']

    total_runs = len(infos)
    # total_string = ('{0} runs, {1:.1f} days (calib: {2:.1f}, img: {3:.1f})'.
    #                 format(total_runs, total_runtime/SECS_TO_DAYS,
    #                        total_runtime_calib/SECS_TO_DAYS,
    #                        total_runtime_img/SECS_TO_DAYS,))
    total_string = ('{0} runs, {1:.1f} days'.format(total_runs,
                                                    total_runtime/SECS_TO_DAYS))
    # TODO: Add opt param string for the plot titles
    gen_bar_plot_summed_times(tasks_full_time, 'full_pl_{0}'.format(name_suffix),
                              total_runs=total_string,
                              title_text='All pipeline stages') # Full (calib+imag) pipeline
    gen_bar_plot_summed_times(tasks_calib_time,
                              'calib_pl_{0}'.format(name_suffix),
                              total_runs=total_string,
                              title_text='Calibration stages')
    gen_bar_plot_summed_times(tasks_imaging_time,
                              'imaging_pl_{0}'.format(name_suffix),
                              total_runs=total_string,
                              title_text='Imaging stages')


    
    pl_full_time = dict()
    #tasks_calib_time = dict()
    #tasks_imaging_time = dict()
    total_runtime = 0
    for _key, run_info in infos.items():
        obj = run_info['_pipe_stages_counter']
        total_runtime += run_info['_total_time']

        total =  run_info['_total_time_casa_tasks']
        for _stg, stg_info in obj.items():
            stg_name = stg_info['_equiv_call']
            pl_full_time.setdefault(stg_name, 0)
            #tasks_calib_time.setdefault(_task, 0)
            #tasks_imaging_time.setdefault(_task, 0)
            pl_full_time[stg_name] += stg_info['_taccum']
            #tasks_calib_time[_task] += task_info['_taccum_calib_1_22']
            #tasks_imaging_time[_task] += task_info['_taccum_imaging_23_']

    gen_bar_plot_summed_times(pl_full_time,
                              'full_pl_{0}'.format(name_suffix),
                              total_runs=total_string,
                              title_text='All pipeline stages', # Full (calib+imag) pipeline
                              figname_base='stages_pl_summed_runtime_barplot',
                              rotation=75)

# TODO: rename to plot_bar_plot_...
def gen_bar_plot_summed_times(tasks_boxplot, figname_suffix,
                              figname_base='tasks_summed_runtime_barplot',
                              ylabel_txt='Runtime (days)',
                              total_runs=None,
                              title_txt_long=None,
                              title_text=None,
                              max_tasks=None,
                              rotation=60):
    import numpy as np
    import matplotlib.pyplot as plt

    data_ll = [val for _key, val in tasks_boxplot.items()]
    task_names = [key for key, _val in tasks_boxplot.items()]
    print(' *** got task_names: {}'.format(task_names))


    sort_indices = list(reversed(np.argsort(data_ll)))
    if max_tasks:
        sort_indices = sort_indices[:max_tasks]

    data_ll = [data_ll[idx]/SECS_TO_DAYS for idx in sort_indices]
    task_names = [task_names[idx] for idx in sort_indices]
    print(' *** sorted task_names: {}'.format(task_names))

    # bar plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(1,1,1)
    x_axis = range(len(task_names))
    width = 0.5
    plt.bar(x_axis, data_ll, width, color='#7777DD', align='edge')
    # plt.xlim(0, max(x_axis)+0.7)
    plt.xticks(x_axis, task_names, rotation=rotation, fontsize=FONTSIZE_TITLE)
    plt.yticks(fontsize=FONTSIZE_TITLE)
    plt.ylabel(ylabel_txt, fontsize=FONTSIZE_TITLE)
    # plt.ylim(0, max(data_ll))

    # TODO: problems in old (RHEL6) matplotlib - generate this on newer RHEL, etc.
    #ax2 = ax.twinx()
    #ax2.set_ylabel('% total runtime', fontsize=FONTSIZE_TITLE)
    #ax2.set_yticks(ax.get_yticks() * 100.0 / sum(data_ll)) # fontsize=FONTSIZE_TITLE

    plt.grid(True)
    #fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    # TODO: gen_csv_summed_times_tasks

    if not title_txt_long:
        title_txt_long = 'Total runtime, {0}, summed up ({1})'.format(total_runs,
                                                                      title_text)
    fig.suptitle(title_txt_long, fontsize=FONTSIZE_TITLE)
    fig.savefig('{0}_runs_{1}'.format(figname_base,
                                      figname_suffix))
    plt.close()

def do_task_runtime_per_pl_stg_plots(infos, name_suffix, task='flagdata'):
    print(' ** Producing plots of runtime of CASA tasks(s) split by pipeline '
          'stage. Name suffix {0}. Number of log infos: {1}'.
          format(name_suffix, len(infos)))

    runtime_per_stage = {}
    runtime_per_stage_indiv = {}
    for _key, run_info in infos.items():
        stgs_counter = run_info['_pipe_stages_counter']
        try:
            obj = run_info['_casa_tasks_counter'][task]['_taccum_pipe_stages']
            for stg_idx, stg_runtime in obj.items():
                stg_name = stgs_counter[stg_idx]['_equiv_call']
                runtime_per_stage.setdefault(stg_name, 0)
                # TODO: SECS_TO_DAYS fix
                runtime_per_stage[stg_name] += SECS_TO_DAYS * stg_runtime / SECS_TO_HOURS
                runtime_per_stage_indiv.setdefault(stg_name, [])
                runtime_per_stage_indiv[stg_name].append(stg_runtime / SECS_TO_HOURS)
        except KeyError as exc:
            print('* WRANING: it looks like task {} was not found for this dataset '
                  '(Proj: {}, MOUS: {}. Exception: {}'.
                  format(task, run_info['_project_tstamp'], run_info['_mous'], exc))

    outname = 'task_{0}_per_pipeline_stage_boxplot_runtimes'.format(task)
    gen_tasks_boxplots(runtime_per_stage_indiv, 'full_pl',
                       ylabel_txt='Runtime (h)',
                       plot_title_txt= 'Runtime of {0}, split by pipeline stage'.
                       format(task),
                       max_tasks=40,
                       figname_base=outname)
    total_string = '{0} runs'.format(len(infos))
    total_runtime = sum([float(runtime) for _key, runtime in runtime_per_stage.items()]) / SECS_TO_DAYS
    gen_bar_plot_summed_times(runtime_per_stage, figname_suffix='full_pl',
                              figname_base='task_{0}_per_pipeline_stage_summed_runtimes'.
                              format(task),
                              title_txt_long='Runtime of {0}, by pipeline '
                              'stage. Total: {2:.1f} h. PL jobs: {1}. '
                              .format(task, len(infos), total_runtime),
                              ylabel_txt = 'Runtime (h)')
    
def do_tasks_stats_plots(infos, name_suffix):

    print(' ** Producing CASA tasks stats (boxplots) plots, variant: {0}'.
          format(name_suffix))
        
    tasks_full_times = {}
    tasks_calib_times = {}
    tasks_imaging_times = {}
    for _key, run_info in infos.items():
        obj = run_info['_casa_tasks_counter']

        total =  run_info['_total_time_casa_tasks']
        for _task, task_info in obj.items():
            if False and total < 4*60*60:
                continue
            tasks_full_times.setdefault(_task, []).append(task_info['_taccum'] /
                                                          total * 100)
            tasks_calib_times.setdefault(_task, []).append(task_info['_taccum_calib_1_22'] /
                                                           total * 100)
            tasks_imaging_times.setdefault(_task, []).append(task_info['_taccum_imaging_23_'] /
                                                             total * 100)


    gen_tasks_boxplots(tasks_full_times, 'full_pl_{0}'.format(name_suffix),
                       plot_title_txt ='Tasks stats - including all stages')
    gen_tasks_boxplots(tasks_calib_times, 'calib_pl_{0}'.format(name_suffix),
                       plot_title_txt ='Tasks stats - including only calibration stages')
    gen_tasks_boxplots(tasks_imaging_times, 'imaging_pl_{0}'.format(name_suffix),
                       plot_title_txt ='Tasks stats - including only imaging stages')

def gen_tasks_boxplots(tasks_boxplot, figname_suffix, max_tasks = 10,
                       ylabel_txt='Percentage of total time in CASA tasks',
                       plot_title_txt='Tasks stats',
                       figname_base='tasks_overall_stats_boxplot'):
    
    import numpy as np
    import matplotlib.pyplot as plt

    # data ready for boxplot, as a list of lists
    # useful?
    # items_ll = map(list, tasks_full_times.items())
    #data_ll = map(list, tasks_full_times.values())
    data_ll = [val for _key, val in tasks_boxplot.items()]
    #task_names = map(list, tasks_full_times.keys())
    task_names = [key for key, _val in tasks_boxplot.items()]

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1)
    #data_ll = [[np.random.rand(200)] for i in range(len(data_ll))]
    #print('Going to medians (len data: {0}): {1}'.format(len(data_ll),data_ll))
    print('Each mean/median is calculated over: {0} elements'.format(len(data_ll[0])))
    medians = [np.mean(row) for row in data_ll]
    #print('medians (len: {0}): {1}'.format(len(medians), medians))
    # data_ll = sorted(data_ll, key=lambda x: np.median(x), reverse=True)
    # Sort descending...
    sort_indices = list(reversed(np.argsort(medians)))
    sort_indices = sort_indices[:max_tasks]
    #print('Got sort indices: {0}'.format(sort_indices))
    data_ll = [data_ll[idx] for idx in sort_indices]
    task_names = [task_names[idx] for idx in sort_indices]

    #print('Going to boxplot: {0}'.format(data_ll))
    bplot = ax.boxplot(data_ll) #, patch_artist=True
    for box in bplot['boxes']:
        # change outline color
        box.set(color='darkgrey',linewidth=2)
        # change fill color
        #box.set(facecolor='#1b9e77' )
        #box.set_facecolor('darkgrey')
        ##box.set_fillstyle('full')
    for whisker in bplot['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)
    for cap in bplot['caps']:
            cap.set(color='#7570b3', linewidth=2)
    for median in bplot['medians']:
            median.set(color='blue', linewidth=2)
    for flier in bplot['fliers']:
        flier.set(marker='o', color='green', alpha=0.5)
            
    #print('Going to use ticks: {0}'.format(task_names))
    plt.xticks(range(1, len(task_names)+1), task_names, rotation=45,
               fontsize=FONTSIZE_TITLE)
    plt.title(plot_title_txt, # (sorted by median % of total pipeline time inside CASA tasks)',
              fontsize=FONTSIZE_TITLE)
    plt.ylabel(ylabel_txt, fontsize=FONTSIZE_TITLE)
    plt.grid('on')
    outfig = '{0}_{1}'.format(figname_base,figname_suffix)
    fig.savefig(outfig, bbox_inches='tight')


def check_sanity_stages_22_calibration_23_imaging(infos):
    for _key, run_info in infos.items():
        obj = run_info['_pipe_stages_counter']

        expected_export = 'hifa_exportdata'
        found_last_calib_stage = LAST_CALIB_STAGE
        print('Trying to find the last calib stage starting from {0}...'.format(LAST_CALIB_STAGE))
        for idx in range(LAST_CALIB_STAGE, LAST_CALIB_STAGE+3):   # there's been 22, 23, 24
            if expected_export == obj[str(idx)]['_equiv_call']:
                found_last_calib_stage = idx
                print('Found it: {0}'.format(idx))
        
        idx_22 = str(found_last_calib_stage)#'22'
        print(' * Using this idx for last calib stage: {0}'.format(idx_22))
        stg22 = obj[idx_22]
        if stg22['_equiv_call'] != expected_export:
            raise RuntimeError('* Failed sanity check. This pipe stage {0} is not {1}. '
                               ' It says {2}: {3}'.
                               format(idx_22, expected_export, stg22['_equiv_call'], obj))
        idx_23 = str(found_last_calib_stage+1) #'23'
        expected_mst = 'hif_mstransform'
        stg23 = obj[idx_23]
        if stg23['_equiv_call'] != expected_mst:
            raise RuntimeError('* Failed sanity check. This pipe stage {0} is not {1}. '
                               'It says {2}: {3}'.
                               format(idx_23, expected_mst, stg23['_equiv_call'], obj))

def log_info_sanity_check(infos):
    check_sanity_stages_22_calibration_23_imaging(infos)

def do_all_batch_plots(serial_infos, parallel_infos):
    #do_serial_parallel_plot(serial_infos, parallel_infos)
    print(' *** There are: {0} serial runs, and {1} parallel runs '.
          format(len(serial_infos), len(parallel_infos)))
    if len(serial_infos) > 0:
        do_serial_parallel_plot_all_tasks(serial_infos, parallel_infos)
        do_serial_parallel_plot_totals(serial_infos, parallel_infos)
        do_serial_parallel_plot_pipe_stages(serial_infos, parallel_infos)
        # checking for example 'hif.heuristics.imageparams_base'
        do_serial_parallel_plot_pipe_tasks_functions(serial_infos, parallel_infos)
    
        do_calib_imaging_time_plots(serial_infos, parallel_infos)

def get_total_runtimes(run_infos):
    runtimes = []
    for key, info in run_infos.items():
        runtimes.append(info['_total_time'])
    return runtimes

def get_total_runtimes_multicore(run_infos):
    runtimes = []
    for key, infos_list in run_infos.items():
        for info in infos_list:
            runtimes.append(info['_total_time'])
    return runtimes

def show_total_runtime_stats(runtimes):
    import numpy as np
    import datetime

    if not runtimes:
        print (' * WARNING: no runtimes available!')
        runtimes = [0.0]

    format_time = lambda t: str(datetime.timedelta(seconds=t))

    total = sum(runtimes)
    t_min = np.min(runtimes)
    t_max = np.max(runtimes)
    t_median = np.median(runtimes)
    t_mean = np.mean(runtimes)
    print('Total: {0} s = {1}, min: {2} = {3}, max: {4} = {5}, median: {6} = {7}'
          ', mean: {8} = {9}'.
          format(total, format_time(total), t_min, format_time(t_min), t_max,
                 format_time(t_max), t_median, format_time(t_median),
                 t_mean, format_time(t_mean)))


def show_basic_stats(serial_infos, par_infos, multi_par_infos, show=True):


    serial_runtimes = get_total_runtimes(serial_infos)
    par_runtimes = get_total_runtimes(par_infos)
    multi_par_runtimes = get_total_runtimes_multicore(multi_par_infos)

    print(' * Basic overall stats for serial runtimes ({0}):'.
          format(len(serial_runtimes)))
    show_total_runtime_stats(serial_runtimes)
    print(' * Basic overall stats for parallel runtimes ({0}):'.
          format(len(par_runtimes)))
    show_total_runtime_stats(par_runtimes)

    print(' * Basic overall stats for parallel runtimes (multicore runs, when available):')
    show_total_runtime_stats(multi_par_runtimes)



    # Last second horror tweaks...

    # convenience: turn into list and sort by project name:
    par_infos = [info for _key, info in par_infos.items()]
    par_infos.sort(key=lambda x: x['_project_tstamp'], reverse=False)
    par_runtimes = [info['_total_time'] for info in par_infos]

    # par_runtimes
    total_tasks = [info['_total_time_casa_tasks'] for info in par_infos]
    total_outside_tasks = [info['_total_time'] - info['_total_time_casa_tasks'] for info in par_infos]
    names_infos = [info['_project_tstamp'][:15] for info in par_infos]
    exclude_first = False
    if exclude_first:
        total_tasks = total_tasks[1:]
        total_outside_tasks = total_outside_tasks[1:]
        par_runtimes = par_runtimes[1:]
    print(' * Total runtime: {0}'.format(sum(par_runtimes)))
    print('* Total inside tasks: {0} ({1}%), list: {2}'.
          format(sum(total_tasks), 100.0*sum(total_tasks)/sum(par_runtimes),  total_tasks))
    print('* Total ouside tasks: {0} ({1}%), list: {2}'.
          format(sum(total_outside_tasks), 100.0*sum(total_outside_tasks)/sum(par_runtimes), total_outside_tasks))

    import numpy as np
    indiv_ratios = 100.0*np.divide(np.array(total_tasks), np.array(par_runtimes))
    median_indiv = np.median(indiv_ratios)
    ratios_out = 100.0*np.divide(np.array(total_outside_tasks), np.array(par_runtimes))
    median_ratios_out = np.median(ratios_out)
    print(' Median % inside: {0}. Median % outside: {1}. ' # individual ratios: {2}
          .format(median_indiv, median_ratios_out)) # , indiv_ratios
    # print(' ratios outside: {0}'.format(ratios_out))
    print(' mean inside: {0}'.format(np.mean(indiv_ratios)))
    print(' mean outside: {0}'.format(np.mean(ratios_out)))
    
    x_axis = range(0, len(indiv_ratios))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(bottom=0.15)
    plt.ylabel('% of runtime inside CASA tasks', fontsize=FONTSIZE_TITLE)
    fig.suptitle('% inside CASA tasks, per job (calib + img)', fontsize=FONTSIZE_TITLE)
    plt.bar(x_axis, indiv_ratios, width=0.5, color='#8888DD', align='edge')
    # plt.axhline(median_indiv)
    # plt.axhline(np.mean(indiv_ratios), color='k')
    #plt.text(max(x_axis)/2, median_indiv,'median: {0:.1f}'.format(median_indiv), rotation=0,
    #         fontsize=FONTSIZE_TITLE+6)
    print(' Projects digested: {0}'.format(names_infos))
    plt.xticks(range(0, len(x_axis)), names_infos, rotation=90)
    fig.savefig('plot_pl_test_datasets_pc_in_out_CASA_tasks.png')
    plt.close()

    fig = plt.figure(figsize=(14, 6.5))
    # plot / semilogx
    x_axis = np.array(par_runtimes) / SECS_TO_HOURS
    plt.semilogx(x_axis, ratios_out, 'o', color='#DD5555')
    plt.xlabel('Total runtime (h)', fontsize=FONTSIZE_TITLE)
    plt.ylabel('% of runtime outside CASA tasks', fontsize=FONTSIZE_TITLE)
    plt.grid('on')
    fig.suptitle('% time outside CASA tasks as a function of job (calib + img) total runtime',
                 fontsize=FONTSIZE_TITLE)
    x_ticks = [3, 6, 12, 24, 48, 96, 192, 1000]
    plt.xticks(x_ticks, x_ticks)
    #plt.show()
    fig.savefig('plot_pl_test_datasets_pc_in_out_CASA_tasks_vs_runtime.png')

    # TODO: gen_csv!



# TODO: can be removed
# def get_multicore_info_files(serial_infos, parallel_infos):
#     multicore_lists = {}   # keys: MOUS
#     # Get serial runs
#     for key, obj in serial_infos.items():
#         mous = obj['_mous']
#         multicore_lists[mous] = [obj]
#     # Get parallel runs
#     for key, obj in parallel_infos.items():
#         print(' - Processing : {0}'.format(key))
#         mous = obj['_mous']
#         if mous in multicore_lists:
#             multicore_lists[mous].append(obj)
#         else:
#             multicore_lists[mous] = [obj]

#     return multicore_lists
def plot_multicore_list_runs(run_infos, metric_lambdas,
                             ptype='full_pipeline',
                             ylabel='Pipeline run time (hours)',
                             legends=None):
    """
    :param run_infos: run info objects in a list, corresponding to pipeline runs of a same
    dataset with different number of cores / MPI servers
    """
    import numpy as np
    import matplotlib.pyplot as plt
    #from matplotlib.ticker import FormatStrFormatter

    if len(run_infos) <= 0:
        print(' * Warning: not generating plot, no run infos available')
        return
    
    fig = plt.figure(figsize=(12,8))
    plt.xlabel('Number of parallel servers', fontsize=FONTSIZE_TITLE)
    plt.ylabel(ylabel, fontsize=FONTSIZE_TITLE)

    # Convenience: sort by # mpi servers
    run_infos.sort(key=lambda x: x['_mpi_servers'], reverse=False)
    
    num_servers = [info['_mpi_servers'] for info in run_infos]
    servers_axis = range(len(num_servers))
    width = 0.3


    # if 'full_pipeline' == ptype:
    #     # we only have in/out tasks
    #     bar1 = plt.bar(servers_axis, times_casa_tasks, width, color='darkblue')
    #     bar2 = plt.bar(servers_axis, times_other, width, bottom=times_casa_tasks,
    #                    color='lightblue')
    #     plt.legend((bar1[0], bar2[0]), ('Time inside CASA tasks', 'Time \'other\''),
    #                loc='upper center', prop={'size': FONTSIZE_TITLE})
    # elif metrics_others:
    bars = []
    # Colors will be used with "-idx-1" so that the first in this list is assigned
    # to the top section of the bar (which is the last in the list of metrics)
    if 1 == len(metric_lambdas):
        colors = ['darkblue']
    elif 2 == len(metric_lambdas):
        colors = ['lightblue', 'darkblue']
    else:
        colors = ['orangered', 'darkgreen', 'darkblue', 'darkorange', 'lightblue',
                  'lightgreen', 'olive', 'lightyellow', 'plum',  'lightseagreen',
                  'k', 'cyan', 'gold', 'magenta', 'darkred', 'red', 'blue', 'orange',
                  'lightred', 'darkred', 'red', 'red', 'red', 'red', 'red', 'red']
    prev_val = None
    sum_val = np.zeros((len(run_infos)))
    for idx, metric in enumerate(metric_lambdas):
        # This was very occasionally producing KeyError exceptions:
        # val_bar = np.array([metric(info) for info in run_infos]) / SECS_TO_HOURS
        val_bar = np.zeros(len(run_infos))
        for idx_info, info in enumerate(run_infos):
            try:
                val_bar[idx_info] = metric(info) / SECS_TO_HOURS
            except KeyError as exc:
                print(' ================ NOTE: got exception: {0}'.format(exc))
                val_bar[idx_info] = 0
        color_idx = -idx -1 + len(metric_lambdas)
        if 0 == idx:
            bar = plt.bar(servers_axis, val_bar, width, color=colors[color_idx],
                          align='edge')
        else:
            bar = plt.bar(servers_axis, val_bar, width, bottom=sum_val,
                          color=colors[color_idx], align='edge')
        bars.append(bar)
        sum_val += val_bar

    if legends:
        get_lines = lambda bars: [b[0] for b in bars]
        # reversed because the first is the lowest in the stacked bars,
        # reverse will put then the first at the bottom of the legend lines
        leg = plt.legend(reversed(get_lines(bars)), reversed(legends), loc='upper right', 
                         prop={'size': FONTSIZE_TITLE})
        leg.get_frame().set_edgecolor('k')


    plt.xlim(0, max(servers_axis)+0.7)
    plt.xticks(servers_axis, num_servers, fontsize=FONTSIZE_TITLE)
    plt.yticks(fontsize=FONTSIZE_TITLE)

    # Labels with "x1.3" multiplying factor:
    ax = fig.axes[0]
    for idx, val in enumerate(sum_val):
            ax.text(idx + .25, val + 0.3, 'x{0:1.2f}'.format(sum_val[0]/val),
                    color='k', fontweight='bold', fontsize=FONTSIZE_TITLE)

    mous = run_infos[0]['_mous']


    gen_csv_multicore_list_run(num_servers, sum_val, mous_short_names[mous], mous, ptype)
    
    fig.suptitle('{0}, MOUS: {1}'.  # could add 'ASDM size: {2:.1f} GB' - mous_sizes[mous]
                 format(mous_short_names[mous]),
                 fontsize=FONTSIZE_TITLE, fontweight='bold')
    fig.savefig('plot_bars_runtime_{0}_parallel_multiple_cores_MOUS_{1}_{2}.png'.
                format(ptype, mous_short_names[mous], mous))

    plt.close()


def filter_output_csv_name(name):
    fname = name.replace(' ', '_')
    fname = fname.replace('/', '_')
    return '{0}'.format(fname)
    
def gen_csv_multicore_list_run(num_servers, runtimes, short_name, mous, ptype,
                               basename='plot_data_runtime'):
    """
    Produces a "plot data" csv file For the speedup-vs-cores or "multicore" sets of runs.
    Would normally run together with plot_multicore_list_runs.
    """
    import csv

    filename = filter_output_csv_name('plot_data_{0}_multiple_cores_{1}_MOUS_{2}.csv'.
                                      format(ptype, short_name, mous))

    print('Going to gen csv, with servers: {0}, \nruntimes: {1}'.format(num_servers, runtimes))
    with open(filename, 'wb') as csvf:
        csvf.write('# mpi_servers, runtime_hours, speedup_ratio_wrt_first {0}\n'.
                   format('## MOUS: {0} - short_name: {1}'.format(mous, short_name)))

        writer = csv.writer(csvf, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)

        for mpi_srv, runt in zip(num_servers, runtimes):
            ratio = 0
            if runt != 0:
                ratio = runtimes[0] / runt
            # writer.writerow([mpi_srv, runt, ratio])
            writer.writerow([mpi_srv, '{0:.4f}'.format(runt), '{0:.4f}'.format(ratio)])


#def do_all_multicore_plots(serial_infos, parallel_infos, min_par=2):
def do_all_multicore_plots(multicore_parallel_infos, min_par=2):
    """
    Plots of runtime varying with the number of cores used in different
    parallel runs for a same dataset.

    :param min_par: minimum number of parallel executions (with different
    number of cores) required to produce a 'multi-core' plot
    """
    print(' -------------- do_all_multicore_plots')

    PIPE_FIRST_IMAGING_STAGE = LAST_CALIB_STAGE+1 #23
    PIPE_LAST_STAGE = 36
    for key, obj in multicore_parallel_infos.items():
        mous = obj[0]['_mous']
        print(' - Multicore list len for {0}: {1}. Num servers: {2}'.
              format(mous, len(obj), [obj[idx]['_mpi_servers'] for idx in range(len(obj))]))
        if len(obj) >= 3:
            # total/full pipeline runtime
            metric_total = lambda x: x['_total_time']
            metric_casa_tasks = lambda x: x['_total_time_casa_tasks']
            metric_other = lambda x: metric_total(x) - metric_casa_tasks(x)
            plot_multicore_list_runs(obj, [metric_casa_tasks, metric_other],
                                     ptype='full_pipeline',
                                     legends=['Time inside CASA tasks', 'Time \'other\''])

            metric_tclean = lambda x: x['_casa_tasks_counter']['tclean']['_taccum']
            plot_multicore_list_runs(obj, [metric_tclean], ptype='in_task_tclean',
                                     ylabel='Task tclean runtime (hours)',
                                     legends=None)

            metric_findcont = lambda x: x['_pipe_stages_counter']['26']['_taccum']
            plot_multicore_list_runs(obj, [metric_findcont],
                                     ptype='in_pipe_stage_26_hif_findcont',
                                     ylabel='Pipeline stage 26 hif_findcont run time (hours)',
                                     legends=None)

            # Plot A: stage 26 (findcont) outside of tasks, B: everything else
            # This might not find the expected tasks any longer in CASA 5.4
            try:
                stg26 = str(LAST_CALIB_STAGE + 4) #'26'
                metric_tclean_in_stg26 = lambda x: x['_casa_tasks_counter']['tclean']['_taccum_pipe_stages'][stg26]
                metric_imhead_in_stg26 = lambda x: x['_casa_tasks_counter']['imhead']['_taccum_pipe_stages'][stg26]
                metric_imstat_in_stg26 = lambda x: x['_casa_tasks_counter']['imstat']['_taccum_pipe_stages'][stg26]
                metric_findcont_other = lambda x: (metric_findcont(x) - metric_tclean_in_stg26(x) - metric_imhead_in_stg26(x) -
                                                   metric_imstat_in_stg26(x))
                metric_all_minus_findcont_other = lambda x: metric_total(x) - metric_findcont_other(x)
                plot_multicore_list_runs(obj, [metric_all_minus_findcont_other, metric_findcont_other],
                                         ptype='outside_tasks_in_pipe_stage_26_hif_findcont',
                                         ylabel='Outside of CASA tasks, pipe stage 26 hif_findcont run time (hours)',
                                         legends=['Everything else',
                                                  'Pipeline stage 26, outside tasks (tclean, imhead, imstat)'])
            except KeyError as exc:
                print('* Ignoring exception about stage {0}: {1}'.format(stg26, exc))

            
            
            #metric_stage_1 = lambda x : x['_pipe_stages_counter']['1']['_taccum']
            #metric_stage_25 = lambda x: x['_pipe_stages_counter']['25']['_taccum']
            metric_stage_27 = lambda x: x['_pipe_stages_counter']['27']['_taccum']
            metric_stage_28 = lambda x: x['_pipe_stages_counter']['28']['_taccum']
            #metric_stage_32 = lambda x: x['_pipe_stages_counter']['32']['_taccum']
            #metric_stage_34 = lambda x: x['_pipe_stages_counter']['34']['_taccum']
            #metric_stage_35 = lambda x: x['_pipe_stages_counter']['35']['_taccum']
            metric_stage_36 = lambda x: x['_pipe_stages_counter']['36']['_taccum']
            metric_stages_3x = lambda x: metric_stage_27(x) + metric_stage_28(x) + metric_stage_36(x)
            metric_total_minus_26 = lambda x: metric_total(x)-metric_findcont(x)
            metric_total_minus_26_3x = lambda x: metric_total(x)-metric_findcont(x)-metric_stages_3x(x)
            plot_multicore_list_runs(obj, [metric_total_minus_26_3x, metric_findcont, metric_stages_3x],
                                     ptype='in_pipe_stage_26_hif_findcont',
                                     ylabel='Pipeline run time (hours)',
                                     legends=['Everything else',
                                              'Pipeline stage 26 hif_findcont',
                                              'Pipeline stages 27,28,36'])  # hifa_importdata, hif_uvcontfit, hif_uvcontsub, hifa_exportdata


            # metric_img_tclean = lambda x: x['_casa_tasks_counter']['tclean']['_taccum_imaging_23_']
            #metric_img_flagdata = lambda x: x['_casa_tasks_counter']['flagdata']['_taccum_imaging_23_']



            # TODO...doing: separate 'other' time and turn everything else into 'all other tasks', 'outside of tasks'
            metric_flagdata = lambda x: x['_casa_tasks_counter']['flagdata']['_taccum']
            metric_plotms = lambda x: x['_casa_tasks_counter']['plotms']['_taccum']
            metric_importasdm = lambda x: x['_casa_tasks_counter']['importasdm']['_taccum']
            metric_gaincal = lambda x: x['_casa_tasks_counter']['gaincal']['_taccum']
            metric_applycal = lambda x: x['_casa_tasks_counter']['applycal']['_taccum']
            metric_setjy = lambda x: x['_casa_tasks_counter']['setjy']['_taccum']
            metric_immoments = lambda x: x['_casa_tasks_counter']['immoments']['_taccum']
            metric_plotbandpass = lambda x: x['_casa_tasks_counter']['plotbandpass']['_taccum']
            metric_included_tasks = lambda x: (metric_tclean(x)+metric_flagdata(x)+metric_plotms(x)+
                                               metric_importasdm(x)+metric_gaincal(x)+metric_applycal(x)+
                                               metric_setjy(x)+metric_immoments(x)+metric_plotbandpass(x))
            metric_other_tasks = lambda x: metric_casa_tasks(x)-metric_included_tasks(x)
            metric_total_minus_tasks = lambda x: metric_total(x)-metric_included_tasks(x)-metric_other_tasks(x)
            plot_multicore_list_runs(obj, [metric_other_tasks, metric_immoments, metric_setjy,
                                           metric_plotbandpass,
                                           metric_applycal, metric_gaincal,
                                           metric_importasdm, metric_plotms,
                                           metric_flagdata, metric_tclean,
                                           metric_total_minus_tasks],
                                     ptype='_various_tasks_stacked_plot',
                                     ylabel='Pipeline run time (hours)',
                                     legends=['all other CASA tasks',
                                              'immoments',
                                              'setjy',
                                              'plotbandpass',
                                              'applycal',
                                              'gaincal',
                                              'importasdm',
                                              'plotms',
                                              'flagdata',
                                              'tclean',
                                              'Outside CASA tasks'
                                              ])  # hifa_importdata

            
            metric_imaging_only = lambda x: sum(
                [x['_pipe_stages_counter'][str(idx)]['_taccum']
                 for idx in range(PIPE_FIRST_IMAGING_STAGE, PIPE_LAST_STAGE+1)])
            plot_multicore_list_runs(obj, [metric_imaging_only],
                                     ptype='imaging_pipeline',
                                     ylabel='Imaging pipeline run time (hours)',
                                     legends=None)

            metric_calib_only = lambda x: sum(
                [x['_pipe_stages_counter'][str(idx)]['_taccum']
                 for idx in range(1, PIPE_FIRST_IMAGING_STAGE)])
            plot_multicore_list_runs(obj, [metric_calib_only],
                                     ptype='calibration_pipeline',
                                     ylabel='Calibration pipeline run time (hours)',
                                     legends=None)


def get_total_runtimes_h(run_infos):
    runtimes = {}
    for key, obj in run_infos.items():
        mous = obj['_mous']
        total = float(obj['_total_time'])/3600.0
        runtimes[mous] = total
    return runtimes

def print_total_runtimes(serial_infos, parallel_infos):
    serial_runtimes = get_total_runtimes_h(serial_infos)
    print(" Dict of total runtimes, serial mode: {0}".format(serial_runtimes))
    par_runtimes = get_total_runtimes_h(parallel_infos)
    print(" Dict of total runtimes, parallel mode: {0}".format(par_runtimes))

def print_html_summary(serial_infos, parallel_infos):
    import os

    subdir_per_run = 'per_run'
    doc_hdr = ('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n'
               '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n')

    # TODO: remove - old serial/parallel stuff
    def gen_runtime_sum_section():
        f_total = 'plot_runtime_serial_vs_parallel__totals_CASA_tasks_total.png'
        # f_calib = 'plot_runtime_serial_vs_parallel__section__Calibration_pipeline.png'
        # f_img = 'plot_runtime_serial_vs_parallel__section__Imaging_pipeline.png'
        txt = ''
        if os.path.isfile(f_total) and os.access(f_total, os.R_OK):
            plot_img_pattern = '<a href="{0}"><img src={0} width="50%"/></a>\n'
            txt = '<p>Full pipeline</p>'
            txt += plot_img_pattern.format(f_total)
            txt += '<p>Imaging and calibration</p>'
            txt += '<table><tr>'
            txt += '<td>'
            txt += plot_img_pattern.format(f_calib)
            txt += '</td>'
            txt += '<td>'
            txt += plot_img_pattern.format(f_img)
            txt += '</td>'
            txt += '</tr></table>'

        return txt

    def gen_overall_histograms():
        histo_runtimes = 'plot_pipeline_test_datasets_histo_runtimes_parallel.png'
        histo_sizes = 'plot_pipeline_test_datasets_histo_sizes.png'
        res = '<p>'
        res += '<a href="{0}"><img width="40%" src="{0}"/></a>'.format(histo_runtimes)
        res += '<a href="{0}"><img width="40%" src="{0}"/></a>'.format(histo_sizes)
        res += '</p>\n'
        # html += '<table class="datasets-tbl">\n<tbody>\n'
        # html += ('<tr>'
        #          '</tr>'
        #          )

        # ===> Summed runtimes of pipeline stages and CASA tasks
        # ===> Overall CASA tasks statistics
        # ===> Advanced/Experimental plots

        # res += '</tbody>\n</table>\n'

        subpage_summed = 'foo'
        subpage_overall_stats = 'foo'
        subpage_adv = 'foo'

        # res += '<div width="30%" style="float:left;">'
        # html = '<table class="datasets-tbl">\n<thead>\n'
        # html += ('<tr>'
        #          '<th><a href="{}">===> Summed runtimes of pipeline stages and CASA tasks</a></th>'
        #          '<th><a href="{}">===> Overall CASA tasks statistics</a></th>'
        #          '<th><a href="{}">===> Advanced/Experimental plots</a></th>'
        #          '</tr>\n'
        #          '</thead>\n'
        #          '</table>'.format(subpage_summed, subpage_overall_stats, subpage_adv)
        #          )
        # res += html
        # res += '</div>'

        return res

    def indiv_run_subpage_name(info):
        run_name = 'performance_{}.html'.format(info['_mous'])
        return os.path.join(subdir_per_run, run_name)

    def gen_subpage_overall_casa_stats():
        """ returns the name of the page produced """

        oname = 'overall_casa_stats.html'
        res = doc_hdr + '<html><head>\n'
        title = 'Overall CASA tasks statistics, as box plots'
        res += '<title>{}</title>'.format(title)
        res += '</head>\n'
        res += '<body>\n'
        res += '<h1>{}</h1>\n'.format(title)
        boxplot_full = 'tasks_overall_stats_boxplot_full_pl_parallel.png'
        boxplot_calib = 'tasks_overall_stats_boxplot_calib_pl_parallel.png'
        boxplot_img = 'tasks_overall_stats_boxplot_imaging_pl_parallel.png'
        res += '<a href="{0}"><img src="{0}"/></a>'.format(boxplot_full)
        res += '<hr/>'
        res += '<a href="{0}"><img src="{0}"/></a>'.format(boxplot_calib)
        res += '<hr/>'
        res += '<a href="{0}"><img src="{0}"/></a>'.format(boxplot_img)
        res += '<hr/>'
        res += '</body>\n</html>'

        with open(oname, "w+") as ofile:
            ofile.write(res)

        return oname

    def gen_subpage_summed_stages_n_tasks():
        """ returns the name of the page produced """

        oname = 'summed_runtimes_pl_stages_n_casa_tasks.html'
        res = doc_hdr + '<html><head>\n'
        title = 'Summed runtimes of pipeline stages and CASA tasks'
        res += '<title>{}</title>'.format(title)
        res += '</head>\n'
        res += '<body>\n'
        res += '<h1>{}</h1>\n'.format(title)
        stages = 'stages_pl_summed_runtime_barplot_runs_full_pl_parallel.png'
        tasks_full = 'tasks_summed_runtime_barplot_runs_full_pl_parallel.png'
        tasks_calib = 'tasks_summed_runtime_barplot_runs_calib_pl_parallel.png'
        tasks_img = 'tasks_summed_runtime_barplot_runs_imaging_pl_parallel.png'
        res += '<a href="{0}"><img src="{0}"/></a>'.format(stages)
        res += '<hr/>'
        res += '<a href="{0}"><img src="{0}"/></a>'.format(tasks_full)
        res += '<hr/>'
        res += '<a href="{0}"><img src="{0}"/></a>'.format(tasks_calib)
        res += '<hr/>'
        res += '<a href="{0}"><img src="{0}"/></a>'.format(tasks_img)
        res += '<hr/>'
        res += '</body>\n</html>'

        with open(oname, "w+") as ofile:
            ofile.write(res)

        return oname

    def gen_subpage_advanced_plots():
        import glob
        oname = 'advanced_plots.html'
        # file names like:
        # task_flagdata_per_pipeline_stage_summed_runtimes_runs_full_pl.png
        # task_flagdata_per_pipeline_stage_boxplot_runtimes_full_pl.png
        
        try:
            summed_plots = glob.glob('task_*_per_pipeline_stage_summed_runtimes_*.png')
        except RuntimeError:
            print('Warning, could not find plot "task_*_per_pipeline_stage_summed_runtimes')
            summed_plots = ''

        try:
            box_plots = glob.glob('task_*_per_pipeline_stage_boxplot_runtimes_*.png')
        except RuntimeError:
            print('Warning, could not find plot "task_*_per_pipeline_stage_summed_runtimes')
            box_plots = ''

        title = 'Advanced or experimental plots - WIP'
        res = doc_hdr + '<html><head>\n'
        res += '<title></title>'.format(title)
        if path_css:
            res += '<link rel="stylesheet" type="text/css" href="{}">\n'.format(path_css)
        res += '</head>\n'
        res += '<body>\n'
        res += '<h1>{}</h1>'.format(title)
        res += ('The plots below show run times (total and box plot statistics) of '
                'selected CASA tasks (tclean, flagdata, etc.), grouped by pipeline stage')
        for img in sorted(summed_plots):
            res += '<a href="{0}"><img src="{0}"/></a>'.format(img)
        res += '<hr/>'
        for img in sorted(box_plots):
            res += '<a href="{0}"><img src="{0}"/></a>'.format(img)
        res += '</body>\n</html>'

        with open(oname, "w+") as ofile:
            ofile.write(res)
        
        return oname

    def find_stages_run(info):
        stgs = info['_pipe_stages_counter']
        stg_idx = sorted(stgs.keys(), key=lambda x: int(x))
        stg_names = [info['_pipe_stages_counter'][key]['_equiv_call'] for key in stg_idx]
        res = ''

        def looks_like_calib(stg_names):
            return (
                stg_names[0] == 'hifa_importdata' and
                stg_names[1] == 'hifa_flagdata' and
                stg_names[2] == 'hifa_fluxcalflag' and
                stg_names[-1] == 'hifa_exportdata' and
                'hifa_bandpassflag' in stg_names and
                'hif_lowgainflag' in stg_names and
                'hifa_timegaincal' in stg_names
            )

        def looks_like_plus_imaging_only(stg_names):
            """ 
            :param stg_names: list of stages names, ordered by run order 
            """
            return (
                stg_names[0] == 'hif_mstransform' and
                stg_names[-1] == 'hifa_exportdata' and
                'hif_findcont' in stg_names and
                'hif_uvcontfit' in stg_names and
                'hif_uvcontsub' in stg_names
            )

        def looks_like_plus_imaging(stg_names):
            """ 
            :param stg_names: list of stages names, ordered by run order 
            """
            return (
                stg_names.count('hifa_exportdata') >= 2 and 
                'hif_mstransform' in stg_names and 
                'hif_findcont' in stg_names and
                'hif_uvcontfit' in stg_names and
                'hif_uvcontsub' in stg_names
            )
        
        if len(stgs) >= 20 and looks_like_calib(stg_names):
            res = 'calibration'
            if len(stgs) >= 34 and looks_like_plus_imaging(stg_names):
                res += ' + imaging'

        elif len(stgs) >= 12 and looks_like_imaging_only(stg_names):  # 14+1 presently
            res = 'imaging'
            
        res += ' ({} stages)'.format(len(info['_pipe_stages_counter']))

        return res
    
    def gen_table_datasets(run_infos):
        res = '<table class="datasets-tbl">\n<thead>\n'
        res += ('<tr><th>Project</th> <th>MOUS</th> <th># EBs</th> <th>ASDM size (GB, total)</th> <th>start</th> '
                '<th>runtime</th> <th>stages run</th> <th>CASA version</th> <th>machine</th>'
                '<th>processes</th> </tr>')
        res += '</thead>\n'
        res += '<tbody>\n'
        def sort_key_func(key_val):
            return key_val[1]['_project_tstamp']
        
        for uid, info in sorted(run_infos.items(), key=sort_key_func):
            res += '<tr>'
            res += '<td>{}</td>'.format(info['_project_tstamp'].split('_')[0])
            subpage_name = indiv_run_subpage_name(info)
            res += '<td><a href="{0}">{1}</a></td>'.format(subpage_name, info['_mous'])
            res += '<td>{0}</td>'.format(casa_logs_mous_props.ebs_cnt.get(uid, 0))
            res += '<td>{0:.1f}</td>'.format(get_asdms_size(uid))
            res += '<td>{}</td>'.format(info['_first_tstamp'])
            res += '<td>{}</td>'.format(format_pl_runtime(float(info['_total_time'])))
            res += '<td>{}</td>'.format(find_stages_run(info))
            res += '<td>{}</td>'.format(info['_casa_version'])
            res += '<td>{}</td>'.format(info['_run_machine'])
            res += '<td>{}</td>'.format(int(info['_mpi_servers'])+1)
            res += '</tr>\n'

        res += '</tbody>\n'
        res += '</table>'
        return res

    def gen_per_run_pages(run_infos, path_css=None):
        import glob

        if not os.path.isdir(subdir_per_run):
            os.mkdir(subdir_per_run)

        for uid, info in sorted(run_infos.items()):
            # Look for a plot named like:
            # plot_pl_stages_tasks_other_barplot_E2E6.1.00038.S_MOUS_uid___A002_Xcff05c_X277_mpi_7.png

            exp_basic = ('The bar chart below shows the time consumed by every CASA task. '
                         'All the executions of a same task, for example plotms, are '
                         'aggregated into a single per-task time.')
            try:
                barplot_basic = glob.glob('plot_per_dataset_casa_tasks_barplot_*_{}_*.png'.
                                    format(info['_mous']))[0]
            except (RuntimeError, IndexError):
                print('Warning, could not find file with the bar plot of single-color '
                      'CASA tasks overall times, for MOUS: {}'.format(info['_mous']))
                barplot_basic = ''

            explanation = ('The bar chart below shows the time consumed by every pipeline '
                           'stage (x axis), and within each stage the time consumed by '
                           'different CASA tasks (colors).')

            try:
                barplot = glob.glob('plot_pl_stages_tasks_other_barplot_*_{}_*.png'.
                                    format(info['_mous']))[0]
            except (RuntimeError, IndexError):
                print('Warning, could not find file with the multi-color bar plot of CASA '
                      'tasks times in pipeline stages, for MOUS: {}'.format(info['_mous']))
                barplot = ''
                
            res = doc_hdr + '<html><head>\n'
            res += ('<title>MOUS {0} - execution {1}</title>'.
                    format(info['_mous'], info['_project_tstamp']))
            if path_css:
                res += '<link rel="stylesheet" type="text/css" href="{}">\n'.format(
                    os.path.join('..', path_css))
            res += '</head>\n'
            res += '<body>\n'
            # TODO: move these plot inside the subdirectory per_run/?
            # TODO: add html-gen-function that takes the list of headers + list of values
            res +='<table class="datasets-tbl">\n<thead>\n'
            res += ('<tr>'
                    '<th>CASA version</th>'
                    '<th>Project</th> <th>MOUS</th>'
                    '<th> # EBs</th> <th>Size of all ASDMs (GB)</th>'
                    '<th>Total runtime</th> <th>Machine</th>'
                    '</tr>\n'
                    '</thead>\n'
            )
            mous = info['_mous']
            res += ('<tbody>\n'
                    '<tr> <td>{}</td>'
                    '<td>{}</td> <td>{}</td>'
                    '<td>{}</td> <td>{:.1f}</td>'
                    '<td>{}</td> <td>{}</td>'
                    '</tr>\n'.format(info['_casa_version'],
                                     info['_project_tstamp'].split('_')[0], mous,
                                     casa_logs_mous_props.ebs_cnt.get(mous, 0),
                                     get_asdms_size(mous),
                                     format_pl_runtime(float(info['_total_time'])),
                                     info['_run_machine'])
            )
            res += '</tbody>\n'
            res += '</table>'

            res += '<p>{}</p>'.format(exp_basic)
            res += '<a href="{0}"><img src="{0}"/></a>'.format(
                os.path.join('..', barplot_basic))

            res += '<p>{}</p>'.format(explanation)
            res += '<a href="{0}"><img src="{0}"/></a>'.format(
                os.path.join('..', barplot))

            res += '</body>\n</html>'
            fname = indiv_run_subpage_name(info)
            with open(fname, "w+") as ofile:
                ofile.write(res)
    
    def cp_css():
        import shutil
        subdir = 'css'
        fname = 'casa_log_info_plots.css'
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
        path_myself = os.path.dirname(os.path.abspath(__file__))
        path_css = os.path.join(subdir, fname)
        shutil.copyfile(os.path.join(path_myself, fname), path_css)
        return path_css
    
    def gen_main_page(serial_infos, parallel_infos, path_css=None,
                      outname='ALMA_pipeline_runs_performance_metrics.html'):
        title = 'Performance metrics from ALMA pipeline runs'
        html = doc_hdr + '<head>\n'
        html += '<title>{0}</title>\n'.format(title)
        if path_css:
            html += '<link rel="stylesheet" type="text/css" href="{}">\n'.format(path_css)
        html += '</head>\n'.format(title)
        html += '<body>\n'
        html += '<h1>{}</h1>\n'.format(title)
        html += '<p>'
        html += ('These pages show statistics and plots for a set of pipeline/CASA '
                 'runs that used different ALMA datasets.')
        html+= '</p>\n'

        first_start = min([info['_first_tstamp'] for _key, info in parallel_infos.items()])
        last_end = max([info['_last_tstamp'] for _key, info in parallel_infos.items()])
        casa_versions = [info['_casa_version'] for _key, info in parallel_infos.items()]
        machines = set([info['_run_machine'] for _key, info in parallel_infos.items()])
        oldest_casa = min(casa_versions)
        oldest_freq = casa_versions.count(oldest_casa)
        newest_casa = max(casa_versions)
        newest_freq = casa_versions.count(newest_casa)        

        html +='<table class="datasets-tbl">\n<thead>\n'
        html += ('<tr>'
                 '<th>Total datasets</th>'
                 '<th>First run started</th> <th>Last run ended</th>'
                 '<th>Oldest CASA version used</th> <th>Newest CASA version used</th>'
                 '<th>Number of different machines or nodes used</th>'
                 '</tr>\n'
                 '</thead>\n'
        )
        html += ('<tbody>\n'
                 '<tr> <td>{}</td> <td>{}</td>'
                 '<td>{} </td> <td>{} ({} times)</td>'
                 '<td>{} ({} times)</td> <td>{}</td>'
                 '</tr>\n'.format(len(serial_infos)+len(parallel_infos), first_start,
                                  last_end, oldest_casa, oldest_freq, newest_casa,
                                  newest_freq, len(machines)) 
        )
        html += '</table>\n'

        html += '<h2>Datasets - runtime, input size</h2>\n'
        overall_histos = gen_overall_histograms()

        html += overall_histos
        # html += overall_histos
        # html += '\n<hr/>\n'
        html += '<h2>Aggregated statistics:</h2>\n'
        subpage_summed = gen_subpage_summed_stages_n_tasks()
        subpage_overall_stats = gen_subpage_overall_casa_stats()
        subpage_adv = gen_subpage_advanced_plots()
        html += '<table class="links-tbl">\n<thead>\n'

        to_summed = '<a href="{}">===> Summed runtimes of pipeline stages and CASA tasks</a>'.format(
            subpage_summed)
        to_overall = '<a href="{}">===> Overall CASA tasks statistics</a>'.format(
            subpage_overall_stats)
        to_adv = '<a href="{}">===> Advanced/Experimental plots</a>'.format(
            subpage_adv)
        html += ('<tr>'
                 '<th>{}</th>'
                 #'</tr><tr>'
                 '<th>{}</th>'
                 #'</tr><tr>'
                 '<th>{}</th>'
                 '</tr>\n'
                 '</thead>\n'
                 '</table>'.format(to_summed, to_overall, to_adv)
                 )
        # html += ('<div style="background-color: #DDEFDD; width: 100%">'
        #          '<span width="30%">{}</span>'
        #          '<span width="30%">{}</span>'
        #          '<span width="30%">{}</span>'
        #          '</div>'.format(to_summed, to_overall, to_adv)
        # )
        

        
        # html += ('<p><a href="{0}">===> Summed runtimes of pipeline stages and CASA tasks</a>'
        #          '</p>\n'.format(subpage_summed))
        # html += ('<p><a href="{0}">===> Overall CASA tasks statistics</a></p>\n'.
        #          format(subpage_overall_stats))

        # html += ('<p><a href="{0}">===> Advanced/Experimental plots</a></p>\n'.
        #          format(subpage_adv))

        # html += '<hr/>\n'
        html += '<h2>Individual runs:</h2>\n'
        html += gen_table_datasets(parallel_infos)

        import platform
        html += '\n<!-- auto-generated by {} on {}, tstamp: {} -->\n'.format(
            os.path.basename(__file__), platform.node(), datetime.datetime.now())
                                                                   
        html += '</body>\n</html>'

        with open(outname, "w") as outf:
            outf.write(html)
            if not os.path.islink('index.html'):
                os.symlink(outname, 'index.html')

    path_css = cp_css()

    gen_main_page(serial_infos, parallel_infos, path_css)
    
    gen_per_run_pages(parallel_infos, path_css)

    # old seria/parallel stuff
    # html += '\n<p>Run time</p>\n'
    # totals = gen_runtime_sum_section()
    # html += totals



def plot_histo(data_val, bin_width, ticks_dist, xlabel, ylabel, title, filename):
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["patch.force_edgecolor"] = True
    except KeyError as exc:
        print(' WARNING, got exception in histo options: {0}'.
              format(exc))
       
    fig = plt.figure(figsize=(16, 9))
    val_range = max(data_val) - min(data_val)
    bin_width = 5
    bins_limits = range(0, int(max(data_val) + bin_width), bin_width)
    counts, bins, patches = plt.hist(data_val, bins=bins_limits, align='mid',
                                facecolor='darkgreen')
    plt.xlabel(xlabel, fontsize=FONTSIZE_TITLE)
    plt.ylabel(ylabel, fontsize=FONTSIZE_TITLE)
    plt.title(title.format(bin_width), fontsize=FONTSIZE_TITLE)
    plt.xticks(fontsize=FONTSIZE_TITLE)
    plt.xticks(np.arange(min(bins_limits), max(bins_limits), ticks_dist))
    plt.xlim(0, max(bins_limits))
    plt.ylim(0, max(counts))
    plt.yticks(fontsize=FONTSIZE_TITLE)
    #plt.xlim(bins[0], 250) #bins[-1])
    plt.grid(True)
    fig.savefig(filename)
    print('* Saving histogram plot: {0}'.format(filename))
    plt.close()

# TODO: organize this mess
def produce_datasets_histograms(serial_infos, parallel_infos):
    sizes = []
    times_serial = []
    times_par = []

    max_time = 400
    for key, info in serial_infos.items():
        # mous = key
        mous = info['_mous']
        try:
            sizes.append(mous_sizes[mous])
        except KeyError:
                print(' WARNING: no size available for mous: {0}'.format(mous))
                sizes.append(-1)

        time = info['_total_time'] / SECS_TO_HOURS
        if time > max_time:
            print(' *** WARN WARN: very long time (in serial). Time: {0}. MOUS:'.
                  format(time, mous))
        times_serial.append(time)

    for key, info in parallel_infos.items():
        time = info['_total_time'] / SECS_TO_HOURS
        if time > max_time:
            print(' *** WARN WARN: very long time (in parallel). Time: {0}, MOUS:'.
                  format(time, info['_mous']))
            continue
        times_par.append(time)

    if not sizes:
        for key, info in parallel_infos.items():
            mous = info['_mous']
            try:
                sizes.append(mous_sizes[mous])
            except KeyError:
                print(' WARNING: no size available for mous: {0}'.format(mous))
                sizes.append(-1)

    # Histo of sizes
    bin_width = 5
    import numpy as np
    print('Plotting histo of sizes: {0}'.format(sizes))
    if len(sizes) > 0 and sum(sizes) > 0:
        plot_histo(sizes, bin_width=bin_width, ticks_dist=20, xlabel='ASDM sizes (GB)',
                   ylabel='Count of datasets (ASDMs)',
                   title='Histogram of input ASDM sizes (bins: {0} GB)'.format(bin_width),
                   filename='plot_pipeline_test_datasets_histo_sizes.png')

    bin_width=4
    if len(times_serial) > 0:
        plot_histo(times_serial, bin_width=bin_width, ticks_dist=24, xlabel='Run time (hours)',
                   ylabel='Count of datasets',
                   title='Histogram of pipeline run times (serial mode) (bins: {0} hours)'.
                   format(bin_width),
                   filename='plot_pipeline_test_datasets_histo_runtimes_serial.png')
    else:
        print(' *** NOTICE: Not producing histogram of serial times - no serial runs available')

    print('Plotting histo of parallel times: {0}'.format(times_par))
    plot_histo(times_par, bin_width=bin_width, ticks_dist=24, xlabel='Run time (hours)',
               ylabel='Count of datasets',
               title='Histogram of pipeline run times (bins: {0} hours)'.
               format(bin_width),
               filename='plot_pipeline_test_datasets_histo_runtimes_parallel.png')

def find_serial_call_by_imgname(task_details_params, imgname):
    for det in task_details_params:
        if 'tclean' == det['_name'] and imgname == det['_params']['imagename']:
            print('Comparing {0} with {1}'.format(imgname, det['_params']['imagename']))
            return det

    return None
    
def do_beam_stats(serial_infos, parallel_infos):
    accum = []

    serial_by_mous = {}
    for key, obj in serial_infos.items():
        mous = obj['_mous']
        serial_by_mous[mous] = obj

    for key, obj in parallel_infos.items():
        print("info: {0}".format(obj['_mous']))
        for tclean_call in obj['_tasks_details_params']:
            if 'tclean' == tclean_call['_name'] and tclean_call['_further_info']:
                mous = obj['_mous']
                imgname = tclean_call['_params']['imagename']

                serial_info = serial_by_mous[mous]
                serial_call = find_serial_call_by_imgname(serial_info['_tasks_details_params'],
                                                          imgname)

                beam = tclean_call['_further_info']['common_beam']
                row = [obj['_project_tstamp'], mous, imgname,
                       '', '', '',
                       obj['_mpi_servers'],
                       beam['_major'], beam['_minor'], beam['_pa']]

                if serial_call:
                    beam_s = serial_call['_further_info']['common_beam']
                    row[3] = beam_s['_major']
                    row[4] = beam_s['_minor']
                    row[5] = beam_s['_pa']

                accum.append(row)

    print(accum)
                     
    fname = 'serial_parallel_beam_parameters.csv'
    hdr = '# Proj, MOUS, image_name, serial_major, serial_minor, serial_pa, parallel_servers, parallel_major, parallel_minor, parallel_pa'
    #with open(fname, 'w') as outf:
    #    outf.write(hdr+'\n')
    #    outf.write(accum)
    import csv
    with open(fname, "wb") as outf:
        writer = csv.writer(outf)
        outf.write(hdr+'\n')
        writer.writerows(accum)

def main_info_plotter(input_dir, make_general_plots=False,
                      make_tasks_stats_plots=False,
                      make_summed_tasks_stages_plots=False,
                      make_task_per_pl_stage_plots=False,
                      make_per_pl_stage_barplots=False,
                      make_per_dataset_casa_tasks_barplots=False,
                      make_multicore_plots=False,
                      make_percentages_plots=False,
                      make_tclean_plots=False,
                      make_datasets_histos=False,
                      make_beam_stats=False,
                      gen_html_summary=False):
    import os

    if os.path.isdir(input_dir):
        subdir = input_dir
        run_info_files = find_info_files(subdir)
        print(' * Found these {0} CASA run info files in {1}: {2}'.
              format(len(run_info_files), subdir, run_info_files))
    elif os.path.isfile(input_dir):
        run_info_files = [input_dir]
        print(' * Processing files given as argument: {0}'.format(input_dir))
    else:
        raise RuntimeError('Give a better msg here')

    serial_infos, parallel_infos, multicore_parallel_infos = parse_info_files(run_info_files)

    do_sanity_checks = False
    if do_sanity_checks:
        log_info_sanity_check(serial_infos)
        log_info_sanity_check(parallel_infos)

    if make_general_plots:
        do_all_batch_plots(serial_infos, parallel_infos)

    if make_tasks_stats_plots:
        if serial_infos:
            do_tasks_stats_plots(serial_infos, 'serial')
        if parallel_infos:
            do_tasks_stats_plots(parallel_infos, 'parallel')
        
    if make_summed_tasks_stages_plots:
        if serial_infos:
            do_summed_runtime_plots(serial_infos, 'serial')
        if parallel_infos:
            do_summed_runtime_plots(parallel_infos, 'parallel')

    # This is for example for flagdata
    if make_task_per_pl_stage_plots:
        tasks = ['flagdata', 'tclean', 'applycal', 'gaincal', 'plotms']
        for tsk in tasks:
            if serial_infos:
                do_task_runtime_per_pl_stg_plots(serial_infos, 'serial', task=tsk)
            if parallel_infos:
                do_task_runtime_per_pl_stg_plots(parallel_infos, 'parallel', task=tsk)
    # only this one:
    # do_serial_parallel_plot_pipe_tasks_functions(serial_infos, parallel_infos)

    if make_multicore_plots:
        do_all_multicore_plots(multicore_parallel_infos)

    if make_per_pl_stage_barplots:
        do_per_pl_stage_barplots_multicore_infos(multicore_parallel_infos)

    if make_per_dataset_casa_tasks_barplots:
        do_per_dataset_casa_tasks_barplots(multicore_parallel_infos)

    # Experimental stuff I don't remember well - 201802
    if make_percentages_plots:
        # This is another type of plot that shows something in "parallel vs. serial"
        if len(serial_infos) > 0:
            do_casa_tasks_percentage_serial_parallel_plot(serial_infos, parallel_infos,
                                                          x_axis='time')
            do_casa_tasks_percentage_serial_parallel_plot(serial_infos, parallel_infos,
                                                          x_axis='mous_size')
    # TODO: in the middle of doing this...
    if make_tclean_plots:
        do_tclean_experimental_plots(serial_infos, parallel_infos)
        do_tclean_experimental_plots(serial_infos, parallel_infos, x_axis='cube_size')

    if True:
        print_total_runtimes(serial_infos, parallel_infos)

    show_basic_stats(serial_infos, parallel_infos, multicore_parallel_infos)

    if gen_html_summary:
        print_html_summary(serial_infos, parallel_infos)

    if make_beam_stats:
        do_beam_stats(serial_infos, parallel_infos)

    if make_datasets_histos:
        produce_datasets_histograms(serial_infos, parallel_infos)       

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Produce plots from info files digested '
                                     'from CASA logs.')
    parser.add_argument('input_directory', nargs=1, help=
                        'All the .json files found in the directory are used to produce '
                        'the plots', type=str)
    parser.add_argument('--bundle-html', action='store_true',
                        help='This is ia meta-option that enables a set of options used '
                        'to produce a set of html pages and plot files. It enables '
                        '--gen-html-summary, --make-datasets-histos, etc. It is meant '
                        'as a simple and single option to use for a one-go command to make '
                        'all usual plots and the html pages that display them.')
    parser.add_argument('--make-general-plots', action='store_true')
    parser.add_argument('--make-datasets-histos', action='store_true',
                        help='histograms of dataset sizes and runtimes')
    parser.add_argument('--make-summed-tasks-stages-plots', action='store_true',
                        help='boxplots of summed up times of  CASA tasks and '
                        'pipeline stages')
    parser.add_argument('--make-tasks-stats-plots', action='store_true',
                        help='boxplots of stats of CASA tasks for one dataset')

    parser.add_argument('--make-task-per-pl-stage-plots', action='store_true',
                        help='bar plots of run time of tasks in different  '
                        'pipeline stages (flagdata, etc.)')
    parser.add_argument('--make-per-pl-stage-barplots',action='store_true',
                        help='barplots per PL stage, with CASA tasks and "other". This '
                        'produces one plot per dataset/execution')
    parser.add_argument('--make-per-dataset-casa-tasks-barplots',action='store_true',
                        help='simple barplots of time per CASA task. This '
                        'produces one plot per dataset/execution')
    parser.add_argument('--make-multicore-plots', action='store_true',
                        help='barplots of runtime per CASA task for a range of '
                        'number of cores')
    parser.add_argument('--make-percentages-plots', action='store_true')
    parser.add_argument('--make-tclean-plots', action='store_true')
    parser.add_argument('--make-beam-stats', action='store_true')
    parser.add_argument('--gen-html-summary', action='store_true')
    parser.add_argument('--casa-5-4', action='store_true')
    
    args = parser.parse_args()

    if args.casa_5_4:
        global LAST_CALIB_STAGE
        print('* CASA 5.4')
        LAST_CALIB_STAGE = 23
        print('* last calib stage: {0}'.format(LAST_CALIB_STAGE))

    def enable_bundle_html(args):
        args.make_datasets_histos = True
        args.make_summed_tasks_stages_plots = True
        args.make_tasks_stats_plots = True
        args.make_task_per_pl_stage_plots = True
        args.make_per_pl_stage_barplots = True
        args.make_per_dataset_casa_tasks_barplots = True
        args.gen_html_summary = True
        
    if args.bundle_html:
        enable_bundle_html(args)

    main_info_plotter(args.input_directory[0], args.make_general_plots,
                      args.make_tasks_stats_plots,
                      args.make_summed_tasks_stages_plots,
                      args.make_task_per_pl_stage_plots,
                      args.make_per_pl_stage_barplots,
                      args.make_per_dataset_casa_tasks_barplots,
                      args.make_multicore_plots, args.make_percentages_plots,
                      args.make_tclean_plots, args.make_datasets_histos,
                      args.make_beam_stats,
                      args.gen_html_summary)
    
if __name__ == '__main__':
    main()
