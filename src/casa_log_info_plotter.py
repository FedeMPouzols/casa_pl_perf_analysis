#!/usr/bin/env python

import datetime

from casa_logs_mous_props import mous_sizes, mous_short_names

too_verbose = False

FONTSIZE_LEGEND = 14
FONTSIZE_AXES = 14
FONTSIZE_TITLE = 16


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
    glob_pattern = os.path.join(subdir,'*.json')
    return [ifile for ifile in glob.glob(glob_pattern)]


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
        print serial_x
        print serial_y
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
    plt.legend(loc='upper left', prop={'size': FONTSIZE_LEGEND})

    name_base = make_output_plot_name_base(file_base_prefix, more_prefix, what_plot)
    fig.savefig('{0}.png'.format(name_base))
    if show_plot:
        plt.show()

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
        title = '{0} run time, pipeline runs in serial (black) and parallel (colors) mode'.format(what_plot)
    plt.title(title,fontsize=FONTSIZE_TITLE)

    if too_verbose:
        print serial_x
        print serial_y

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
    plt.grid('on')
    if not legend_loc:
        legend_loc = 'upper left'
    plt.legend(loc=legend_loc, prop={'size': FONTSIZE_LEGEND}) # 'best' 'upper right'

    name_base = make_output_plot_name_base(file_base_prefix, more_prefix, what_plot)
    fig.savefig('{0}.png'.format(name_base))
    if show_plot:
        plt.show()

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
    first_obj = serial_infos.itervalues().next()
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

    # hours
    time_div = 3600    
    
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

def do_calib_imaging_time_plots(serial_infos, parallel_infos, plot_type='dual'):
    """
    This produces separate plots for the calibration and imaging
    phases of the pipeline.

    Exploits the fact that calibration happens within a known range of stages. At
    the time this doc was written, calibration goes from stage 1 up to 22 (included).
    After that, the next stages are considered as imaging stuff.
    """
    do_pipe_stages_ranges_plot(serial_infos, parallel_infos,
                              [(1,22), (23, 36)], ['Calibration', 'Imaging'],
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

    # hours
    time_div = 3600

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

    # hours
    time_div = 3600    
    
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
    plt.title('{0} run time, pipeline runs in serial (black) and parallel (colors) mode'.
              format(what_plot),
              fontsize=FONTSIZE_TITLE)

    if too_verbose:
        print serial_x
        print serial_y
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
    plt.grid('on')
    plt.legend(loc='upper left', prop={'size': FONTSIZE_LEGEND}) # 'best' 'upper right'

    name_base = make_output_plot_name_base(file_base_prefix, more_prefix, what_plot)
    fig.savefig('{0}.png'.format(name_base))
    if show_plot:
        plt.show()

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
    time_div = 3600    

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

    time_div = 3600

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

def check_sanity_stages_22_calibration_23_imaging(infos):
    for _key, run_info in infos.items():
        obj = run_info['_pipe_stages_counter']

        idx_22 = '22'
        stg22 = obj['22']
        expected_export = 'hifa_exportdata'
        if stg22['_equiv_call'] != expected_export:
            raise RuntimeError('* Failed sanity check. This pipe stage {0} is not {1}. '
                               ' It says {2}: {3}'.
                               format(idx_22, expected_export, stg22['_equiv_call'], obj))
        idx_23 = '23'
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
    print('Total: {0} s = {1}, min: {2} = {3}, max: {4} = {5}, median: {6} = {7}'.
          format(total, format_time(total), t_min, format_time(t_min), t_max,
                 format_time(t_max), t_median, format_time(t_median)))

def show_basic_stats(serial_infos, par_infos, multi_par_infos, show=True):
    serial_runtimes = get_total_runtimes(serial_infos)
    par_runtimes = get_total_runtimes(par_infos)
    multi_par_runtimes = get_total_runtimes_multicore(multi_par_infos)

    print(' * Basic overall stats for serial runtimes:')
    show_total_runtime_stats(serial_runtimes)
    print(' * Basic overall stats for parallel runtimes:')
    show_total_runtime_stats(par_runtimes)

    print(' * Basic overall stats for parallel runtimes (multicore runs, when available):')
    show_total_runtime_stats(multi_par_runtimes)

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

def plot_multicore_list_runs(run_infos, metric_lambda,
                             ptype='full_pipeline',
                             ylabel='Pipeline run time (hours)',
                             metrics_others=None,
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
    
    SECS_TO_HOURS = 3600
    num_servers = [info['_mpi_servers'] for info in run_infos]
    servers_axis = range(len(num_servers))
    #times_total = np.array([info['_total_time'] for info in run_infos]) / SECS_TO_HOURS
    times_total = np.array([metric_lambda(info) for info in run_infos]) / SECS_TO_HOURS
    times_casa_tasks = np.array([info['_total_time_casa_tasks'] for info in run_infos]) / SECS_TO_HOURS
    times_other = times_total - times_casa_tasks
    width = 0.3


    if 'full_pipeline' == ptype:
        # we only have in/out tasks
        bar1 = plt.bar(servers_axis, times_casa_tasks, width, color='darkblue')
        bar2 = plt.bar(servers_axis, times_other, width, bottom=times_casa_tasks,
                       color='lightblue')
        plt.legend((bar1[0], bar2[0]), ('Time inside CASA tasks', 'Time \'other\''),
                   loc='upper center', prop={'size': FONTSIZE_TITLE})
    elif metrics_others:
        bars = []
        colors = ['darkblue', 'g', 'y', 'm']
        prev_val = None
        sum_bar = np.zeros((len(run_infos)))
        for idx, metric in enumerate(metrics_others):
            val_bar = np.array([metric(info) for info in run_infos]) / SECS_TO_HOURS
            if 0 == idx:
                bar = plt.bar(servers_axis, val_bar, width, color=colors[idx])
            else:
                bar = plt.bar(servers_axis, val_bar, width, bottom=prev_val,
                              color=colors[idx])
            bars.append(bar)
            prev_val = val_bar
            sum_bar += val_bar
        get_lines = lambda bars: [b[0] for b in bars]
        plt.legend(get_lines(bars), (legends), loc='upper center', 
                   prop={'size': FONTSIZE_TITLE})
    else:
        bar1 = plt.bar(servers_axis, times_total, width, color='darkblue')

    plt.xlim(0, max(servers_axis)+0.7)
    plt.xticks(servers_axis, num_servers, fontsize=FONTSIZE_TITLE)
    plt.yticks(fontsize=FONTSIZE_TITLE)

    # Labels with factor:
    ax = fig.axes[0]
    if not metrics_others:
        top_bar = times_total
    else:
        top_bar = sum_bar
    for idx, val in enumerate(top_bar):
            ax.text(idx + .25, val + 0.3, 'x{0:1.2f}'.format(times_total[0]/val),
                    color='k', fontweight='bold', fontsize=FONTSIZE_TITLE)

    mous = run_infos[0]['_mous']
    fig.suptitle('{0}, MOUS: {1}, ASDM size: {2} GB'.
                 format(mous_short_names[mous], mous, mous_sizes[mous]),
                 fontsize=FONTSIZE_TITLE, fontweight='bold')
    fig.savefig('plot_bars_runtime_{0}_parallel_multiple_cores_MOUS_{1}_{2}.png'.
                format(ptype, mous_short_names[mous], mous))

#def do_all_multicore_plots(serial_infos, parallel_infos, min_par=2):
def do_all_multicore_plots(multicore_parallel_infos, min_par=2):
    """
    Plots of runtime varying with the number of cores used in different
    parallel runs for a same dataset.

    :param min_par: minimum number of parallel executions (with different
    number of cores) required to produce a 'multi-core' plot
    """
    print(' -------------- do_all_multicore_plots')

    PIPE_FIRST_IMAGING_STAGE = 23
    PIPE_LAST_STAGE = 36
    for key, obj in multicore_parallel_infos.items():
        mous = obj[0]['_mous']
        print(' - Multicore list len for {0}: {1}. Num servers: {2}'.
              format(mous, len(obj), [obj[idx]['_mpi_servers'] for idx in range(len(obj))]))
        if len(obj) >= 3:
            # total/full pipeline runtime
            metric_total = lambda x: x['_total_time']
            plot_multicore_list_runs(obj, metric_total, ptype='full_pipeline')

            metric_tclean = lambda x: x['_casa_tasks_counter']['tclean']['_taccum']
            plot_multicore_list_runs(obj, metric_tclean, ptype='in_task_tclean',
                                     ylabel='Task tclean runtime (hours)')

            metric_findcont = lambda x: x['_pipe_stages_counter']['26']['_taccum']
            plot_multicore_list_runs(obj, metric_findcont,
                                     ptype='in_pipe_stage_26_hif_findcont',
                                     ylabel='Pipeline stage 26 hif_findcont run time (hours)')
            plot_multicore_list_runs(obj, metric_findcont,
                                     ptype='in_pipe_stage_26_hif_findcont',
                                     ylabel='Pipeline stage 26 hif_findcont run time (hours)')

            metric_total_minus_26 = lambda x: metric_total(x)-metric_findcont(x)
            plot_multicore_list_runs(obj, metric_total_minus_26,
                                     ptype='in_pipe_stage_26_hif_findcont',
                                     ylabel='Pipeline run time (hours)',
                                     metrics_others=[metric_total_minus_26, metric_findcont],
                                     legends=['Everything else',
                                              'Pipeline stage 26 hif_findcont'])

            metric_imaging_only = lambda x: sum(
                [x['_pipe_stages_counter'][str(idx)]['_taccum']
                 for idx in range(PIPE_FIRST_IMAGING_STAGE, PIPE_LAST_STAGE+1)])
            plot_multicore_list_runs(obj, metric_imaging_only,
                                     ptype='imaging_pipeline',
                                     ylabel='Imaging pipeline run time (hours)')

            metric_calib_only = lambda x: sum(
                [x['_pipe_stages_counter'][str(idx)]['_taccum']
                 for idx in range(1, PIPE_FIRST_IMAGING_STAGE)])
            plot_multicore_list_runs(obj, metric_calib_only,
                                     ptype='calibration_pipeline',
                                     ylabel='Calibration pipeline run time (hours)')

    # for (old) multi_as_dict
    # print(' - Total datasets: {0}'.format(len(multicore_parallel_infos)))
    # for mous_key, ds_infos in multicore_parallel_infos.items():
    #     key0 = list(ds_infos.keys())[0]
    #     print ('key0: {0}'.format(key0))

    #     mous = ds_infos[key0]['_mous']
    #     print(' - Dict len for {0}/{1}: {2}'.format(mous_key, mous, len(ds_infos)))
    #     for key, obj in ds_infos.items():
    #         #mous = obj[0]['_mous']
    #         print(' - mpi: {0}'.format(obj['_mpi_servers']))

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

def main_info_plotter(input_dir, make_general_plots=False, make_multicore_plots=False):
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

    log_info_sanity_check(serial_infos)
    log_info_sanity_check(parallel_infos)

    if make_general_plots:
        do_all_batch_plots(serial_infos, parallel_infos)
    # only this one:
    # do_serial_parallel_plot_pipe_tasks_functions(serial_infos, parallel_infos)

    if make_multicore_plots:
        do_all_multicore_plots(multicore_parallel_infos)
       
    do_percentages = False
    if do_percentages:
        do_casa_tasks_percentage_serial_parallel_plot(serial_infos, parallel_infos,
                                                      x_axis='time')
        do_casa_tasks_percentage_serial_parallel_plot(serial_infos, parallel_infos,
                                                      x_axis='mous_size')


    if True:
        print_total_runtimes(serial_infos, parallel_infos)

    # TODO: in the middle of doing this...
    do_tclean_plots = False
    if do_tclean_plots:
        do_tclean_experimental_plots(serial_infos, parallel_infos)
        do_tclean_experimental_plots(serial_infos, parallel_infos, x_axis='cube_size')

    show_basic_stats(serial_infos, parallel_infos, multicore_parallel_infos)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Produce plots from info files digested '
                                     'from CASA logs.')
    parser.add_argument('input_directory', nargs=1, help=
                        'All the .json files found in the directory are used to produce '
                        'the plots', type=str)
    parser.add_argument('--make-general-plots', action='store_true')
    parser.add_argument('--make-multicore-plots', action='store_true')

    args = parser.parse_args()

    main_info_plotter(args.input_directory[0], args.make_general_plots,
                      args.make_multicore_plots)
    
if __name__ == '__main__':
    main()
