#!/usr/bin/python

import os
import rospy
import psutil
import threading
import numpy as np

from colorama import init, Fore
init(autoreset=True)


def log_cpu_usage_thread(trace_dir, stop_event):
    cpu_measurements = list()
    mem_measurements = list()
    while not stop_event.is_set():
        cpu_perc = psutil.cpu_percent(1, percpu=True)
        cpu_measurements.append(cpu_perc)
        mem_measurements.append(
            [psutil.virtual_memory().percent, psutil.swap_memory().percent])
        stop_event.wait(1.0)
    m_cpu = np.array(cpu_measurements)
    m_mem = np.array(mem_measurements)
    np.savetxt(os.path.join(trace_dir, 'log_cpu_usage.txt'), m_cpu, fmt='%.2e')
    np.savetxt(os.path.join(trace_dir, 'log_memory_usage.txt'),
               m_mem, fmt='%.2e')


class RosNode:
    def __init__(self, package, executable):
        self._package = package
        self._executable = executable
        self._param_string = ''

    def add_parameters(self, namespace, parameter_dictionary):
        for key in parameter_dictionary.keys():
            if key == 'flags':
                continue
            if type(parameter_dictionary[key]) is dict:
                self.add_parameters(namespace+key+'/',
                                    parameter_dictionary[key])
            else:
                self._param_string += ' _'+namespace + \
                    key+':='+str(parameter_dictionary[key])

    def add_flags(self, flag_dictionary):
        for key, value in flag_dictionary.iteritems():
            self._param_string += ' --'+key+'='+str(value)

    def clear_all_parameters(self):
        params = rospy.get_param_names()
        for p in params:
            rospy.delete_param(p)

    def run(self, parameter_dictionary, namespace='',
            log_cpu_usage=True, log_cpu_usage_folder='/tmp'):
        self.clear_all_parameters()
        self.add_parameters(namespace, parameter_dictionary)
        if 'flags' in parameter_dictionary:
            self.add_flags(parameter_dictionary['flags'])
            print("Flags are {0} ".format(parameter_dictionary['flags']))

        if log_cpu_usage:
            print('Will log CPU Usage')
            t_stop = threading.Event()
            t = threading.Thread(target=log_cpu_usage_thread,
                                 args=(log_cpu_usage_folder, t_stop))
            t.start()

        command = 'rosrun ' + self._package + ' ' + \
            self._executable + ' ' + self._param_string
        # print(Fore.RED + 'Executing Process: '+command)
        print(Fore.RED + 'Start ROS node')
        os.system(command)
        print(Fore.GREEN+'ROS node finished processing.')

        if log_cpu_usage:
            t_stop.set()
            t.join()
