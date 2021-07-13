#!/usr/bin/python

import subprocess, re

def get_cpu_info():
	command = "cat /proc/cpuinfo"
	all_info = subprocess.check_output(command, shell=True).strip()
	for line in all_info.split("\n"):
		if "model name" in line:
			model_name = re.sub(".*model name.*:", "", line,1).strip()
			return model_name.replace("(R)","").replace("(TM)", "")