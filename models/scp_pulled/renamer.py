import time
import os
import subprocess
import decimal 
import argparse
import re

dirs =  os.listdir()
path = "\/home\/user\/projects\/neural_program_synthesis\/models\/scp_pulled"
for di in dirs:
	if di == "tmpd": continue
	cmd = "sed -ie 's/\/home\/rb7e15\/2\.7v\/.*model/"+path+"\/"+di+"\/model/' "+di+"/model/checkpoint"
	print(cmd)
	p = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)
	p.wait()