import subprocess
import sys 

def scores(path):
  bashCommand = 'perl conlleval'
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stdin=open(path))
  output, error = process.communicate()
  output = output.decode().split('\n')[1].split('%; ')
  output = [out.split(' ')[-1] for out in output]
  acc, prec, recall, fb1 = tuple(output)
  return float(acc), float(prec), float(recall), float(fb1)
  
