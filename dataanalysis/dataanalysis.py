import numpy as np
import matplotlib.pyplot as plt
import re
import glob

maxIter=500
data=np.zeros([maxIter,3])

file_paths = glob.glob(r'C:\Users\qqsup\Desktop\COSSE\master thesis\parallelcode\slurm-2372462.out')
for file_path in file_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()[25:25+maxIter]
        
for i in range(maxIter):
    data[i,:]=re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", lines[i])
        
iteration=data[:,0]
resnorm=data[:,1]

plt.figure(figsize=(8, 6))
plt.plot(iteration,resnorm)
plt.yscale('log')
plt.grid()
plt.xlabel('number of iterations')
plt.ylabel(r'residual norm $||r||_2$')