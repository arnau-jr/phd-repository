import numpy as np
import matplotlib.pyplot as plt
import random

data = np.loadtxt('ener_comp.dat')
Nconfs = data.shape[0]

indexes = random.sample(range(Nconfs),50)
print(len(set(indexes))==len(indexes))

plt.xlabel("Configuration index")
plt.ylabel(r"E (cm$^{-1}$)")
# plt.xlim([130,140])
plt.xlim([1,len(indexes)])
# plt.ylim([0,10000.])
# plt.plot(range(1,len(indexes)+1),((data[indexes,0]-data[indexes,1])/data[indexes,0])*100.,'r-',label="E(Ab initio)- E(Fit)")
plt.bar(range(1,len(indexes)+1),data[indexes,0],color='red',label="E(Ab initio)")
plt.bar(range(1,len(indexes)+1),data[indexes,1],color='blue',label="Fit",alpha=0.5)
plt.legend()
plt.savefig("comp.pdf")
plt.close()
