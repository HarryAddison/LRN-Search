import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv

from sklearn.mixture import GaussianMixture
from astroML.plotting.tools import draw_ellipse
from astropy.io import fits
from astropy.table import Table
from astroquery.gaia import Gaia
from extinction import fitzpatrick99

from sklearn import metrics


'''
This script is for BIC. It is a method of selecting the best number of 
components for a model
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions 

def SelBest(arr:list, X:int)->list:
    '''
    Returns the set of X configurations with shorter distance
    '''
    
    dx=np.argsort(arr)[:X]
    return arr[dx]
    
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variables

# Version
version = "V3.1.1"

# Training data set path.
train_data_path = "D:/addis/Documents/Uni work/Placement Year/Project/Stuff on this OS/Project Files/Versions/V3/Data/HRD Data/Training Data/7-result.csv"

# Plot save location.
plot_save_path = "D:/addis/Documents/Uni work/Placement Year/Project/Stuff on this OS/Project Files/Versions/V3/Results/Component Selection/"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loading training data and correcting for extinction/reddening.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loading in training data.

train_plx, train_ap_mag, train_color, train_extinc = np.loadtxt(
                       train_data_path, skiprows=1, delimiter=',', unpack=True)
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gaia DR2 corrections.
train_ap_mag_correct = []

for i in range(len(train_ap_mag)):
    temp = train_ap_mag[i]
    
    if train_ap_mag[i] > 2 and train_ap_mag[i] <= 6:
        temp = (-0.047344 + 1.16405 * train_ap_mag[i] - 0.046799 * 
                train_ap_mag[i]**2 + 0.0035015 * train_ap_mag[i]**3)
        
    if train_ap_mag[i] > 6 and train_ap_mag[i] <= 16:
        temp = train_ap_mag[i] - 0.0032 * (train_ap_mag[i] - 6)
    
    if train_ap_mag[i] > 16:
        temp = train_ap_mag[i] - 0.032
            
    train_ap_mag_correct.append(temp)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Correcting for extinction.

train_ap_mag = train_ap_mag_correct - train_extinc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Correcting for reddening

# rp basspand wavelength 7570
rp_wave = np.array([6600]) #Angstroms

train_redden = []

for i in range(len(train_extinc)):

    a_v = train_extinc[i] / 0.87511505

    rp_extinc = fitzpatrick99(rp_wave, a_v, 3.1, unit="aa")
    
    train_redden.append(train_extinc[i] - rp_extinc[0])

train_color = train_color - train_redden

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calulating absolute magnitue from aparent and parallax.

train_abs_mag = train_ap_mag + 5 * np.log10(train_plx) - 10

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Readying the data set for use.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Reducing the data set to the required parameter space.

reduc_train_color = []
reduc_train_abs_mag = []

for i in range(len(train_color)):
    if train_color[i] < 1.6 and train_abs_mag[i] < 2:
        reduc_train_abs_mag.append(train_abs_mag[i])
        reduc_train_color.append(train_color[i])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Putting the colour and absolute magnitude into an array for as required by the GMM  code 
# ((c,mag), (c,mag)...).

temp = []

for i in range(len(reduc_train_color)):
    temp.append((reduc_train_color[i], reduc_train_abs_mag[i]))

train_data = np.array(temp)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Selecting the best munber of component. 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BIC

n_clusters=np.arange(2, 20)
bics=[]
bics_err=[]
iterations=20
for n in n_clusters:
    tmp_bic=[]
    for _ in range(iterations):
        
        gmm=GaussianMixture(n, n_init=2).fit(train_data) 
        
        tmp_bic.append(gmm.bic(train_data))
    val=np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
    err=np.std(tmp_bic)
    bics.append(val)
    bics_err.append(err)

fig = plt.figure(figsize=(6, 5))
plt.errorbar(n_clusters,bics, yerr=bics_err, label='BIC')
plt.title("BIC Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Score")

plt.savefig("%sBIC_%s.png" %(plot_save_path, version), dpi=fig.dpi)

fig = plt.figure(figsize=(6, 5))
plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
plt.title("Gradient of BIC Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("grad(BIC)")

plt.savefig("%sBIC_Grad_%s.png" %(plot_save_path, version), dpi=fig.dpi)