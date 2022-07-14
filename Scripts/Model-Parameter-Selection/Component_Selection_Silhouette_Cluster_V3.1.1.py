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
This script is for finding candidate LRN precursors for further 
analysis.

Uses Gaia data to form a HR diagram which then has its density modelled 
using a gaussian mixture model.
A data set is then fitted to this model and "scores" are found. 
Candidates are then selected based upon thier "score".

> Inputs:
    > Training Data Set (csv file)
    > Data Set (fits file)
        
> Outputs:
    > Plot of the model ("Results/model")
    > Plot of the data set with scores
    > Plot of the candidates
    > File of the data set ("Corrected_Data.csv")
            (scores appended).
    > Files of the candidates
            (scores and ra and dec (J2000) appended).
            
> Variables:
    > data_set_path == path to the data set.
    > plot_id == string for uniquely naming plot files.
    > thres_score == Threshold score for candidate slection. 
                          
> Outline of Code:
    > Line  : Functions
                > Line : Saving data to fits file
    > Line 46: Variables
    > Line 61: Loading training data
    > Line 86: Model Creation
    > Line : Plotting the model
    > Line : Loading the data set 
    > Line : Scoring the data set to the model
    > Line : Plotting the scores
    > Line : Saving the data
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def SelBest(arr:list, X:int)->list:
    '''
    Returns the set of X configurations with shorter distance
    '''
    
    dx=np.argsort(arr)[:X]
    return arr[dx]
    
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variables

# Version
version = "V3.1.1"

# Training data set path.
train_data_path = "/home/haddison/MPhys-Project/Candidate-Selection/V3/Data/HRD-Data/Training-Data/7-result.csv"

# Plot save location.
plot_save_path = "/home/haddison/MPhys-Project/Candidate-Selection/V3/Results/Component-Selection/"

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
# Silhouette Score

n_clusters=np.arange(2, 20)
sils=[]
sils_err=[]
iterations=10
for n in n_clusters:
    
    tmp_sil=[]
    for _ in range(iterations):
        
        gmm=GaussianMixture(n, n_init=2, max_iter=1000, 
                        covariance_type="full").fit(train_data) 
        labels=gmm.predict(train_data)
        sil=metrics.silhouette_score(train_data, labels, metric='euclidean')
        tmp_sil.append(sil)
       
    val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
    err=np.std(tmp_sil)
    sils.append(val)
    sils_err.append(err)

fig = plt.figure(figsize=(5, 5))
plt.errorbar(n_clusters, sils, yerr=sils_err)
plt.title("Silhouette Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("Number of Components")
plt.ylabel("Score")

plt.savefig("%sSilhouette_%s.png" %(plot_save_path, version), dpi=fig.dpi)