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
from sklearn.model_selection import train_test_split


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
#Train - Test ditance check

def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)



n_clusters=np.arange(2, 20)
iterations=20
results=[]
res_sigs=[]
for n in n_clusters:
    dist=[]
    
    for iteration in range(iterations):
        train, test=train_test_split(train_data, test_size=0.5)
        
        gmm_train=GaussianMixture(n, n_init=2).fit(train) 
        gmm_test=GaussianMixture(n, n_init=2).fit(test) 
        dist.append(gmm_js(gmm_train, gmm_test))
    selec=SelBest(np.array(dist), int(iterations/5))
    result=np.mean(selec)
    res_sig=np.std(selec)
    results.append(result)
    res_sigs.append(res_sig)
    
fig = plt.figure(figsize=(5, 5))
plt.errorbar(n_clusters, results, yerr=res_sigs)
plt.title("Distance Test Results", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("Number of Components")
plt.ylabel("Distance")

plt.savefig("%sDistance_%s.png" %(plot_save_path, version), dpi=fig.dpi)