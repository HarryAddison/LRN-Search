'''
Author: Harry Addison
Date: 11/07/2022

License: MIT License

Overview:
        Functions used to perform a search for the next Galactic 
        luminous red nova.
        Full method is explained in the paper "Searching for the Next
        Galactic Luminous Red Nova, Harry Addison et. al" 
       (https://ui.adsabs.harvard.edu/abs/2022arXiv220607070A/abstract).
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing

from astroML.plotting.tools import draw_ellipse
from astropy.io import fits
from astropy.table import Table
from astroquery.gaia import Gaia
from extinction import fitzpatrick99
from datetime import datetime


def ap_abs_mag(ap_mag, plx=0, dist=0, type="plx"):
    '''
    Calculate the absolute magnitude from apparent magnitudes and 
    distance/parallax. 
    
    > If you want to use parallax:
        plx = array of parallax data 
        type="plx"
    
        Note: parallaxes must be given in milli-arcseconds (mas).
    
    > If you want to use distance:
        dist = array of distance data
        type="dist"
        
        Note: distances must be given in parsecs (pc).
    '''
    
    if type == "plx":
        abs_mag = ap_mag + 5 * np.log10(plx) - 10
    
    if type == "dist":
        abs_mag = ap_mag - 5 * np.log10(dist/10)
    
    return abs_mag
    

def a_g_to_a_v(a_g):
    '''
    Estimate the V-band extinction correlating to the Gaia G-band 
    extinction.
    
    Input G-band extinction must be in units of mag.
    '''
    
    # Array of V-band extinctions to convert and compare to the true G-band extinctions.
    a_v_trial = np.linspace(0.00, 10, 1000, endpoint=True)
    
    # Effective wavelength of the Gaia DR2 G-band.
    g_wave = np.array([6230.00]) #Angstrom
    
    # Converting the V-band extinctions to G-band.
    a_g_result = []
    
    for a_v in a_v_trial:
        a_g_result.append(fitzpatrick99(g_wave, a_v, 3.1, unit="aa"))
    
    # Comparing the the converted V-band extinctions to the true G-band extinctions.
    abs_diff = abs(np.array(a_g_result) - a_g)
    
    # Selecting the closest converted V-band extinction
    best_ind = np.argmin(abs_diff)
    a_v_best = a_v_trial[best_ind]
    
    return a_v_best
    
    
def redden_correct(data, num_proc):
    '''
    Correcting the Gaia G-RP colour for reddening effects.
    
    First calculates the reddening based on the G-band extinction.
    Then it corrects for the calculated reddening.
    
    The input data ("data") is a table consisting of the columns "a_g"
    and "g_rp".
    
    num_proc = number of processors to use for multiprocessing.
    '''
    
    p = multiprocessing.Pool(processes = num_proc)
    
    # Converting the G-band extinction to V-band extinctions
    a_v = p.starmap(a_g_to_a_v, [([data["a_g"][i]]) for i in range(len(data))])
    
    p.close()
    p.join()
    
    # Effective wavelength of the Gaia RP-band.
    rp_wave = np.array([7730.00])
    
    # Converting the V-band extinction to the RP-band.
    a_rp = []
    for i in a_v:
        a_rp.append(fitzpatrick99(rp_wave, i, 3.1, unit="aa")[0])
    
    # Calculating the reddening.
    a_g_rp = data["a_g"] - a_rp
    
    # Correcting the colour for the reddening.
    data["g_rp"] = data["g_rp"] - a_g_rp
    
    return data


def ms_def(mist_data_dir):
    '''
    Defining the oldest limit of the main sequence based on the MIST
    stellar evolution tracks.
    
    mist_data_dir = data directory containing the data of the MIST
    stellar evolution tracks.
    '''
    
    # Masses of the stars to be used.
    mass = ["001", "002", "003", "004", "005", "006", "007", "008", "009", 
            "010", "011", "012", "013", "014", "015", "016", "017", "018", 
            "019", "020", "021", "022", "023", "024", "025", "026"] 
    
    
    xs = []
    ys = []
        
    # Loading in the track data and finding the last data point of the main sequence phase.
    for i in range(len(mass)):
        
        data_path = ("%s%s0000M.track.eep.cmd" %(mist_data_dir, mass[i]))

        eeps_data = Table.read(data_path, format="ascii", header_start=14)
            
        x = []
        y = []
            
        for j in range(len(eeps_data)):
            if eeps_data["phase"][j] == 0:
                x.append(eeps_data['Gaia_G_DR2Rev'][j] - 
                            eeps_data['Gaia_RP_DR2Rev'][j])
            
                y.append(eeps_data['Gaia_G_DR2Rev'][j])
            
        xs.append(x[-1])
        ys.append(y[-1])
     
    # Fitting a 3rd order polynomial to the last data points of the Main sequence tracks.
    coeffs = np.polyfit(xs, ys, 3)
    
    return xs, ys, coeffs

    
def cand_select(data, model, thres_log_likeli, mist_data_dir):
    '''
    Selecting progenitor candidates
    
    Inputs:
    > data: Table consisting of the columns: "g_rp", "g_mag"
    > model: The produced Gaussian mixture model
    > thres_log_likeli: Threshold log likelihood values
    > mist_data_dir: directory containing the data for the MIST stellar
                     evolution tracks.
    '''
    
    #Formattigng the data required for "scoring" the data.
    data_format = np.array([data["g_rp"], data["g_mag"]]).transpose()
    
    # "Scoring" the full data set compared to the model.
    data["log_likeli"] = model.score_samples(data_format)
    
    # defining the MS edge using the MIST stellar evolution tracks
    xs, ys, coeffs = ms_def(mist_data_dir)
    
    # Selecting candidates based on thier score, and removing sources below MS and RGB.
    ms_flag = np.ones(len(data)) # 1 = right of MS, 0 = left of MS
    
    for i in range(len(data)):
        
        if data["g_rp"][i] < xs[0] and data["g_rp"][i] > xs[-1]:
            y_predict = (coeffs[0] * data["g_rp"][i]**3 + 
                         coeffs[1] * data["g_rp"][i]**2 + 
                         coeffs[2] * data["g_rp"][i] + 
                         coeffs[3])
            
            if data["g_mag"][i] >= y_predict:
                ms_flag[i] = 0
    
    cand_data = Table(names=data.colnames)
    
    for i in range(len(data)):   
        if (data["log_likeli"][i] <= thres_log_likeli 
            and data["g_mag"][i] <= 1.25 
            and not (data["g_rp"][i] >= 0.6 and data["g_mag"][i] >= -2)
            and data["g_rp"][i] <= 1 and not (data["g_rp"][i] <= xs[-1])
            and ms_flag[i] == 1):
            
            cand_data.add_row(data[i])
    
    return data, cand_data


def cone_search(data):
    '''
    Conduct a cone search of the Gaia on the given coordinates.
    
    data is a table containing the columns: "ra", "dec".
    '''
    
    complete_state = False
        
    while complete_state == False:
        try:
            # cone serach radius=2 arecseconds around the source at ra and dec.
            query = ('''
                     SELECT count(*)
                     FROM gaiaedr3.gaia_source
                     WHERE
                     CONTAINS(POINT('ICRS',gaiaedr3.gaia_source.ra,
                                    gaiaedr3.gaia_source.dec),
                     CIRCLE('ICRS', %s, %s, 0.0005555555555555556))=1
                     ''' %(data["ra"], data["dec"]))
            
            job = Gaia.launch_job_async(query, "source check", dump_to_file=False)
            results = job.get_results()
                
            num_sources = float(results["count_all"])
            
            complete_state = True
        
        except:
            print('''ERROR: Gaia query was not successful. 
                            Will retry the query.''')
            
    return num_sources
    
    
def neighbour_limitation(data, num_proc):
    '''
    Check to see if a progenitor source has any neighbours within 2 
    arcseconds. If so it is removed from the progenitor sample.
    This is nessassary check due to a limitation of ZTF where the survey
    cannot distinguish between close neighbours.
    
    data is a table of the progenitors' data.
    num_proc is the number of processors to be used for multiprocessing.
    '''
    
    p = multiprocessing.Pool(processes = num_proc)
    
    # Cone search on each progenitor retuning the number of sources within 2 arcseconds.
    num_sources = p.starmap(cone_search, 
                            [([data[i]]) for i in range(len(data))])
 
    p.close()
    p.join()
    
    # Removing progenitors that have another source within 2 arcseconds.
    cand_data = Table(names=data.colnames)
    
    for i in range(len(data)):
        if num_sources[i] == 1:
            cand_data.add_row(data[i])
        
    return cand_data
    

def plot_evo_tracks(ax, dir, linestyles=[], colors=[]):
    '''
    Plotting the MIST stellar evolution tracks.
    
    ax = axis to be plotted to.
    dir = directory containing the MIST stellar evolution track data.
    linestyles = line_styles to be used for plotting. Leave blank to use
                 the defaults.
    colors = colours to be used for plotting. LEave blank to use the
             defaults.
    '''
    
    # Range of masses to be plotted.
    mass = ["002", "003", "005", "007", "010"]
    # Labels for each track.
    mass_label = [r"$2 \,\rm{M_{\odot}}$", r"$3 \,\rm{M_{\odot}}$", 
                  r"$5 \,\rm{M_{\odot}}$", r"$7 \,\rm{M_{\odot}}$",
                  r"$10 \,\rm{M_{\odot}}$"]
    # Default linestyles
    if linestyles == []:
        linestyles = ["solid", "dashed", "dashdot", "dotted", (0, (5, 1))]
    # Default colours
    if colors == []:
        colors = ["purple", "orange", "black", "darkgreen", "red"]

    # Loading in the tracks and plotting the MS to RGB phases.
    for i in range(len(mass)):
        # Laoding the data.
        data_path = ("%s%s0000M.track.eep.cmd" %(dir, mass[i]))

        eeps_data = Table.read(data_path, format="ascii", header_start=14)
        
        # Selecting the MS to RGB phase
        x = []
        y = []
        reduced_phases = []
        
        for j in range(len(eeps_data)):
            if eeps_data["phase"][j] >= 0 and eeps_data["phase"][j] <= 3:
                x.append(eeps_data['Gaia_G_DR2Rev'][j] - eeps_data['Gaia_RP_DR2Rev'][j])
                y.append(eeps_data['Gaia_G_DR2Rev'][j])
                reduced_phases.append(eeps_data["phase"][j])
        
        # Plotting the track
        ax.plot(x, y, linestyle=linestyles[i], label=mass_label[i], c=colors[i])
                
        # Highlighting MS
        xs = []
        ys = []
        
        for k in range(len(reduced_phases)):
            if reduced_phases[k] == 0:
                xs.append(x[k])
                ys.append(y[k])
                
        if len(xs) > 0:
            if mass[i] == "010":
                ax.plot(xs, ys, color="Red", linewidth=4.0, alpha=0.5, label="Main Sequence")
            else:
                ax.plot(xs, ys, color="Red", linewidth=4.0, alpha=0.5)
                    
    # Shrink current axis's height by 12% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.12,
                        box.width, box.height * 0.85])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    
    return
    

def plot_cmd_likeli(data, model, thres_log_likeli):
    '''
    Plotting the CMD of the fitting data set with the log likelihood
    values of belonging to the model.
    
    data = fitting data set after fitting to the model.
    model = Gaussian mixture model.
    thres_log_likeli = threshold log likelihood value.
    '''
    
    # Creating fake data to plot contours on HRD.
    x = np.linspace(min(data["g_rp"]), max(data["g_rp"]), 100, endpoint=True)
    y = np.linspace(min(data["g_mag"]), max(data["g_mag"]), 100, endpoint=True)

    z = np.zeros((100,100))

    for i in range(100):
        fake_data = []
        for j in range(100):
            fake_data.append([x[i], y[j]])
        
        fake_data = np.array(fake_data)
        
        # fiiting the fake data to the model to produce log likelihood contours
        z[i] = model.score_samples(fake_data)

    z = np.transpose(z)

    # CMD of data set with scores.
    fig = plt.figure(figsize=(6, 5))

    plt.scatter(data["g_rp"], data["g_mag"], c=data["log_likeli"], 
                cmap="hot", s=2, lw=2, marker="o")

    plt.colorbar(label="Log Likelihood")

    contours = plt.contour(x,y,z, 
                          levels=np.array([-100.0,-50.0,-30.0,
                                           -20.0,-10.0, -5.0,0]),
                          colors="black", linestyles="solid")                      
    plt.clabel(contours, inline=True, fmt="%1.0f")

    thres_contours = plt.contour(x,y,z, levels=np.array([thres_log_likeli]), 
                                 colors="b",linestyles="solid")
    plt.clabel(thres_contours, inline=True, fmt="%1.2f")

    plt.xlabel("G-RP (mag)")
    
    plt.ylabel("Absolute G Magnitude (mag)")
    plt.ylim(top=1.5)
    plt.gca().invert_yaxis()
    
    plt.title("Candidate Threshold = %s" %(thres_log_likeli))
    
    return fig
 

def plot_cmd_train(data, mist_data_dir):
    '''
    Plotting CMD of the training data with density overplotted.
    Also plotting the MIST stellar evolution tracks. 
    
    data = training data set after corrections/processing.
    mist_data_dir = directory containing MIST data.
    '''
    
    # Histogram to find the density of the data.
    hist, xedges, yedges = np.histogram2d(data["g_rp"], data["g_mag"], 
                                          bins=300)

    xidx = np.clip(np.digitize(data["g_rp"], xedges), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(data["g_mag"], yedges), 0, hist.shape[1]-1)
    col = hist[xidx, yidx]

    # Scatter plot of g_mag mag vs g_rp, with a density map
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    max_col = max((col+10**-200)**(1/3))
    norm_col = (col+10**-200)**(1/3) / max_col

    im = plt.scatter(data["g_rp"], data["g_mag"], 
                     c=norm_col, cmap="hot", 
                     lw=0.5, s=0.5)


    cbar = plt.colorbar(im, ticks=[0, (250/max_col**3), (500/max_col**3), 
                                      (750/max_col**3), (1000/max_col**3), 
                                      (1250/max_col**3)], 
                                      label="Sources per Bin")
    cbar.ax.set_yticklabels(["0", "%d"%(250), 
                          "%d"%(500), "%d"%(750), "%d"%(1000), "%d"%(1250)])

    ax.set_xlabel("G-RP (mag)")
    ax.set_ylabel("Absolute G Magnitude (mag)")
    ax.invert_yaxis()
    
    plot_evo_tracks(ax, mist_data_dir, 
                        linestyles = ["solid", "dashed", "dashdot",
                                       (0, (5, 1)), "dotted"],
                        colors = ["tab:blue","tab:blue","tab:blue",
                                  "tab:blue","tab:blue"])
    
    return fig, ax
 
    
def plot_cmd_cand(data, mist_data_dir):
    '''
    Plotting CMD of the progenitor candidates.
    
    data = candidate data.
    mist_data_dir = directory of MIST data.
    '''
    
    # HRD of candidates with log likelihood values overplotted.
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    
    data.add_row([0,0,0,0,0,0,0,0,0,-20])
    
    im = ax.scatter(data["g_rp"], data["g_mag"], c=data["log_likeli"], 
                cmap="hot", s=2, lw=2, marker="o")

    plt.colorbar(im, ax=ax, label="Log Likelihood")

    ax.set_xlabel("G-RP (mag)")

    ax.set_ylabel("Absolute G Magnitude (mag)")
    ax.set_ylim(top=1.5)
    ax.invert_yaxis()
    
    plot_evo_tracks(ax, mist_data_dir)
        
    return fig, ax
    
    
def plot_model(train_data, model, num_comp):
    '''
    Plotting the Gaussian mixture model.
    
    train_data = training data set.
    model = constructed Gaussian mixture model.
    num_comp = number of components used in the model.
    '''
    
    # Computting 2d density for plot
    color_bins = 1000
    mag_bins = 1000

    H, color_bins, mag_bins = np.histogram2d(train_data["g_rp"], 
                                             train_data["g_mag"],
                                             (color_bins, mag_bins))

    Xgrid = np.array(list(map(np.ravel,
                              np.meshgrid(0.5 * (color_bins[:-1]
                                                 + color_bins[1:]),
                                          0.5 * (mag_bins[:-1]
                                                 + mag_bins[1:]))))).T

    log_dens = model.score_samples(Xgrid).reshape((1000, 1000))

    # Plotting.
    fig = plt.figure(figsize=(6, 6))

    im = plt.imshow((1.2**(log_dens)), 
                     origin='lower', interpolation='nearest', aspect='auto', 
                     extent=[color_bins[0], color_bins[-1],
                     mag_bins[0], mag_bins[-1]],
                     cmap="hot_r")

    # Change "scales" to 1 or 2. it represents sigma of the gaussian distribution
    plt.scatter(model.means_[:, 0], model.means_[:, 1], c='w') 
    
    for mu, C, w in zip(model.means_, model.covariances_, model.weights_):
        draw_ellipse(mu, C, scales=[1], fc='none', ec='k')

    plt.ylim(mag_bins[0], mag_bins[-1])

    plt.xlabel("G-RP (mag)")
    plt.ylabel("Absolute G Magnitude (mag)")
    plt.gca().invert_yaxis()

    plt.title("%s Component Model" %num_comp)

    plt.colorbar(im, label="Density")
        
    return fig