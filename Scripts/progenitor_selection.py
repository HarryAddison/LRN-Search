'''
Author: Harry Addison
Date: 11/07/2022

License: MIT License

Overview:
        Code used to perform a search for the next Galactic 
        luminous red nova.
        This code is used to produce a sample of candidate luminous red
        novae progenitors.        
        Full method is explained in the paper "Searching for the Next
        Galactic Luminous Red Nova, Harry Addison et. al" 
       (https://ui.adsabs.harvard.edu/abs/2022arXiv220607070A/abstract).
'''


if __name__ == "__main__":
    
    import lrne_search_functions as lsf
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn.mixture import GaussianMixture
    from datetime import datetime
    from astropy.table import Table
    from extinction import fitzpatrick99
    
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Recording the start time
    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Reading in input variables from input_variables.txt
    with open("../Input/input_variables.txt") as f:
        lines = f.readlines()
        
        fig_save = eval(lines[16].split("\n")[0])
        fig_show = eval(lines[21].split("\n")[0])
        
        train_data_path = str(lines[35].split("\n")[0])
        fit_data_path = str(lines[39].split("\n")[0])
        mist_data_dir = str(lines[47].split("\n")[0])
        
        data_save_dir = str(lines[51].split("\n")[0])
        fig_save_dir = str(lines[55].split("\n")[0])
        
        num_comp = int(lines[59].split("\n")[0])
        thres_log_likeli = float(lines[63].split("\n")[0])
        num_proc = int(lines[67].split("\n")[0])
    
    # Printing input variables to screen
    print(
    '''
    Initialisation time: %s
                    
    Input variables:
        > fig_save: %s
        > fig_show: %s
        > train_data_path: %s
        > fit_data_path: %s
        > mist_data_dir: %s
        > data_save_dir: %s
        > fig_save_dir: %s
        > num_comp: %d
        > thres_log_likeli: %f
        > num_proc: %d
    '''%(start_time, 
         fig_save, fig_show, 
         train_data_path, fit_data_path, mist_data_dir, 
         data_save_dir, fig_save_dir,
         num_comp, thres_log_likeli, num_proc))
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Loading in the training data.
    train_data = Table.read(train_data_path)
    
    # Applying nessassary corrections due to known issues with Gaia DR2 data.
    for i in range(len(train_data)):
        if train_data["g_mag"][i] > 2 and train_data["g_mag"][i] <= 6:
            train_data["g_mag"][i] = (-0.047344 + 
                                      1.16405 * train_data["g_mag"][i] - 
                                      0.046799 * train_data["g_mag"][i]**2 + 
                                      0.0035015 * train_data["g_mag"][i]**3)
    
        if train_data["g_mag"][i] > 6 and train_data["g_mag"][i] <=16:
            train_data["g_mag"][i] = (train_data["g_mag"][i] - 
                                     0.0032 * (train_data["g_mag"][i] -6))
                                    
        if train_data["g_mag"][i] > 16:
            train_data["g_mag"][i] = train_data["g_mag"][i]- 0.032
 
    # Applying extinction correction to training data "g_mag".
    train_data["g_mag"] = train_data["g_mag"] - train_data["a_g"]
    
    # Applying reddening correction to training data "g_rp".
    data = lsf.redden_correct(train_data, num_proc)
    
    # Converting apparent g magnitude (g_mag) to absolutue magnitudes using the parallax.
    train_data["g_mag"] = lsf.ap_abs_mag(train_data["g_mag"], 
                                         plx=train_data["plx"],
                                         type="plx")

    # Removing data from unwanted parameter space.
    temp_data = Table(names=train_data.colnames)
    
    for i in range(len(train_data)):
        if train_data["g_rp"][i] < 1.6 and train_data["g_mag"][i] < 1.5:
            temp_data.add_row(train_data[i])
    
    train_data = temp_data
    
    print(
    '''
    %s
    Training data loaded:
        > Number of Sources = %d
    
    Corrections to training data set applied:
        > Corrections to the raw data due to known issues
        > De-reddening/de-extinction
        > Conversion of apparent to absolute magnitudes
    '''%(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(train_data)))
    
    # Plotting the training data
    fig, ax = lsf.plot_cmd_train(train_data, mist_data_dir)
    
    if fig_save == True:
        plt.savefig("%sCMD_Training_Data.png" %(fig_save_dir), 
                    dpi=fig.dpi)

    if fig_show == True:
        plt.show()
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # formatting the training data so it can be used in the GasussianMixture function.
    train_data_gmm = np.array([train_data["g_rp"], 
                               train_data["g_mag"]]).transpose()
    
    # Constructing the Gaussian mixture model.
    model = GaussianMixture(n_components=num_comp, max_iter=1000, 
                            covariance_type="full")
    model.fit(train_data_gmm)
    
    print(
    '''
    %s
    Model constructed:
        > Number of components = %d
    '''%(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), num_comp))
    
    # Plotting the model.
    lsf.plot_model(train_data, model, num_comp)
    
    if fig_save == True:
        plt.savefig("%sModel.png" %(fig_save_dir), dpi=fig.dpi)
    
    if fig_show == True:
        plt.show()
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Loading the fitting data set
    fit_data = Table.read(fit_data_path)
    
    # Converting apparent to absoulte magnitudes.
    fit_data["g_mag"] = lsf.ap_abs_mag(fit_data["g_mag"], 
                                       dist=fit_data["dist"], 
                                       type="dist")
                                       
    print(
    '''
    %s
    Fitting data loaded
        > Number of Sources = %d
    
    Corrections applied:
        > Conversion of apparent to absolute magnitudes
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(fit_data)))
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Selecting Hertzsprung gap candidates
    fit_data, cand_data = lsf.cand_select(fit_data, model, 
                                          thres_log_likeli, mist_data_dir)
    
    print(
    '''
    %s
    Fitting data fitted to the model.
    Progenitor candidates selected.
        > Number of progenitor candidates: %d
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(cand_data)))
    
    # Plotting CMD of the fitting data set with the log likelihoods of belonging to the model
    fig = lsf.plot_cmd_likeli(fit_data, model, thres_log_likeli)
    
    if fig_save == True:
        plt.savefig("%sCMD_Scores.png" %(fig_save_dir), dpi=fig.dpi)
    
    if fig_show == True:
        plt.show()
        
    # Plotting CMD of the progenitor candidates with log likelihood values.
    fig, ax = lsf.plot_cmd_cand(cand_data, mist_data_dir)
    
    if fig_save == True:
        plt.savefig("%sCMD_Progenitors.png" %(fig_save_dir), dpi=fig.dpi)
    
    if fig_show == True:
        plt.show()
        
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    print(
    '''
    %s
    Initialising Gaia cone searches
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    
    # Converting positions from the gaia epoch to J2000.
    cand_data["ra"] = (cand_data["ra"] - 
                       ((cand_data["epoch"] - 2000) * 
                        (cand_data["pmra"]/3600) * 
                         10**-3))
    
    cand_data["dec"] = (cand_data["dec"] - 
                       ((cand_data["epoch"] - 2000) * 
                        (cand_data["pmdec"]/3600) * 
                         10**-3))
                         
    # Removing candidates if they have another source within 2 arcsec. (Limitation due to ZTF)
    ztf_cand_data = lsf.neighbour_limitation(cand_data, num_proc)
    
    print(
    '''
    %s
    Completed Gaia cone searches.
        > Candidates with no source within 2": %d
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(ztf_cand_data)))
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Saving data sets
    # Training data with corrections
    train_data.write("%strain_data_set_corrections.fits"%(data_save_dir), 
                     format="fits", overwrite=True)
        
    fit_data.write("%sfit_data_set_altered.fits"%(data_save_dir), 
                   format="fits", overwrite=True)
    
    cand_data.write("%sprogen_data_set.fits"%(data_save_dir), 
                    format="fits", overwrite=True)
    
    ztf_cand_data.write("%ssingle_progen_data_set.fits"%(data_save_dir), 
                        format="fits", overwrite=True)
                        
    print(
    '''
    %s
    All data saved.
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))