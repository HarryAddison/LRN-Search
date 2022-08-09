'''
Author: Harry Addison
Date: 11/07/2022

License: MIT License

Overview:
        Code used to perform a search for the next Galactic 
        luminous red nova.
        This code is used to collect and analyse the lightcurves of the 
        progenitor candidates found using "progenitor_selection.py". 
        This code produces a sample of candidate luminous red
        novae precursors for follow-up invesitgations. 
        Full method is explained in the paper "Searching for the Next
        Galactic Luminous Red Nova, Harry Addison et. al" 
       (https://ui.adsabs.harvard.edu/abs/2022arXiv220607070A/abstract).
'''


if __name__ == "__main__":

    import lrne_search_functions as lsf
    import numpy as np
    import matplotlib.pyplot as plt
    import multiprocessing
    
    from astropy.table import Table
    from datetime import datetime
    
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Recording the start time
    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
    # Reading in input variables from input_variables.txt
    with open("../Input/input_variables.txt") as f:
        lines = f.readlines()
        
        lc_fig_save = eval(lines[26].split("\n")[0])
        lc_fig_show = eval(lines[31].split("\n")[0])
        
        progen_data_path = ("%ssingle_progen_data_set.fits"
                            %str(lines[51].split("\n")[0]))
        error_file_path = str(lines[43].split("\n")[0])
        
        data_save_dir = str(lines[51].split("\n")[0])
        fig_save_dir = str(lines[55].split("\n")[0])
        
        window_width_percent = float(lines[71].split("\n")[0])
        outlier_thres = float(lines[75].split("\n")[0])
        mag_change_rate = float(lines[79].split("\n")[0])
        num_points_thres = int(lines[83].split("\n")[0])
        time_span_thres = float(lines[87].split("\n")[0])
        
        num_proc = int(lines[67].split("\n")[0])

    # Printing input variables to screen
    print(
    '''
    Initialisation time: %s
                    
    Input variables:
        > lc_fig_save: %s
        > lc_fig_show: %s
        > progen_data_path: %s
        > error_file_path: %s
        > data_save_dir: %s
        > fig_save_dir: %s
        > window_width_percent: %f
        > outlier_thres: %f
        > mag_change_rate: %f
        > num_points_thres: %d
        > time_span_thres: %f
        > num_proc: %d
    '''%(start_time,
         lc_fig_save, lc_fig_show,
         progen_data_path, error_file_path, 
         data_save_dir, fig_save_dir,
         window_width_percent, outlier_thres, mag_change_rate, 
         num_points_thres, time_span_thres,
         num_proc))
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Loading in the progenitor data
    progen_data = Table.read(progen_data_path)
    
    print(
    '''
    %s
    Progenitor data loaded
        > Number of Sources = %d
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(progen_data)))
    
    # Collecting, processing, and analysing progenitor ZTF g, i, and r band lightcurves.
    
    print(
    '''
    %s
    Initialising collection, processing, and analysis of the 
    progenitor's ZTF g, i, and r band lightcurves.
    
    This process can take a few minutes per source, resulting in
    a long compute time. 
    
    Please be patient.
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    
    bands = ["g", "i", "r"]
    
    for band in bands:
        
        p = multiprocessing.Pool(processes = num_proc)
        
        results = p.starmap(lsf.precursor_selection,[(data, band, 
                                                      error_file_path, 
                                                      data_save_dir, 
                                                      window_width_percent, 
                                                      outlier_thres, 
                                                      mag_change_rate,
                                                      num_points_thres,
                                                      time_span_thres)
                                                      for data in progen_data])
       
        p.close()
        p.join()
        
        lc_stage_flag, sum_diff, grad_sum_diff = np.transpose(results)
        
        progen_data["%s-band_candidate_flag"%band] = np.array(lc_stage_flag, 
                                                              dtype="U")
        
        progen_data["%s-band_sum_diff"%band] = np.array(sum_diff, dtype=float)
        
        progen_data["%s-band_grad_sum_diff"%band] = np.array(grad_sum_diff, 
                                                             dtype=float)
        
        
    print(
    '''
    %s
    ZTF lightcurves of progenitors collected, processed, and analysed.
    
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))        
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Plotting of lightcurves
    if lc_fig_save == True or lc_fig_show == True:
        
        for source_data in progen_data:
            
            avail_bands = []
            avail_lc_data = []
            # check if lightcurve data is available per band
            for band in bands:
                lc_stage_flag = source_data["%s-band_candidate_flag"%band]
                
                if (lc_stage_flag == "False: Insufficient brightening" or
                    lc_stage_flag == "True"):
                
                    avail_bands.append(band)
                    
                    lc_path = ("%sZTF-LC-Data/%s_band_%d.fits"%(data_save_dir, 
                                band, int(source_data["source_id"])))
                    
                    avail_lc_data.append(Table.read(lc_path))
                    
            if len(avail_bands) != 0:
                fig, ax = lsf.plot_lc(avail_lc_data, avail_bands)
                
                if lc_fig_save == True:
                    plt.savefig("%s/ZTF-LC/LC_Gaia_id_%d.png"%(fig_save_dir,
                                 int(source_data["source_id"])), 
                                 dpi=fig.dpi)
        
                if lc_fig_show == True:
                    plt.show()
                
                plt.close()
                
        print(
    '''
    %s
    ZTF lightcurves of progenitors plotted.
    
    ''' %(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # Saving data
    
    # Making a table of just precursor candidates.
    precur_data = Table(names=progen_data.colnames, dtype=progen_data.dtype)
    
    for data in progen_data:
        if (data["g-band_candidate_flag"] == "True" 
            or data["i-band_candidate_flag"] == "True"
            or data["r-band_candidate_flag"] == "True"):
            
            precur_data.add_row(data)
   
    # Saving the data
    progen_data.write("%ssingle_progen_data_set_lc_analysis.fits"%(data_save_dir), 
                        format="fits", overwrite=True)
    
    precur_data.write("%sprecur_cand_data_set.fits"%(data_save_dir), 
                        format="fits", overwrite=True)

    print(
    '''
    %s
    All data saved.
    
    Number of precursor candidates: %s
    '''
    %(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(precur_data)))