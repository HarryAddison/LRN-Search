# LRN-Search
Code created for the purpose of searching for the next Galactic luminous red novae. Associated with research paper: https://ui.adsabs.harvard.edu/abs/2022arXiv220607070A/abstract


## About the Project
The aim of the project was to identify luninous red nova (LRN) progenitors using observational data from the Gaia mission and data from time-domain surveys such as the Zwicky Transient Facility (ZTF). To achieve this we began by creating a sample of progenitor systems. As the progenitors of LRNe are thought to be binary systems with a yellow giant/yellow super giant primary component, we searched for Hertzsprung gap stars. Our method models the observational CMD constructed from Gaia DR2 and then fits a data set from Gaia EDR3, selecting stars that exist within the gap as LRNe progenitors. We then obtain and analyse the ZTF lightcurves of these progenitors, applying a slow transient detection method to detect any lightcurves that exhibit a slow increase in brightness that one would expect from a LRNe progenitor. In our paper we conduct follow-up investigations into our most likely LRNe precursor candidates, discussing their possible natures.


## Code Details
### Scripts
> lrne_search_functions.py:
      Contains the functions used in all other scripts

> progenitor_selection.py:
      Contains the code used to model the observational CMD and select LRNe progenitors.

> precursor_selection.py: 
      Contains the code used to collect and analyse the progenitors' lightcurve data, and the selection of LRNe precursors.

## Inputs
> Input Variables
        File Name: input_variables.txt
        Contains: List of input variables such as file paths and directories that correspond to where data should be read from/written to. You may change the variables                   within this file.
        
> Training Data Set:
      File Name: train_data_set.fits
      Contains: Observational data from Gaia DR2 used to train the Gaussian mixture model. This data is provided in the repositry, but if you wish to construct your                   own data set it must consist of the specific columns as outlined below:
                Column Name : Descrition 
                > g_mag : Gaia G band apparent magnitude in units of mag.
                > g_rp : Gaia G-RP colour in units of mag.
                > plx : Gaia parallax in units of milli-arcesconds (mas).
                > a_g : Gaia G band extinction in units of mag.

> Fitting Data Set:
      File Name: fit_data_set.fits
      Contains: Observational data from Gaia EDR3 used to select the progenitor systems from. This data is provided in the repositry, but if you wish to construct your                 own data set it must consist of the specific columns as outlined below:
                Column Name : Description
                > g_mag : Gaia G band apparent magnitude in units of mag.
                > g_rp : Gaia G-RP colour in units of mag.
                > dist : Gaia Bailer-Jones distances in units of parsecs.
                > 

> Stellar Evolution Tracks:
      Folder Name: Stellar-Evo-Data
      Contains: Stellar evolutionary tracks from MIST.
      
> 

## File Structure
> Project 
      >> Input
            >>> input_variables.txt
            >>> Data
                   >>>> Gaia-Data
                          >>>>> train_data_set.fits
                          >>>>> fit_data_set.fits
                   >>>> Stellar-Evo-Data
                            >>>>> MIST Stellar Evolutionary Track files
      >> Output
            >>> Data
                   >>>> files output by the code
            >>> Figures
                   >>>> saved figures
      >> Scripts
            >>> lrne_search_functions.py
            >>> progenitor_selection.py
            >>> Model-Parameter-Selection
                   >>>> 

## Dependencies
This list of dependicies does not include their respective dependicies. For information on the dependicies of these packages please see thier respective documentations. It should also be noted that the versions listed have been shown to work, however, other versions of these packages may also work.

> astroML 
> astropy
> astroquery
> datetime
> extinction
> matplotlib
> multiprocessing
> numpy
> sklearn


## Known Problems/Issues
> ZTF lightcurve collection is slow.
      This is a limitation of the ZTF servers.
