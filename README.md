# Event Selection at the Galacic Center Region with IceCube 
This project is an event selection on level 3 tack events at the Galactic Center region (GC declination +- 10 degrees) using boosted decison tree with classification. 
The model indents to select well-reconstructed events and reduce background rate. Simulated Monte Carlo (MC) signal events and real collected data are used as 
signal and background respectively as training data.

## Description
The burn sample uses 24 days of data and 3000 MC events. The energy spectrum of astrophysical neutrinos can be modeled by a power law and the MC is weighted to E^-2.7 
power law. 

A pre-training selection is done to get rid of poorly reconstructed events. The notebook Prepare_Datasets.ipynb is to do this preselection for the burn sample. 
The precuts include:

- Declination filter: only keep events within GC dec +- 10 degrees
- Keep events with successful SplineMPE reconstruction
- rlogl < 9
- Length of track > 250m
- Angular resolution < 4.5 degrees
- Keep events with successful Millipede energy reconstruction
- (Optional) Keep non-corner-clipper events

After precuts, BDT_Train.ipynb trains different models and compare them. XGBoost stands out and is used in later selection. 

Precuts and the trained XGBoost model are included in BDT.py to select on 24 days of data and 4000 MC files. A BDT cut is done to keep the data rate at 90.7mHz.
This BDT cut gets rid of 81% of background and keeps 74% of signal after the precuts. Finally, the sensitivity of the selected data is tested and a sensitivity 
of 5.672e-10 TeV/cm2/s at 100 TeV is observed.

## Dependencies and libraries
- python 3.7.5
- XGBoost 1.3.3
- LightGBM 3.1.1
- sklearn 0.24.1
- pandas 0.23.4
- numpy 1.91.2
