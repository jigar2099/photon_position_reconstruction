# Photon Position Reconstruction
## Introduction
This repository is created in order to predict position of photons from given data.
Intensity Interferometry is the branch of Astronomy which aims to calculate diameter of Stars.
It's experimental setup includes two Photo Multiplier Tubes (PMTs) which detects photons for selected in the form of photo current( depending on selected time binnings ). Thus, the generated data from two PMTs represent the photon emission pattern from the source with respect to the time. Overall shapes in generated data, from both PMTs, can be different depending on the experimental setup and the used PMTs. However, for the case of the correlated photons, location of the photon pulse is expected to be the same.

This project is the part of research going on in the Erlangen Center of Astroparticlephysics(ECAP), and the data used in this project is from H.E.S.S. campaign-I. It is also extention of my masters thesis, in which I attempted to 
<a href="https://dl.gi.de/handle/20.500.12116/39542" target="_blank">calculated the flux from PMTs using Deep Learning</a>.

## Dataset

There are two types of data generated from the experimental run.
1. Calibration data
2. Actual measurements

The calibration data is basically very low rate measurements. It is useful to extract out the photon pulse shapes (complete). After extracting the pulse shapes, it is important to do EDA (exploratory data analysis), in order to figure out the unexpected photon pulse shapes, which can be discarded. Now, the set of shapes might have identical shapes. Depending on the selection one can remove or keep identical shapes (here identical shapes are removed). The resulting set of shapes can be used to create Monte Carlo datasets for Neural Networks(NNs).

### Monte Carlo (MC) simulations

The simulation focuses on addition of photon pulse (complete of partial), rather than first selecting rate (in MHz) and then determining number of pulses to be added. It also considers the scenario where pulses are partial ( at the edge of the sample). Following are the key points for using MC.

1. Single channel MC simulation :

    - First requirement is to select the desired sample size (here 256).
    - One needs to select range of rates (number of pulses to be added) in samples. For example, if selected range is from 0 to 100, then the simulation will generate a dataset, and in this dataset one can find samples with pulses ranging from 0 to 100.
    - Then one needs to mention number of example samples to be generated for each selected rate. This creates balanced dataset, and helps to avoide biased scenario for training.
    - Finally, in the output, MC provides two types of data
        1. Input data: consists of added pulses
        2. Lable data: consists number of pulses added, for example fully added pulse/s are represented by integer values, while float value indicates sliced pulses, which are mainly cases closer to the edges of the sample.
    - A single run of simulation produces single dataset. Therefore, one needs to use it three times for training data, testing data and validation data.
2. Double channel MC simulation : This simulation is created in order to mimic the data geretated from the two PMTs used in the experiment. It generates single channel data first, and then in second channel data generation, it randomly determines correlated pulse positions in both channels and add the pulse into second channel data, while rest of the pulse positions could be uncorrelated. In this way, it applies the consideration of correlation from entire range of selected rates.

## How to use
1. clone the repository
2. add calibration file to src/data/ 
3. go to notebooks folder and use EDA_data_gen.ipynb for data analysis and desired data generation, which creates different datasets and saves them into DATASETS/sh06x2/
4. in the same folder run.ipynb can be used for data loading and model training. It saves model in models/sh06x2/, and tranining performance in model_performance/sh06x2/.