# Brain network study during resting states

## Bioinformatics project @Sapienza Univeristy of Rome, academic year: 2019/2020

### Authors: Francesco Russo, Iason Tsardanidis, Michele Cernigliaro

![alt text](https://github.com/cerniello/Neuroscience_EEG_Graph_Analysis/blob/master/electroencephalogram_eeg.jpg)


This study takes its foundations in the field of network neuroscience. More specifically, we are interested in the analysis of the electrical activity of the brain, recorded through Electroencephalography (EEG), using the mathematical formalism provided by the graph theory. The aim of this paper is to infer functional connectivity (causal and temporal relation) among different cerebral sites, monitored through 64 EEG channels located uniformly along the scalp. The analysis is performed during two different resting states (open and closed eyes). It is shown that brain areas which are not necessarily connected by physical links can have functional connections. We used the EEG signal recorded on one patient selected among 109 test subjects from PhysioNet's dataset "EEG Motor Movement/Imagery Dataset", focusing our attention on the first two runs corresponding to eyes open (EO) and eyes closed (EC) resting states.


### Material in this repository 

* Bioinformatics_proj1(neuro)_ay1920: PDF with the related Guidelines
* data/: Folder containing
  * EEG signal during resting states of patient nr. 4 in .edf format ('S004R01.edf', 'S004R02.edf')
  * channel_locations.txt: channels mapping
  * Other .txt files: files related to motifs analysis (mfinder format)
* BioInformatics - Project_1 - Neuroscience.ipynb: jupyter notebook with all the runs for this project
* Neuroscience_Project_modules.py: Python file containing the class and the functions implemented for this project
* BioInformatics - Project_1 - Neuroscience.html: html file of the jupyter notebook runned and executed 
