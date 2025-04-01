# Correct-TDS




## With the pip install package :

- Install pyOpt from https://github.com/madebr/pyOpt (optional)

- Install anaconda (https://docs.anaconda.com/anaconda/install/index.html)

- Create an environment

- Open the environment terminal

- To install the package, write in the command line :

	```
	pip install Correct-TDS
	```
- To run Correct@TDS, write in the command line :

	```
 	Correct-TDS-interface
	```
- The gui can freeze while computing, it's normal, it's still in development

## Manual instalation

- Install anaconda (https://docs.anaconda.com/anaconda/install/index.html)

- Install mpi4py with the command

	```
	conda install -c conda-forge mpi4py
	```
- Install pyswarm with the command

	```
	pip install pyswarm
	```
	
- Install numba with the command

	```
	pip install numba
	```	
- Install pyOpt from https://github.com/madebr/pyOpt (optional)

- To apply Tuckey window with alpha= 0.05, modify "apply_window = 0" to "apply_window = 1" in interface.py
	
- To run Correct@TDS, write in the command line:

	```
	python interface.py
	```
- The gui can freeze while computing, it's normal, it's still in development

## Usage

A THz software that performs statistical analysis and error correction on .h5 files obtained with TeraHertz Time Domain Spectroscopy (THz-TDS).
Input file should contains :
 - One dataset with label "timeaxis", containing the sampling times in picoseconds
 - One dataset for each time signal to correct, labeled with their index starting from 0


The optimization process can also be launched from a python code as shown in the example below:

	import Correct_TDS

	opti = Correct_TDS.Optimization()


	#Compulsory initialization step:
	opti.setData(toCorrect)																			#indicates the list of pulses to correct
	opti.setReference(ref)																			#reference signal
	opti.setTimeAxis(t)																				#time axis
	opti.setParameters(fitDelay = True, delayLimit = 1e-14, fitAmplitude=True, amplitudeLimit=1)	#Optimisation choices

	#Launch optimization
	opti.optimize()

	#Get the corrected signals:
	corrected = opti.getCorrectedTraces()
	#Get the correction parameters as a dictionary
	parameters = opti.getCorrectionParameters()



## Related software
Below is a list of other software projects that address related tasks in
THz-TDS analysis, with summaries taken from the project documentation.
- https://github.com/THzbiophotonics/Fit-TDS
  - "Python code aiming at the retrieving of material parameter from a
    TeraHertz time domain spectroscopy (TDS) measurements from a fit in the time
    domain."
- https://github.com/puls-lab/phoeniks
  - "A free and open-source (FOSS) Python class to extract the refractive index
    and absorption coefficient from time traces of a THz Time Domain
    Spectrometer (THz-TDS)."
- https://github.com/dotTHzTAG/CaTSper
  - "The CaTSper tool extracts the frequency-dependent optical constants from
    terahertz time-domain waveforms." (uses MATLAB)
- https://github.com/YaleTHz/nelly
  - "Nelly is a package for extracing refractice indices and conductivities from
    time-domain terahertz spectroscopy data." (uses MATLAB)
- https://github.com/dodge-research-group/thztools
  - "THzTools is an open-source Python package for data analysis in
    terahertz time-domain spectroscopy (THz-TDS)."
