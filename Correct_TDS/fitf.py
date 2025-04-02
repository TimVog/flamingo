#!/usr/bin/python
# -*- coding: latin-1 -*-

## This two lines is to chose the econding
# =============================================================================
# Standard Python modules
# =============================================================================
import numpy as np
from numpy.fft import rfft
from numpy.fft import irfft
import h5py
import warnings
import os, time
import pickle
from pyswarm import pso   ## Library for optimization
import scipy.optimize as optimize  ## Library for optimization
from scipy import signal
import json

###############################################################################

j = 1j

###############################################################################
## Parallelization that requieres mpi4py to be installed, if mpi4py was not installed successfully comment frome line 32 to line 40 (included)
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    size = comm.Get_size()
except:
    print('mpi4py is required for parallelization')
    myrank=0


# =============================================================================
# classes we will use
# =============================================================================

class globalparameters: 
    def __init__(self, t = None, freq = None, w = None):
        self.t = t
        self.freq = freq
        self.w = w

# =============================================================================

class inputdatafromfile: 
    def __init__(self, path, trace_start, trace_end, time_start, time_end, sample = 1):
        # sample = 1 if it is a sample and 0 if not and we want to compute the covariance with
        # trace_start, trace_end for the number of traces to study
        with h5py.File(path, "r") as f:
            self.time = np.array(f["timeaxis"])
            if trace_end == -1:
                self.numberOfTrace = len(f)-1 # on enleve timeaxis
                trace_end = len(f)-2
            else:
                self.numberOfTrace = trace_end-trace_start+1

            self.Pulseinit = [np.array(f[str(trace)]) for trace in range(trace_start, trace_end+1)] #list of all time trace without the ref, traces are ordered
            if sample:
                self.ref_number = self.choose_ref_number()
            else:
                self.ref_number = None
            self.timestamp = None
            try:
                self.timestamp = [f[str(trace)].attrs["TIMESTAMP"] for trace in range(trace_start, trace_end+1) ]
            except:
                pass
    
    def mean(array):
        moyenne = np.zeros(len(array[-1]))
        for i in array:
            moyenne = moyenne+ i
        moyenne = moyenne/len(array)
        return moyenne
    
    def choose_ref_number(self):
        norm = []
        pseudo_norm = []
        temp = inputdatafromfile.mean(self.Pulseinit)
        
        # I added the "mean" function to inputdatafromfile so now we use "inputdatafromfile.mean" instead of just "mean"

        for i in range(self.numberOfTrace):
            pseudo_norm.append(np.dot(self.Pulseinit[i], temp))
            
        for i in range(self.numberOfTrace):
            norm.append(np.dot(self.Pulseinit[i], self.Pulseinit[i]))
        proximity = np.array(pseudo_norm)/np.array(norm)

        return np.abs(proximity - 1).argmin()  #proximite la plus proche de 1
    

class ReferenceData:
    def __init__(self, *args):
        #Initialistion can be done from a file, or directly from a time trace
        self.Pulseinit = None
        self.SpulseInit = None
        if len(args) != 0:          
            if type(args[0])==str and len(args) == 4:
                path, ref_number, trace_start, time_start = args       
                with h5py.File(path, "r") as f:
                    self.Pulseinit = np.array(f[str(trace_start+ref_number)])
            elif type(args[0])==np.ndarray:
                self.Pulseinit = args[0].copy()
            else:
                raise TypeError("Invalid arguments")
        
        if self.Pulseinit is not None:
            self.Spulseinit = rfft(self.Pulseinit)  ## We compute the spectrum of the measured pulse
            
    def setFromFile(self, path, ref_number, trace_start, time_start):
        with h5py.File(path, "r") as f:
            self.Pulseinit = np.array(f[str(trace_start+ref_number)])
        self.Spulseinit = rfft(self.Pulseinit)
    
    def setPulse(self, pulse):
        self.Pulseinit = pulse.copy()
        self.Spulseinit = rfft(self.Pulseinit)
        
    def setSpectrum(self, spectrum, n = None):
        self.Spulseinit = spectrum.copy()
        if n is not None:
            self.Pulseinit = irfft(spectrum, n)
        else:
            self.Pulseinit = irfft(spectrum)
        
        
    def transferfunction(self, myinputdata):
        if self.Pulseinit is not None:
            return self.Spulseinit/myinputdata.Spulse
        else:
            raise Exception("Reference pulse was not initialized")
        


# =============================================================================

class mydata:
    def __init__(self, pulse):
        self.pulse = pulse.copy() # pulse
        self.Spulse = rfft((pulse)) # spectral field
        
        
        
class datalist:
    def __init__(self, pulseList = []):
        self.pulse = pulseList.copy()
        self.mean = []
        self.time_std = []
        self.freq_std = []
        self.freq_std_with_window = []
        self.freq_std_to_save = []  #if superresolution save the std of the cutted traces as output

        self.covariance = None
        self.covariance_inverse = None

        
    def add_trace(self, pulse):
        self.pulse.append(pulse) # pulse with sample
        
        






#### Classes for callbacks

class Callback_bfgs(object):
    def __init__(self,monerreur):
        self.nit = 0
        self.monerreur=monerreur
        
    def __call__(self, par, convergence=0):
        self.nit += 1
        with open('algo_bfgs_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, self.monerreur(par), par))
            
class Callback_slsqp(object):
    def __init__(self,monerreur):
        self.nit = 0
        self.monerreur=monerreur
        
    def __call__(self, par, convergence=0):
        self.nit += 1
        with open('algo_slsqp_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, self.monerreur(par), par))


class Callback_annealing(object):
    def __init__(self,monerreur):
        self.nit = 0
        self.monerreur=monerreur
        
    def __call__(self, par, f, context):
        self.nit += 1
        with open('algo_dualannealing_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, self.monerreur(par), par))




class Optimization():
    """
    Optimisation tool to correct delay variation, amplitude variation and periodic sampling error in a serie of repeated measurement
    Use method setData, setReference, setTimeAxis and setParameters to initialize the object
    Use method optimize to run the optimisation
    Recover the corrected traces and correction parameters with getCorrectedTraces and getCorrectionParameters
    For citation and detailed exmplanation, see : E. Denakpo, T. Hannotte, N. Osseiran, F. Orieux and R. Peretti, "Signal estimation and uncertainties extraction in terahertz time-domain spectroscopy," in IEEE Transactions on Instrumentation and Measurement, doi: 10.1109/TIM.2025.3554287
    """
    def __init__(self):
        
        #Storing all the variables which were in temp files, in majority used to optimize
        

        self.reference_number = None
        self.fit_dilatation = None
        self.dilatation_limit = None
        self.dilatationmax_guess = None
        self.fit_delay = None
        self.delaymax_guess = None
        self.delay_limit = None
        self.nsample = None
        self.fit_periodic_sampling = None
        self.periodic_sampling_freq_limit = None
        self.fit_leftover_noise = None
        self.leftcoef_guess = None
        self.leftcoef_limit = None
              
        self.datacorrection = None
        self.fopt = None
        self.result = None
        
        
        self.algo = 6
        self.swarmsize = None
        self.maxiter = None
        self.maxiter_ps = None
        

        self.data = None
        self.ref = None
        self.globalparameters = None
        self.apply_window = None
        
        self.myreferencedata=None
        self.minval=None
        self.maxval=None
        self.myinputdata=None
        self.myglobalparameters=None
        self.dt=None
        self.mymean=None
        
        self.interrupt=False #Becoming True when break button pressed
    
    def setData(self, pulses):
        """
        Compulsory before using method optimize.
        Define the list of signal to correct.
        Signal should be real numbers.
        
        Parameters:
        ----------
        pulses: array like 2d
            list of time domain measurement to correct.
        
        """
        self.data = datalist(pulses)
        
    def setReference(self, referenceTrace, referenceNumber=None): #if the reference is also part of the input dataset, indicate its index in the input list with referenceNumber
        """
        Compulsory before using method optimize.
        Define a reference signal for delay and amplitude correction.
        
        Parameters:
        ----------
        pulses: array like 1d
            Single signal to use as a reference
        
        referenceNumber: int, optional
            if the reference is part of the input dataset, indicate its index in the list.
        """
        self.ref = ReferenceData(referenceTrace)
        self.reference_number = referenceNumber

        
    def setGlobalParameters(self, globalParameters):
        self.globalParameters = globalParameters
        self.dt = globalParameters.t[1] -globalParameters.t[0]
        self.nsample = len(globalParameters.t)
    
    def setTimeAxis(self, t):
        """
        Compulsory before using method optimize.
        Define the time axis, assumed to be uniform and common for all measurement in the input dataset
        
        Parameters:
        ----------
        t: array like 1d
            Uniformly sampled time value (in seconds)
        
        """
        f = np.fft.rfftfreq(len(t), t[1]-t[0])
        w = f*2*np.pi
        self.globalparameters = globalparameters(t,f,w)
        self.dt = t[1]-t[0]
        self.nsample = len(t)
        
    def setParameters(self, fitDelay = False, delayLimit = 1e-12, fitAmplitude = False, amplitudeLimit = 0.1, fitPeriodicSampling = False, periodicSamplingFreqLimit = 7e12, maxIter = 1000, maxIterPS = 1000):
        """
        Compulsory before using method optimize.
        Define which correction to apply, and the parameters upper bounds
        
        Parameters:
        ----------
        fitDelay: bool, optional
            if True, delay will be corrected (default is False)
        delayLimit: float, optional
            Define an upper bound for the delay in absolute value, in seconds (default is 1e-12)
        fitAmplitude: bool, optional
            if True, amplitude will be corrected (default is False)
        amplitudeLimit: float, optional
            Define an upper bound for the amplitude error in absolute value,  (default is 0.1)
        fitPeriodicSampling: bool, optional
            if True, periodic samplig error will be corrected (default is False)
        delayLimit: float, optional
            Define an upper bound for the central frequency of the perdiodic sampling error (default is 7e12)   
        maxIter: int, optional
            Maximum number of iteration for the optimisation of delay and amplitude (default is 1000)
        maxIterPS: int, optional
            Maximum number of iteration for the optimisation of periodic sampling (default is 1000)
        """
        self.fit_dilatation = False
        self.dilatation_limit = np.zeros(2)
        self.dilatationmax_guess = np.zeros(2)
        self.fit_delay = fitDelay
        self.delaymax_guess = 0
        self.delay_limit = delayLimit
        self.fit_periodic_sampling = fitPeriodicSampling
        self.periodic_sampling_freq_limit = periodicSamplingFreqLimit
        self.fit_leftover_noise = fitAmplitude
        self.leftcoef_guess = np.zeros(2)
        self.leftcoef_limit = [amplitudeLimit, 1e-100]
        self.maxiter = maxIter
        self.maxiter_ps = maxIterPS

    def setAlgo(self, algo):
        self.algo = algo
        
    def getCorrectedTraces(self):
        """
        After optimisation, returns the list of corrected signals
        
        Returns
        -------
        list of 1d array:
            List of corrected time signal.
        """
        return self.datacorrection.pulse
    
    def getCorrectionParameters(self):
        """
        After optimisation, returns a dictionary with correction parameters applied
        
        Returns
        -------
        2 channel dictionary:
            Channel "amplitude" and "delay" each contains the list of corection applied to each signal.
        """
        resultDict = {}
        resultDict["delay"] = np.zeros(len(self.result))
        resultDict["amplitude"] = np.zeros(len(self.result))
        for i, result in enumerate(self.result):
            xopt = result[0]
            if self.fit_leftover_noise:
                resultDict["amplitude"][i] = xopt[-2]
            if self.fit_delay:
                resultDict["delay"][i] = xopt[0]
        return resultDict

        
    def errorchoice(self):
        nsample=self.nsample
        fit_dilatation=self.fit_dilatation
        fit_leftover_noise=self.fit_leftover_noise
        fit_delay=self.fit_delay
        
        def monerreur(x):                                             
            xopt = x*(self.maxval-self.minval)+self.minval
            myinputdatacorrected = self.applyCorrection(xopt)
            erreur = np.linalg.norm(self.myreferencedata.Pulseinit[:nsample] - myinputdatacorrected[:nsample] )/np.linalg.norm(self.myreferencedata.Pulseinit[:nsample])
            return erreur
                
        return monerreur
        

        
    def applyCorrection(self, x):
        
        fit_dilatation=self.fit_dilatation
        fit_leftover_noise=self.fit_leftover_noise
        fit_delay=self.fit_delay
        
        leftover_guess = np.zeros(2)
        delay_guess = 0
        dilatation_coefguess = np.zeros(2)
        
        coef = np.zeros(2) #[a,c]
                
        if fit_delay:
            delay_guess = x[0]
            
        if fit_dilatation:
            if fit_delay:
                dilatation_coefguess = x[1:3]
            else:
                dilatation_coefguess = x[0:2]
            dilatation_coefguess[1] = 0
        if fit_leftover_noise:
                leftover_guess = x[-2:]

        coef[0] = leftover_guess[0] #a
        coef[1] = leftover_guess[1] #c

        Z = np.exp(j*self.myglobalparameters.w*delay_guess)
        myinputdatacorrected_withdelay = irfft(Z*self.myinputdata.Spulse, n = len(self.myglobalparameters.t))

        leftnoise = np.ones(len(self.myglobalparameters.t)) - coef[0]*np.ones(len(self.myglobalparameters.t))   #(1-a)    
        myinputdatacorrected = leftnoise*(myinputdatacorrected_withdelay  
                                          - (dilatation_coefguess[0]*self.myglobalparameters.t)*np.gradient(myinputdatacorrected_withdelay, self.dt))

        return myinputdatacorrected
    


    # =============================================================================
        
    def optimize(self,nb_proc = 1):
        """
        Use this method after initiaization to launch the optimisation process.
        """
    
        # =============================================================================
        # We load the model choices
        # =============================================================================
        reference_number = self.reference_number
        fit_dilatation = self.fit_dilatation
        dilatation_limit = self.dilatation_limit
        dilatationmax_guess = self.dilatationmax_guess
        fit_delay = self.fit_delay
        delaymax_guess = self.delaymax_guess
        delay_limit = self.delay_limit
        fit_periodic_sampling = self.fit_periodic_sampling
        periodic_sampling_freq_limit = self.periodic_sampling_freq_limit
        fit_leftover_noise = self.fit_leftover_noise
        leftcoef_guess = self.leftcoef_guess
        leftcoef_limit = self.leftcoef_limit
        
        data = self.data
        
        self.myreferencedata=self.ref
        
        self.myglobalparameters = self.globalparameters
        apply_window = self.apply_window
        nsamplenotreal=len(self.myglobalparameters.t)

        if apply_window == 1:
            windows = signal.tukey(nsamplenotreal, alpha = 0.05)


        out_dir="temp"
        algo = self.algo
        swarmsize = self.swarmsize
        maxiter = self.maxiter
        maxiter_ps = self.maxiter_ps


        # Load fields data
        out_opt_filename = "optim_result"
        out_opt_full_info_filename = f"{out_dir}/{out_opt_filename.split('.')[0]}_full_info.out"

        datacorrection = datalist()

            # =============================================================================
        myvariables = []
        nb_param = len(myvariables)
            
        myVariablesDictionary = {}
        minDict = {}
        maxDict = {}
        totVariablesName = myvariables
            
            
        if fit_delay == 1:
            myVariablesDictionary['delay']=delaymax_guess
            minDict['delay'] = -delay_limit
            maxDict['delay'] =  delay_limit
            totVariablesName = np.append(totVariablesName,'delay')
            
            
        if fit_dilatation == 1:
            tab=[]
            for i in range (0,len(dilatationmax_guess)):                    
                myVariablesDictionary['dilatation '+str(i)]=dilatationmax_guess[i]#leftcoef[count-1]
                minDict['dilatation '+str(i)] = -dilatation_limit[i]
                maxDict['dilatation '+str(i)] = dilatation_limit[i]
                tab = np.append(tab,'dilatation '+str(i))
            totVariablesName = np.append(totVariablesName,tab)
            
            
        if (fit_leftover_noise == 1):
            tab=[]
            for i in range (0,len(leftcoef_guess)):                    
                myVariablesDictionary['leftover '+str(i)]=leftcoef_guess[i]#leftcoef[count-1]
                minDict['leftover '+str(i)] = -leftcoef_limit[i]
                maxDict['leftover '+str(i)] = leftcoef_limit[i]
                tab = np.append(tab,'leftover '+str(i))
            totVariablesName = np.append(totVariablesName,tab)
            ## We take into account the thicknesses and delay as optimization parameters
            # so we put the values and their uncertainty in the corresponding lists
            
            
            #=============================================================================#
            # Instantiate Optimization Problem
            #=============================================================================#


        #*************************************************************************************************************
            # Normalisation
        self.minval = np.array(list(minDict.values()))
        self.maxval = np.array(list(maxDict.values()))
        guess = np.array(list(myVariablesDictionary.values()))

        x0=np.array((guess-self.minval)/(self.maxval-self.minval))
        lb=np.zeros(len(guess))
        up=np.ones(len(guess))

        self.dt=self.myglobalparameters.t.item(2)-self.myglobalparameters.t.item(1)   ## Sample rate

        numberOfTrace = len(data.pulse)
        fopt_init = []
        exposant_ref = 4  # au lieu d'avoir x0 = 0.5 pour la ref, qui est dejà optimal et donc qui fait deconné l'ago d'optim, on aura x0 = 0.5-1e^exposant_ref



        if fit_periodic_sampling: #need an init point for optimization after correction
            # print("Periodic sampling optimization")

            self.mymean = np.mean(data.pulse, axis = 0)

            nu = periodic_sampling_freq_limit*1e12   # 1/s   Hz
            delta_nu = self.myglobalparameters.freq[-1]/(len(self.myglobalparameters.freq)-1) # Hz
            index_nu=int(nu/delta_nu)
            
            maxval_ps = np.array([self.dt/10, 12*2*np.pi*1e12, np.pi])
            minval_ps = np.array([0, 6*2*np.pi*1e12, -np.pi])
            guess_ps = np.array([0,0,0])
            
            x0_ps = (guess_ps-minval_ps)/(maxval_ps-minval_ps)
            lb_ps=np.zeros(len(guess_ps))
            ub_ps=np.ones(len(guess_ps))
            
            def error_periodic(x):
                x = x*(maxval_ps-minval_ps)+minval_ps
                
                ct = x[0]*np.cos(x[1]*self.myglobalparameters.t + x[2])    # s 
                corrected = self.mymean - np.gradient(self.mymean, self.dt)*ct
                
                error = sum(abs((rfft(corrected)[index_nu:])))
                
                return error
            
            res_ps = optimize.dual_annealing(error_periodic, x0 = x0_ps, maxiter = maxiter_ps, bounds=list(zip(lb_ps, ub_ps)))
            
            xopt_ps = res_ps.x*(maxval_ps-minval_ps)+minval_ps




        if fit_delay or fit_leftover_noise or fit_dilatation:
            # print("Delay and amplitude and dilatation error optimization")
            for trace in range(numberOfTrace) :
                # Checking if break button is pressed
                if self.interrupt and self.optimization_process.is_alive():
                    self.optimization_process.terminate()

                self.myinputdata=mydata(data.pulse[trace])    ## We create a variable containing the data related to the measured pulse

                
                monerreur = self.errorchoice()

                
                if fit_leftover_noise:
                    if fit_dilatation:
                        if fit_delay:
                            x0[1] = 0.505 #coef a on evite de commencer l'init à 0 car parfois probleme de convergence
                        else:
                            x0[0] = 0.505  # coef a 
                    elif not fit_dilatation and trace ==0: # si on fit pas la dilatation, on peut utiliser les anciens result d'optim, a part pour la trace 0
                        if fit_delay:
                            x0[1] = 0.505 #coef a on evite de commencer l'init à 0 car parfois probleme de convergence
                        else:
                            x0[0] = 0.505  # coef a 
                
                if trace == reference_number: # on part pas de 0.5 car il diverge vu que c'est la ref elle meme
                    ref_x0= [0.5 - 0.1**exposant_ref]*len(totVariablesName)
                    fopt_init.append(monerreur(ref_x0))
                else:
                    guess= x0*(self.maxval-self.minval)+self.minval
                    fopt_init.append(monerreur(x0))


                
                
                # =============================================================================
                # solving the problem with the function in scipy.optimize
                # =============================================================================
                
                
                if  algo==0: 
                    start = time.process_time()
                    xopt,fopt=pso(monerreur,lb,up,swarmsize=swarmsize,minfunc=1e-18,minstep=1e-8,debug=1,phip=0.5,phig=0.5,maxiter=maxiter) ## 'monerreur' function that we want to minimize, 'lb' and 'up' bounds of the problem
                    elapsed_time = time.process_time()-start
                    
                if algo == 5:
                    start = time.process_time()
                    cback=Callback_bfgs(monerreur)
                    if trace == reference_number:
                        res = optimize.minimize(monerreur,ref_x0,method='L-BFGS-B',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter})
                    else:
                        res = optimize.minimize(monerreur,x0,method='L-BFGS-B',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter})
                    elapsed_time = time.process_time()-start
                    xopt = res.x
                    fopt = res.fun
                    
                if algo == 6:
                    start = time.process_time()
                    cback=Callback_slsqp(monerreur)
                    if trace == reference_number:
                        res = optimize.minimize(monerreur,ref_x0,method='SLSQP',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter, 'ftol': 1e-20})
                    else:
                        res = optimize.minimize(monerreur,x0,method='SLSQP',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter})
                    elapsed_time = time.process_time()-start
                    xopt = res.x
                    fopt = res.fun
                    
                if algo==7:
                    start = time.process_time()
                    cback=Callback_annealing(monerreur)
                    res = optimize.dual_annealing(monerreur, bounds=list(zip(lb, up)),callback=cback,maxiter=maxiter)
                    elapsed_time = time.process_time()-start
                    xopt = res.x
                    fopt = res.fun
                
                
                
                if myrank == 0:
                    xopt = xopt*(self.maxval-self.minval)+self.minval  #denormalize

                    
                    
                    
                    datacorrection.add_trace(self.applyCorrection(xopt))
                    
                    # =========================================================================
                    # saving the results
                    # ========================================================================
            
                    result_optimization=[xopt,fopt]
                    if(trace == 0):   # write first time
                        result_optimization = [item.tolist() if isinstance(item, np.ndarray) else item for item in result_optimization]
                        result_optimization = [result_optimization]
                        self.result = result_optimization

                    else:  #append after first time
                        result_optimization = [item.tolist() if isinstance(item, np.ndarray) else item for item in result_optimization]
                        self.result.append(result_optimization) # use .append instead of =



            #TOADD: progressbar   
            ################################### After the optimization loop #############
            
            #Loop for is done : interrupt variable back to False
            self.interrupt=False
            
            if myrank == 0 and not fit_periodic_sampling:  
                datacorrection.mean = np.mean(datacorrection.pulse, axis = 0)
                datacorrection.time_std = np.std(datacorrection.pulse, axis = 0)
                datacorrection.freq_std = np.std(rfft(datacorrection.pulse, axis = 1),axis = 0)
                if apply_window == 1:
                    datacorrection.freq_std_with_window = np.std(rfft(datacorrection.pulse*windows, axis = 1),axis = 0)

                self.datacorrection = datacorrection
                self.fopt=fopt_init
                
            
                    


        ###################################################
                 #  ****************************************** PERIODIC SAMPLING *******************************************   


        if fit_periodic_sampling:
            # print("Periodic sampling optimization")            

            if fit_delay or fit_leftover_noise or fit_dilatation:
                self.mymean = np.mean(datacorrection.pulse, axis = 0)   
            else:
                self.mymean = np.mean(data.pulse, axis = 0)

            nu = periodic_sampling_freq_limit*1e12   # 1/s   Hz
            delta_nu = self.myglobalparameters.freq[-1]/(len(self.myglobalparameters.freq)-1) # Hz
            index_nu=int(nu/delta_nu)
            
            maxval_ps = np.array([self.dt/10, 12*2*np.pi*1e12, np.pi])
            minval_ps = np.array([0, 6*2*np.pi*1e12, -np.pi])
            guess_ps = xopt_ps
            
            x0_ps = (guess_ps-minval_ps)/(maxval_ps-minval_ps)
            lb_ps=np.zeros(len(guess_ps))
            ub_ps=np.ones(len(guess_ps))
            
            def error_periodic(x):
                # x = A, v, phi
                x = x*(maxval_ps-minval_ps)+minval_ps
                
                ct = x[0]*np.cos(x[1]*self.myglobalparameters.t + x[2])    # s 
                corrected = self.mymean - np.gradient(self.mymean, self.dt)*ct
                
                error = sum(abs((rfft(corrected)[index_nu:])))
                


                return error
            

            res_ps = optimize.dual_annealing(error_periodic, x0 = x0_ps, maxiter = maxiter_ps, bounds=list(zip(lb_ps, ub_ps)))
            
            xopt_ps = res_ps.x*(maxval_ps-minval_ps)+minval_ps
            fopt_ps = res_ps.fun
            
            result_optimization = [xopt_ps, fopt_ps]
            
            if fit_delay or fit_leftover_noise or fit_dilatation:
                result_optimization = [item.tolist() if isinstance(item, np.ndarray) else item for item in result_optimization]
                self.result.append(result_optimization) # use .append instead of =

            else:
                result_optimization = [item.tolist() if isinstance(item, np.ndarray) else item for item in result_optimization]
                self.result = result_optimization
            
            ct = xopt_ps[0]*np.cos(xopt_ps[1]*self.myglobalparameters.t + xopt_ps[2])
            
            if fit_delay or fit_leftover_noise or fit_dilatation:
                for i in range(numberOfTrace):
                    datacorrection.pulse[i]= datacorrection.pulse[i] - np.gradient(datacorrection.pulse[i], self.dt)*ct
            else:
                for i in range(numberOfTrace):
                    # print(f"correction of trace {i}")
                    temp = data.pulse[i] - np.gradient(data.pulse[i], self.dt)*ct
                    datacorrection.add_trace(temp)

            


            datacorrection.mean = np.mean(datacorrection.pulse, axis = 0)
            datacorrection.time_std = np.std(datacorrection.pulse, axis = 0)
            datacorrection.freq_std = np.std(rfft(datacorrection.pulse, axis = 1),axis = 0) 
            if apply_window == 1:
                datacorrection.freq_std_with_window = np.std(rfft(datacorrection.pulse*windows, axis = 1),axis = 0)
                        

            self.datacorrection = datacorrection
            self.fopt=fopt_init

        ###################################################