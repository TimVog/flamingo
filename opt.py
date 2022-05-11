# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:34:04 2019

@author: nayab, juliettevl
"""
#from memory_profiler import profile
# =============================================================================
# Standard Python modules
# =============================================================================
import os, time
import pickle
from pyswarm import pso   ## Library for optimization
import numpy as np   ## Library to simplify the linear algebra calculations
import scipy.optimize as optimize  ## Library for optimization
import fitf as TDS
import h5py #Library to import the noise matrix
import warnings
from scipy import signal
#warnings.filterwarnings("ignore") #this is just to remove the 'devided by zero' runtime worning for low frequency
#we stricly advise to comment the above line as soon as you modify the code!


# =============================================================================
j = 1j


# =============================================================================
# External Python modules (serves for optimization algo #3)
# =============================================================================
## Parallelization that requieres mpi4py to be installed, if mpi4py was not installed successfully comment frome line 32 to line 40 (included)
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    size = comm.Get_size()
except:
    print('mpi4py is required for parallelization')
    myrank=0


#end
# =============================================================================
# Extension modules
# =============================================================================

try:
    from pyOpt import Optimization   ## Library for optimization
    from pyOpt import ALPSO  ## Library for optimization
except:
    if myrank==0:
        print("Error importing pyopt")
    
try:
    from pyOpt import SLSQP  ## Library for optimization
except:
    if myrank==0:
        print("Error importing pyopt SLSQP")


# =============================================================================
# classes we will use
# =============================================================================
        
class myfitdata: #ok
    def __init__(self, myinput, x):
        #self.mytransferfunction = TDS.transferfunction(myglobalparameters.w,delay_guess,leftover_guess)
        self.pulse = fit_input(myinputdata, x)
        self.Spulse = (TDS.torch_rfft((self.pulse)))
        # =============================================================================
    # function that returns the convolved pulse to the transfer function, it does it by different Drude model with one oscillator, n oscillators, etc
    # =============================================================================
    """def calculedpulse(self,delay_guess,leftover_guess, myinput):
        global myglobalparameters
        
        Z = TDS.transferfunction(myglobalparameters.w,delay_guess=delay_guess,leftover_guess=leftover_guess)
        Spectrumtot=Z*myinput.Spulse
        Pdata=(TDS.torch_irfft((np.array(Spectrumtot)), n = len(myinput.pulse)))
        return Pdata"""

    

# =============================================================================
class Callback_bfgs(object): #ok
    def __init__(self):
        self.nit = 0
        
    def __call__(self, par, convergence=0):
        self.nit += 1
        with open('algo_bfgs_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, monerreur(par), par))
            
class Callback_slsqp(object):
    def __init__(self):
        self.nit = 0
        
    def __call__(self, par, convergence=0):
        self.nit += 1
        with open('algo_slsqp_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, monerreur(par), par))


class Callback_annealing(object):
    def __init__(self):
        self.nit = 0
        
    def __call__(self, par, f, context):
        self.nit += 1
        with open('algo_dualannealing_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, monerreur(par), par))

           



#=============================================================================
def errorchoice(): #basic #ok  #on veut coller le input à la ref
    global myglobalparameters, myinputdata, mode, fit_delay, fit_leftover_noise, fit_dilatation, dt
    
    if mode == "basic":
        def monerreur(x):
            
            leftover_guess = np.zeros(2)
            delay_guess = 0
            dilatation_coefguess = np.zeros(2)
            
            coef = np.zeros(2) #[a,c]
    
            x = denormalize(x)
            
            if fit_delay:
                delay_guess = x[0]
                
            if fit_dilatation:
                if fit_delay:
                    dilatation_coefguess = x[1:3]
                else:
                    dilatation_coefguess = x[0:2]

            if fit_leftover_noise:
                    leftover_guess = x[-2:]

            coef[0] = leftover_guess[0] #a
            coef[1] = leftover_guess[1] #c

            Z = np.exp(j*myglobalparameters.w*delay_guess)
            myinputdatacorrected_withdelay = np.fft.irfft(Z*myinputdata.Spulse, n = len(myglobalparameters.t))

            leftnoise = np.ones(len(myglobalparameters.t)) - coef[0]*np.ones(len(myglobalparameters.t))   #(1-a)    
            myinputdatacorrected = leftnoise*(myinputdatacorrected_withdelay + (dilatation_coefguess[0]*myglobalparameters.t+dilatation_coefguess[1])*np.gradient(myinputdatacorrected_withdelay, dt))
            erreur = np.linalg.norm(myreferencedata.Pulseinit - myinputdatacorrected )/np.linalg.norm(myreferencedata.Pulseinit)
            return erreur
            
    elif mode == "superresolution":
        def monerreur(x):
            Z = fit_transfer_function(x)
            Spectrumtot=Z*myinputdata.Spulse
            pulse_theo=(TDS.torch_irfft((np.array(Spectrumtot)), n = len(myinputdata.pulse))) # calcul from calculedpulse. In fact it is the same calcul as in the basic mode for i!=0
            erreur=np.linalg.norm(myreferencedata.Pulseinit-pulse_theo)/np.linalg.norm(myreferencedata.Pulseinit)
            return erreur
    return monerreur


def fit_input(myinputdata, x):
    global myglobalparameters, mode, fit_delay, fit_leftover_noise, fit_dilatation, dt
    
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

    if fit_leftover_noise:
            leftover_guess = x[-2:]

    coef[0] = leftover_guess[0] #a
    coef[1] = leftover_guess[1] #c

    Z = np.exp(j*myglobalparameters.w*delay_guess)
    myinputdatacorrected_withdelay = np.fft.irfft(Z*myinputdata.Spulse, n = len(myglobalparameters.t))

    leftnoise = np.ones(len(myglobalparameters.t)) - coef[0]*np.ones(len(myglobalparameters.t))   #(1-a)    
    myinputdatacorrected = leftnoise*(myinputdatacorrected_withdelay + (dilatation_coefguess[0]*myglobalparameters.t+dilatation_coefguess[1])*np.gradient(myinputdatacorrected_withdelay, dt))

    return myinputdatacorrected

def denormalize(x):
    return x*(maxval-minval)+minval

def fit_transfer_function(x): #ok                ref_x0 = [0.5 - 0.1**exposant_ref]
    global myglobalparameters, fit_delay, delaymax_guess, delay_limit, fit_leftover_noise, leftcoef_guess, leftcoef_limit
    x1 = x*(maxval-minval)+minval
    delay_guess = 0
    leftover_guess = np.zeros(2)
    if(fit_leftover_noise == 1):
        count=-1
        for i in range (0,len(leftcoef_guess)):                    
            count = count+1
            leftover_guess[i] = x1[-len(leftcoef_guess)+count]
    if (fit_delay == 1):
        if (fit_leftover_noise == 1):
            delay_guess = x1[-len(leftcoef_guess)-1]
        else:
            delay_guess = x1[-1]
            
    return TDS.transferfunction(myglobalparameters.w, delay_guess, leftover_guess)
# =============================================================================
def errorchoice_pyOpt(): #ok                ref_x0 = [0.5 - 0.1**exposant_ref]
    def objfunc(x):  ## Function used in the Optimization function from pyOpt. For more details see http://www.pyopt.org/quickguide/quickguide.html
        monerreur = errorchoice()
        f = monerreur(x)
        fail = 0
        return f, 1, fail
    return objfunc


# =============================================================================

def optimALPSO(opt_prob, swarmsize, maxiter,algo,out_opt_full_info_filename): #ok
    if algo == 2:
        alpso_none = ALPSO(pll_type='SPM')
    else:
        alpso_none = ALPSO()
    #alpso_none.setOption('fileout',1)
    #alpso_none.setOption('filename',out_opt_full_info_filename)
    alpso_none.setOption('SwarmSize',swarmsize)
    alpso_none.setOption('maxInnerIter',6)
    alpso_none.setOption('etol',1e-5)
    alpso_none.setOption('rtol',1e-10)
    alpso_none.setOption('atol',1e-10)
    alpso_none.setOption('vcrazy',1e-4)
    alpso_none.setOption('dt',1e0)
    alpso_none.setOption('maxOuterIter',maxiter)
    alpso_none.setOption('stopCriteria',0)#Stopping Criteria Flag (0 - maxIters, 1 - convergence)
    alpso_none.setOption('printInnerIters',1)               
    alpso_none.setOption('printOuterIters',1)
    alpso_none.setOption('HoodSize',int(swarmsize/100))
    return(alpso_none(opt_prob))
    
def optimSLSQP(opt_prob,maxiter):#ok
    slsqp_none = SLSQP()
    #slsqp_none.setOption('IPRINT',1)
    #slsqp_none.setOption('IFILE',out_opt_full_info_filename)
    slsqp_none.setOption('MAXIT',maxiter)
    slsqp_none.setOption('IOUT',15) 
    slsqp_none.setOption('ACC',1e-20)
    return(slsqp_none(opt_prob))

def optimSLSQPpar(opt_prob,maxiter): # arecopierdansdoublet #ok
          
    slsqp_none = SLSQP() # arecopierdansdoublet

    #slsqp_none.setOption('IPRINT',1)
    #slsqp_none.setOption('IFILE',out_opt_full_info_filename)
    slsqp_none.setOption('MAXIT',maxiter)
    slsqp_none.setOption('IOUT',12) 
    slsqp_none.setOption('ACC',1e-24)
    return(slsqp_none(opt_prob,sens_mode='pgc')) 

            

# =============================================================================                ref_x0 = [0.5 - 0.1**exposant_ref]
# Change that if the GPU is needed
# =============================================================================
def fft_gpu(y): #ok
#    global using_gpu
#    if using_gpu==1:
#        ygpu = cp.array(y)
#        outgpu=cp.fft.rfft(ygpu)  # implied host->device
#        out=outgpu.get()
#    else:
    out = np.fft.rfft(y)
    return(out)

def ifft_gpu(y): #ok
#    global using_gpu
#    if using_gpu==1:  ## Only works if the number of elements before doing the fft is pair
#        ygpu = cp.array(y)
#        outgpu=cp.fft.irfft(ygpu)  # implied host->device
#        out=outgpu.get()                            
#    else:
    out = np.fft.irfft(y)
    return(out)

# =============================================================================
# We load the model choices
# =============================================================================                ref_x0 = [0.5 - 0.1**exposant_ref]
f=open(os.path.join("temp",'temp_file_1_ini.bin'),'rb') #ok
[path_data, path_data_ref, reference_number, fit_dilatation, dilatation_limit, dilatationmax_guess, freqWindow, timeWindow, fit_delay, delaymax_guess, delay_limit, mode, 
 fit_periodic_sampling, periodic_sampling_freq_limit, fit_leftover_noise, leftcoef_guess, leftcoef_limit]=pickle.load(f)
f.close()

data = TDS.datalist()
f=open(os.path.join("temp",'temp_file_6.bin'),'rb')
data = pickle.load(f)
myreferencedata=pickle.load(f) # champs
f.close()

myglobalparameters = TDS.globalparameters()
f=open(os.path.join("temp",'temp_file_7.bin'),'rb')
myglobalparameters = pickle.load(f)
f.close()


#f=open(os.path.join("temp",'temp_file_4.bin'),'rb') #TODO
out_dir="temp"
#f.close() #où mettre les optim result

f=open(os.path.join("temp",'temp_file_5.bin'),'rb') #ok
[algo,swarmsize,maxiter, maxiter_ps]=pickle.load(f)
f.close()

#using_gpu = 0


# Load fields data
out_opt_filename = "optim_result"
out_opt_full_info_filename=os.path.join(out_dir,'{0}_full_info.out'.format(out_opt_filename.split('.')[0]))

#data = TDS.inputdatafromfile(path_data)    ## We load the signal of the measured pulse with sample
datacorrection = TDS.datalist()

"""myglobalparameters=TDS.globalparameters   # t freq w
myglobalparameters.t=data.time*1e-12 #this assumes input files are in ps ## We load the list with the time of the experiment
nsample=len(myglobalparameters.t)
dt=myglobalparameters.t.item(2)-myglobalparameters.t.item(1)   ## Sample rate
myglobalparameters.freq = np.fft.rfftfreq(nsample, dt)        ## We create a list with the frequencies for the spectrum
myglobalparameters.w=myglobalparameters.freq*2*np.pi


if mode == "superresolution":
    frep=99.991499600e6 # repetition frequency of the pulse laser used in the tds measurments in Hz, 99
    nsampleZP=np.round(1/(frep*dt)) #number of time sample betwen two pulses. IT has to be noted that it could be better to have an integer number there then the rounding does not change much
    nsamplenotreal=nsampleZP.astype(int)
    myglobalparameters.t=np.arange(nsampleZP)*dt  # 0001 #
    myglobalparameters.freq = np.fft.rfftfreq(nsamplenotreal, dt)
    myglobalparameters.w = 2*np.pi*myglobalparameters.freq
    data.time = myglobalparameters.t*1e12 #on met à jour le time, pas vraiment nessessair emais on sait jamais si va etre reutilise plus tard
"""

    # =============================================================================
myvariables = []
nb_param = len(myvariables)
    
myVariablesDictionary = {}
minDict = {}
maxDict = {}
totVariablesName = myvariables
    
    # =============================================================================


#=================================================
    
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
minval = np.array(list(minDict.values()))
maxval = np.array(list(maxDict.values()))
guess = np.array(list(myVariablesDictionary.values()))

x0=np.array((guess-minval)/(maxval-minval))
lb=np.zeros(len(guess))
up=np.ones(len(guess))

dt=myglobalparameters.t.item(2)-myglobalparameters.t.item(1)   ## Sample rate

numberOfTrace = len(data.pulse)
fopt_init = []
exposant_ref = 4  # au lieu d'avoir x0 = 0.5 pour la ref, qui est dejà optimal et donc qui fait deconné l'ago d'optim, on aura x0 = 0.5-1e^exposant_ref



if fit_periodic_sampling:
    print("Periodic sampling optimization")

    mymean = np.mean(data.pulse, axis = 0)

    nu = periodic_sampling_freq_limit*1e12   # 1/s   Hz
    delta_nu = myglobalparameters.freq[-1]/(len(myglobalparameters.freq)-1) # Hz
    index_nu=int(nu/delta_nu)
    
    maxval_ps = np.array([dt/10, 12*2*np.pi*1e12, np.pi])
    minval_ps = np.array([0, 6*2*np.pi*1e12, -np.pi])
    guess_ps = np.array([0,0,0])
    
    x0_ps = (guess_ps-minval_ps)/(maxval_ps-minval_ps)
    lb_ps=np.zeros(len(guess_ps))
    ub_ps=np.ones(len(guess_ps))
    
    def error_periodic(x):
        # x = A, v, phi
        global mymean
        x = x*(maxval_ps-minval_ps)+minval_ps
        
        ct = x[0]*np.cos(x[1]*myglobalparameters.t + x[2])    # s 
        corrected = mymean - np.gradient(mymean, dt)*ct
        
        error = sum(abs((TDS.torch_rfft(corrected)[index_nu:])))
        
        #error = 0
        #for i in range(index_nu,len(myglobalparameters.freq)):
         #   error += abs(np.real(np.exp(-j*np.angle(np.fft.rfft(corrected)[i-index_nu])) * np.fft.rfft(corrected)[i]))

        return error
    
    res_ps = optimize.dual_annealing(error_periodic, x0 = x0_ps, maxiter = maxiter_ps, bounds=list(zip(lb_ps, ub_ps)))
    #res_ps = optimize.minimize(error_periodic,x0_ps, method='SLSQP',bounds=list(zip(lb_ps, ub_ps)), options={'maxiter':maxiter_ps})
    #res_ps = optimize.minimize(error_periodic,x0_ps,method='L-BFGS-B',bounds=list(zip(lb_ps, ub_ps)), options={'maxiter':1000})
    #res_ps = pso(error_periodic,lb_ps,ub_ps,swarmsize=100,minfunc=1e-18,minstep=1e-8,debug=1,phip=0.5,phig=0.5,maxiter=100)
    
    xopt_ps = res_ps.x*(maxval_ps-minval_ps)+minval_ps




if fit_delay or fit_leftover_noise or fit_dilatation:
    print("Delay and amplitude and dilatation error optimization")
    for trace in  range(numberOfTrace) :
        print("Time trace "+str(trace))
        myinputdata=TDS.mydata(data.pulse[trace])    ## We create a variable containing the data related to the measured pulse with sample
        data.pulse[trace] = []
        
        monerreur = errorchoice()
        objfunc = errorchoice_pyOpt()
        
        if fit_leftover_noise:
            if fit_delay:
                x0[1] = 0.51 #coef a on evite de commencer l'init à 0 car parfois probleme de convergence
            else:
                x0[0] = 0.51  # coef a  
        
        if trace == reference_number:
            ref_x0= [0.5 - 0.1**exposant_ref]*len(totVariablesName)
            # on print pas c
            if fit_leftover_noise:  
                print('guess')
                print((np.array(ref_x0)*(maxval-minval)+minval)[:-1])
                print('x0')
                print(ref_x0[:-1])
            else:
                print('guess')
                print(np.array(ref_x0)*(maxval-minval)+minval)
                print('x0')
                print(ref_x0)
            print('errorguess')
            fopt_init.append(monerreur(ref_x0))
            print(fopt_init[-1])
        else:
            guess= x0*(maxval-minval)+minval
            # on print seulemnt delay et a , pas c
            if fit_leftover_noise:
                print('guess')
                print(guess[:-1])
                print('x0')
                print(x0[:-1])
            else:
                print('guess')
                print(guess)
                print('x0')
                print(x0)
            print('errorguess')
            fopt_init.append(monerreur(x0))
            print(fopt_init[-1])

        
        
        ## Optimization dans le cas PyOpt
        if algo in [1,2,3,4]:
            opt_prob = Optimization('Dielectric modeling based on TDS pulse fitting',objfunc)
            icount = 0
            for nom,varvalue in myVariablesDictionary.items():
                #if varvalue>=0:
                if trace == reference_number:
                    opt_prob.addVar(nom,'c',lower = 0,upper = 1,
                            value = ref_x0[icount] #normalisation
                            )
                else:
                    opt_prob.addVar(nom,'c',lower = 0,upper = 1,
                            value = (varvalue-minDict.get(nom))/(maxDict.get(nom)-minDict.get(nom)) #normalisation
                            )
                icount+=1
                #else:
                #    opt_prob.addVar(nom,'c',lower = 0,upper = 1,
                #                value = -(varvalue-minDict.get(nom))/(maxDict.get(nom)-minDict.get(nom)) #normalisation
                 #               )    
            opt_prob.addObj('f')
            #opt_prob.addCon('g1','i') #possibility to add constraints
            #opt_prob.addCon('g2','i')
        
        
        # =============================================================================
        # solving the problem with the function in scipy.optimize
        # =============================================================================
        
        
        if  algo==0: ## xopt is a list we the drudeinput's parameters that minimize 'monerreur', fopt is a list with the optimals objective values
            start = time.process_time()
            xopt,fopt=pso(monerreur,lb,up,swarmsize=swarmsize,minfunc=1e-18,minstep=1e-8,debug=1,phip=0.5,phig=0.5,maxiter=maxiter) ## 'monerreur' function that we want to minimize, 'lb' and 'up' bounds of the problem
            elapsed_time = time.process_time()-start
            print("Time taken by the optimization:",elapsed_time)
            
        if algo == 5:
            start = time.process_time()
            cback=Callback_bfgs()
            if trace == reference_number:
                res = optimize.minimize(monerreur,ref_x0,method='L-BFGS-B',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter})
            else:
                res = optimize.minimize(monerreur,x0,method='L-BFGS-B',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter})
            elapsed_time = time.process_time()-start
            xopt = res.x
            fopt = res.fun
            print(res.message,"\nTime taken by the optimization:",elapsed_time)
            
        if algo == 6:
            start = time.process_time()
            cback=Callback_slsqp()
            if trace == reference_number:
                res = optimize.minimize(monerreur,ref_x0,method='SLSQP',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter, 'ftol': 1e-20})
            else:
                res = optimize.minimize(monerreur,x0,method='SLSQP',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter})
            elapsed_time = time.process_time()-start
            xopt = res.x
            fopt = res.fun
            print(res.message,"\nTime taken by the optimization:",elapsed_time)
            
        if algo==7:
            start = time.process_time()
            cback=Callback_annealing()
            res = optimize.dual_annealing(monerreur, bounds=list(zip(lb, up)),callback=cback,maxiter=maxiter)
            elapsed_time = time.process_time()-start
            xopt = res.x
            fopt = res.fun
            print(res.message,"\nTime taken by the optimization:",elapsed_time)
        
        
        
        # =============================================================================
        # solving the problem with pyOpt
        # =============================================================================
        
        
        if  (algo==1)|(algo == 2):
            start = time.process_time()
            [fopt, xopt, inform] = optimALPSO(opt_prob, swarmsize, maxiter,algo,out_opt_full_info_filename)
            elapsed_time = time.process_time()-start
            print(inform,"\nTime taken by the optimization:",elapsed_time)
            
        if algo ==3:
                try:
                    start = time.process_time()
                    [fopt, xopt, inform] = optimSLSQP(opt_prob,maxiter)
                    elapsed_time = time.process_time()-start
                    print(inform,"\nTime taken by the optimization:",elapsed_time)
                except Exception as e:
                    print(e)
        
        if algo ==4:
                try:
                    start = time.process_time()
                    [fopt, xopt, inform] = optimSLSQPpar(opt_prob,maxiter)
                    elapsed_time = time.process_time()-start
                    print(inform,"\nTime taken by the optimization:",elapsed_time)
                except Exception as e:
                    print(e)
          
        #x0 = xopt
        # =============================================================================
        
        if myrank == 0:
            xopt = xopt*(maxval-minval)+minval  #denormalize
            print('The best error was: \t{}'.format(fopt))
            if(fit_leftover_noise):
                print('the best parameters were: \t{}\n'.format(xopt[:-1]))
            else:
                print('the best parameters were: \t{}\n'.format(xopt))
            # =========================================================================
                    
            myfitteddata=myfitdata(myinputdata, xopt)
            
            datacorrection.add_trace(myfitteddata.pulse)
            # =========================================================================
            # saving the results
            # ========================================================================
    
            result_optimization=[xopt,fopt]   #text result à revoir ecrire seulemnet 1 fois
            if(trace == 0):   
                f=open(os.path.join("temp",'temp_file_3.bin'),'wb')
                pickle.dump(result_optimization,f,pickle.HIGHEST_PROTOCOL)
                f.close()
            else:
                f=open(os.path.join("temp",'temp_file_3.bin'),'ab')
                pickle.dump(result_optimization,f,pickle.HIGHEST_PROTOCOL)
                f.close()
        
            ## Save the data obtained via this program
            # Save optimization parameters
            #outputoptim = fopt
            #outputoptim = np.append(outputoptim,xopt)
            
            #out_opt_h5 = h5py.File(out_opt_filename+'.h5', 'w')
            #dset = out_opt_h5.create_dataset("output", (len(outputoptim),),dtype='float64')
            #dset[:]=outputoptim
        
            #out_opt_h5.close()


            # =========================================================================
            # History and convergence
            # =========================================================================
        
            
            """if  (algo==1)|(algo == 2):
                f = open("{0}_print.out".format(out_opt_full_info_filename.split('.')[0]),'r')
            
                # Find maxiter
                line = f.readline()
                while (line !=''):
                    line = f.readline()
                    lineSplit = line.split()
                    if len(lineSplit)>2:
                        if (lineSplit[0:3]==['NUMBER','OF','ITERATIONS:']):
                            maxiter = int(lineSplit[3])
                    
                f.close()
                f = open("{0}_print.out".format(out_opt_full_info_filename.split('.')[0]),'r') # Go back to the beginning
                
                # To find number of parameters:
                j = 0
                while (f.readline()!= 'OBJECTIVE FUNCTION VALUE:\n') & (j<50):
                    j = j+1
                P = [(f.readline())[4:]]
                
                while (f.readline()!= 'BEST POSITION:\n') & (j<100):
                    j =  j+1
                line = (f.readline())
                line = (line.split())
                nLines = 0
                while(len(line)>0):#(line[0][0] == 'P'):
                    P.extend(line[2::3])
                    line = (f.readline())
                    line = (line.split())
                    nLines = nLines+1
                bestPositions = np.zeros((maxiter,len(P)))
                bestPositions[0]=P
                
                
                for i in range(1,maxiter):
                    j = 0
                    while (f.readline()!= 'OBJECTIVE FUNCTION VALUE:\n') & (j<100):
                        j = j+1 # to avoid infinite loop
                        # One could use pass instead
                    P = [(f.readline())[4:]]
                    while (f.readline()!= 'BEST POSITION:\n') & (j<200):
                        j =  j+1
                    for nLine in range(nLines):
                        line = (f.readline())
                        line = (line.split())
                        P.extend(line[2::3])
                    bestPositions[i]=P
                f.close()
                
                # Write and save file
                historyHeaderFile = os.path.join(out_dir,"convergence.txt")
                historyHeader = 'objective function value'
                for name in myVariablesDictionary:
                    historyHeader = '{}{}\t'.format(historyHeader, name)
                np.savetxt(historyHeaderFile, bestPositions, header = historyHeader)"""
                
    
    ###################################
    
    
    if myrank == 0 and not fit_periodic_sampling:
        #on ajoute la ref  
        
        #SAVE the result in binary for other modules
        f=open(os.path.join("temp",'temp_file_2.bin'),'wb')
        pickle.dump(datacorrection,f,pickle.HIGHEST_PROTOCOL)
        pickle.dump(fopt_init,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        
    
            


###################################################
         #  ****************************************** PERIODIC SAMPLING *******************************************   


if fit_periodic_sampling:
    print("Periodic sampling optimization")

    if fit_delay or fit_leftover_noise or fit_dilatation:
        mymean = np.mean(datacorrection.pulse, axis = 0)   
    else:
        mymean = np.mean(data.pulse, axis = 0)

    nu = periodic_sampling_freq_limit*1e12   # 1/s   Hz
    delta_nu = myglobalparameters.freq[-1]/(len(myglobalparameters.freq)-1) # Hz
    index_nu=int(nu/delta_nu)
    
    maxval_ps = np.array([dt/100, 12*2*np.pi*1e12, np.pi])
    minval_ps = np.array([0, 6*2*np.pi*1e12, -np.pi])
    guess_ps = xopt_ps
    
    x0_ps = (guess_ps-minval_ps)/(maxval_ps-minval_ps)
    lb_ps=np.zeros(len(guess_ps))
    ub_ps=np.ones(len(guess_ps))
    
    def error_periodic(x):
        # x = A, v, phi
        global mymean
        x = x*(maxval_ps-minval_ps)+minval_ps
        
        ct = x[0]*np.cos(x[1]*myglobalparameters.t + x[2])    # s 
        corrected = mymean - np.gradient(mymean, dt)*ct
        
        error = sum(abs((TDS.torch_rfft(corrected)[index_nu:])))
        
        #error = 0
        #for i in range(index_nu,len(myglobalparameters.freq)):
         #   error += abs(np.real(np.exp(-j*np.angle(np.fft.rfft(corrected)[i-index_nu])) * np.fft.rfft(corrected)[i]))

        return error
    
    print('guess')
    print(guess_ps)
    print('x0')
    print(x0_ps)
    res_ps = optimize.dual_annealing(error_periodic, x0 = x0_ps, maxiter = maxiter_ps, bounds=list(zip(lb_ps, ub_ps)))
    #res_ps = optimize.minimize(error_periodic,x0_ps, method='SLSQP',bounds=list(zip(lb_ps, ub_ps)), options={'maxiter':maxiter_ps})
    #res_ps = optimize.minimize(error_periodic,x0_ps,method='L-BFGS-B',bounds=list(zip(lb_ps, ub_ps)), options={'maxiter':1000})
    #res_ps = pso(error_periodic,lb_ps,ub_ps,swarmsize=100,minfunc=1e-18,minstep=1e-8,debug=1,phip=0.5,phig=0.5,maxiter=100)
    
    xopt_ps = res_ps.x*(maxval_ps-minval_ps)+minval_ps
    fopt_ps = res_ps.fun
    
    result_optimization = [xopt_ps, fopt_ps]
    if fit_delay or fit_leftover_noise or fit_dilatation:
    	f=open(os.path.join("temp",'temp_file_3.bin'),'ab')
    else:
    	f=open(os.path.join("temp",'temp_file_3.bin'),'wb')
    pickle.dump(result_optimization,f,pickle.HIGHEST_PROTOCOL)
    f.close()
    
    ct = xopt_ps[0]*np.cos(xopt_ps[1]*myglobalparameters.t + xopt_ps[2])
    
    if fit_delay or fit_leftover_noise or fit_dilatation:
        for i in range(numberOfTrace):
            print("correction of trace {}".format(i))
            datacorrection.pulse[i]= datacorrection.pulse[i] - np.gradient(datacorrection.pulse[i], dt)*ct
    else:
        for i in range(numberOfTrace):
            print("correction of trace {}".format(i))
            temp = data.pulse[i] - np.gradient(data.pulse[i], dt)*ct
            datacorrection.add_trace(temp)

    
    print('The best error was: \t{}'.format(fopt_ps))
    print('the best parameters were: \t{}\n'.format(xopt_ps))

                
    f=open(os.path.join("temp",'temp_file_2.bin'),'wb')
    pickle.dump(datacorrection,f,pickle.HIGHEST_PROTOCOL)
    pickle.dump(fopt_init,f,pickle.HIGHEST_PROTOCOL)
    f.close()

###################################################
         #  ****************************************** PERIODIC SAMPLING *******************************************   


