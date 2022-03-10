#!/usr/bin/python
# -*- coding: latin-1 -*-

## This two lines is to chose the econding
# =============================================================================
# Standard Python modules
# =============================================================================
import numpy as np
import h5py
import warnings
#import torch
#warnings.filterwarnings("ignore") #this is just to remove the 'devided by zero' runtime worning for low frequency
#we stricly advise to comment the above line as soon as you modify the code!

#from memory_profiler import profile
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
    
###############################################################################
def transferfunction(w,delay_guess = 0,leftover_guess = np.zeros(2)):
     #Transfer function with no layer
    coef = np.zeros(2)

    for i in range(0,len(leftover_guess)):
        coef[i] = leftover_guess[i]
    wnorm = w*1e-12
    leftnoise = (np.ones(len(wnorm))-(coef[0]*np.ones(len(wnorm))+coef[1]*(j*wnorm)**2))         
    Z = np.exp(j*w*delay_guess)*leftnoise 
    return Z

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
            
    def choose_ref_number(self):
        norm = []
        pseudo_norm = []
        temp = mean(self.Pulseinit)

        for i in range(self.numberOfTrace):
            pseudo_norm.append(np.dot(self.Pulseinit[i], temp))
            
        for i in range(self.numberOfTrace):
            norm.append(np.dot(self.Pulseinit[i], self.Pulseinit[i]))
        proximity = np.array(pseudo_norm)/np.array(norm)

        return np.abs(proximity - 1).argmin()  #proximite la plus proche de 1
    

class getreferencetrace:
    def __init__(self, path, ref_number, trace_start, time_start):
        with h5py.File(path, "r") as f:
            self.Pulseinit = np.array(f[str(trace_start+ref_number)])
            self.Spulseinit = torch_rfft(self.Pulseinit)  ## We compute the spectrum of the measured pulse
            
    def transferfunction(self, myinputdata):
        return self.Spulseinit/myinputdata.Spulse
        


# =============================================================================

class mydata:
     def __init__(self, pulse):
        self.pulse = pulse # pulse with sample
        self.Spulse = torch_rfft((pulse)) # spectral field with sample
        
        
class datalist:
     def __init__(self):
        self.pulse = []# pulse with sample
        
     def add_trace(self, pulse):
        self.pulse.append(pulse) # pulse with sample
        
     def add_ref(self, ref_number, refpulse):
        self.pulse.insert(ref_number, refpulse) # pulse with sample
        


# =============================================================================
# Change that if the GPU is needed
# =============================================================================
def fft_gpu(y):
#    global using_gpu
#    if using_gpu==1:
#        ygpu = cp.array(y)
#        outgpu=cp.fft.rfft(ygpu)  # implied host->device
#        out=outgpu.get()
#    else:
    out = np.fft.rfft(y)
    return(out)

def ifft_gpu(y):
#    global using_gpu
#    if using_gpu==1:  ## Only works if the number of elements before doing the fft is pair
#        ygpu = cp.array(y)
#        outgpu=cp.fft.irfft(ygpu)  # implied host->device
#        out=outgpu.get()                            
#    else:
    out = np.fft.irfft(y)
    return(out)

def torch_rfft(signal, axis = -1):
    #return torch.fft.rfft(torch.from_numpy(np.array(signal)), dim = axis).numpy()
    return np.fft.rfft(np.array(signal), axis = axis)
 

def torch_irfft(signal, axis = -1, n = None):
    #return torch.fft.irfft(torch.from_numpy(np.array(signal)), dim = axis, n = n).numpy()
    return np.fft.irfft(np.array(signal), axis = axis, n = n)


def mean(array):
    moyenne = np.zeros(len(array[-1]))
    for i in array:
        moyenne = moyenne+ i
    moyenne = moyenne/len(array)
    return moyenne

def std(array,moyenne):
    somme = np.zeros(len(array[-1]))
    for i in array:
        somme+= abs(i - moyenne)**2
    ecart = np.sqrt((somme/len(array)))
    return ecart

    
"""def torch_cov(X, rowvar=True, bias=False):
    tensor = torch.from_numpy(np.array(X))
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    X =  factor * tensor @ tensor.transpose(-1, -2).conj()
    return X.numpy()"""