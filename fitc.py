#!/usr/bin/python
# -*- coding: latin-1 -*-

## This two lines is to chose the econding
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys
import pickle
import subprocess
import numpy as np
import h5py
import fitf as TDS
import warnings
#warnings.filterwarnings("ignore") #this is just to remove the 'devided by zero' runtime worning for low frequency
#we stricly advise to comment the above line as soon as you modify the code!

###############################################################################

j = 1j

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
    

# =============================================================================
# classes we will use
# =============================================================================


class ControlerBase:
    def __init__(self):
        self.clients_tab1 = list()
        self.clients_tab3 = list()
        self.message = ""


    def addClient(self, client):
        self.clients_tab1.append(client)

    def addClient3(self, client):
        self.clients_tab3.append(client)
        

    def refreshAll(self, message):
        self.message = message
        for client in self.clients_tab1:
            client.refresh()
            
    def refreshAll3(self, message):
        self.message = message
        for client in self.clients_tab3:
            client.refresh()


class Controler(ControlerBase):
    def __init__(self):
        super().__init__()
        # Initialisation:
        
        self.myreferencedata=TDS.getreferencetrace
        self.data=TDS.inputdatafromfile
        self.data_without_sample=TDS.inputdatafromfile
        
        self.myglobalparameters=TDS.globalparameters()
        
        self.nsample=None
        self.nsamplenotreal=None
        self.dt=None   ## Sample rate
        
        self.myinput = TDS.datalist()
        self.mydatacorrection=TDS.fitdatalist()
        self.myinput_cov = None
        self.mydatacorrection_cov = None        
        
        self.delay_correction = []
        self.leftover_correction = []
        self.periodic_correction = []
        self.dilatation_correction = []
        self.errorIndex=0
        
        ## parameters for the optimization algorithm
        self.swarmsize=1000
        self.maxiter=20

        # Variables for existence of temp Files
        self.is_temp_file_3 = 0 # temp file storing optimization results
        self.is_temp_file_4 = 0
        self.is_temp_file_5 = 0 # temp file storing algorithm choices
        

        # Variable to see if initialisation is done
        self.initialised = 0
        self.initialised_param = 0
        self.optim_succeed = 0
        
        self.path_data = None
        self.path_data_ref = None
        self.reference_number = None
        self.fit_delay = 0
        self.delaymax_guess = None
        self.delay_limit = None
        self.fit_leftover_noise = 0
        self.leftcoef_guess = None
        self.leftcoef_limit = None
        self.fit_periodic_sampling = 0
        self.periodic_sampling_freq_limit = None
        self.fit_dilatation = 0
        self.dilatation_limit = None
        self.dilatationmax_guess = None
        
        self.Lfiltering = None
        self.Hfiltering = None
        self.set_to_zeros = None
        self.dark_ramp = None
        
        self.algo = None
        self.mode = None
        
        self.Freqwindow = None
        self.timeWindow = None
        self.fopt = []
        self.fopt_init = []
        
# =============================================================================
# Initialisation tab
# =============================================================================

    def init(self):
        self.refreshAll("Initialisation: Ok")
        
        """    def error_message_path(self):
        self.refreshAll("Error: Please enter a valid file(s).")"""

    def loading_text(self):
        self.refreshAll("\n Processing... \n")

    def choices_ini(self, path_data, path_data_ref, Lfiltering_index, 
                    Hfiltering_index, zeros_index, dark_index, cutstart, cutend,sharpcut, slope, intercept,modesuper):
        """Process all the informations given in the first panel of initialisation: 
            create instances of classes to store data, apply filters"""

        self.path_data = path_data
        self.optim_succeed = 0
        
        self.myinput = TDS.datalist() # Warning: Don't remove the parentheses

        self.mydatacorrection=TDS.fitdatalist()
        
        self.myinput_cov = None
        self.mydatacorrection_cov = None    
        self.delay_correction = []
        self.leftover_correction = []
        self.periodic_correction = []
        self.dilatation_correction = []
        self.fopt = []
        self.fopt_init = []
            
        self.data= TDS.inputdatafromfile(path_data)    ## We load the signal of the measured pulse with sample
        #self.myinput_cov = TDS.torch_cov(self.data.Pulseinit, rowvar=False)
        #print(np.shape(self.myinput_cov))
        if path_data_ref:
            self.data_without_sample= TDS.inputdatafromfile(path_data_ref, sample = 0)    ## We load the signal of the measured pulse with sample

        self.reference_number = self.data.ref_number
        self.myreferencedata = TDS.getreferencetrace(path_data, self.reference_number)

        self.myglobalparameters.t = self.data.time*1e-12 # this assumes input files are in ps ## We load the list with the time of the experiment
        self.nsample = len(self.myglobalparameters.t)
        self.dt=self.myglobalparameters.t.item(2)-self.myglobalparameters.t.item(1)  ## Sample rate
        self.myglobalparameters.freq = np.fft.rfftfreq(self.nsample, self.dt)        ## We create a list with the frequencies for the spectrum
        self.myglobalparameters.w = self.myglobalparameters.freq*2*np.pi
        
                    # if one changes defaults values in TDSg this also has to change:
        self.Lfiltering = Lfiltering_index # 1 - Lfiltering_index 
        self.Hfiltering = Hfiltering_index # 1 - Hfiltering_index
        self.set_to_zeros = zeros_index # 1 - zeros_index
        self.dark_ramp = dark_index
        
        if modesuper == 1:
            frep=99.991499600e6 # repetition frequency of the pulse laser used in the tds measurments in Hz, 99
            nsampleZP=np.round(1/(frep*self.dt)) #number of time sample betwen two pulses. IT has to be noted that it could be better to have an integer number there then the rounding does not change much
            self.nsamplenotreal=nsampleZP.astype(int)
            self.myglobalparameters.t=np.arange(nsampleZP)*self.dt  # 0001 #
            self.myglobalparameters.freq = np.fft.rfftfreq(self.nsamplenotreal, self.dt)
            self.myglobalparameters.w = 2*np.pi*self.myglobalparameters.freq
            #self.data.time = self.myglobalparameters.t*1e12 #on met à jour le time, pas vraiment nessessair emais on sait jamais si va etre reutilise plus tard

        else:
            self.nsamplenotreal = self.nsample 
            
        Freqwindowstart = np.ones(len(self.myglobalparameters.freq))
        Freqwindowend = np.ones(len(self.myglobalparameters.freq))
        if self.Lfiltering:
            stepsmooth = cutstart/sharpcut
            Freqwindowstart = 0.5+0.5*np.tanh((self.myglobalparameters.freq-cutstart)/stepsmooth)
        if self.Hfiltering:
            #cutend = comm.bcast(cutend,root=0) #for parralellisation
            #sharpcut = comm.bcast(sharpcut,root=0)
            stepsmooth = cutend/sharpcut
            Freqwindowend = 0.5-0.5*np.tanh((self.myglobalparameters.freq-cutend)/stepsmooth)                                 
        
        self.Freqwindow = Freqwindowstart*Freqwindowend
        self.timeWindow = np.ones(self.nsamplenotreal)
        
        for trace in  range(self.data.numberOfTrace):
            #print(trace)
            myinputdata=TDS.mydata(self.data.Pulseinit[trace])    ## We create a variable containing the data related to the measured pulse with sample

            if modesuper == 1:
                self.mode = "superresolution"
                myinputdata=TDS.mydata(np.pad(myinputdata.pulse,(0,self.nsamplenotreal-self.nsample),'constant',constant_values=(0)))

                if trace == 0: # on fait le padding une seule fois sur la ref
                    self.myreferencedata.Pulseinit=np.pad(self.myreferencedata.Pulseinit,(0,self.nsamplenotreal-self.nsample),'constant',constant_values=(0))
                    self.myreferencedata.Spulseinit=(TDS.fft_gpu((self.myreferencedata.Pulseinit)))    # fft computed with GPU
            else:
                self.mode = "basic"
                
            # Filter data
            if trace == 0:
                self.myreferencedata.Spulseinit = self.myreferencedata.Spulseinit*self.Freqwindow
                self.myreferencedata.Pulseinit  = TDS.torch_irfft(self.myreferencedata.Spulseinit, n = self.nsamplenotreal)

            myinputdata.Spulse         = myinputdata.Spulse        *self.Freqwindow
            myinputdata.pulse          = TDS.torch_irfft(myinputdata.Spulse, n = self.nsamplenotreal)
                    
            if self.dark_ramp:
                #Enleve la rampe du dark noise du signal
                if trace == 0:
                    self.myreferencedata.Pulseinit = self.myreferencedata.Pulseinit - slope*self.myglobalparameters.t*1e12+intercept
                myinputdata.pulse = myinputdata.pulse - slope*self.myglobalparameters.t*1e12+intercept
                
            if self.set_to_zeros:
                #Remplace la fin du pulse d'input par des 0 (de la longueur du decalage entre les 2 pulses)
                #on veut coller le input à la ref
                imax1 = np.argmax(myinputdata.pulse)
                imax2 = np.argmax(self.myreferencedata.Pulseinit)
                tmax1 = self.myglobalparameters.t[imax1]
                tmax2 = self.myglobalparameters.t[imax2]
                deltaTmax = tmax2-tmax1
                    
                tlim1 = self.myglobalparameters.t[self.nsample-1]-(5*deltaTmax/4)
                tlim2 = self.myglobalparameters.t[self.nsample-1]-(deltaTmax)
                for k in range(self.nsample):
                    if self.myglobalparameters.t[k] < tlim1:
                        self.timeWindow[k] = 1
                    elif self.myglobalparameters.t[k] >= tlim2:
                        self.timeWindow[k] = 0
                    else:
                        term = (4/deltaTmax)*(self.myglobalparameters.t[k]-tlim1)
                        self.timeWindow[k] = 1-(3*term**2-2*term**3)
                myinputdata.pulse = myinputdata.pulse*self.timeWindow
                myinputdata.Spulse = (TDS.torch_rfft((myinputdata.pulse)))

            self.myinput.add_trace(myinputdata.pulse)
        #if modesuper:
         #   self.myinput_cov = np.cov(np.array(self.myinput.pulse)[:,:self.nsample], rowvar=False)/np.sqrt(self.data.numberOfTrace)
        #else:
         #   self.myinput_cov = np.cov(self.myinput.pulse, rowvar=False)/np.sqrt(self.data.numberOfTrace)
            
        if not os.path.isdir("temp"):
            os.mkdir("temp")
        f=open(os.path.join("temp",'temp_file_6.bin'),'wb')
        pickle.dump(self.myinput,f,pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.myreferencedata,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        
        f=open(os.path.join("temp",'temp_file_7.bin'),'wb')
        pickle.dump(self.myglobalparameters,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        
        self.data.Pulseinit = [] #don't forget important


    def choices_ini_param( self, fit_delay, delaymax_guess, delay_limit, fit_dilatation, dilatation_limit,dilatationmax_guess,
                    fit_periodic_sampling, periodic_sampling_freq_limit, fit_leftover_noise, leftcoef_guess, leftcoef_limit):
        """Process all the informations given in the first panel of initialisation: 
            create instances of classes to store data, apply filters, store choices in temp file 1_ini.
            Note that data is reload in creation of class myfitdata and before optimization, 
            any operations on data, like filters, has to be applied in the 3 cases."""


        self.fit_delay = fit_delay
        self.delaymax_guess = delaymax_guess
        self.delay_limit = delay_limit
        self.fit_leftover_noise = fit_leftover_noise
        self.leftcoef_guess = leftcoef_guess
        self.leftcoef_limit = leftcoef_limit
        self.fit_periodic_sampling = fit_periodic_sampling
        self.periodic_sampling_freq_limit = periodic_sampling_freq_limit
        self.fit_dilatation = fit_dilatation
        self.dilatation_limit = dilatation_limit
        self.dilatationmax_guess = dilatationmax_guess

            # files for choices made
        mode_choicies_opt=[self.path_data, self.path_data_ref, self.reference_number, self.fit_dilatation, self.dilatation_limit, self.dilatationmax_guess,
                               self.Freqwindow,self.timeWindow, self.fit_delay, self.delaymax_guess, self.delay_limit,  self.mode, 
                               self.fit_periodic_sampling, self.periodic_sampling_freq_limit, self.fit_leftover_noise, self.leftcoef_guess, self.leftcoef_limit]
    
        if not os.path.isdir("temp"):
            os.mkdir("temp")
        f=open(os.path.join("temp",'temp_file_1_ini.bin'),'wb')
        pickle.dump(mode_choicies_opt,f,pickle.HIGHEST_PROTOCOL)
        f.close()



# =============================================================================
# Model parameters
# =============================================================================
   
    def invalid_swarmsize(self):
        self.refreshAll3("Invalid swarmsize. \n")

    def invalid_niter(self):
        self.refreshAll3("Invalid number of iterations. \n")
        
        
# =============================================================================
# Optimization
# =============================================================================
        
    def algo_parameters(self,choix_algo,swarmsize,niter, niter_ps):
        """Save algorithm choices in temp file 5"""
        self.algo=choix_algo
        mode_choicies_opt=[choix_algo,int(swarmsize),int(niter), int(niter_ps)]
        if not os.path.isdir("temp"):
            os.mkdir("temp")

        f=open(os.path.join("temp",'temp_file_5.bin'),'wb')
        pickle.dump(mode_choicies_opt,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_5 = 1

        self.refreshAll3("")
        
        
    def begin_optimization(self,nb_proc):
        """Run optimization and update layers"""
        output=""
        error=""
        returncode=0
        if sys.platform=="win32" or sys.platform=="cygwin":
            print("OS:Windows \n")
            if not os.path.isdir("temp"):
                os.mkdir("temp")
            optimization_filename = os.path.join('temp',"opt.bat")
            try:
                with open(optimization_filename, 'w') as OPATH:
                    OPATH.writelines(['call set Path=%Path%;C:\ProgramData\Anaconda3 \n',
                   'call set Path=%Path%;C:\ProgramData\Anaconda3\condabin \n',
                   'call set Path=%Path%;C:\ProgramData\Anaconda3\Scripts \n',
                   #'call conda activate \n', 
                   'call mpiexec -n {0} python opt.py'.format(nb_proc)])
#                    OPATH.writelines([f'call mpiexec -n {nb_proc} opt.exe'])
                subprocess.call(optimization_filename)
                returncode = 0
                error = ""
                output = ""
            except:
                print("No parallelization! You don't have MPI installed or there's a problem with your MPI.")
                with open(optimization_filename, 'w') as OPATH:
                    OPATH.writelines([f'call opt.exe'])
                subprocess.call(optimization_filename)
                returncode = 0
                error = ""
                output = ""
        elif sys.platform=="linux" or sys.platform=="darwin":
            print("OS:Linux/MacOS \n")
            optimization_filename = os.path.join('temp',"opt.sh")
            try:
                # Check if Open MPI is correctly installed
                try:
                    command = 'mpiexec --version'
                    process=subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    output_mpi,error_mpi = process.communicate()
                    returncode_mpi=process.returncode
                except:
                    returncode_mpi = 1
                    error_mpi = "Command mpiexec not recognized."

                try:
                    command = './py3-env/bin/python --version'
                    process=subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    output_py3,error_py3 = process.communicate()
                    returncode_py3=process.returncode
                    python_path = "./py3-env/bin/python"
                except:
                    try:
                        command = "python3 --version"
                        process=subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                        output_py3,error_py3 = process.communicate()
                        returncode_py3=process.returncode
                        python_path = "python3"
                    except:
                        returncode_py3 = 1
                        error_py3 = "Command python3 not recognized."

                # Run optimization
                if returncode_mpi==0:
                    if returncode_py3==0:
                        command = 'mpiexec -n {0} {1} opt.py'.format(nb_proc, python_path)
                    else:
                        print("Problem with python command : \n {} \n".format(error_py3))
                        return(0)
                else:
                    print("No parallelization! You don't have MPI installed or there's a problem with your MPI: \n {}".format(error_mpi))
                    if returncode_py3==0:
                        command = '{0} opt.py'.format(python_path)
                    else:
                        print("Problem with python command : \n {} \n".format(error_py3))
                        return(0)

                try:
                    with open(optimization_filename, 'w') as OPATH:
                        OPATH.writelines(command)
                    returncode = subprocess.call('chmod +x ./{}'.format(optimization_filename),shell=True)
                    if returncode == 0:
                        returncode = subprocess.call('./{}'.format(optimization_filename),shell=True)
                    if returncode == 1:
                        command = ""
                        with open("launch_opt.py", 'w') as OPATH:
                            OPATH.writelines(command)
                        try:
                            import launch_opt
                            try:
                                f=open(os.path.join("temp",'temp_file_3.bin'),'rb')
                                f.close()
                                returncode=0
                            except:
                                print("Unknown problem.")
                                sys.exit()
                        except:
                            print("Unknown problem.")
                            sys.exit()
                except:
                    returncode = 1
                    error = "Unknow problem."
                    output = ""
            except:
                print("Unknown problem.")
                sys.exit()

        else:
            print("System not supported.")
            return(0)

        if returncode==0:
            f=open(os.path.join("temp",'temp_file_3.bin'),'rb')
            self.is_temp_file_3 = 1
            self.delay_correction = []
            self.leftover_correction = []
            self.periodic_correction = []
            self.dilatation_correction = []
            self.fopt = []
            sum_fopt = 0
            if self.fit_delay or self.fit_leftover_noise or self.fit_dilatation:
                for i in range(self.data.numberOfTrace):
                    var_inter=pickle.load(f)
                    xopt=var_inter[0]
                    fopt=var_inter[1]
                    self.fopt.append(fopt)
                    if self.fit_leftover_noise:
                        self.leftover_correction.append(xopt[-2:])
                    if self.fit_delay:
                        self.delay_correction.append(xopt[0])
                    if self.fit_dilatation:
                        if self.fit_delay:
                            self.dilatation_correction.append(xopt[1:3])
                        else:
                            self.dilatation_correction.append(xopt[0:2])
                            
                sum_fopt = np.sum(self.fopt)
            if self.fit_periodic_sampling:
                var_inter=pickle.load(f)
                xopt_ps=var_inter[0]
                fopt_ps=var_inter[1]
                self.periodic_correction.append(xopt_ps)
            f.close()
            
            f=open(os.path.join("temp",'temp_file_2.bin'),'rb')
            self.mydatacorrection = pickle.load(f)
            self.fopt_init = pickle.load(f)
            f.close()
            
            """data_for_cov = []
            if self.path_data_ref:
                if self.mode == "superresolution":
                    transfer_function = TDS.torch_rfft(np.mean(self.mydatacorrection, axis = 0)[:self.nsample])/TDS.torch_rfft(np.mean(self.data_without_sample.Pulseinit, axis = 0))
                else:
                    transfer_function = TDS.torch_rfft(np.mean(self.mydatacorrection, axis = 0))/TDS.torch_rfft(np.mean(self.data_without_sample.Pulseinit, axis = 0))
                for i in range(self.data.numberOfTrace):
                    data_for_cov.append(TDS.torch_irfft(self.data_without_sample.Spulseinit*transfer_function), n = len(self.myglobalparameters.t))
                self.mydatacorrection_cov = np.cov(data_for_cov,rowvar = False)/np.sqrt(self.data.numberOfTrace)
            else:
                if self.mode == "superresolution":
                    self.mydatacorrection_cov = np.cov(np.array(self.mydatacorrection.pulse)[:,:self.nsample], rowvar=False)/np.sqrt(self.data.numberOfTrace)
                else:
                    self.mydatacorrection_cov = np.cov(self.mydatacorrection.pulse, rowvar=False)/np.sqrt(self.data.numberOfTrace)"""

            message = "Optimization terminated successfully\nCheck the output directory for the result\n\n"
            """if self.fit_leftover_noise and self.fit_delay:
                message += 'For delay and amplitude\n    The last best error was:     {}'.format(fopt) + '\n    The last best parameters were: \t{}\n'.format(xopt[:-1]) + "\n"
            elif self.fit_delay:
                message += 'For delay\n    The last best error was:     {}'.format(fopt) + '\n    The last best parameters were:     {}\n'.format(xopt) + "\n"
            elif self.fit_leftover_noise:
                message += 'For amplitude\n    The last best error was:     {}'.format(fopt) + '\n    The last best parameters were:     {}\n'.format(xopt[:-1]) + "\n"
             """   
            if self.fit_periodic_sampling:
               message += 'For periodic sampling: \n The best error was:     {}'.format(fopt_ps) + '\nThe best parameters were:     {}\n'.format(xopt_ps) + "\n"
            if self.fit_leftover_noise or self.fit_delay or self.fit_dilatation:
                message+= "Sum of std Pulse E field (sqrt(sum(std^2))):\n   before correction \t{}\n".format(np.sqrt(sum(np.std(self.myinput.pulse, axis = 0)**2)))
                message+= "   after correction \t{}\n\n".format(np.sqrt(sum(np.std(self.mydatacorrection.pulse, axis = 0)**2)))
                
                #message+= "Sum of errors :\n   before correction \t{}\n".format(sum_fopt)
                #message+= " Sum of errors after correction \t{}\n\n".format(sum_fopt)

            citation= "Please cite this paper in any communication about any use of Correct@TDS : \nComming soon..."
            message += citation
            
            self.refreshAll3(message)
        else:
            self.refreshAll3("Output : \n {} \n".format(output))
            print("System not supported. \n")
            print('Output : \n {0} \n Error : \n {1} \n'.format(output, error))
            return(0)
        
    def save_data(self, save_all):
      
        citation= "Please cite this paper in any communication about any use of Correct@TDS : \n Coming soon..."
        title = "\n timeaxis \t E-field"
        try:
            if self.initialised: 
                    if self.optim_succeed:
                        if self.mode == "superresolution":
                            out = np.column_stack((self.data.time, np.mean(self.mydatacorrection.pulse, axis = 0)[:self.nsample]))
                            print("here")
                        else:
                            out = np.column_stack((self.myglobalparameters.t*1e12, np.mean(self.mydatacorrection.pulse, axis = 0)))
                        np.savetxt(os.path.join(self.outputdir,"corrected_mean.txt"),out, header= citation+title, delimiter = "\t)
                        
                        #hdf =  h5py.File(os.path.join(self.outputdir,"corrected_covariance_inverse.h5"),"w")
                        #dataset = hdf.create_dataset("covariance_inverse", data = np.linalg.inv(self.mydatacorrection_cov))
                        #dataset.attrs["CITATION"] = citation
                        
                        to_save = []
                        if self.fit_delay:
                            to_save.append(self.delay_correction)
                        else:
                            to_save.append([0]*self.data.numberOfTrace) #empty
                            
                        if self.fit_dilatation:
                            to_save.append([self.dilatation_correction[i][0] for i in range(self.data.numberOfTrace)])
                            to_save.append([self.dilatation_correction[i][1] for i in range(self.data.numberOfTrace)])
                        else:
                            to_save.append([0]*self.data.numberOfTrace) #empty
                            to_save.append([0]*self.data.numberOfTrace) #empty

                            
                        if self.fit_leftover_noise:
                            to_save.append([self.leftover_correction[i][0] for i in range(self.data.numberOfTrace)])
                        else:
                            to_save.append([0]*self.data.numberOfTrace) #empty
                            
                        if self.fit_periodic_sampling:
                            to_save.append([self.periodic_correction[0][0]]*self.data.numberOfTrace)  
                            to_save.append([self.periodic_correction[0][1]]*self.data.numberOfTrace)  
                            to_save.append([self.periodic_correction[0][2]]*self.data.numberOfTrace)  
                        else:
                            to_save.append([0]*self.data.numberOfTrace) #empty
                            to_save.append([0]*self.data.numberOfTrace) #empty
                            to_save.append([0]*self.data.numberOfTrace) #empty

                        np.savetxt(os.path.join(self.outputdir,"correction_parameters.txt"), np.transpose(to_save), delimiter = "\t", header=citation+"\n delay \t alpha_dilatation \t beta_dilatation \t a_amplitude \t A \t nu \t phi")
                
                        if save_all:
                            #save in hdf5
                            hdf =  h5py.File(os.path.join(self.outputdir,"corrected_traces.h5"),"w")
                            dataset = hdf.create_dataset(str("0"), data = self.mydatacorrection.pulse[0])
                            dataset.attrs["CITATION"] = citation
                            if self.mode == "superresolution":
                                hdf.create_dataset('timeaxis', data = self.data.time)
                            else:
                                hdf.create_dataset('timeaxis', data = self.myglobalparameters.t*1e12)
                            count = 1
                            for i in self.mydatacorrection.pulse[1:]:
                                if self.mode == "superresolution":
                                    dataset = hdf.create_dataset(str(count), data = i[:self.nsample])
                                else:
                                    dataset = hdf.create_dataset(str(count), data = i)
                                dataset.attrs["CITATION"] = citation
                                count+=1
                            hdf.close()
                    else:
                        if self.mode == "superresolution":
                            out = np.column_stack((self.data.time, np.mean(self.myinput.pulse, axis = 0)[:self.nsample]))
                        else:
                            out = np.column_stack((self.myglobalparameters.t*1e12, np.mean(self.myinput.pulse, axis = 0)))
                        np.savetxt(os.path.join(self.outputdir,"mean.txt"),out, delimiter = "\t", header= citation+title)
                        
                        #hdf =  h5py.File(os.path.join(self.outputdir,"covariance_inverse.h5"),"w")
                        #dataset = hdf.create_dataset("covariance_inverse", data = np.linalg.inv(self.myinput_cov))
                        #dataset.attrs["CITATION"] = citation
                        
                        if save_all:
                            #save in hdf5
                            hdf =  h5py.File(os.path.join(self.outputdir,"traces.h5"),"w")
                            dataset = hdf.create_dataset(str("0"), data = self.myinput.pulse[0])
                            dataset.attrs["CITATION"] = citation
                            if self.mode == "superresolution":
                                hdf.create_dataset('timeaxis', data = self.data.time)
                            else:
                                hdf.create_dataset('timeaxis', data = self.myglobalparameters.t*1e12)
                               
                            count = 1
                            for i in self.myinput.pulse[1:]:
                                if self.mode == "superresolution":
                                    dataset = hdf.create_dataset(str(count), data = i[:self.nsample])
                                else:
                                    dataset = hdf.create_dataset(str(count), data = i)
                                dataset.attrs["CITATION"] = citation
                                count+=1
                            hdf.close()
                    return 1
            else:
                self.refreshAll3("Please enter initialization data first")
                return 0
        except Exception as e:
            self.refreshAll3("Please enter initialization data first")
            print(e)
            return 0

            
    def loading_text3(self):
        self.refreshAll3("\n Processing... \n")

    def message_log_tab3(self,message):
        self.refreshAll3(message)

    def error_message_path3(self):
        self.refreshAll3("Error: Please enter a valid file(s).")

    def error_message_output_paths(self):
        self.refreshAll3("Invalid output paths.")

    def error_message_output_filename(self):
        self.refreshAll3("Invalid output filename.")

    def get_output_paths(self,outputdir):
        """Check output names and path and stores them in temp file 4 if they are valid. """
        try:
            self.outputdir = str(outputdir)
        except:
            self.refreshAll3("Invalid output directory.")
            return(0)
       
        output_paths = [self.outputdir]
        if not os.path.isdir("temp"):
            os.mkdir("temp")
        f=open(os.path.join("temp",'temp_file_4.bin'),'wb')
        pickle.dump(output_paths,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_4 = 1


    def ploting_text3(self,message):
        self.refreshAll3(message)
        
    def no_temp_file_1(self):
        self.refreshAll3("Unable to execute without running step Initialization.")

    def no_temp_file_4(self):
        self.refreshAll3("Unable to execute without selecting path for output data first.")
    
    def no_temp_file_5(self):
        self.refreshAll3("Unable to execute without optimization parameters")
