#!/usr/bin/env python
# coding: utf-8



#!/usr/bin/python
# -*- coding: latin-1 -*-

import os
import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import sip
import shutil
import fitf as TDS
import h5py
import seaborn as sns
import time
import fitf as TDS
from fitc import Controler
from scipy import signal
plt.rcParams.update({'font.size': 14})


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    size = comm.Get_size()
except:
    print('mpi4py is required for parallelization')
    myrank=0


"""    def on_click(self):
        textboxValue = self.textbox.text()
        QMessageBox.question(self, 'Message - pythonspot.com', "You typed: " + textboxValue, QMessageBox.Ok, QMessageBox.Ok)
        self.textbox.setText("")"""


def deleteLayout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                deleteLayout(item.layout())
        sip.delete(layout)

def to_sup(s):
    """Convert a string of digit to supescript"""
    sups = {u'0': u'\u2070',
            u'1': u'\u00b9',
            u'2': u'\u00b2',
            u'3': u'\u00b3',
            u'4': u'\u2074',
            u'5': u'\u2075',
            u'6': u'\u2076',
            u'7': u'\u2077',
            u'8': u'\u2078',
            u'9': u'\u2079'}

    return ''.join(sups.get(char, char) for char in s)


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   
   

graph_option_2=None
preview = 1


class MyTableWidget(QWidget):

    def __init__(self, parent,controler):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tab = Initialisation_tab(self,controler)

        # Add tabs to widget
        self.layout.addWidget(self.tab)
        self.setLayout(self.layout)




###############################################################################
###############################################################################
#######################   Initialisation tab   ################################
###############################################################################
###############################################################################

class Initialisation_tab(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.setMinimumSize(640, 480)
        hlayout  = QHBoxLayout()
        vlayout = QVBoxLayout()
        
        self.init_param_widget = InitParam_handler(self, controler)
        self.init_param_widget.refresh()

        vlayout.addWidget(self.init_param_widget)
        hlayout.addLayout(vlayout)
        self.setLayout(hlayout)
        

        
class InitParam_handler(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        #self.setMaximumWidth(500)

    def refresh(self):
        init_instance = InitParamWidget(self.parent,self.controler)
        try:
            deleteLayout(self.layout())
            self.layout().deleteLater()
        except:
            main_layout = QVBoxLayout()
            main_layout.addWidget(init_instance)
            self.setLayout(main_layout)


class InitParamWidget(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.controler.addClient(self)
        self.path_data = None
        self.path_data_ref = None
        
        label_width=1500
        text_box_width=150
        text_box_height=25
        
        # We create the text associated to the text box

        self.label_path_data = QLabel('Select data (hdf5) \u00b9')
        self.label_path_data.setAlignment(Qt.AlignVCenter)
        self.label_path_data.resize(200, 100)
        self.label_path_data.resize(self.label_path_data.sizeHint())
        self.label_path_data.setMaximumHeight(30)
        
        self.button_ask_path_data = QPushButton('browse')
        self.button_ask_path_data.resize(200, 100)
        self.button_ask_path_data.resize(self.button_ask_path_data.sizeHint())
        self.button_ask_path_data.clicked.connect(self.get_path_data)
        
        self.label_path_without_sample = QLabel('Select traces without sample (optional) \u00b2')
        self.label_path_without_sample.setAlignment(Qt.AlignVCenter)
        self.label_path_without_sample.resize(200, 100)
        self.label_path_without_sample.resize(self.label_path_without_sample.sizeHint())
        self.label_path_without_sample.setMaximumHeight(30)
        
        
        self.button_ask_path_without_sample = QPushButton('browse')
        self.button_ask_path_without_sample.resize(200, 100)
        self.button_ask_path_without_sample.resize(self.button_ask_path_without_sample.sizeHint())
        self.button_ask_path_without_sample.clicked.connect(self.get_path_data_ref)
        
        self.button_preview = QPushButton('Preview')
        self.button_preview.resize(200, 100)
        self.button_preview.resize(self.button_preview.sizeHint())
        #self.button_preview.clicked.connect(self.preview) todo

        self.button_data = QPushButton('Submit data')

        self.button_parameters = QPushButton('Submit correction parameters')
        self.button_parameters.clicked.connect(self.on_click_param)
        self.button_parameters.pressed.connect(self.pressed_loading)

        # We create a button to extract the information from the text boxes
        self.button = QPushButton('Submit / Preview')
        self.button.pressed.connect(self.pressed_loading)
        self.button.clicked.connect(self.on_click)
        
        # Filter or not filter
        self.LFfilter_label = QLabel('Filter low frequencies?')
        self.LFfilter_choice = QComboBox()
        self.LFfilter_choice.addItems(['No','Yes'])
        self.LFfilter_choice.setMaximumWidth(text_box_width)
        self.LFfilter_choice.setMaximumHeight(text_box_height)
        
        self.HFfilter_label = QLabel('Filter high frequencies?')
        self.HFfilter_choice = QComboBox()
        self.HFfilter_choice.addItems(['No','Yes'])
        self.HFfilter_choice.setMaximumWidth(text_box_width)
        self.HFfilter_choice.setMaximumHeight(text_box_height)
        
        self.label_start = QLabel('\tStart (Hz)')
        self.label_end   = QLabel('\tEnd (Hz)')
        self.label_sharp = QLabel('Sharpness of frequency filter \u2074')
        self.start_box = QLineEdit()
        self.end_box   = QLineEdit()
        self.sharp_box = QLineEdit()
        self.start_box.setMaximumWidth(text_box_width)
        self.start_box.setMaximumHeight(text_box_height)
        self.start_box.setText("0.18e12")
        self.end_box.setMaximumWidth(text_box_width)
        self.end_box.setMaximumHeight(text_box_height)
        self.end_box.setText("6e12")
        self.sharp_box.setMaximumWidth(text_box_width)
        self.sharp_box.setMaximumHeight(text_box_height)
        self.sharp_box.setText("10")
        
        # remove end of reference pulse
        self.label_zeros = QLabel('Set end of time trace to zero? \u2074')
        self.zeros_choice = QComboBox()
        self.zeros_choice.addItems(['No','Yes'])
        self.zeros_choice.setMaximumWidth(text_box_width-7)
        self.zeros_choice.setMaximumHeight(text_box_height)
        
        #Remove baseline "dark" noise
        self.label_dark = QLabel('Remove dark noise ramp? \u00b3')
        self.dark_choice = QComboBox()
        self.dark_choice.addItems(['No','Yes']) #Use a function to add
        self.dark_choice.setMaximumWidth(text_box_width-7)
        self.dark_choice.setMaximumHeight(text_box_height)
        self.dark_choice.currentIndexChanged.connect(self.superresolution)
        
        self.label_slope = QLabel('\tSlope')
        self.slope_box = QLineEdit()
        self.slope_box.setMaximumWidth(text_box_width-7)
        self.slope_box.setMaximumHeight(text_box_height)
        self.slope_box.setText("4e-6")
        
        self.label_intercept = QLabel('    Intercept')
        self.intercept_box = QLineEdit()
        self.intercept_box.setMaximumWidth(text_box_width-7)
        self.intercept_box.setMaximumHeight(text_box_height)
        self.intercept_box.setText("0.3e-3")
        
        # Delay
        self.label_delay = QLabel("Fit delay?")
        self.label_delay.setMaximumWidth(label_width)
        self.options_delay = QComboBox()
        self.options_delay.addItems(['No','Yes'])
        self.options_delay.setMaximumWidth(text_box_width-75)
        self.delayvalue_label = QLabel("Delay absolute value")
        self.delay_limit_box = QLineEdit()
        self.delay_limit_box.setMaximumWidth(text_box_width-24)
        self.delay_limit_box.setText("10e-12")
        
        # Leftover noise
        self.label_leftover = QLabel("Fit amplitude noise?")
        self.label_leftover.setMaximumWidth(label_width)
        self.options_leftover = QComboBox()
        self.options_leftover.addItems(['No','Yes'])
        self.options_leftover.setMaximumWidth(text_box_width-75)
        self.leftovervaluea_label = QLabel("Absolute value of         a")
        self.leftovera_limit_box = QLineEdit()
        self.leftovera_limit_box.setMaximumWidth(text_box_width-24)
        self.leftovera_limit_box.setText("10e-2")
        #self.leftovervaluec_label = QLabel(" c")
        #self.leftoverc_limit_box = QLineEdit()
        #self.leftoverc_limit_box.setMaximumWidth(text_box_width-24)  
             
        # Dilatation
        self.label_dilatation = QLabel("Fit dilatation?")
        self.label_dilatation.setMaximumWidth(label_width)
        self.options_dilatation = QComboBox()
        self.options_dilatation.addItems(['No','Yes'])
        self.options_dilatation.setMaximumWidth(text_box_width-75)
        self.dilatationvaluea_label = QLabel("Absolute value of         a")
        self.dilatationa_limit_box = QLineEdit()
        self.dilatationa_limit_box.setMaximumWidth(text_box_width-90)
        self.dilatationa_limit_box.setText("10e-3")
        self.dilatationvalueb_label = QLabel("b")
        self.dilatationb_limit_box = QLineEdit()
        self.dilatationb_limit_box.setMaximumWidth(text_box_width-90) 
        self.dilatationb_limit_box.setText("10e-12")
        
        #periodic sampling
        self.label_periodic_sampling = QLabel("Fit periodic sampling?")
        self.label_periodic_sampling.setMaximumWidth(label_width)
        self.options_periodic_sampling = QComboBox()
        self.options_periodic_sampling.addItems(['No','Yes'])
        self.options_periodic_sampling.setMaximumWidth(text_box_width-75)
        self.periodic_sampling_freq_label = QLabel("Frequency [THz]")
        self.periodic_sampling_freq_limit_box = QLineEdit()
        self.periodic_sampling_freq_limit_box.setMaximumWidth(text_box_width-24)
        
        # Super resolution
        self.label_super = QLabel("\tSuper resolution ")
        self.options_super = QComboBox()
        self.options_super.addItems(['No','Yes'])
        self.options_super.setMaximumWidth(text_box_width-24)


        # Organisation layout
        self.hlayout6=QHBoxLayout()
        self.hlayout7=QHBoxLayout()
        self.hlayout8=QHBoxLayout()
        self.hlayout9=QHBoxLayout()
        self.hlayout10=QHBoxLayout()
        self.hlayout11=QHBoxLayout()
        self.hlayout12=QHBoxLayout()
        self.hlayout13=QHBoxLayout()
        self.hlayout14=QHBoxLayout()
        self.hlayout17=QHBoxLayout()
        self.hlayout18=QHBoxLayout()
        self.hlayout19=QHBoxLayout()
        self.hlayout20=QHBoxLayout()
        self.hlayout21=QHBoxLayout()
        self.hlayout22=QHBoxLayout()
        self.hlayout23=QHBoxLayout()
        self.vlayoutmain=QVBoxLayout()
        
        
        self.hlayout6.addWidget(self.label_path_data,20)
        self.hlayout6.addWidget(self.button_ask_path_data,17)
        #self.hlayout6.addWidget(self.button_preview,17)


        self.hlayout7.addWidget(self.label_path_without_sample,20)
        self.hlayout7.addWidget(self.button_ask_path_without_sample,17)
        
        self.hlayout8.addWidget(self.button_data)
        #self.hlayout8.addWidget(self.options_super,1)
        
        self.hlayout9.addWidget(self.label_delay,0)
        self.hlayout9.addWidget(self.options_delay,1)

        self.hlayout9.addWidget(self.delayvalue_label,0)
        self.hlayout9.addWidget(self.delay_limit_box,1)
        
        self.hlayout12.addWidget(self.label_leftover,0)
        self.hlayout12.addWidget(self.options_leftover,0)
        self.hlayout12.addWidget(self.leftovervaluea_label,0)
        self.hlayout12.addWidget(self.leftovera_limit_box,1)
        #self.hlayout14.addWidget(self.leftovervaluec_label,0)
        #self.hlayout14.addWidget(self.leftoverc_limit_box,1)
        
        self.hlayout13.addWidget(self.label_dilatation,1)
        self.hlayout13.addWidget(self.options_dilatation,1)
        self.hlayout13.addWidget(self.dilatationvaluea_label,0)
        self.hlayout13.addWidget(self.dilatationa_limit_box,0)
        self.hlayout13.addWidget(self.dilatationvalueb_label,0)
        self.hlayout13.addWidget(self.dilatationb_limit_box,0)
        
        self.hlayout14.addWidget(self.label_periodic_sampling,0)
        self.hlayout14.addWidget(self.options_periodic_sampling,0)
        self.hlayout14.addWidget(self.periodic_sampling_freq_label,0)
        self.hlayout14.addWidget(self.periodic_sampling_freq_limit_box,1)
        
        self.hlayout10.addWidget(self.button_parameters)
        
        self.hlayout17.addWidget(self.LFfilter_label,1)
        self.hlayout17.addWidget(self.LFfilter_choice,0)
        self.hlayout17.addWidget(self.label_start,1)
        self.hlayout17.addWidget(self.start_box,0)
        
        self.hlayout18.addWidget(self.HFfilter_label,1)
        self.hlayout18.addWidget(self.HFfilter_choice,0)
        self.hlayout18.addWidget(self.label_end,1)
        self.hlayout18.addWidget(self.end_box,0)

        self.hlayout19.addWidget(self.label_sharp,1)
        self.hlayout19.addWidget(self.sharp_box,0)


        self.hlayout20.addWidget(self.label_zeros,0)
        self.hlayout20.addWidget(self.zeros_choice,0)
        
        self.hlayout21.addWidget(self.label_dark,0)
        self.hlayout21.addWidget(self.dark_choice,1)
        self.hlayout21.addWidget(self.label_super,0)
        self.hlayout21.addWidget(self.options_super,1)
        self.label_super.hide()
        self.options_super.hide()
        
        
        #self.hlayout8.addWidget(self.label_super,0)
        #self.hlayout8.addWidget(self.options_super,1)
        
        self.hlayout22.addWidget(self.label_slope,0)
        self.hlayout22.addWidget(self.slope_box,0)
        self.hlayout22.addWidget(self.label_intercept,0)
        self.hlayout22.addWidget(self.intercept_box,0)
        self.label_slope.hide()
        self.slope_box.hide()
        self.label_intercept.hide()
        self.intercept_box.hide()
        
        sub_layoutv2 = QVBoxLayout()
        sub_layoutv3 = QVBoxLayout()
        sub_layoutv2.addLayout(self.hlayout6)
        sub_layoutv2.addLayout(self.hlayout7)
        #sub_layoutv2.addLayout(self.hlayout8)
        
        sub_layoutv3.addLayout(self.hlayout9)
        #sub_layoutv2.addLayout(self.hlayout11)
        sub_layoutv3.addLayout(self.hlayout13)
        sub_layoutv3.addLayout(self.hlayout12)
        sub_layoutv3.addLayout(self.hlayout14)
        sub_layoutv3.addLayout(self.hlayout10)

        #sub_layoutv2.addLayout(self.hlayout23)
        init_group = QGroupBox()
        init_group.setTitle('Data Initialization')
        init_group.setLayout(sub_layoutv2)
        

        self.vlayoutmain.addWidget(init_group)
        
        init_group2 = QGroupBox()
        init_group2.setTitle('Correction parameters')
        init_group2.setLayout(sub_layoutv3)
        
        sub_layoutv = QVBoxLayout()
        sub_layoutv.addLayout(self.hlayout21)
        sub_layoutv.addLayout(self.hlayout22)

        sub_layoutv.addLayout(self.hlayout17)
        sub_layoutv.addLayout(self.hlayout18)
        sub_layoutv.addLayout(self.hlayout19)
        #sub_layoutv.addLayout(self.hlayout20)
        filter_group = QGroupBox()
        filter_group.setTitle('Filters')
        filter_group.setLayout(sub_layoutv)
        self.vlayoutmain.addWidget(filter_group)
        self.vlayoutmain.addWidget(self.button)
        self.vlayoutmain.addWidget(init_group2)

        
        self.action_widget = Action_handler(self,controler)
        self.action_widget.refresh()
        self.save_param = Saving_parameters(self,controler)
        #self.save_param.refresh()
        self.graph_widget = Graphs_optimisation(self,controler)
        
        
        self.vlayoutmain.addWidget(self.action_widget)
        self.vlayoutmain.addWidget(self.save_param)
        
        self.text_box = TextBoxWidget(self, controler)
        self.vlayoutmain.addWidget(self.text_box,0)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(self.vlayoutmain,0)
        main_layout.addWidget(self.graph_widget,1)
        self.setLayout(main_layout)

    def superresolution(self):
        if self.dark_choice.currentIndex() == 1:
            self.label_super.show()
            self.options_super.show()
            self.label_slope.show()
            self.slope_box.show()
            self.label_intercept.show()
            self.intercept_box.show()
        else:
            self.label_super.hide()
            self.options_super.setCurrentIndex(0)
            self.options_super.hide()
            self.label_slope.hide()
            self.slope_box.hide()
            self.label_intercept.hide()
            self.intercept_box.hide()

    def pressed_loading1(self):
        self.controler.loading_text()

    def on_click(self):
        global graph_option_2, preview
        try:
            Lfiltering_index = self.LFfilter_choice.currentIndex()
            Hfiltering_index = self.HFfilter_choice.currentIndex()
            #zeros_index = self.zeros_choice.currentIndex()
            zeros_index = 0
            dark_index = self.dark_choice.currentIndex()
            cutstart = float(self.start_box.text())
            cutend   = float(self.end_box.text())
            cutsharp = float(self.sharp_box.text())
            modesuper = 0
            if(dark_index):
                modesuper = self.options_super.currentIndex()
            slope = float(self.slope_box.text())
            intercept = float(self.intercept_box.text())

            try:
                self.controler.choices_ini(self.path_data, self.path_data_ref,
                                               Lfiltering_index, Hfiltering_index, zeros_index, dark_index, cutstart, 
                                               cutend, cutsharp, slope, intercept, modesuper)
                graph_option_2='Pulse (E_field)'
                preview = 1
                self.graph_widget.refresh()
                self.controler.refreshAll3(" Data initialization done")
            except Exception as e:
                print(e)
                self.controler.error_message_path3()
                return(0)
        except:
            self.controler.refreshAll3("Invalid parameters, please enter real values only")
        self.controler.initialised=1
        self.controler.optim_succeed = 0

    def on_click_param(self):
        global preview, graph_option_2
        if not self.controler.initialised:
            self.controler.refreshAll3("Please submit initialization data first")
            return(0)
        try:
            fit_delay = 0
            delay_guess = 0
            delay_limit = 0
            fit_leftover_noise = 0
            leftover_guess = np.zeros(2)
            leftover_limit = np.zeros(2)
            fit_periodic_sampling = 0
            periodic_sampling_freq_limit = 0
            fit_dilatation = 0
            dilatation_limit = np.zeros(2)
            dilatationmax_guess = np.zeros(2)

            if self.options_delay.currentIndex():
                fit_delay = 1
                delay_limit = float(self.delay_limit_box.text())
                if delay_limit <=0:
                    raise ValueError
            if self.options_leftover.currentIndex():
                fit_leftover_noise = 1
                leftover_limit[0] = float(self.leftovera_limit_box.text())
                #leftover_limit[1] = float(self.leftoverc_limit_box.text())
                leftover_limit[1] = 10e-100
                
                if leftover_limit[0]<=0 or leftover_limit[1]<=0:
                    raise ValueError

            if self.options_dilatation.currentIndex():
                fit_dilatation = 1
                dilatation_limit[0] = float(self.dilatationa_limit_box.text())
                dilatation_limit[1] = float(self.dilatationb_limit_box.text())
                
                if dilatation_limit[0]<=0 or dilatation_limit[1]<=0:
                    raise ValueError
            if self.options_periodic_sampling.currentIndex():
                fit_periodic_sampling = 1
                periodic_sampling_freq_limit = float(self.periodic_sampling_freq_limit_box.text())
                if periodic_sampling_freq_limit < 0 or periodic_sampling_freq_limit > self.controler.myglobalparameters.freq[-1]*1e-12:
                    raise ValueError
                    
            try:
                self.controler.choices_ini_param( fit_delay, delay_guess, delay_limit, fit_dilatation, dilatation_limit, dilatationmax_guess,
                                               fit_periodic_sampling, periodic_sampling_freq_limit,
                                              fit_leftover_noise ,leftover_guess, leftover_limit)
                self.controler.refreshAll3(" Parameters initialization done")
                self.controler.initialised_param = 1
                self.controler.optim_succeed = 0
            except Exception as e:
                print(e)
                return(0)
        except:
            self.controler.refreshAll3("Invalid parameters, please enter real positive values only and valid frequency range")
            
        try:
            preview = 1
            self.graph_widget.refresh()
        except:
            print("unknown error")
            
    def get_path_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Data (hdf5 file)", options=options, filter="Hdf5 File(*.h5)")
        try:
            self.path_data=fileName
            name=os.path.basename(fileName)
            if name:
                self.button_ask_path_data.setText(name)
            else:
                self.button_ask_path_data.setText("browse")
            self.controler.initialised = 0
        except:
            self.controler.error_message_path()

    def get_path_data_ref(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Reference Data (hdf5 file)", options=options,  filter="Hdf5 File(*.h5)")
        try:
            self.path_data_ref=fileName
            name=os.path.basename(fileName)
            if name:
                self.button_ask_path_without_sample.setText(name)
            else:
                self.button_ask_path_without_sample.setText("browse")
            self.controler.initialised = 0
            
        except:
            self.controler.error_message_path()

    def pressed_loading(self):
        self.controler.loading_text3()
        
    def refresh(self):# /!\ Optimization_tab is not a client of the controler, 
    #this is not called by the controler, only when action_choice is changed.
        deleteLayout(self.action_widget.layout())
        self.action_widget.refresh()
        self.controler.refreshAll3('')

class TextBoxWidget(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient(self)
        self.controler.addClient3(self)
        self.setReadOnly(True)
        #self.append("Log")
        self.setMinimumWidth(560)
        
        references = ['\u00b9 Time should be in ps.\nDatasets in the hdf5 must be named ["timeaxis", "0", "1", ..., "N-1"]\nExample: if we have 1000 time traces, N-1 = 999',
                      '\u00b2 Use to take into account the reference traces in the covariance computation \n(only if the initial data are measures with a sample)',
                      '\u00b3 Use a linear function (or custom function) to remove the contribution of the dark noise after 200GHz.',
                      '\u2074 Sharpness: 100 is almost a step function, 0.1 is really smooth. See graphs in optimization tab.'
                      
                      ]
        
        references_2=["\u2075 Error options:",
                    "Constant weight: error = \u2016Efit(w)-E(w)\u2016 / \u2016E(w)\u2016",
                    "or error = \u2016Efit(t)-E(t)\u2016 / \u2016E(t)\u2016 in super resolution"]
            
        for reference in references:
            self.append(reference)
            self.append('')
            
        #for i in references_2:
        #    self.append(i)
            
    def refresh(self):
        message = self.controler.message
        if type(message)==list:
            for i in message:
                self.append(i)
        else:
            self.append(message)
            



###############################################################################
###############################################################################
#########################   Optimisation tab   ################################
###############################################################################
###############################################################################



class Action_handler(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        #self.setMaximumWidth(500)

    def refresh(self):
        self.action_widget = Optimization_choices(self,self.controler)

        try:
            deleteLayout(self.layout())
            self.layout().deleteLater()
        except:
            main_layout = QVBoxLayout()
            main_layout.addWidget(self.action_widget)
            self.setLayout(main_layout)

class Optimization_choices(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler=controler
        self.parent = parent
        self.emptycellfile = None
        self.setTitle("Optimization")
        self.algo_index = 0
        # Corrective factors
        label_width=200
        action_widget_width=150
        corrective_width_factor=-12
        
        
        # Algorithm choice
        self.label_algo = QLabel("Algorithm - delay/amplitude/dilatation")
        #self.label_algo.setMaximumWidth(label_width)
        self.options_algo = QComboBox()
        self.options_algo.addItems(['NumPy optimize swarm particle',
                                    'ALPSO without parallelization',
                                    'ALPSO with parallelization',
                                    'SLSQP (pyOpt)',
                                    'SLSQP (pyOpt with parallelization)',
                                    'L-BFGS-B',
                                    'SLSQP (scipy)',
                                    'Dual annealing'])
        #self.options_algo.setMaximumWidth(action_widget_width+40)
        self.options_algo.currentIndexChanged.connect(self.refresh_param)
        
        
        self.label_algo_ps = QLabel("Algorithm - periodic sampling")
        #self.label_algo.setMaximumWidth(label_width)
        self.options_algo_ps = QComboBox()
        self.options_algo_ps.addItems(['Dual annealing'])
        #self.options_algo.setMaximumWidth(action_widget_width+40)
        self.options_algo_ps.currentIndexChanged.connect(self.refresh_param)
        
        
        # Error choice
        self.label_error = QLabel("Error function weighting \u2075")
        #self.label_error.setMaximumWidth(label_width)
        self.options_error = QComboBox()
        self.options_error.addItems(['Constant'])
       # self.options_error.setMaximumWidth(action_widget_width-12)
        self.options_error.currentIndexChanged.connect(self.refresh_param)
        
            # Number of iterations
        self.label_niter = QLabel("\tIterations")
        self.enter_niter = QLineEdit()
        self.enter_niter.setMaximumWidth(action_widget_width +corrective_width_factor)
        self.enter_niter.setMaximumHeight(30)  
    
            # Number of iterations ps
        self.label_niter_ps = QLabel("Iterations")
        self.enter_niter_ps = QLineEdit()
        self.enter_niter_ps.setMaximumWidth(action_widget_width +corrective_width_factor)
        self.enter_niter_ps.setMaximumHeight(30)   
        self.enter_niter_ps.setText("1000")
        
                    # SwarmSize
        self.label_swarmsize = QLabel("    Swarmsize")
        self.enter_swarmsize = QLineEdit()
        self.enter_swarmsize.setMaximumWidth(action_widget_width +corrective_width_factor)
        self.enter_swarmsize.setMaximumHeight(30)
        
        
            # Button to launch optimization
        self.begin_button = QPushButton("Begin Optimization")
        self.begin_button.clicked.connect(self.begin_optimization)
        self.begin_button.pressed.connect(self.pressed_loading)
        #self.begin_button.setMaximumWidth(50)
        #self.begin_button.setMaximumHeight(20)
        
        # Wiget to see how many process are going to be used for the omptimization
        self.label_nb_proc = QLabel("How many process do you want to use?")
        self.enter_nb_proc = QLineEdit()
        self.enter_nb_proc.setText('1')
        self.enter_nb_proc.setMaximumWidth(50)
        self.enter_nb_proc.setMaximumHeight(25)

        # Creation layouts
        sub_layout_h=QHBoxLayout()
        sub_layout_h_2=QHBoxLayout()
        sub_layout_h_3=QHBoxLayout()
        sub_layout_h_7=QHBoxLayout()
        sub_layout_h_8=QHBoxLayout()
        main_layout=QVBoxLayout()
        

        # Organisation layouts
        
        
        # Organisation layouts for optimisation
        sub_layout_h_7.addWidget(self.label_algo,0)
        sub_layout_h_7.addWidget(self.options_algo,0)
        sub_layout_h_3.addWidget(self.label_niter,0)
        sub_layout_h_3.addWidget(self.enter_niter,0)
        sub_layout_h_3.addWidget(self.label_swarmsize,0)
        sub_layout_h_3.addWidget(self.enter_swarmsize,0)
        sub_layout_h_2.addWidget(self.label_algo_ps,0)
        sub_layout_h_2.addWidget(self.options_algo_ps,0)
        sub_layout_h_2.addWidget(self.label_niter_ps,0)
        sub_layout_h_2.addWidget(self.enter_niter_ps,0)
        sub_layout_h_8.addWidget(self.begin_button)
        #sub_layout_h_8.addWidget(self.options_error)

        # Vertical layout   
        main_layout.addLayout(sub_layout_h_7)
        main_layout.addLayout(sub_layout_h_3)
        main_layout.addLayout(sub_layout_h_2)
        main_layout.addLayout(sub_layout_h_8)
        
        self.setLayout(main_layout)



    def pressed_loading(self):
        self.controler.loading_text3()
        
    def begin_optimization(self):
        global preview
        t1 = time.time()
        global graph_option_2
        submitted = self.submit_algo_param()    #get values from optimisation widget
        if submitted == 1:
            try:
                from mpi4py import MPI
                nb_proc=int(self.enter_nb_proc.text())
            except:
                self.controler.message_log_tab3("You don't have MPI for parallelization, we'll use only 1 process")
                nb_proc=1
            if self.controler.is_temp_file_5 == 1:
                self.controler.begin_optimization(nb_proc)
                graph_option_2='Pulse (E_field)'
            else:
                self.controler.no_temp_file_5()
        preview = 0
        print("\nTime taken by the optimization:")
        self.controler.optim_succeed = 1
        self.controler.refreshAll3("")
        print(time.time()-t1)
    
    def refresh_param(self):
        #self.parent.parent.save_param.refresh()
        #self.controler.errorIndex = self.options_error.currentIndex() # for graph
        self.algo_index = self.options_algo.currentIndex()
        if self.algo_index < 3:
            self.label_swarmsize.show()
            self.enter_swarmsize.show()
        else :
            self.label_swarmsize.hide()
            self.enter_swarmsize.hide()
            
            
    def submit_algo_param(self):
        choix_algo=self.algo_index
        swarmsize = 0
        if not self.controler.initialised  or not self.controler.initialised_param:
            self.controler.refreshAll3("Please submit initialization and/or correction parameters first")
            return(0)
        
        if not self.controler.fit_delay  and not self.controler.fit_leftover_noise and not self.controler.fit_periodic_sampling and not self.controler.fit_dilatation:
            self.controler.refreshAll3("Please submit correction parameters first")
            return(0)
        if self.controler.fit_delay or self.controler.fit_leftover_noise or self.controler.fit_dilatation:
            if (choix_algo == 3 or choix_algo == 4):
                try:
                    from pyOpt import SLSQP
                except:
                    self.controler.refreshAll3("SLSQP was not imported successfully and can't be used")
                    return(0)
            if choix_algo <3: #particle swarm
                try:
                    swarmsize=int(self.enter_swarmsize.text())
                    if swarmsize<0:
                        self.controler.invalid_swarmsize()
                        return(0)
                except:
                    self.controler.invalid_swarmsize()
                    return(0)
        try:
            niter = 0
            niter_ps = 0
            if self.controler.fit_delay or self.controler.fit_leftover_noise or self.controler.fit_dilatation:
                niter=int(self.enter_niter.text())
                if niter<0:
                    self.controler.invalid_niter()
                    return(0)
            if self.controler.fit_periodic_sampling :
                niter_ps=int(self.enter_niter_ps.text())
                if niter_ps<0:
                    self.controler.invalid_niter()
                    return(0)
            self.controler.algo_parameters(choix_algo,swarmsize,niter,niter_ps)
        except Exception as e:
            print(e)
            print(e)
            self.controler.invalid_niter()
            return(0)
        return(1)
        
class Saving_parameters(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setTitle("Saving results")
        
        # Corrective factors
        action_widget_width=150
        corrective_width_factor=-12

        
        # Widget to see output directory
        self.label_outputdir = QLabel('Output directory: ')
        #self.label_outputdir.setMaximumWidth(label_width)
        self.button_outputdir = QPushButton('browse', self)
        self.button_outputdir.clicked.connect(self.get_outputdir)
       # self.button_outputdir.setMaximumWidth(action_widget_width +
                                        #   corrective_width_factor)
        #self.button_outputdir.setMaximumHeight(30)    
        
        self.label_save = QLabel('Save each traces?')
        self.save_choice = QComboBox()
        self.save_choice.addItems(['No','Yes'])
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_data)
        self.save_button.pressed.connect(self.pressed_loading)

        
        sub_layout_h1=QHBoxLayout()
        sub_layout_h2=QHBoxLayout()
        sub_layout_h3=QHBoxLayout()            

        sub_layout_h3.addWidget(self.label_outputdir,0)
        sub_layout_h3.addWidget(self.button_outputdir,0)
        sub_layout_h3.addWidget(self.label_save,0)
        sub_layout_h3.addWidget(self.save_choice,0)
        sub_layout_h2.addWidget(self.save_button,0)
        
        self.main_layout=QVBoxLayout()
        self.main_layout.addLayout(sub_layout_h3)
        self.main_layout.addLayout(sub_layout_h2)
        
        self.setLayout(self.main_layout)
    
    def save_data(self):
        global preview
        try:
            self.controler.get_output_paths(self.outputdir)
        except:
            self.controler.error_message_output_paths()
            return(0)
        try:
            saved = self.controler.save_data(self.save_choice.currentIndex())
            if saved:
                if not self.controler.optim_succeed:
                    preview = 1
                self.controler.refreshAll3("Done")
            else:
                print("Something went wrong")
        except:
            print("Unknown error")
            
    def refresh():
        pass
        
    def pressed_loading(self):
        self.controler.loading_text3()
        
    def get_outputdir(self):
        DirectoryName = QFileDialog.getExistingDirectory(self,"Select Directory")
        try:
            self.outputdir=str(DirectoryName)
            name=os.path.basename(str(DirectoryName))
            if name:
                self.button_outputdir.setText(name)
            else:
                self.button_outputdir.setText("browse")
        except:
            self.controler.error_message_path3()        
                




class Graphs_optimisation(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient3(self)
        self.setTitle("Graphs")
        # Create objects to plot graphs
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.draw() 
        
        # Create buttons to chose what to plot
        # E field
        #self.button_E_field = QPushButton('E field', self)
        #self.button_E_field.clicked.connect(self.E_field_graph)
        # E field [dB]
        self.button_E_field_dB = QPushButton('\nE field  [dB]\n', self)
        self.button_E_field_dB.clicked.connect(self.E_field_dB_graph)
        # Pulse (E field)
        self.button_Pulse_E_field = QPushButton('\nPulse E field\n', self)
        self.button_Pulse_E_field.clicked.connect(self.Pulse_E_field_graph)
        # Pulse (E field) [dB]
        """self.button_Pulse_E_field_dB = QPushButton('Pulse E field [dB]', self)
        self.button_Pulse_E_field_dB.clicked.connect(self.Pulse_E_field_dB_graph)
        # E field residue
        self.button_E_field_residue = QPushButton('E field residue', self)
        self.button_E_field_residue.clicked.connect(self.E_field_residue_graph)
        # E field residue [dB]
        self.button_E_field_residue_dB = QPushButton('E field residue [dB]', self)
        self.button_E_field_residue_dB.clicked.connect(self.E_field_residue_dB_graph)
        # Pulse residue (E field)
        self.button_Pulse_E_field_residue = QPushButton('E(t) residue', self)
        self.button_Pulse_E_field_residue.clicked.connect(self.Pulse_E_field_residue_graph)
        # Pulse residue (E field) [dB]
        self.button_Pulse_E_field_residue_dB = QPushButton('E(t) residue [dB]', self)
        self.button_Pulse_E_field_residue_dB.clicked.connect(self.Pulse_E_field_residue_dB_graph)
                # delay
        self.button_Delay = QPushButton('Delay', self)
        self.button_Delay.clicked.connect(self.Delay_graph)
                # coef a
        self.button_CoefA = QPushButton('Coef a', self)
        self.button_CoefA.clicked.connect(self.CoefA_graph)
                # coef c
        self.button_CoefC = QPushButton('Coef c', self)
        self.button_CoefC.clicked.connect(self.CoefC_graph)
        
                # delay
        #self.button_Delay_by_index = QPushButton('Delay by index', self)
        #self.button_Delay_by_index.clicked.connect(self.Delay_by_index_graph)"""
        
                # Pulse (E field) std
        self.button_Pulse_E_field_std = QPushButton('\nStd Pulse E field\n', self)
        self.button_Pulse_E_field_std.clicked.connect(self.Pulse_E_field_std_graph)
        # Pulse (E field) [dB] std
        self.button_E_field_dB_std = QPushButton('\nStd E field [dB]\n', self)
        self.button_E_field_dB_std.clicked.connect(self.E_field_std_dB_graph)
        
        #Phase
        self.button_Phase = QPushButton('\nPhase\n', self)
        self.button_Phase.clicked.connect(self.Phase_graph)

        #self.button_Phase_std = QPushButton('Std Phase', self)
        #self.button_Phase_std.clicked.connect(self.Phase_std_graph)
        
        #Parameters
        self.button_Correction_param = QPushButton('\nCorrection parameters\n', self)
        self.button_Correction_param.clicked.connect(self.Correction_param_graph)
        
        #Parameters
        #self.button_Correction_param2 = QPushButton('\nCorrection parameters2\n', self)
        #self.button_Correction_param2.clicked.connect(self.Correction_param_graph2)

        #Parameters
        #self.button_Errors = QPushButton('\nErrors\n', self)
        #self.button_Errors.clicked.connect(self.Errors)
        
        #Covariance
        #self.button_Cov_Pulse_E_field= QPushButton('(@_@)\nCovariance matrix\n!!can take time!!', self)
        #self.button_Cov_Pulse_E_field.clicked.connect (self.Covariance_Pulse)
        
        self.label_window = QLabel("Window")  

        # Organisation layout
        self.vlayoutmain = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.hlayout2 = QHBoxLayout()
        self.hlayout3=QHBoxLayout()
        #self.hlayout.addWidget(self.button_E_field)
        self.hlayout.addWidget(self.button_Pulse_E_field)
        self.hlayout.addWidget(self.button_E_field_dB)
        #self.hlayout.addWidget(self.button_Pulse_E_field_dB)
        self.hlayout.addWidget(self.button_Pulse_E_field_std)
        self.hlayout.addWidget(self.button_E_field_dB_std)
        self.hlayout.addWidget(self.button_Phase)
        self.hlayout.addWidget(self.button_Correction_param)
        #self.hlayout.addWidget(self.button_Correction_param2)
        #self.hlayout.addWidget(self.button_Errors)
        #self.hlayout.addWidget(self.button_Cov_Pulse_E_field)
        
        self.hlayout3.addWidget(self.label_window,0)
        window_group = QGroupBox()
        window_group.setLayout(self.hlayout3)
        window_group.setMaximumWidth(100)
        window_group.setMaximumHeight(60)


        #self.hlayout.addWidget(self.button_Delay)
        #self.hlayout.addWidget(self.button_CoefA)
        #self.hlayout2.addWidget(self.button_E_field_residue)
        #self.hlayout2.addWidget(self.button_E_field_residue_dB)
        #self.hlayout2.addWidget(self.button_Pulse_E_field_residue)
        #self.hlayout2.addWidget(self.button_Pulse_E_field_residue_dB)
        #self.hlayout2.addWidget(self.button_Phase_std)
        #self.hlayout.addWidget(self.button_Delay_by_index)
        #self.hlayout2.addWidget(self.button_CoefC)

        self.vlayoutmain.addWidget(self.toolbar)
        #self.vlayoutmain.addWidget(window_group,Qt.AlignRight)

        self.vlayoutmain.addWidget(self.canvas)
        self.vlayoutmain.addLayout(self.hlayout)
        self.vlayoutmain.addLayout(self.hlayout2)
        self.setLayout(self.vlayoutmain)
        #self.drawgraph()


    def draw_graph_init(self,myinput, myreferencedata, ref_number, mydatacorrection, delay_correction, dilatation_correction, leftover_correction,myinput_cov, mydatacorrection_cov,
                        myglobalparameters, fopt, fopt_init, mode, preview):
        global graph_option_2
        self.figure.clf()
        
        apply_window = 1
        nsample = len(myinput.pulse[-1])
        windows = np.ones(nsample)

        if apply_window and mode == "basic":
            windows = signal.tukey(nsample, alpha = 0.05)

        if graph_option_2=='E_field [dB]':
            self.figure.clf()
            ax1 = self.figure.add_subplot(111)
            if mode == "basic":
                ax1.set_title('E_field ', fontsize=10)
            else:
                ax1.set_title('E_field', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('E_field [dB]',color=color)
            ax1.plot(myglobalparameters.freq,20*np.log(abs(TDS.torch_rfft(np.mean(myinput.pulse, axis = 0)*windows)))/np.log(10), 'b-', label='mean spectre (log)')
            if not preview:
                ax1.plot(myglobalparameters.freq,20*np.log(abs(np.fft.rfft(myreferencedata.Pulseinit*windows)))/np.log(10), 'g-', label='reference spectre (log)')
                ax1.plot(myglobalparameters.freq,20*np.log(abs(np.fft.rfft(np.mean(mydatacorrection.pulse, axis = 0)*windows)))/np.log(10), 'r-', label='corrected mean spectre (log)')
                #ax1.plot(myglobalparameters.freq,20*np.log(abs(np.std(np.fft.rfft(myinput.pulse, axis = 1), axis = 0)))/np.log(10), 'b--', label='std spectre (log)')
                #ax1.plot(myglobalparameters.freq,20*np.log(abs(np.std(np.fft.rfft(mydatacorrection.pulse, axis = 1), axis = 0)))/np.log(10), 'r--', label='corrected std spectre (log)')
            ax1.legend()
            ax1.grid()

            
        elif graph_option_2=='Pulse (E_field)':
            self.figure.clf()
            ax1 = self.figure.add_subplot(111)
            if mode == "basic":
                ax1.set_title('Pulse (E_field) - ', fontsize=10)
            else:
                ax1.set_title('Pulse (E_field)', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Amplitude',color=color)
            ax1.plot(myglobalparameters.t, np.mean(myinput.pulse, axis = 0)*windows, 'b-', label='mean pulse')
            if not preview:
                ax1.plot(myglobalparameters.t, myreferencedata.Pulseinit*windows, 'g-', label='reference pulse')
                ax1.plot(myglobalparameters.t, np.mean(mydatacorrection.pulse, axis = 0)*windows, 'r-', label='corrected mean pulse')
            ax1.legend()
            ax1.grid()
           
                
        elif graph_option_2=='Correction parameters':
            self.figure.clf()
            ax1 = self.figure.add_subplot(221)
            ax2 = self.figure.add_subplot(222)
            ax3 = self.figure.add_subplot(223)
            ax4 = self.figure.add_subplot(224)

            ax1.set_title('Delay', fontsize=10)
            ax2.set_title('Coef a - amplitude', fontsize=10)
            
            color = 'tab:red'
            ax1.set_xlabel("Trace index")
            ax1.set_ylabel('Delay [s]',color=color)
            
            ax2.set_xlabel("Trace index")
            ax2.set_ylabel('Coef a',color=color)
            
            ax3.set_title('alpha- dilatation', fontsize=10)
            ax4.set_title('beta - dilatation', fontsize=10)
            
            color = 'tab:red'
            ax3.set_xlabel("Trace index")
            ax3.set_ylabel('alpha',color=color)
            
            ax4.set_xlabel("Trace index")
            ax4.set_ylabel('beta',color=color)
            
            if not preview:
                if delay_correction and len(delay_correction)>1:
                    ax1.plot(delay_correction, "b.-")
                    ax1.plot(ref_number, delay_correction[ref_number], "rv")
                    ax1.annotate(" Reference", (ref_number,delay_correction[ref_number]))
                if leftover_correction and len(leftover_correction)>1:
                    ax2.plot([leftover_correction[i][0] for i in range(len(leftover_correction))], "b.-")
                    ax2.plot(ref_number, leftover_correction[ref_number][0], "rv")
                    ax2.annotate(" Reference", (ref_number,leftover_correction[ref_number][0]))
                if dilatation_correction and len(dilatation_correction)>1:
                    ax3.plot([dilatation_correction[i][0] for i in range(len(dilatation_correction))], "b.-")
                    ax3.plot(ref_number, dilatation_correction[ref_number][0], "rv")
                    ax3.annotate(" Reference", (ref_number,dilatation_correction[ref_number][0]))
                    
                    ax4.plot([dilatation_correction[i][1] for i in range(len(dilatation_correction))], "b.-")
                    ax4.plot(ref_number, dilatation_correction[ref_number][1], "rv")
                    ax4.annotate(" Reference", (ref_number,dilatation_correction[ref_number][1]))
                    
                ax1.grid()
                ax2.grid() 
                ax3.grid()
                ax4.grid()
                
        elif graph_option_2=='Errors':
            self.figure.clf()
            ax1 = self.figure.add_subplot(111)
            
            color = 'tab:red'
            ax1.set_xlabel("Trace index")
            ax1.set_ylabel('Error',color=color)
            
            if not preview:
                if len(fopt_init)>1 and len(fopt)>1:
                    ax1.plot(fopt_init, "b.-", label = "before correction (with initial guess)" )
                    ax1.plot(ref_number, fopt_init[ref_number], "bv")
                    ax1.annotate(" Reference", (ref_number,fopt_init[ref_number]))
                    
                    ax1.plot(fopt, "r.--", label = "after correction" )
                    ax1.plot(ref_number, fopt[ref_number], "rv")
                    ax1.annotate(" Reference", (ref_number,fopt[ref_number]))
                    
                ax1.grid()
            
                
        elif graph_option_2=='Std Pulse (E_field)':
            self.figure.clf()
            ax1 = self.figure.add_subplot(111)
            if mode == "basic":
                ax1.set_title('Std Pulse (E_field) ', fontsize=10)
            else:
                ax1.set_title('Std Pulse (E_field)', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Standard deviation Pulse (E_field)',color=color)
            ax1.plot(myglobalparameters.t, np.std(myinput.pulse*windows, axis = 0), 'b-', label='mean pulse')
            if not preview:
                ax1.plot(myglobalparameters.t, np.std(mydatacorrection.pulse*windows, axis = 0), 'r-', label='corrected mean pulse')
            ax1.legend()
            ax1.grid()
            
        elif graph_option_2 == 'Std E_field [dB]':
            self.figure.clf()
            ax1 = self.figure.add_subplot(111)
            if mode == "basic":
                ax1.set_title('Standard deviation E_field [dB] ', fontsize=10)
            else:
                ax1.set_title('Standard deviation E_field [dB]', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Std E_field [dB]',color=color)
            ax1.plot(myglobalparameters.freq,20*np.log(abs(np.std(np.fft.rfft(myinput.pulse*windows, axis = 1), axis = 0)))/np.log(10), 'b-', label='std spectre (log)')
            if not preview:
                ax1.plot(myglobalparameters.freq,20*np.log(abs(np.std(np.fft.rfft(mydatacorrection.pulse*windows, axis = 1), axis = 0)))/np.log(10), 'r-', label='corrected std spectre (log)')
            ax1.legend()
            ax1.grid()
            
            
        elif graph_option_2 == 'Phase':
            self.figure.clf()
            ax1 = self.figure.add_subplot(111)
            if mode == "basic":
                ax1.set_title('Phase ', fontsize=10)
            else:
                ax1.set_title('Phase', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Phase',color=color)
            ax1.plot(myglobalparameters.freq,np.unwrap(np.angle(np.fft.rfft(np.mean(myinput.pulse, axis = 0)*windows)))/myglobalparameters.freq, 'b-', label='mean phase')
            if not preview:
                ax1.plot(myglobalparameters.freq,np.unwrap(np.angle(np.fft.rfft(myreferencedata.Pulseinit*windows)))/myglobalparameters.freq, 'g-', label='reference phase')
                ax1.plot(myglobalparameters.freq,np.unwrap(np.angle(np.fft.rfft(np.mean(mydatacorrection.pulse, axis = 0)*windows)))/myglobalparameters.freq, 'r-', label='corrected mean phase')
            ax1.legend()
            ax1.grid()
        
        elif graph_option_2 == "Covariance Pulse E field":
            self.figure.clf()
            ax1 = self.figure.add_subplot(121)
            ax2 = self.figure.add_subplot(122)
            
            ax1.set_title('Covariance matrix (log) - before correction', fontsize=10)
            ax2.set_title('Covariance matrix (log) - after correction', fontsize=10)
            
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Time [s]',color=color)
            
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Time [s]',color=color)
            temp = 10*np.log(abs(myinput_cov)**2)/np.log(10)
            #mini = np.partition(temp.flatten(), 1000)[1000]
            maxi = np.max(temp)
            mini = np.min(temp)
            im = ax1.imshow(temp, vmin=mini, vmax=maxi, cmap = sns.cm.rocket)
            plt.colorbar(im, ax = ax1)            
            if not preview:
                temp = 10*np.log(abs(mydatacorrection_cov)**2)/np.log(10)
                im = ax2.imshow(temp, vmin=mini, vmax=maxi, cmap = sns.cm.rocket)
                plt.colorbar(im, ax = ax2)

            
        self.figure.tight_layout()
        self.canvas.draw()


    def E_field_graph(self):
        global graph_option_2
        graph_option_2='E_field'
        self.controler.ploting_text3('Ploting E_field')

    def E_field_dB_graph(self):
        global graph_option_2
        graph_option_2='E_field [dB]'
        self.controler.ploting_text3('Ploting E_field [dB]')

    def Pulse_E_field_graph(self):
        global graph_option_2
        graph_option_2='Pulse (E_field)'
        self.controler.ploting_text3('Ploting pulse E_field')

    def Pulse_E_field_dB_graph(self):
        global graph_option_2
        graph_option_2='Pulse (E_field) [dB]'
        self.controler.ploting_text3('Ploting pulse E_field [dB]')
        

    def E_field_residue_graph(self):
        global graph_option_2
        graph_option_2='E_field residue'
        self.controler.ploting_text3('Ploting E_field residue')

    def E_field_residue_dB_graph(self):
        global graph_option_2
        graph_option_2='E_field residue [dB]'
        self.controler.ploting_text3('Ploting E_field residue [dB]')

    def Pulse_E_field_residue_graph(self):
        global graph_option_2
        graph_option_2='Pulse (E_field) residue'
        self.controler.ploting_text3('Ploting pulse E_field residue')

    def Pulse_E_field_residue_dB_graph(self):
        global graph_option_2
        graph_option_2='Pulse (E_field) residue [dB]'
        self.controler.ploting_text3('Ploting pulse E_field residue [dB]')
    
    def Delay_graph(self):
        global graph_option_2
        graph_option_2='Delay'
        self.controler.ploting_text3('Ploting Delay histogram')
        
    def Delay_by_index_graph(self):
        global graph_option_2
        graph_option_2='Delay by indice'
        self.controler.ploting_text3('Ploting Delay by index')
        
    def CoefA_graph(self):
        global graph_option_2
        graph_option_2='Coef a'
        self.controler.ploting_text3('Ploting Coef a histogram')
        
    def CoefC_graph(self):
        global graph_option_2
        graph_option_2='Coef c'
        self.controler.ploting_text3('Ploting Coef c histogram')
    
    def Phase_graph(self):
        global graph_option_2
        graph_option_2='Phase'
        self.controler.ploting_text3('Ploting Phase')
        
    def Phase_std_graph(self):
        global graph_option_2
        graph_option_2='Std Phase'
        self.controler.ploting_text3('Ploting Std Phase')
        
    def Correction_param_graph(self):
        global graph_option_2
        graph_option_2='Correction parameters'
        self.controler.ploting_text3('Ploting Correction parameters')
        
    def Correction_param_graph2(self):
        global graph_option_2
        graph_option_2='Correction parameters2'
        self.controler.ploting_text3('Ploting Correction parameters')    
        
    def Errors(self):
        global graph_option_2
        graph_option_2='Errors'
        self.controler.ploting_text3('Ploting Errors')
        
    def Covariance_Pulse(self):
        global graph_option_2
        graph_option_2='Covariance Pulse E field'
        self.controler.ploting_text3('Ploting Covariance')  
        
    def Pulse_E_field_std_graph(self):
        global graph_option_2
        graph_option_2='Std Pulse (E_field)'
        self.controler.ploting_text3('Ploting Std pulse E_field')

    def E_field_std_dB_graph(self):
        global graph_option_2
        graph_option_2='Std E_field [dB]'
        self.controler.ploting_text3('Ploting Std E_field [dB]')

    def refresh(self):
        try:
            self.draw_graph_init(self.controler.myinput, self.controler.myreferencedata, self.controler.reference_number, 
                                 self.controler.mydatacorrection, self.controler.delay_correction, self.controler.dilatation_correction,
                                 self.controler.leftover_correction,
                                 self.controler.myinput_cov, self.controler.mydatacorrection_cov,
                                     self.controler.myglobalparameters,self.controler.fopt, self.controler.fopt_init,
                                      self.controler.mode, preview)
                
        except Exception as e:
            print(e)
        """except:
            print("There is a refresh problem")
            pass"""





###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class MainWindow(QMainWindow):
    def __init__(self, controler):
        super().__init__()
        self.setWindowTitle("Correct@TDS")
        self.mainwidget = MyTableWidget(self,controler)
        self.setCentralWidget(self.mainwidget)
        self.setWindowIcon(QtGui.QIcon('icon.png'))

    def closeEvent(self,event):
        try:
            shutil.rmtree("temp")
        except:
            pass

def main():

    sys._excepthook = sys.excepthook 
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback) 
        sys.exit(1) 
    sys.excepthook = exception_hook 
    
    app = QApplication([])
    controler = Controler()
    win = MainWindow(controler)
    #controler.init()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

