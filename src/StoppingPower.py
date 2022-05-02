#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:17:12 2022

@author: alejandrobertolet
"""

from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('pyMicroDose', 'nist/')

class StoppingPower:
    def __init__(self):
        self.logEE = np.log10(self.starData.Energy)
        self.logEER = np.log10(self.rangeData.Energy)
        self.optionsElectronic = ['electronic', 'Electronic', 'e', 'E']
        self.optionsNuclear = ['nuclear', 'Nuclear', 'n', 'N']
        self.optionsTotal = ['total', 'Total', 't', 'T']
        
    def getStoppingPower(self, E, mode = 'electronic'):
        if type(E) == int or type(E) == float:
            return self.__getStoppingPowerValue(E, mode)
        if type(E) == list or type(E) == np.ndarray:
            sps = []
            for e in E:
                sps.append(self.__getStoppingPowerValue(e, mode))
            return np.array(sps)
        
    def __getStoppingPowerValue(self, E, mode = 'electronic'):
        logE = np.log10(E)
        if mode in self.optionsElectronic:
            csInterp = CubicSpline(self.logEE, self.starData.ElecStopPower/10) #To kev/um
            return csInterp(logE)
        if mode in self.optionsNuclear:
            csInterp = CubicSpline(self.logEE, self.starData.NucStopPower/10) #To kev/um
            return csInterp(logE)
        if mode in self.optionsTotal:
            csInterp = CubicSpline(self.logEE, self.starData.TotalStopPower/10) #To kev/um
            return csInterp(logE)
    
    def getEnergyForStoppingPower(self, sp, mode = 'electronic', side = 'high'):
        if type(sp) == int or type(sp) == float:
            return self.__getEnergyForStoppingPowerValue(sp, mode, side)
        if type(sp) == list or type(sp) == np.ndarray:
            es = []
            for s in sp:
                es.append(self.__getEnergyForStoppingPowerValue(s, mode, side))
            return np.array(es)
        
    def getCSDARange(self, E):
        if type(E) == int or type(E) == float:
            return self.__getRangeValue(E)
        if type(E) == list or type(E) == np.ndarray:
            r = []
            for e in E:
                r.append(self.__getRangeValue(e))
            return np.array(r)
    
    def __getRangeValue(self, E):
        logE = np.log10(E)
        csInterp = CubicSpline(self.logEER, self.rangeData.CSDARange*1e4) # to um
        return csInterp(logE)
    
    def __getEnergyForStoppingPowerValue(self, sp, mode = 'electronic', side = 'high'):
        if sp > self.maxStopPowerVal:
            print("There is no energy for such a high stopping power value.")
            return
        if sp < self.minStopPowerVal:
            print("There is no energy for such a low stopping power value.")
            return
        if mode in self.optionsElectronic:
            imax = np.argmax(self.starData.ElecStopPower)
            if side == 'high':
                psp = np.flip(self.starData.ElecStopPower[imax:]/10)
                e = np.flip(self.starData.Energy[imax:])
            else:
                psp = self.starData.ElecStopPower[:imax]/10
                e = self.starData.Energy[:imax]
        if mode in self.optionsNuclear:
            imax = np.argmax(self.starData.NucStopPower)
            if side == 'high':
                psp = np.flip(self.starData.NucStopPower[imax:]/10)
                e = np.flip(self.starData.Energy[imax:])
            else:
                psp = self.starData.NucStopPower[:imax]/10
                e = self.starData.Energy[:imax]
        if mode in self.optionsTotal:
            imax = np.argmax(self.starData.TotalStopPower)
            if side == 'high':
                psp = np.flip(self.starData.TotalStopPower[imax:]/10)
                e = np.flip(self.starData.Energy[imax:])
            else:
                psp = self.starData.TotalStopPower[:imax]/10
                e = self.starData.Energy[:imax]
        csInterp = CubicSpline(psp, e)
        return csInterp(sp)
        
class AlphaStoppingPower(StoppingPower):
    def __init__(self):
        self.starData = pd.read_csv(DATA_PATH+'astardata.csv')
        self.rangeData = pd.read_csv(DATA_PATH+'alpharange.csv')
        self.maxStopPowerVal = 226.21
        self.minStopPowerVal = 1.56
        StoppingPower.__init__(self)
    
class ProtonStoppingPower(StoppingPower):
    def __init__(self):
        self.starData = pd.read_csv(DATA_PATH+'pstardata.csv')
        self.rangeData = pd.read_csv(DATA_PATH+'protonrange.csv')
        self.maxStopPowerVal = 82.41
        self.minStopPowerVal = 0.2001
        StoppingPower.__init__(self)
        