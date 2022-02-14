#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:08:36 2021

@author: alejandrobertolet
"""

import numpy as np
import pandas as pd
from scipy import special as spf
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

def linearFunction(x, m, b):
    return b + m * x

class AlphaCoefficients:
    def __init__(self):
        self.Dimensions = np.array([0.1 ,0.5, 1.0, 5.0, 10.0])
        self.C_e = np.array([3.645, 24.43, 51.21, 261.9, 383.0])
        self.e_e = np.array([0.515, 1.425, 2.114, 3.074, 7.961])
        self.k_e = np.array([0.897, 0.7678, 0.7787, 0.817, 0.3453])
        self.b_e = np.array([9.146, 5.726, 4.487, 23.97, 50.5])
        self.q_e = np.array([1.123, 0.9049, 0.7585, 1.353, 2.043])
        self.C_se = np.array([2.44, 21.3, 26.95, 62.42, 88.42])
        self.b_se = np.array([0.8088, 2.59, 4.473, 8.5, 10.0])
        self.f_se = np.array([0.9189, 1.056, 0.8575, 0.6949, 0.6422])
        self.k_se = np.array([0.678, 0.6498, 0.9635, 0.6988, 0.4939])
        self.e_se = np.array([2.95, 0.4742, 1.022, 8.102, 17.47])
        self.q_se = np.array([1.014, 0.5825, 0.676, 1.099, 1.473])
        self.C_d1 = np.array([0,0,0.4169, 0.6564, 0.593])
        self.k1_d1 = np.array([0,0,1.58, 2.858, 0.9468])
        self.k2_d1 = np.array([0,0,2.216, 2.746, 2.619])
        self.p_d1 = np.array([0,0,0.0295, 0.0025, 0.0024])
        self.C_sc = np.array([0,0,10.93, 0.2494, 0.2678])
        self.k1_sc = np.array([0,0,0.3731, 0.1952, 0.1337])
        self.k2_sc = np.array([0,0,4.188, 0.1009, 0.0029])
        self.p_sc = np.array([0,0,0.0066, 0.0025, 0.0113])
        self.k1_sf = np.array([0.1, 0.08928, 0.188, 0.1358, 0.1075])
        self.k2_sf = np.array([360, 171.5, 93.52, 6.537, 2.431])
        self.k1_sd = np.array([0.1, 0.1055,0.2279, 0.161, 0.1255])
        self.k2_sd = np.array([360, 171.3,81.29, 5.557, 2.222])

class AlphaParticle:
    def __init__(self, siteDim = 1):
        #Initialize parameters
        self.Coeffs = AlphaCoefficients()
        
        self.dim = siteDim
        
        #Parameters fixed for alpha particles
        self.h = 7.017
        self.a = 1.946
        self.p = 1.752
        self.D = 2/3 * siteDim
        
    # Interpolation within limits, extrapolation based on linear models    
    def meanE(self, E, siteDim = None, mode = "Interpolation"):
        if siteDim is not None:
            self.dim = siteDim
            self.D = 2/3 * siteDim
        if type(E) == int or type(E) == float:
            meanEForAllSites = self.Coeffs.C_e * spf.erf(self.Coeffs.k_e * pow(E, self.Coeffs.q_e)) * \
                            np.log(self.h*E +self.Coeffs.b_e) * (self.a*pow(E, self.p-1)+self.Coeffs.e_e)/ \
                            (self.a*pow(E,self.p)+self.D)
            if mode == "Fit":
                f = np.polyfit(self.Coeffs.Dimensions, meanEForAllSites, 2)
                p = np.poly1d(f)
                return p(self.dim)
            if mode == "Interpolation":
                if (self.dim >= self.Coeffs.Dimensions[0] and self.dim <= self.Coeffs.Dimensions[len(self.Coeffs.Dimensions)-1]):
                    cs = CubicSpline(self.Coeffs.Dimensions, meanEForAllSites)
                    return cs(self.dim)
                else:
                    fit, _ = curve_fit(linearFunction, self.Coeffs.Dimensions[0:2], meanEForAllSites[0:2])
                    return linearFunction(self.dim, fit[0], fit[1])
            
        if type(E) == list or type(E) == np.ndarray:
            meanEs = []
            for e in E:
                m = self.Coeffs.C_e * spf.erf(self.Coeffs.k_e * pow(e, self.Coeffs.q_e)) * \
                            np.log(self.h*e +self.Coeffs.b_e) * (self.a*pow(e, self.p-1)+self.Coeffs.e_e)/ \
                            (self.a*pow(e,self.p)+self.D)
                if mode == "Fit":
                    f = np.polyfit(self.Coeffs.Dimensions, m, 2)
                    p = np.poly1d(f)
                    meanEs.append(p(self.dim))
                if mode == "Interpolation":
                    if (self.dim >= self.Coeffs.Dimensions[0] and self.dim <= self.Coeffs.Dimensions[len(self.Coeffs.Dimensions)-1]):
                        cs = CubicSpline(self.Coeffs.Dimensions, m)
                        meanEs.append(cs(self.dim))
                    else:
                        fit, _ = curve_fit(linearFunction, self.Coeffs.Dimensions, m)
                        meanEs.append(linearFunction(self.dim, *fit))
            meanEs = np.array(meanEs)
            return meanEs                               
    
    def stdE(self, E, siteDim = None, mode = "Interpolation"):
        if siteDim is not None:
            self.dim = siteDim
            self.D = 2/3 * siteDim
        if type(E) == int or type(E) == float:
            stdEForAllSites = self.Coeffs.C_se * spf.erf(self.Coeffs.k_se * pow(E, self.Coeffs.q_se)) * \
                            np.log(self.h*E +self.Coeffs.b_se) * (self.a*pow(E, self.p-self.Coeffs.f_se)+self.Coeffs.e_se)/ \
                                (self.a*pow(E,self.p)+self.D)
            if mode == "Fit":
                f = np.polyfit(self.Coeffs.Dimensions, stdEForAllSites, 2)
                p = np.poly1d(f)
                return p(self.dim)
            if mode == "Interpolation":
                if (self.dim >= self.Coeffs.Dimensions[0] and self.dim <= self.Coeffs.Dimensions[len(self.Coeffs.Dimensions)-1]):
                    cs = CubicSpline(self.Coeffs.Dimensions, stdEForAllSites)
                    return cs(self.dim)
                else:
                    fit, _ = curve_fit(linearFunction, self.Coeffs.Dimensions, stdEForAllSites)
                    return linearFunction(self.dim, *fit)
        if type(E) == list or type(E) == np.ndarray:
            stdEs = []
            for e in E:
                s = self.Coeffs.C_se * spf.erf(self.Coeffs.k_se * pow(e, self.Coeffs.q_se)) * \
                            np.log(self.h*e +self.Coeffs.b_se) * (self.a*pow(e, self.p-self.Coeffs.f_se)+self.Coeffs.e_se)/ \
                                (self.a*pow(e,self.p)+self.D)
                if mode == "Fit":
                    f = np.polyfit(self.Coeffs.Dimensions, s, 2)
                    p = np.poly1d(f)
                    stdEs.append(p(self.dim))
                if mode == "Interpolation":
                    if (self.dim >= self.Coeffs.Dimensions[0] and self.dim <= self.Coeffs.Dimensions[len(self.Coeffs.Dimensions)-1]):
                        cs = CubicSpline(self.Coeffs.Dimensions, s)
                        stdEs.append(cs(self.dim))
                    else:
                        fit, _ = curve_fit(linearFunction, self.Coeffs.Dimensions, s)
                        stdEs.append(linearFunction(self.dim, *fit))
            stdEs = np.array(stdEs)
            return stdEs
        
    
    def meanS(self, E, siteDim = None, mode = "Interpolation"):
        if siteDim is not None:
            self.dim = siteDim
            self.D = 2/3 * siteDim
        if type(E) == int or type(E) == float:
            meanSForAllSites = self.D * (1 - np.exp(-self.Coeffs.k1_sf * np.exp(self.Coeffs.k2_sf*E)))
            if mode == "Fit":
                f = np.polyfit(self.Coeffs.Dimensions, meanSForAllSites, 1)
                p = np.poly1d(f)
                return p(self.dim)
            if mode == "Interpolation":
                if (self.dim >= self.Coeffs.Dimensions[0] and self.dim <= self.Coeffs.Dimensions[len(self.Coeffs.Dimensions)-1]):
                    cs = CubicSpline(self.Coeffs.Dimensions, meanSForAllSites)
                    return cs(self.dim)
                else:
                    fit, _ = curve_fit(linearFunction, self.Coeffs.Dimensions, meanSForAllSites)
                    return linearFunction(self.dim, *fit)
        if type(E) == list or type(E) == np.ndarray:
            meanSs = []
            for e in E:
                m = self.D * (1 - np.exp(-self.Coeffs.k1_sf * np.exp(self.Coeffs.k2_sf*e)))
                if mode == "Fit":
                    f = np.polyfit(self.Coeffs.Dimensions, m, 1)
                    p = np.poly1d(f)
                    meanSs.append(p(self.dim))
                if mode == "Interpolation":
                    if (self.dim >= self.Coeffs.Dimensions[0] and self.dim <= self.Coeffs.Dimensions[len(self.Coeffs.Dimensions)-1]):
                        cs = CubicSpline(self.Coeffs.Dimensions, m)
                        meanSs.append(cs(self.dim))
                    else:
                        fit, _ = curve_fit(linearFunction, self.Coeffs.Dimensions, m)
                        meanSs.append(linearFunction(self.dim, *fit))
            meanSs = np.array(meanSs)
            return meanSs
    
    def sD(self, E, siteDim = None, mode = "Interpolation"):
        if siteDim is not None:
            self.dim = siteDim
            self.D = 2/3 * siteDim
        if type(E) == int or type(E) == float:
            sDForAllSites = 9/8 * self.D * (1 - np.exp(-self.Coeffs.k1_sd * np.exp(self.Coeffs.k2_sd*E)))
            if mode == "Fit":
                f = np.polyfit(self.Coeffs.Dimensions, sDForAllSites, 1)
                p = np.poly1d(f)
                return p(self.dim)
            if mode == "Interpolation":
                if (self.dim >= self.Coeffs.Dimensions[0] and self.dim <= self.Coeffs.Dimensions[len(self.Coeffs.Dimensions)-1]):
                    cs = CubicSpline(self.Coeffs.Dimensions, sDForAllSites)
                    return cs(self.dim)
                else:
                    fit, _ = curve_fit(linearFunction, self.Coeffs.Dimensions, sDForAllSites)
                    return linearFunction(self.dim, *fit)
        if type(E) == list or type(E) == np.ndarray:
            sDs = []
            for e in E:
                sd = 9/8 * self.D * (1 - np.exp(-self.Coeffs.k1_sd * np.exp(self.Coeffs.k2_sd*e)))
                if mode == "Fit":
                    f = np.polyfit(self.Coeffs.Dimensions, sd, 1)
                    p = np.poly1d(f)
                    sDs.append(p(self.dim))
                if mode == "Interpolation":
                    if (self.dim >= self.Coeffs.Dimensions[0] and self.dim <= self.Coeffs.Dimensions[len(self.Coeffs.Dimensions)-1]):
                        cs = CubicSpline(self.Coeffs.Dimensions, sd)
                        sDs.append(cs(self.dim))
                    else:
                        fit, _ = curve_fit(linearFunction, self.Coeffs.Dimensions, sd)
                        sDs.append(linearFunction(self.dim, *fit))
            sDs = np.array(sDs)
            
            return sDs
    
    def yF(self, E, siteDim = None, mode = "FixedSite"):
        if mode == "FixedSite":
            return self.meanE(E, self.dim, 'Interpolation') / self.meanS(E, self.dim, 'Interpolation')
        else:
            return self.meanE(E, siteDim, mode) / self.meanS(E, siteDim, mode)
    
    def yD(self, E, siteDim = None, mode = "FixedSite"):
        if mode == "FixedSite":
            meanE = self.meanE(E, self.dim, 'Interpolation')
            meanS = self.meanS(E, self.dim, 'Interpolation')
            return meanE/meanS * (1 + pow(self.stdE(E, self.dim, 'Interpolation')/meanE, 2))
        else:
            meanE = self.meanE(E, siteDim, mode)
            meanS = self.meanS(E, siteDim, mode)
            return meanE/meanS * (1 + pow(self.stdE(E, siteDim, mode)/meanE, 2))
    
    def z1F(self, E, siteDim = None, mode = "FixedSite"):
        return self.GetZ(self.yF(E, siteDim, mode), siteDim)
    
    def z1D(self, E, siteDim = None, mode = "FixedSite"):
        return self.GetZ(self.yD(E, siteDim, mode), siteDim)
    
    def stdDevZ1(self, E, siteDim = None, mode = "FixedSite"):
        return self.GetZ(self.stdE(E, siteDim, mode) / self.meanS(E, siteDim, mode), siteDim)
    
    def GetZ(self, y, siteDim = None):
        if siteDim is not None:
            r = siteDim / 2 * 1e-6
        else:
            r = self.dim / 2 * 1e-6 # radius of the site
        e = 1.602176634e-19
        rho = 997 # kg/m3
        # Convert site dimension into m
        # Convert y into J/m
        yJm = y * 1e6 * e * 1e3
        return yJm / (rho * np.pi * r**2)


class AlphaStoppingPower:
    def __init__(self):
        self.astarData = pd.read_csv('nist/astardata.csv')
        
    def getStoppingPower(self, E, mode = 'electronic'):
        logEE = np.log10(self.astarData.Energy)
        logE = np.log10(E)
        optionsElectronic = ['electronic', 'Electronic', 'e', 'E']
        optionsNuclear = ['nuclear', 'Nuclear', 'n', 'N']
        optionsTotal = ['total', 'Total', 't', 'T']
        if mode in optionsElectronic:
            csInterp = CubicSpline(logEE, self.astarData.ElecStopPower/10) #To kev/um
            return csInterp(logE)
        if mode in optionsNuclear:
            csInterp = CubicSpline(logEE, self.astarData.NucStopPower/10) #To kev/um
            return csInterp(logE)
        if mode in optionsTotal:
            csInterp = CubicSpline(logEE, self.astarData.TotalStopPower/10) #To kev/um
            return csInterp(logE)
        