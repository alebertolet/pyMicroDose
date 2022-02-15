#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:08:36 2021

@author: alejandrobertolet
"""

import numpy as np
from scipy import special as spf
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

def linearFunction(x, m, b):
    return b + m * x

class ProtonCoefficients:
    def __init__(self):
        self.Dimensions = np.array([1.0, 5.0, 10.0])
        self.C_e = np.array([4.957, 26.61, 48.95])
        self.e_e = np.array([0, 1.383, 5.121])
        self.k_e = np.array([16.72, 25.14, 8.762])
        self.b_e = np.array([0.9837, 1.897, 4])
        self.q_e = np.array([1.303, 2.606, 2.932])
        self.C_se = np.array([2.504, 10.01, 22.45])
        self.b_se = np.array([0.8018, 0.7301, 2.0])
        self.f_se = np.array([0.9092, 0.8927, 0.952])
        self.k_se = np.array([10.36, 15.81, 22.5])
        self.e_se = np.array([-0.179, 2.695, 2.0])
        self.q_se = np.array([1, 2.033, 2.961])
        self.C2_se = np.array([1, 35.49, 93.48])
        self.kt_se = np.array([5.059, 177.6, 97.74])
        self.c_se = np.array([0.05, 0.23, 0.38])
        self.Cgt20_se = np.array([1.156, 10.05, 17.6])
        self.fgt20_se = np.array([0.6396, 0.8877, 0.8687])
        self.C_d1 = np.array([0.0729, 0.0646, 0.0564])
        self.k1_d1 = np.array([0.3516, 0.1318, 0.0072])
        self.k2_d1 = np.array([2.996, 1.664, 1.398])
        self.p_d1 = np.array([0.1092, 0.1992, 0.6781])
        self.C_sc = np.array([0.1808, 0.2771, 0.3198])
        self.k1_sc = np.array([0.0238, 0.015, 0.0075])
        self.k2_sc = np.array([1.187, 0.515, 0.3834])
        self.p_sc = np.array([0.4779, 0.4797, 0.5449])
        self.k1_sf = np.array([0.1937, 0.1094, 0.0758])
        self.k2_sf = np.array([140.5, 11.93, 7.162])
        self.k1_sd = np.array([0.3948, 0.1343, 0.0907])
        self.k2_sd = np.array([69.41, 10.65, 6.812])

class Proton:
    def __init__(self, siteDim = 1):
        #Initialize parameters
        self.Coeffs = ProtonCoefficients()
        self.dim = siteDim
        
        #Parameters fixed for protons
        self.h = 27.9
        self.a = 19.53
        self.p = 1.799
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
            if E <= 20:
                C = self.Coeffs.C_se
                f = self.Coeffs.f_se
            else:
                C = self.Coeffs.Cgt20_se
                f = self.Coeffs.fgt20_se
            stdEForAllSites = C * spf.erf(self.Coeffs.k_se * pow(E, self.Coeffs.q_se)) * \
                            np.log(self.h*E +self.Coeffs.b_se) * (self.a*pow(E, self.p-f)+self.Coeffs.e_se)/ \
                                (self.a*pow(E,self.p)+self.D)-self.Coeffs.C2_se*np.exp(-self.Coeffs.kt_se*(E-self.Coeffs.c_se)**2)
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
                if e <= 20:
                    C = self.Coeffs.C_se
                    f = self.Coeffs.f_se
                else:
                    C = self.Coeffs.Cgt20_se
                    f = self.Coeffs.fgt20_se
                s = C * spf.erf(self.Coeffs.k_se * pow(e, self.Coeffs.q_se)) * \
                            np.log(self.h*e +self.Coeffs.b_se) * (self.a*pow(e, self.p-f)+self.Coeffs.e_se)/ \
                                (self.a*pow(e,self.p)+self.D)-self.Coeffs.C2_se*np.exp(-self.Coeffs.kt_se*(e-self.Coeffs.c_se)**2)
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
        