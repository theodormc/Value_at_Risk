# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:20:37 2022

@author: XYZW
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

def order_statistics(data,k):
    """
    "Compute the kth smallest element of an array"
    
    INPUT: 
        data is an array
    """
    
    if isinstance(data,list) or isinstance(data,np.ndarray) and len(np.shape(data))==1:
        sorted_data = sorted(data,reverse = False)
    else:
        sorted_data = sorted(data[:,0],reverse = False)
    return sorted_data[k]

def VaR_historical(values,alpha,order = 'desc'):
    if order == 'desc':
        losses = np.diff(values)
    else:
        losses = -np.diff(values)
    n = len(losses)
    stat_ord = order_statistics(losses,int(n*(1-alpha)))
    stat_ord2 = order_statistics(losses,int(n*(1-alpha))+1)
    weight = n*(1-alpha) - int(n*(1-alpha))
    return (stat_ord+weight*(stat_ord2-stat_ord))*(-1)

def historical_VaR(data,alpha):
    """
    INPUT:
        data = 1D array
    
    OUTPUT:
        
    """
    n = len(data)
    stat_ord = order_statistics(data,int(n*(1-alpha)))
    stat_ord2 = order_statistics(data,int(n*(1-alpha))+1)
    weight = n*(1-alpha) - int(n*(1-alpha))
    return (stat_ord+weight*(stat_ord2-stat_ord))*(-1)

def ES_historical(values,alpha,order = 'desc'):
    if order == 'desc':
        losses = np.diff(values)
    else:
        losses = -np.diff(values)
    
    sorted_losses = sorted(losses,reverse = False)
    n = len(losses)
    k = int(n*alpha)
    return np.mean(np.array(sorted_losses[k:n]))*(-1)

def historical_ES(data,alpha):
    """
    INPUT:
        data = 1D array
        
    """
    if isinstance(data,list) or isinstance(data,np.ndarray) and len(np.shape(data))==1:
        sorted_data = sorted(data,reverse = False)
    else:
        sorted_data = sorted(data[:,0],reverse = False)
    n = len(data)
    k = int(n*(1-alpha))
    return np.mean(np.array(sorted_data[0:k]))*(-1)

def historical_VaR_chgs(data,alpha = 0.95,no_periods = 1):
    m = len(data)
    data_diff = -(np.array(data)[0:m-no_periods]-np.array(data)[no_periods:m])
    return historical_VaR(data_diff,alpha)

def historical_ES_chgs(data,alpha = 0.95, no_periods = 1):
    m = len(data)
    data_diff = -(np.array(data)[0:m-no_periods]-np.array(data)[no_periods:m])
    return historical_ES(data_diff,alpha)

def historical_VaR_FX(data,investments,alpha=0.95,no_periods = 1):
    r"""
    INPUTS:
        data: 2D array of rates
    """
    portf_values = np.dot(data,investments)
    return historical_VaR_chgs(portf_values,alpha,no_periods = no_periods)

def historical_ES_FX(data,investments,alpha=0.95,no_periods = 1):
    portf_values = np.dot(data,investments)
    return historical_ES_chgs(portf_values,alpha,no_periods = no_periods)

def analytical_ES_FX(data,investments,alpha = 0.95,horizon = 1):
    chgs = np.diff(data,axis = 0)
    cov_mat = np.cov(chgs.T)
    def std_dev_port(exposures,cov_mat):
        return np.sqrt(np.dot(np.dot(exposures,cov_mat),exposures.T))
    sig = std_dev_port(investments,cov_mat)
    return stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)*sig*np.sqrt(horizon)

def analytical_VaR_FX(data,investments,alpha=0.95,horizon = 1):
    #weights = np.array(investments)/sum(investments)
    chgs = np.diff(data,axis = 0)
    cov_mat = np.cov(chgs.T)
    def std_dev_port(exposures,cov_mat):
        return np.sqrt(np.dot(np.dot(exposures,cov_mat),exposures.T))
    sig = std_dev_port(investments,cov_mat)
    return stats.norm.isf(1-alpha)*np.sqrt(horizon)*sig

def analytical_VaR_FX2(data,investments,alpha=0.95,horizon = 1):
    portf_values = np.dot(data,investments)
    chgs = np.diff(portf_values)
    sig = np.std(chgs)
    return stats.norm.isf(1-alpha)*np.sqrt(horizon)*sig

def risk_contribution_VaR(data,investments,alpha = 0.95,horizon = 1,include_VaR = 'no'):
    """
    
    """
    chgs = np.diff(data,axis = 0)
    cov_mat = np.cov(chgs.T)
    def std_dev_port(exposures,cov_mat):
        return np.sqrt(np.dot(np.dot(exposures,cov_mat),exposures.T))
    sig = std_dev_port(investments,cov_mat)
    risk_exposures = np.dot(investments,cov_mat)*investments
    risk_contribs = risk_exposures * stats.norm.isf(1-alpha)*np.sqrt(horizon)/sig
    
    if include_VaR == 'no':
        return risk_contribs
    else:
        VaR = stats.norm.isf(1-alpha)*np.sqrt(horizon)*sig
        return risk_contribs,VaR

def risk_contribution_VaR2(data,init_cap,prcs,alpha = 0.95,horizon = 1,
                                   include_VaR='no'):
    investments = init_cap*prcs
    return risk_contribution_VaR(data,investments,alpha = 0.95,horizon = 1,
                                 include_VaR = include_VaR)

#%%
def VaR_FX_capital(init_capital,prcs,data,alpha = 0.95,h = 1):
    """
    Compute analytical VaR starting from an initial capital.
    
    init_capital = initial capital
    
    data = 2D array numpy of data. Each column represents a time series of data
    
    
    """
    n = np.shape(data)[0]
    init_portf = np.array(prcs)*init_capital/data[n-1,:]
    portf_values = np.dot(data,init_portf)
    def VaR_analytic(mu,sig,alpha = 0.95,h = 1):
        return -mu+sig*np.sqrt(h)*stats.norm.isf(1-alpha)
    sig = np.std(np.diff(portf_values)/portf_values[1:len(portf_values)])
    return VaR_analytic(0,sig,alpha = 0.95,h = h)*init_capital

def ES_FX_capital(init_capital,prcs,data,alpha = 0.95,h = 1):
    n = np.shape(data)[0]
    init_portf = np.array(prcs)*init_capital/data[n-1,:]
    portf_values = np.dot(data,init_portf)
    def ES_analytic(mu,sig,alpha = 0.95,h = 1):
        return -mu+sig*np.sqrt(h)*stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)
    sig = np.std(np.diff(portf_values)/portf_values[1:len(portf_values)])
    return ES_analytic(0,sig,alpha = 0.95,h = h)*init_capital

def VaR_FX_capital_hist(init_capital,prcs,data,alpha = 0.95,h=1):
    n = np.shape(data)[0]
    init_portf = np.array(prcs)*init_capital/data[n-1,:]
    portf_values = np.dot(data,init_portf)
    return historical_VaR_FX(portf_values,alpha,no_periods = h)

def ES_FX_capital_hist(init_capital,prcs,data,alpha = 0.95,h = 1):
    n = np.shape(data)[0]
    init_portf = np.array(prcs)*init_capital/data[n-1,:]
    portf_values = np.dot(data,init_portf)
    return historical_ES_FX(portf_values,alpha,no_periods = h)

    