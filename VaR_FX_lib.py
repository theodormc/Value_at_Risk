# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 06:04:45 2023

@author: XYZW
"""

import xlwings as xw
import pandas as pd
import numpy as np
import scipy.stats as stats
#import mixture_distributions as MD
import matplotlib.pyplot as plt
import matplotlib
import sys
import VaR_FX
sys.path.append(r'C:\Users\XYZW\Documents\Python Scripts\Market risk\FX_tests')
#%%

#%%
def portfolio_value(init_cap,prc_values,data):
    """
    INIT CAP is on a base currency
    prc_values = % of initial capital
    """
    exposures = init_cap*np.array(prc_values)/data[-1,:]
    n = np.shape(data)[0]
    portf_values = [np.dot(exposures,data[n-i,:]) for i in range(1,n+1)]
    return portf_values

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

def empirical_surv(values,x):
    return sum([values[i]>x for i in range(len(values))])

def VaR_historical(values,alpha,order = 'desc',no_periods = 1,
                   type_ret = 'absolute'):
    """
    values must be a numpy array. If it a list, it must be first converted into a 
    numpy array before used as input in the function
    """
    n = len(values)
    if order == 'desc':
        losses = values[0:n-no_periods]-values[no_periods:n]
    else:
        losses = -(values[0:n-no_periods]-values[no_periods:n])
    m = len(losses)
    stat_ord = order_statistics(losses,int(m*(1-alpha)))
    stat_ord2 = order_statistics(losses,int(m*(1-alpha))+1)
    weight = m*(1-alpha) - int(m*(1-alpha))
    if type_ret == 'absolute':
        return (stat_ord+weight*(stat_ord2-stat_ord))*(-1)
    else:
        if order=='desc':
            return (stat_ord+weight*(stat_ord2-stat_ord))*(-1)/values[-1]
        else:
            return (stat_ord+weight*(stat_ord2-stat_ord))*(-1)/values[0]

def VaR_historical_returns(values,alpha,order = 'desc',no_periods = 1):
    n = len(values)
    if order == 'desc':
        loss_returns = (values[0:n-no_periods]-values[no_periods:n])/values[no_periods:n]
    else:
        loss_returns = -(values[0:n-no_periods]-values[no_periods:n])/values[no_periods:n]
    m = len(loss_returns)
    stat_ord = order_statistics(loss_returns,int(m*(1-alpha)))
    stat_ord2 = order_statistics(loss_returns,int(m*(1-alpha))+1)
    weight = m*(1-alpha) - int(m*(1-alpha))
    return (stat_ord + weight*(stat_ord2-stat_ord))*(-1)
    
def VaR_analytical(values,alpha,order = 'desc',no_periods = 1,type_ret = 'absolute'):
    if order == 'desc':
        losses = np.diff(values)
    else:
        losses = -(np.diff(values))
    sig = np.std(losses)
    if type_ret == 'absolute':
        return stats.norm.isf(1-alpha)*sig*np.sqrt(no_periods)
    else:
        if order == 'desc':
            return stats.norm.isf(1-alpha)*sig*np.sqrt(no_periods)/values[-1]
        else:
            return stats.norm.isf(1-alpha)*sig*np.sqrt(no_periods)/values[0]

def VaR_analytical_returns(values,alpha,order = 'desc',no_periods = 1):
    if order == 'desc':
        loss_returns = np.diff(values)/np.array(values)[1:]
    else:
        loss_returns = -np.diff(values)/np.array(values)[0:-1]
    sig = np.std(loss_returns)
    return stats.norm.isf(1-alpha)*sig*np.sqrt(no_periods)

def resample_VaR_historical(values,alpha,order = 'desc',no_periods = 1,no_times = 100):
    n = len(values)
    if order == 'desc':
        losses = values[0:n-no_periods]-values[no_periods:n]
    else:
        losses = -(values[0:n-no_periods]-values[no_periods:n])
    m = len(losses)
    resampled_losses = np.random.choice(losses,m*no_times,p = [1/m]*m)
    stat_ord = order_statistics(resampled_losses,int(m*(1-alpha)))
    stat_ord2 = order_statistics(resampled_losses,int(m*(1-alpha))+1)
    weight = m*(1-alpha) - int(m*(1-alpha))
    return (stat_ord+weight*(stat_ord2-stat_ord))*(-1)

def confidence_interval_VaR_hist(values,alpha,order = 'desc',no_periods = 1,no_times = 100):
    n = len(values)
    if order == 'desc':
        losses = values[0:n-no_periods]-values[no_periods:n]
    else:
        losses = -(values[0:n-no_periods]-values[no_periods:n])
    m = len(losses)
    resampled_losses = np.random.choice(losses,m*no_times,p = [1/m]*m)
    resampled_losses = np.reshape(resampled_losses,(m,no_times))
    VaRs = [order_statistics(resampled_losses[:,i],int(alpha*m)) for i in range(no_times)]
    return VaRs

def ES_analytical(values,alpha,order = 'desc',no_periods = 1):
    if order == 'desc':
        losses = np.diff(values)
    else:
        losses = -(np.diff(values))
    sig = np.std(losses)
    return stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)*sig*np.sqrt(no_periods)

def ES_historical(values,alpha,order = 'desc',no_periods = 1):
    n = len(values)
    if order == 'desc':
        losses = values[0:n-no_periods]-values[no_periods:n]
    else:
        losses = -(values[0:n-no_periods]-values[no_periods:n])
    
    sorted_losses = sorted(losses,reverse = False)
    m = len(losses)
    k = int(m*alpha)
    return np.mean(np.array(sorted_losses[k:m]))

def risk_return_ratio(values):
    """
    Return / risk ratio/profile
    INPUTS: 
        values = a column (univariate time-series)
    """
    ret = values[-1]/values[0]-1
    risk = np.std(np.diff(values)/values[0:-1])*np.sqrt(252)
    return ret/risk

def risk_return_ratio_alloc(init_cap,prcs,values_data):
    """
    GIVEN a 2D array of values, an initial capital (init_cap) and a given 
    allocation (given by prcs), we find the risk_return_ratio
    """
    portf_values = portfolio_value(init_cap,prcs,values_data)
    return risk_return_ratio(portf_values)

def VaR_analytical2(cov_mat_returns,weights,alpha,no_periods = 1):
    """
    no_periods = Number of days for the analytical VaR
    """
    sig = np.sqrt(np.dot(weights,np.dot(cov_mat_returns,
                                        np.array(weights,ndmin = 2).T))[0])
    
    value_at_risk = stats.norm.isf(1-alpha)*np.sqrt(no_periods)*sig
    return value_at_risk

def risk_contribs_VaR_hist(returns,weights,alpha):
    loss_returns = - np.dot(returns,weights)
    m = len(loss_returns)
    weight = m*(1-alpha) - int(m*(1-alpha))
    k = int(m*(1-alpha))
    sorted_loss_returns = sorted(loss_returns,reverse = False)
    ind = list(loss_returns).index(sorted_loss_returns[k])
    ind2 = list(loss_returns).index(sorted_loss_returns[k+1])
    risk_contribs = -returns[ind,:]*weights-weight*(returns[ind2,:]-\
                                returns[ind,:])*weights
    return risk_contribs

def risk_contribs_VaR_hist2(values,weights,order = 'desc',alpha = 0.95):
    portf_values = np.dot(values,weights)
    m = len(portf_values)
    if order == 'desc':
        losses = np.diff(values,axis = 0)
    else:
        losses = -np.diff(values,axis = 0)
    sorted_losses = sorted(losses,reverse = True)
    k = int(m*(1-alpha))
    ind = list(sorted_losses).index(sorted_losses[k])
    ind2 = list(sorted_losses).index(sorted_losses[k])
    
    
        
    
def VaR_historical2(returns,weights,alpha,order = 'desc'):
    loss_returns = -np.dot(returns,weights)
    m = len(loss_returns)
    return order_statistics(loss_returns,int(alpha*m))

def ES_analytical2(cov_mat_returns,weights,alpha,no_periods = 1):
    sig = np.sqrt(np.dot(weights,np.dot(cov_mat_returns,
                                        np.array(weights,ndmin = 2).T))[0])
    ES = stats.norm.pdf(stats.norm.cdf(1-alpha))/(1-alpha)*sig*np.sqrt(no_periods)
    return ES
    
def ES_historical2(returns,weights,alpha,order = 'desc'):
    loss_returns = -np.dot(returns,weights)
    m = len(loss_returns)
    sorted_losses = sorted(loss_returns,reverse = False)
    m = len(loss_returns)
    k = int(m*alpha)
    return np.mean(np.array(sorted_losses[k:m]))
        
def VaR_FX_capital(init_cap, prcs, values_data, order = 'desc', alpha = 0.95, h = 1,
                   type_VaR = 'analytical', type_ret = 'absolute'):
    portf_values = portfolio_value(init_cap,prcs,values_data)
    if type_VaR == 'analytical':
        return VaR_analytical(np.array(portf_values),alpha,order = order,
                              no_periods = h,type_ret = type_ret)
    else:
        return VaR_historical(np.array(portf_values),alpha,order = order,
                              no_periods = h,type_ret = type_ret)


        
def VaR_FX_capital_returns(init_cap,prcs,values_data,order = 'desc',
                           alpha = 0.95, h = 1, type_VaR = 'analytical'):
    portf_values = portfolio_value(init_cap,prcs,values_data)
    if type_VaR == 'analytical':
        return VaR_analytical_returns(np.array(portf_values),alpha,order = order,
                              no_periods = h)
    else:
        return VaR_historical_returns(np.array(portf_values),alpha,order = order,
                            no_periods = h)
    
def VaR_FX_resampled(init_cap,prcs,values_data,order = 'desc',alpha = 0.95,h = 1,
                     no_times = 100):
    portf_values = portfolio_value(init_cap,prcs,values_data)
    return resample_VaR_historical(np.array(portf_values),alpha,order = 'desc',
                                   no_periods = 1,no_times = no_times)
    
def ES_FX_capital(init_cap,prcs,values_data,order = 'desc',alpha = 0.95,h = 1,
                   type_ES = 'analytical'):
    portf_values = portfolio_value(init_cap,prcs,values_data)
    if type_ES == 'analytical':
        return ES_analytical(np.array(portf_values),alpha,order = order,
                              no_periods = h)
    else:
        return ES_historical(np.array(portf_values),alpha,order = order,
                              no_periods = h)

def risk_portf(cov_mat,W):
    """
    W = vector of wealths on each component
    cov_mat = covariance matrix of the returns
    """
    return np.sqrt(np.dot(W,np.dot(cov_mat,np.array(W,ndmin = 2).T))[0])

def risk_contribs(cov_mat,W):
    return W*np.dot(cov_mat,np.array(W,ndmin = 2).T).T[0]/risk_portf(cov_mat,W)



#%%
def backtest_VaR(values,n,m,alpha = 0.95,no_periods = 1,type_VaR = 'historical'):
    """
    For a 1D time series of values we backtest the VaR based on the first n values 
    by using the registered results on the next m results.
    
    Use VaR_analytical(1D)
    """
    if type_VaR == 'historical':
        VaR = VaR_historical(np.array(values)[0:n],alpha,no_periods = no_periods)
    else:
        VaR = VaR_analytical(np.array(values)[0:n],alpha , no_periods = no_periods)
    losses = -np.diff(np.array(values)[n:n+m])
    return sum(losses>VaR)/m

def backtest_VaR_student(values,n,m,alpha = 0.95,no_periods = 1,
                         type_VaR = 'historical'):
    if type_VaR in ['historical','Historical','Hist','hist']:
        VaR = VaR_historical(np.array(values)[0:n],alpha,no_periods = no_periods)
    else:
        losses1 = -np.diff(np.array(values)[n:n+m])
        df,loc,scale = stats.t.fit(losses1)
        VaR = loc+scale*stats.t.isf(1-alpha,df)*np.sqrt((df-2)/df)
    losses2 = -np.diff(np.array(values)[n:n+m])
    return VaR,sum(losses2>VaR)/m