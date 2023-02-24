# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 08:36:44 2023

@author: XYZW
"""

import numpy as np
import pandas as pd
import sys
import scipy.stats as stats
sys.path.append(r'C:\Users\XYZW\Documents\Python Scripts\equity exotics')
import option_price_BS as opt_BS
#%%
def loss_rank(data,k,order = 'desc'):
    """
    Return the rank of the biggest kth value of the losses represented by 
    variable data
    INPUTS:
        data = list (univariate )
    """
    if order=='desc':
        sorted_data  = sorted(data,reverse = True)
    else:
        sorted_data = sorted(data,reverse = False)
    pos = data.index(sorted_data[k-1])
    return pos+1

def VaR_historical(data,alpha):
    sorted_losses = sorted(data)
    q = alpha*len(sorted_losses)
    weight = q - int(q)
    VaR_hist = sorted_losses[int(q)]+ \
                    weight*(sorted_losses[int(q+1)]-sorted_losses[int(q)])
    return VaR_hist
       

#%%
def VaR_FX_capital2(init_cap,prcs,values_data,order = 'desc',alpha = 0.95,h = 1,
                    type_VaR = 'analytical',type_ret = 'absolute'):
    n = np.shape(values_data)
    if order == 'desc':
        exposures = init_cap*np.array(prcs)/values_data[0,:]
        returns_FX = values_data[0:n-1,:]/values_data[1:n,:]-1
    else:
        exposures = init_cap*np.array(prcs)/values_data[-1,:]
        returns_FX = values_data[1:n,:]/values_data[0:n-1,:]-1
    cov_mat = np.cov(returns_FX)
    return stats.norm.isf(1-alpha)*np.sqrt(h)*VaR_FX_lib.risk_portf(cov_mat,exposures)

#%%

def VaR_historical_FX_weighted(FX_data,alpha,init_cap,prcs,lbd,no_periods = 1,
                               dataframe = 'No'):
    """
    I assume the data is in ascending order.
    
    The simulated portfolio values are taken from the oldest recorded returns 
    to the newest recorded returns. So are the losses. 
    
    """
    n = np.shape(FX_data)[0]
    returns_FX = FX_data[no_periods:n,:]/FX_data[0:n-no_periods,:]-1
    m = np.shape(returns_FX)[0]
    simulated_FX_vals = np.array([FX_data[-1,:]*(1+returns_FX[i,:])
                            for i in range(0,m)],ndmin = 2)
    exposures = np.array(prcs)*init_cap/FX_data[-1,:]
    simulated_portf_values = [np.dot(simulated_FX_vals[i,:],exposures)
                                for i in range(m)]
    

    losses = init_cap - np.array(simulated_portf_values)
    weights = [lbd**(m-i)*(1-lbd)/(1-lbd**m) for i in range(1,m+1)][::-1]
    
    ranks = [loss_rank(list(losses),i)-1 for i in range(1,m+1)]
    
    losses_weights = np.array([sorted(list(losses)),
                               list(np.array(weights)[ranks])],ndmin = 2).T
    global df
    df = pd.DataFrame(np.array([losses,weights,ranks],ndmin = 2).T,
                      columns = ['Losses','Weight','Position'])
    print(df)
    losses_weights= np.insert(losses_weights,2,np.cumsum(losses_weights[:,1]),
                              axis = 1)
    logicals = [(losses_weights[i,2]>alpha) for i in range(m)]
    pos = list(logicals).index(True)
    VaR_weighted = losses_weights[pos,0]
    if dataframe == 'No':
        return VaR_weighted
    else:
        return VaR_weighted,losses,ranks

#%%
def VaR_historical_FX(FX_data,alpha,init_cap,prcs,no_periods = 1):
    n = np.shape(FX_data)[0]
    returns_FX = FX_data[no_periods:n,:]/FX_data[0:n-no_periods,:]-1
    simulated_FX_vals = np.array([FX_data[-1,:]*(1+returns_FX[i,:])
                            for i in range(0,n-no_periods)],ndmin = 2)
    exposures = np.array(prcs)*init_cap/FX_data[-1,:]
    simulated_portf_values = [np.dot(simulated_FX_vals[i,:],exposures)
                                for i in range(n-no_periods)]
    losses = init_cap - np.array(simulated_portf_values)
    VaR_hist = VaR_historical(losses,alpha)
    return VaR_hist

def VaR_analytical(capitals,cov_mat,alpha,h = 1/252,type_VaR = 'percentage'):
    if type_VaR == 'percentage':
        if sum(capitals)==1:
            weights = capitals
        else:
            weights = np.array(capitals)/sum(capitals)
        sig = np.sqrt(np.dot(weights,np.dot(cov_mat,np.array(weights,ndmin = 2).T)))[0]
    else:
        sig = np.sqrt(np.dot(capitals,np.dot(cov_mat,np.array(capitals,ndmin = 2).T)))[0]
    return sig*stats.norm.isf(1-alpha)*np.sqrt(h)

def ES_analytical(capitals,cov_mat,alpha,h = 1/252,type_ES = 'percentage'):
    if type_ES == 'percentage':
        if sum(capitals)==1:
            weights = capitals
        else:
            weights = np.array(capitals)/sum(capitals)
        sig = np.sqrt(np.dot(weights,np.dot(cov_mat,np.array(weights,ndmin = 2).T)))[0]
    else:
        sig = np.sqrt(np.dot(capitals,np.dot(cov_mat,np.array(capitals,ndmin = 2).T)))[0]
    return sig*stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)*np.sqrt(h)

#%%
def VaR_FX_options(capitals,exch_rates,vols,corr,r0,rf_rates,strikes,expiries,option_types,
                   underlyings,positions,alpha,h = 1/252,type_VaR = 'percentage'):
    """
    Parameters:
        ------------
        
    positions: array of number of options
    
    underlyings:.
    
    """
    cov_mat = np.array(vols,ndmin = 2).T*np.array(vols,ndmin = 2)*corr
    prices = [0]*len(vols)
    deltas = [0]*len(vols)
    exposures_opt = [0]*len(vols)
    for i in range(len(option_types)):
        prices[underlyings[i]],deltas[underlyings[i]] = opt_BS.option_price_BS(
              exch_rates[underlyings[i]],
              strikes[i],r0,vols[underlyings[i]],expiries[i],rf_rates[underlyings[i]],
              option =  option_types[i],greeks = 'yes')[0:2]
        exposures_opt[underlyings[i]] = deltas[underlyings[i]]*positions[i]
    #print("EXPOsures options",exposures_opt)
    exposures_FX = np.array(capitals)/np.array(exch_rates)
    exposures_total = np.array(exposures_opt+exposures_FX)*np.array(exch_rates)
    VaR_portf = VaR_analytical(exposures_total,cov_mat,alpha,h = h,type_VaR = type_VaR)
    return VaR_portf
    
def ES_FX_options(capitals,exch_rates,vols,corr,r0,rf_rates,strikes,expiries,option_types,
                   underlyings,positions,alpha,h = 1/252,type_ES = 'percentage'):
    cov_mat = np.array(vols,ndmin = 2).T*np.array(vols,ndmin = 2)*corr
    prices = [0]*len(vols)
    deltas = [0]*len(vols)
    exposures_opt = [0]*len(vols)
    for i in range(len(option_types)):
        prices[underlyings[i]],deltas[underlyings[i]] = opt_BS.option_price_BS(
              exch_rates[underlyings[i]],
              strikes[i],r0,vols[underlyings[i]],expiries[i],rf_rates[underlyings[i]],
              option =  option_types[i],greeks = 'yes')[0:2]
        exposures_opt[underlyings[i]] = deltas[underlyings[i]]*positions[i]
    #print("EXPOsures options",exposures_opt)
    exposures_FX = np.array(capitals)/np.array(exch_rates)
    exposures_total = np.array(exposures_opt+exposures_FX)*np.array(exch_rates)
    ES_portf = ES_analytical(exposures_total,cov_mat,alpha,h = h,type_ES = type_ES)
    return ES_portf

    