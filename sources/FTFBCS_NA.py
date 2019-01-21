# -*- coding: utf-8 -*-
"""
Created on December 3, 2011

This program solves the diffusion and reaction part of the ADRE 
using finite difference method.

@author: Quanghee Yi
"""
# calculate each term in mass balance equation with diffusion and reaction
def firstTerm(load, vol, t):
    value = load/vol*t
    return value 
    #print value
def secondTerm(E, c, vol, t):
    value = - t/vol*(- E)*c
    return value
def thirdTerm(E, c, vol, t, k):    
    value_dif = -t/vol*(E + E + k*vol)*c
    value_rct = -t/vol*(k*vol)*c
    value = value_dif + value_rct
    return value
def fourthTerm(E, c, vol, t):     
    value = - t/vol*(-E)*c
    return value
    
def explicitFDM(ti, i, n, loading, vol, disp_prime, k, del_t, ci, c):  
    # i = time index and j = space index    
            
    if ti == 0:
        if i == 0: 
            new_conc = firstTerm(loading[ti][i],vol,del_t) + \
            secondTerm(disp_prime,ci,vol,del_t) + \
            thirdTerm(disp_prime,c[i],vol,del_t, k) + \
            fourthTerm(disp_prime,c[i+1],vol,del_t)   
            return new_conc
        elif i == n-1:
            new_conc = firstTerm(loading[ti][i],vol,del_t) + \
            secondTerm(disp_prime,c[i-1],vol,del_t) + \
            thirdTerm(disp_prime,c[i],vol,del_t, k) + \
            fourthTerm(disp_prime,c[i-1],vol,del_t) 
            return new_conc
        else:
            new_conc = firstTerm(loading[ti][i],vol,del_t) + \
            secondTerm(disp_prime,c[i-1],vol,del_t) + \
            thirdTerm(disp_prime,c[i],vol,del_t, k) + \
            fourthTerm(disp_prime,c[i+1],vol,del_t)
            return new_conc
    else:
        if i == 0:
            new_conc = firstTerm(loading[ti][i],vol,del_t) + \
            secondTerm(disp_prime,ci,vol,del_t) + \
            thirdTerm(disp_prime,c[ti-1][i],vol,del_t, k) + \
            fourthTerm(disp_prime,c[ti-1][i+1],vol,del_t)
            return new_conc
        elif i == n-1:
            new_conc = firstTerm(loading[ti][i],vol,del_t) + \
            secondTerm(disp_prime,c[ti-1][i-1],vol,del_t)+ \
            thirdTerm(disp_prime,c[ti-1][i],vol,del_t, k) + \
            fourthTerm(disp_prime,c[ti-1][i-1],vol,del_t)
            return new_conc
        else:
            new_conc = firstTerm(loading[ti][i],vol,del_t) + \
            secondTerm(disp_prime,c[ti-1][i-1],vol,del_t)+ \
            thirdTerm(disp_prime,c[ti-1][i],vol,del_t, k) + \
            fourthTerm(disp_prime,c[ti-1][i+1],vol,del_t) 
            return new_conc
           
  
        








