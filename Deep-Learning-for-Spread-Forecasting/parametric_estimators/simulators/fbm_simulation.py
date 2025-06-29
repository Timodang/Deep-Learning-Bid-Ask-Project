# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:41:14 2023

@author: xavie
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def davies_harte(T, N, H):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    g = [gamma(k,H) for k in range(0,N)];    r = g + [0] + g[::-1][0:N-1]

    # Step 1 (eigenvalues)
    j = np.arange(0,2*N);   k = 2*N-1
    lk = np.fft.fft(r*np.exp(2*np.pi*complex(0,1)*k*j*(1/(2*N))))[::-1]

    # Step 2 (get random variables)
    Vj = np.zeros((2*N,2), dtype=np.complex128); 
    Vj[0,0] = np.random.standard_normal();  Vj[N,0] = np.random.standard_normal()
    
    for i in range(1,N):
        Vj1 = np.random.standard_normal();    Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1; Vj[i][1] = Vj2; Vj[2*N-i][0] = Vj1;    Vj[2*N-i][1] = Vj2
    
    # Step 3 (compute Z)
    wk = np.zeros(2*N, dtype=np.complex128)   
    wk[0] = np.sqrt((lk[0]/(2*N)))*Vj[0][0];          
    wk[1:N] = np.sqrt(lk[1:N]/(4*N))*((Vj[1:N].T[0]) + (complex(0,1)*Vj[1:N].T[1]))       
    wk[N] = np.sqrt((lk[0]/(2*N)))*Vj[N][0]       
    wk[N+1:2*N] = np.sqrt(lk[N+1:2*N]/(4*N))*(np.flip(Vj[1:N].T[0]) - (complex(0,1)*np.flip(Vj[1:N].T[1])))
    
    Z = np.fft.fft(wk);     fGn = Z[0:N] 
    fBm = np.cumsum(fGn)*(N**(-H))
    fBm = (T**H)*(fBm).real 
    path = np.array([0] + list(fBm))
    return path

def gfbm_simulation_HF(P0,sigma_day,Spread,nb_days, Hurst):
    dico_prices={}
    #sigma = sigma_day / np.sqrt(28880) 
    for days in tqdm(range(1,nb_days)):  ## (1,253) usually
        fbm_sim = davies_harte(1, 28880, Hurst)
        #fbm_cum = fbm_sim.cumsum()
        compt=0
        df_prices=pd.DataFrame(index=np.arange(1,481),columns=["Open","High","Low","Close"])
        for minutes in range(1,481):
            tab_prices=[]
            for secondes in range(1,61):
                price=P0*np.exp(-0.5*(sigma_day**2)*(compt/28880)**(2*Hurst) + sigma_day*fbm_sim[compt])
                compt+=1
                tab_prices.append(price*(1+(2*np.random.randint(2)-1)*Spread/2))
            df_prices["Open"].at[minutes]=tab_prices[0]
            df_prices["High"].at[minutes]=max(tab_prices)
            df_prices["Low"].at[minutes]=min(tab_prices)
            df_prices["Close"].at[minutes]=tab_prices[-1]
        dico_prices[days]=df_prices
    return dico_prices

def gfbm_simulation_LF(P0,sigma_month,Spread, nb_month, Hurst):  
    dico_prices={}
    #sigma = sigma_day / np.sqrt(390)
    for months in tqdm(range(1,nb_month)):
        df_prices=pd.DataFrame(index=np.arange(1,22),columns=["Open","High","Low","Close"])
        fbm_sim = davies_harte(1,8190, Hurst)
        compt = 0
        for days in range(1,22):
            tab_prices=[]
            for minutes in range(390):
                price=P0*np.exp(-0.5*(sigma_month**2)*(compt/8190)**(2*Hurst) + sigma_month*fbm_sim[compt])
                compt+=1
                tab_prices.append(price*(1+(2*np.random.randint(2)-1)*Spread/2))
            df_prices["Open"].at[days]=tab_prices[0]
            df_prices["High"].at[days]=max(tab_prices)
            df_prices["Low"].at[days]=min(tab_prices)
            df_prices["Close"].at[days]=tab_prices[-1]
        dico_prices[months]=df_prices
    return dico_prices
