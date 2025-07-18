# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:50:13 2022

@author: xavie
"""

import numpy as np
import pandas as pd


def edge(open: np.array, high: np.array, low: np.array, close: np.array, use_oposed=False) -> float:
    """
    Efficient Estimation of Bid-Ask Spreads from OHLC Prices
    Implements an efficient estimation procedure of the bid-ask spread from Open, High, Low, and Close
    prices as proposed in Ardia, Guidotti, Kroencke (2021): https://www.ssrn.com/abstract=3892335
    Prices must be sorted in ascending order of the timestamp.
    :param open: array-like vector of Open prices.
    :param high: array-like vector of High prices.
    :param low: array-like vector of Low prices.
    :param close: array-like vector of Close prices.
    :return: The spread estimate.
    """

    n = len(open)
    if len(high) != n or len(low) != n or len(close) != n:
        raise Exception("open, high, low, close must have the same length")

    o = np.log(np.asarray(open))
    h = np.log(np.asarray(high))
    l = np.log(np.asarray(low))
    c = np.log(np.asarray(close))
    m = (h + l) / 2.

    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]

    x1 = (m - o) * (o - m1) + (m - c1) * (c1 - m1)
    x2 = (m - o) * (o - c1) + (o - c1) * (c1 - m1)

    e1 = np.nanmean(x1)
    e2 = np.nanmean(x2)

    v1 = np.nanvar(x1)
    v2 = np.nanvar(x2)

    w1 = v2 / (v1 + v2)
    w2 = v1 / (v1 + v2)
    k = 4 * w1 * w2

    n1 = np.nanmean(o == h)
    n2 = np.nanmean(o == l)
    n3 = np.nanmean(c1 == h1)
    n4 = np.nanmean(c1 == l1)
    n5 = np.nanmean(np.logical_and(h == l, l == c1))

    s2 = -4 * (w1 * e1 + w2 * e2) / ((1 - k * (n1 + n2) / 2) + (1 - n5) * (1 - k * (n3 + n4) / 2))
    rep = 0
    if use_oposed==False:
        rep = float(max(0, s2) ** 0.5)
    if use_oposed==True:
        rep = np.sqrt(np.abs(s2))
    return rep

def edge_close_price(log_close, use_oposed=False):
    log_diff = [log_close[i] - log_close[i-1] for i in range(2, len(log_close))]
    log_diff_2 = [log_close[i] - log_close[i-1] for i in range(1, len(log_close)-1)]
    cov = np.cov(log_diff, log_diff_2)[0][1]
    n1 = np.nanmean(log_close[1:] == log_close[:-1])
    rep = -4*cov / ((1-n1)**2)
    if use_oposed==False:
        if rep>0:
            spread=np.sqrt(rep)
        else:
            spread = 0
    if use_oposed==True:
        spread=np.sqrt(np.abs(rep))
    return spread
