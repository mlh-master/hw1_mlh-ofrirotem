# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = {f:pd.to_numeric(CTG_features[f], errors='coerce').fillna(1000) for f in CTG_features if f != extra_feature}
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for f in CTG_features:
        if f != extra_feature:
            c_cdf[f] = pd.to_numeric(CTG_features[f], errors='coerce').fillna(np.random.choice(CTG_features[f].convert_dtypes(convert_integer=True)))
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {f: {'min': c_feat[f].min(), 'Q1': c_feat[f].quantile(0.25), 'median': c_feat[f].median(), 'Q3': c_feat[f].quantile(0.75), 'max': c_feat[f].max()} for f in c_feat}
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for feat in c_feat:
        q1 = d_summary[feat]['Q1']
        q3 = d_summary[feat]['Q3']
        iqr = q3 - q1
        min_value = q1 - 1.5 * iqr
        max_value = q3 + 1.5 * iqr
        c_no_outlier[feat] = [i if i >= min_value and i <= max_value else np.nan for i in c_feat[feat]]
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = [i if i <= thresh and i >= 0 else np.nan for i in c_cdf[feature]]
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    from sklearnearn.preprocessing import StandardScaler, MinMaxScaler
    modes = {'Standard': StandardScaler(),
             'MinMax': MinMaxScaler(),
             'mean': None,
             'none': None
    }
    
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
