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
    removes an unnecessary feature and drops the NaN values for the rest of the features
    
    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = {f: [v for v in CTG_features[f] if v != 'NaN'] for f in CTG_features if f != extra_feature}
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """
    replaces NaN values with a random values from the feature vector
    
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
    create a dictionary of statistics summary for the features of the data
    
    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {f: {'min': c_feat[f].min(), 'Q1': c_feat[f].quantile(0.25), 'median': c_feat[f].median(), 'Q3': c_feat[f].quantile(0.75), 'max': c_feat[f].max()} for f in c_feat}
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """
    removes the outliers of the data
    
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
    applys a physiological logic rules on a feature

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
    normalize or standartize the data

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = CTG_features.copy()
    
    std = lambda feat: pd.DataFrame([i for i in map(lambda v: (v - feat.mean()) / feat.std(), feat)])
    minmax = lambda feat: pd.DataFrame([i for i in map(lambda v: (v - feat.min()) / (feat.max() - feat.min()), feat)])
    mean = lambda feat: pd.DataFrame([i for i in map(lambda v: (v - feat.mean()) / (feat.max() - feat.min()), feat)])
    
    modes = {'mean': lambda x, y: (mean(CTG_features[x]), mean(CTG_features[y])),
             'MinMax': lambda x, y: (minmax(CTG_features[x]), minmax(CTG_features[y])),
             'standard': lambda x, y: (std(CTG_features[x]), std(CTG_features[y])),
             'none': lambda x, y: (CTG_features[x], CTG_features[y])
    }
    
    nsd_res[x], nsd_res[y] = modes[mode](x, y)
    
    if flag:
        nsd_res[[x, y]].plot(kind='hist', bins=100)
        plt.ylabel('Count')
        plt.xlabel('Value')
        plt.legend()
        plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
