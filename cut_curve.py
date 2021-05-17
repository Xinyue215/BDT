#!/usr/bin/env python


import numpy as np
import tables, pylab, sys, operator,dashi, os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from icecube import NewNuFlux, dataclasses,astro
from xgboost import XGBRFClassifier
from xgboost import XGBClassifier
import random, time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

start = time.time()


gamma = 2.7



Data_array_train = '/data/user/xk35/BDT/2016/Precut_files/Data_post_cut_gamma_2.7_24days_v2.npy'
Data_array_test = '/data/user/xk35/BDT/2016/Precut_files/Data_post_cut_gamma_2.7_24days_test_v2.npy'

MC_array = '/data/user/xk35/BDT/2016/Precut_files/MC_post_cut_gamma_2.7_3000_v2.npy'


Data_train = np.load(Data_array_train)
MC = np.load(MC_array)
MC['ow'] = MC['ow']*2
MC_train = MC[:int(len(MC)/2)]

Data_test = np.load(Data_array_test)
MC_test = MC[int(len(MC)/2):]



def calculate_LT(Data):

	LT_day = 0
	for run in np.unique(Data['run']):
		mask = (Data['run'] == run)
		min_time = np.min(Data[mask]['time'])
		max_time = np.max(Data[mask]['time'])
		run_time = max_time - min_time
		#print(run_time)
		LT_day += run_time
    
	LT = LT_day*86400

	#print(LT_day)
	return LT

LT_train = calculate_LT(Data_train)
print('train day' , LT_train/86400.)
LT_test = calculate_LT(Data_test)
print('test day' , LT_test/86400.)


def calculate_weight(MC, LT):

	fluxNorm = 1.0e-18
	mcWeights_27 = fluxNorm * MC['ow'] * (MC['trueE']/1.0e5)**(-2.7) * LT
	mcWeights_20 = fluxNorm * MC['ow'] * (MC['trueE']/1.0e5)**(-2.0) * LT

	if gamma == 2.7:
		MC['weights'] = mcWeights_27
	elif gamma == 2.0:
		MC['weights'] = mcWeights_20

	
calculate_weight(MC_train, LT_train)
calculate_weight(MC_test, LT_test)


def makeFeatures (feature_name, MC, Data):
	Features = np.zeros((len(MC) + len(Data), len(feature_name)))
	Data['weights'] = 10*np.ones(len(Data))
	for count, value in enumerate(feature_name):
		Features.T[count][:] = np.concatenate((MC[value],Data[value]))
	
	Event_Type = np.concatenate((np.ones(len(MC)), np.zeros(len(Data))))
	return Features, Event_Type

feature_name = np.array(['spMPE_ndirE', 'spMPE_rlogl','spMPE_parabo','MPEHighNoise_delta_angle',
			'logE','BayesLLHRatio','cog_r2','cog_z','spMPE_ldirE','LineFit_delta_angle',
			'NEarlyNCh','logMuEx','MuExrllt','weights','run','event','subevent','trueRa',
			'trueDec','spMPE_azimuth','spMPE_zenith','time','logE','spMPE_parabo','ow',
			'trueE'])

Features_train, Event_Type_train = makeFeatures(feature_name, MC_train, Data_train)
Features_test, Event_Type_test = makeFeatures(feature_name, MC_test, Data_test)

	

X_train = Features_train.copy()
np.random.shuffle(X_train)     
X_test = Features_test.copy()  
np.random.shuffle(X_test)      

#train testsplit                
#X_train, X_test, y_train, y_test = train_test_split(Features_train, Event_Type_train, test_size = 0.5, random_state = 1)

#pull out weight
X_train_weights = X_train.T[13]
mask_mc = (X_train_weights != 10)
X_train_weights = X_train_weights[mask_mc]
y_train = Event_Type_train.copy()
y_train[mask_mc] = 1
y_train[~mask_mc] = 0
print(y_train)

X_test_weights = X_test.T[13]
mask_mc = (X_test_weights != 10)
X_test_weights = X_test_weights[mask_mc]
y_test = Event_Type_test.copy()
y_test[mask_mc] = 1             
y_test[~mask_mc] = 0            
print(y_test)


MC_test = X_test[mask_mc]
Data_test = X_test[~mask_mc]
np.save('/data/user/xk35/BDT/2016/Trained/MC_24_3000_v1.npy',MC_test)
np.save('/data/user/xk35/BDT/2016/Trained/Data_24_3000_v1.npy', Data_test)
	
#get rid of weights before training, 13 trainingf features
tot_fet = len(feature_name)
X_train = np.delete(X_train, np.arange(13,tot_fet), axis = 1)
X_test = np.delete(X_test, np.arange(13,tot_fet), axis = 1)


	
def compute_models(*args):
	names = []
	probs = []
	for classifier, kwargs in args:
		print(classifier.__name__)
		clf = classifier(**kwargs)
		clf.fit(X_train, y_train)
        
        	#Note that we are outputing the probabilities [of class 1], not the classes
		y_probs = clf.predict_proba(X_test)[:, 1]

		names.append(classifier.__name__)
		probs.append(y_probs)
	return names, probs


#name_cutcurve, prob_cutcurve = compute_models((XGBClassifier, dict(n_estimators = 700,eta=0.1, max_depth=12))
name_cutcurve, prob_cutcurve = compute_models((XGBRFClassifier, dict(n_estimators = 400, learning_rate = 0.1, subsample=0.8, colsample_bynode=0.277, max_depth=12))

#DisionTreeClassifier,dict(max_depth=12,criterion='entropy'))
                             )

MC_score = prob_cutcurve[0][y_test == 1]
np.save('/data/user/xk35/BDT/2016/Trained/MC_score_v1.npy', MC_score)
Data_score = prob_cutcurve[0][y_test == 0]
np.save('/data/user/xk35/BDT/2016/Trained/Data_score_v1.npy', Data_score)
	
#calculayte eff	
score_mins = np.linspace(0.4,0.6,300)
#score_mins = np.r_[np.linspace(0, .9, 200), 1 - np.logspace(-5, -1,100)[::-1]] 
sig_eff = [X_test_weights[MC_score > score_min].sum()/X_test_weights.sum() for score_min in score_mins]

data_eff = [(Data_score[Data_score > score_min]).sum()/Data_score.sum() for score_min in score_mins]

np.save('/data/user/xk35/BDT/2016/Precut_files/BDT_cut_vars/2.7/eff/sig_eff_24day3000_xgbrf_v1.npy', sig_eff)
np.save('/data/user/xk35/BDT/2016/Precut_files/BDT_cut_vars/2.7/eff/data_eff_24day3000_xgbrf_v1.npy', data_eff)


print (time.time()-start)
