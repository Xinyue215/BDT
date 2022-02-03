#!/usr/bin/env python 

import sys, getopt
from icecube import dataio
import numpy as np
import glob
from icecube import icetray, dataio, tableio,dataclasses,simclasses,recclasses, gulliver, finiteReco, paraboloid,phys_services, lilliput, gulliver_modules, rootwriter
from I3Tray import *
from icecube.hdfwriter import I3SimHDFWriter
from icecube import millipede, linefit
from icecube.hdfwriter import I3HDFWriter
from icecube import astro,icetray
import time,pickle
import sklearn, xgboost
import subprocess
from csky.utils import ensure_dir
import gc
import joblib 
from icecube import hdfwriter,bayesian_priors



#clf = pickle.load(open('/home/xk35/BDT_corrected/Train/BDT_I_GC.pkl', "rb"))

#clf = joblib.load('/home/xk35/BDT_corrected/Train/xgb_Train_nobayes.joblib') 
GC_ra , GC_dec = astro.gal_to_equa(0., 0.)
def zenith_cut(frame, minDec=GC_dec-np.radians(10), maxDec = GC_dec+np.radians(10)):
    dec = frame['SplineMPE'].dir.zenith - np.radians(90.)

    return ((dec >= minDec) and (dec <= maxDec))

def splineMPE_cut(frame):
    status = frame['SplineMPE'].fit_status    
    return  (status == 0) 

def rlogl_cut(frame):
    rlogl = frame['SplineMPEFitParams'].rlogl
    return rlogl<9.

def ldirE_cut(frame):
    ldirE = frame['SplineMPEDirectHitsE'].dir_track_length
    return ldirE >250.

def sigma_cut(frame):
    print('doing sigma cut')
    err1 = frame['MPEFitParaboloidFitParams'].pbfErr1
    err2 = frame['MPEFitParaboloidFitParams'].pbfErr2
    sigma = np.sqrt((err1**2+err2**2)/2)
    return sigma <np.radians(4.5)
    sys.stdout.flush()

def delta_angle(pt1_zen, pt1_azi=None, pt2_zen=None, pt2_azi=None):


    z1=np.array(pt1_zen)
    z2=np.array(pt2_zen)
    a1=np.array(pt1_azi)
    a2=np.array(pt2_azi)
    pi = np.pi
    # haversine!
    cos_alpha = np.cos(z1-z2) - np.cos(pi/2.0 - z1)*np.cos(pi/2.0 - z2)*(1-np.cos(a1-a2))
    alpha=180/pi*np.arccos(cos_alpha)
    if np.isnan(alpha):
        alpha = 180
    return alpha

def SplitMinZenithFunc(zen1,zen2,zen3,zen4):
    zen1[(zen1<0)|(zen1!=zen1)]=400
    zen2[(zen2<0)|(zen2!=zen2)]=400
    zen3[(zen3<0)|(zen3!=zen3)]=400
    zen4[(zen4<0)|(zen4!=zen4)]=400
    return np.degrees(np.min(np.transpose((zen1,zen2,zen3,zen4)),axis=1))


#BayesianFunc = lambda blogl,logl: np.nan_to_num(blogl - logl)

def BDT_score(frame,cut_score):
    ndirE = frame['SplineMPEDirectHitsE'].n_dir_pulses
    rlogl = frame['SplineMPEFitParams'].rlogl
    ##
    err1 = frame['MPEFitParaboloidFitParams'].pbfErr1
    err2 = frame['MPEFitParaboloidFitParams'].pbfErr2
    sigma = np.sqrt((err1**2+err2**2)/2)
    ##
    highNoise_zen = frame['MPEFitHighNoise'].dir.zenith
    highNoise_azi = frame['MPEFitHighNoise'].dir.azimuth
    TWHV_zen = frame['MPEFit_TWHV'].dir.zenith
    TWHV_azi = frame['MPEFit_TWHV'].dir.azimuth
    MPEHighNoise_delta_angle = delta_angle(highNoise_zen,highNoise_azi,TWHV_zen,TWHV_azi)
    logE = np.log10(frame['SplineMPEMuEXDifferential'].energy)
    
#    Bayesian_logl = frame['SPEFit2BayesianFitParams'].logl
#    TWHV_logl = frame['SPEFit2_TWHVFitParams'].logl
    #BayesLLHRatio = BayesianFunc(Bayesian_logl,TWHV_logl)
    LLH = frame['SPEFit2_TWHVFitParams'].logl
    
    cog_z = frame['HitStatisticsValues'].cog.z
    cog_r2 = frame['HitStatisticsValues'].cog.x**2 + frame['HitStatisticsValues'].cog.y**2
    
    ldirE = frame['SplineMPEDirectHitsE'].dir_track_length
    spMPE_zen = frame['SplineMPE'].dir.zenith
    spMPE_azi = frame['SplineMPE'].dir.azimuth
    LineFit_TWHV_zen = frame['LineFit_TWHV'].dir.zenith
    LineFit_TWHV_azi = frame['LineFit_TWHV'].dir.azimuth
    LineFit_delta_angle = delta_angle(spMPE_zen,spMPE_azi,LineFit_TWHV_zen,LineFit_TWHV_azi)
    nearlyE = frame['SplineMPEDirectHitsE'].n_early_pulses
    NEarlyNCh = nearlyE/ndirE
    MuExrllt = frame['MuEXAngular4_rllt'].value
    
    feature = np.zeros((1,12))
    feature[0][:] = (np.array([ndirE, rlogl, sigma,MPEHighNoise_delta_angle,logE,LLH,cog_r2,cog_z,ldirE,LineFit_delta_angle,NEarlyNCh,MuExrllt]))
    print(feature)
    #clf = joblib.load('/home/xk35/BDT_corrected/Train/xgb_Train_nobayes.joblib')    
    clf = xgboost.XGBClassifier()                                 
    clf.load_model('/home/xk35/BDT_corrected/Train/BDT_I_GC_xgb1.4.json')

    score = clf.predict_proba(feature)[0, 1] 
    print(float(score))
    sys.stdout.flush()    
    frame['BDT_score'] = dataclasses.I3Double(float(score)) 

    return score >= float(cut_score)
    #return True


def main(argv):
    inputfile = ''
    outputfile = ''
    #move_file = ''
    folder_group = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:s:",["ifile=","ofile=","cut_score="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile> -s <cutscore>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <input file name> -o <output file directory> <True/False> -s <cut score>')
            sys.exit()
        elif opt in  ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--cut_score"):
            cut_score = arg


    print('Input file is "', inputfile)
    print('Output file is "', outputfile)

#    files = [gcdfile,inputfile]
    
    tray = I3Tray()

    output_name = inputfile.split('/')[-1]
    #data
    #output_name = output_name.replace('Level3pass2','postBDTI')
    #mc                                                      
    output_name = output_name.replace('Level3','postBDTI')
    #output_name = output_name.replace('zst','bz2')
    outfile = (outputfile + '/' + output_name)

#    tray.AddModule('I3Reader', 'reader', FilenameList=files)
    tray.AddModule('I3Reader', 'reader', filename=inputfile)
    tray.AddModule(zenith_cut,     # function name
                   "MPEFitZenithFilter",     # module specifier
                   minDec=GC_dec-np.radians(10), maxDec = GC_dec+np.radians(10))
    tray.AddModule(splineMPE_cut,   
                    "MPEFitFilter")
    tray.AddModule(rlogl_cut,   
                "MPEFitrloglFilter")
    tray.AddModule(ldirE_cut,   
                    "MPEFitldirEFilter")
    tray.AddModule(sigma_cut,   
                    "SigmaFilter")
    
#    NIts = 2    
#    try:  
#        tray.AddSegment(DoReconstructions,'level3_recos',Pulses='SRTHVInIcePulses', If = lambda frame: frame.Stop==icetray.I3Frame.Physics)
    
    tray.AddModule(BDT_score,"BDT_cut", cut_score = cut_score)
    #clf.__del__()
    tray.Add('I3Writer','EventWriter',
        Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics],
        DropOrphanStreams=[icetray.I3Frame.DAQ],
        Filename=outfile)
    tray.AddModule('TrashCan','can')
    tray.Execute()
    tray.Finish()

if __name__ == "__main__":
    main(sys.argv[1:])

