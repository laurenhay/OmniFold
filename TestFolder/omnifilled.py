import numpy as np
import uproot
import aiohttp
from matplotlib import pyplot as plt
import awkward as ak
from keras.layers import Dense, Input
from keras.models import Model

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from parse import *

import omnifold as of
import os
import tensorflow as tf
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

from tensorflow.python.client import device_lib

root = tk.Tk()
root.withdraw()

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]    
   

def chooseroot():  #call stacker for this instead
    directory = filedialog.askopenfilename(filetypes=[("Root", "*.root")])
    filename = parse('{}/{}', directory)
    while filename is not None:
        filename = parse('{}/{}', filename[1])
        if filename is not None:
            outputname = filename
    outputfile = "Folded_" + outputname[1]
    return directory, outputfile


def stacker(): #calls chooseroot() internally 

    rfile, ofile = chooseroot()
    events = uproot.open(rfile)
    events = events['Events']
    #events = uproot.open("QCD_pt.root:Events")
    #print(events.keys())

    #qcd_mc = uproot.open('QCD_pt.root')
    #uprootevents = qcd_mc['Events']

    testquants = ['FatJet_pt','FatJet_mass','FatJet_phi', 'FatJet_eta','GenJetAK8_eta','GenJetAK8_phi','GenJetAK8_pt','GenJetAK8_mass']
    quantvals = []

    for quantity in range(len(testquants)):
        #print(testquants[quantity])
        #print(testquants[-(1+quantity)])
        fk8 = events[testquants[-(1+quantity)]].array() # array of all 0s in secondary jet
        tk8 = ak.num(fk8) > 0
        #print('tk8')
        #print(tk8)
        #print(fpt[tpt])
        
        
        fpt = events[testquants[quantity]].array() # output jet with all secondary jet 0s removed 
        tes = fpt[tk8]
        #print('tes')
        #print(tes)
        
        tpt = ak.num(tes) > 0  # primary jet with both its own 0s and secondary jet 0s removed
        #print('tpt')
        #print(tpt)
        #print(tes[tpt][:,:1])
        quantvals.append(tes[tpt][:,:1])
        

    #print("LOOP DONE")
    #print(uprootevents.keys('FatJet*'))
    #### getting leading jet pt and mass as storing as numpy-like array
    jetr = tes[tpt][:,:1]

    jet_recopt = quantvals[0]
    jet_recomass = quantvals[1]
    jet_recophi = quantvals[2]
    jet_recoeta = quantvals[3]
    jet_geneta = quantvals[-4]
    jet_genphi = quantvals[-3]
    jet_genpt = quantvals[-2]
    jet_genmass = quantvals[-1]

    amt = len(jet_genmass)//2

    theta0_G = np.column_stack((
        jet_genmass[:amt],   # 1 dee
        jet_genpt[:amt]    # 2 dee
    ))

    theta0_S = np.column_stack((
        jet_recomass[:amt],   # 1 dee
        jet_recopt[:amt]    # 2 dee
    ))


    # Natural
    theta_unknown_G = np.column_stack([
        jet_genmass[amt:],   # 1 dee
        jet_genpt[amt:]    # 2 dee
    ])

    theta_unknown_S = np.column_stack([
        jet_recomass[amt:],   # 1 dee
        jet_recopt[amt:]    # 2 dee
    ])

    theta0 = np.stack((theta0_G, theta0_S), axis=1)
    return theta0, theta_unknown_S, theta0_G, theta_unknown_G, ofile
    
def plotit():

    _,_,_=plt.hist(theta0_G[:,0],bins=np.linspace(-3,1000,200),color='blue',alpha=0.5,label="MC, gen")
    _,_,_=plt.hist(theta0_S[:,0],bins=np.linspace(-3,1000,200),histtype="step",color='black',ls=':',label="MC, reco")
    _,_,_=plt.hist(theta_unknown_G[:,0],bins=np.linspace(-3,1000,200),color='orange',alpha=0.5,label="MCTest, gen")
    _,_,_=plt.hist(theta_unknown_S[:,0],bins=np.linspace(-3,1000,200),histtype="step",color='black',label="MCTest, reco")
    plt.xlabel("x1")
    plt.ylabel("events")
    plt.legend(frameon=False)
    
    _,_,_=plt.hist(theta0_G[:,1],bins=np.linspace(-3,1000,200),color='blue',alpha=0.5,label="MC, gen")
    _,_,_=plt.hist(theta0_S[:,1],bins=np.linspace(-3,1000,200),histtype="step",color='black',ls=':',label="MC, reco")
    _,_,_=plt.hist(theta_unknown_G[:, 1],bins=np.linspace(-3,1000,200),color='orange',alpha=0.5,label="MCTest, gen")
    _,_,_=plt.hist(theta_unknown_S[:, 1],bins=np.linspace(-3,1000,200),histtype="step",color='black',label="MCTest, reco")
    plt.xlabel("x2")
    plt.ylabel("events")
    plt.legend(frameon=False)
    
    _,_,_=plt.hist(jet_geneta[:],bins=np.linspace(-3,3,200),color='blue',alpha=0.5,label="gen, eta")
    _,_,_=plt.hist(jet_genphi[:],bins=np.linspace(-3,3,200),histtype="step",color='black',ls=':',label="gen, phi")
    _,_,_=plt.hist(jet_recoeta[:],bins=np.linspace(-3,3,200),color='orange',alpha=0.5,label="reco, eta")
    _,_,_=plt.hist(jet_recophi[:],bins=np.linspace(-3,3,200),histtype="step",color='green',ls=':',label="reco, phi")
    #_,_,_=plt.hist(jet_genmass[:],bins=np.linspace(-3,1000,200),color='orange',alpha=0.5,label="MCTest, gen")
    #_,_,_=plt.hist(jet_genpt[:],bins=np.linspace(-3,1000,200),histtype="step",color='black',label="MCTest, reco")
    plt.xlabel("parts")
    plt.ylabel("events")
    plt.legend(frameon=False)
    
def runomni(train, test): #train should be gen and reco, test is just reco

    inputs = Input((2,))  # 2D input
    hidden_layer_1 = Dense(50, activation='relu')(inputs)
    hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
    model = Model(inputs=inputs, outputs=outputs)
    
    myweights = of.omnifold(ak.to_numpy(train),ak.to_numpy(test),20,model)
    return myweights
    
    
def postwplot(theta0_G, theta_unknown_G, myweights):


    _,_,_=plt.hist(theta0_G[:,0],bins=np.linspace(-3,1000,200),color='blue',alpha=0.5,label="MC, gen")
    _,_,_=plt.hist(theta_unknown_G[:,0],bins=np.linspace(-3,1000,200),color=['orange'],alpha=0.5,label="MCTest, gen")
    _,_,_=plt.hist(theta0_G[:,0],bins=np.linspace(-3,1000,200),weights=myweights[-1, 0, :],color='black',histtype="step",label="OmniFolded")
    plt.xlabel("X1")
    plt.ylabel("events")
    plt.legend(frameon=False)
    
    _,_,_=plt.hist(theta0_G[:,1],bins=np.linspace(-3,1000,200),color=['blue'],alpha=0.5,label="MC, gen")
    _,_,_=plt.hist(theta_unknown_G[:,1],bins=np.linspace(-3,1000,200),color=['orange'],alpha=0.5,label="MCTest, gen")
    _,_,_=plt.hist(theta0_G[:,1],bins=np.linspace(-3,1000,200),weights=myweights[-1, 0, :],color='black',histtype="step",label="OmniFolded")
    plt.xlabel("X2")
    plt.ylabel("events")
    plt.legend(frameon=False)    


def unfold(): # optional pass in 
    train, test, traingen, datagen, ofile = stacker()
    ofweights = runomni(train, test)
    postwplot(traingen, datagen, ofweights) 
    weightfile = open(ofile,"w")
    #weightfile.close()
    ofweights[-1,0,:].tofile(weightfile, ",")
    weightfile.close()
    print("Weights Saved to:")
    print(ofile)
    
    return 
    
