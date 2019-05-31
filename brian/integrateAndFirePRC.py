# coding: utf-8

# #Integrate and fire study for the Phase Response Curve 
# __Goal:__ Assess the effect of flat and phase dependent response curves in the overall synchronization of groups of neurons receiving common inputs.
# 
# Joao Couto - jpcouto@gmail.com
# 

# ### Find the external current for 30Hz firing frequency
# Creates non-leaky and leaky integrate and fire models as example of flat and phase dependent response curves.
# Uses scipy.optimize.fmin to find the baseline current needed to evoke 30Hz firing in each models for use in subsequent steps. 

# In[2]:

import brian as b
import pylab as plt
import numpy as np
b.set_global_preferences(use_weave=True)

tau = 10*b.ms # membrane time constant
Vr = 0*b.mV # Reset voltage
Vt = 15*b.mV # Threshold voltage

eqs_leak = '''
dV/dt = (-V+Iext)/tau : volt
Iext: volt
'''
eqs_noleak = '''
dV/dt = (Iext)/tau : volt
Iext: volt
'''
N = 1
# Neuron groups for leaky and non-leaky integrate and fire
G_leak=b.NeuronGroup(N, model=eqs_leak, reset=Vr, threshold=Vt)
G_noleak=b.NeuronGroup(N, model=eqs_noleak, reset=Vr, threshold=Vt)

targetF = 30.0
spks_leak = b.SpikeMonitor(G_leak,record=True)
spks_noleak = b.SpikeMonitor(G_noleak,record=True)

def search_current_for_freq(x,target,model,spkcounter):
    '''
    Minimization for the firing frequency parameter estimation
    '''
    model.Iext = x*b.mA
    spkcounter.reinit()
    b.run(0.1*b.second)
    if not len(spkcounter.spiketimes[0]):
        rate = 0.0
    else:
        rate = np.mean(1.0/np.diff(spkcounter.spiketimes[0]))
    #print(rate,model.Iext)
    return abs(rate-target)

from scipy.optimize import fmin

P = fmin(search_current_for_freq, 17.0, 
         args=(targetF,G_leak,spks_leak), disp = 0, full_output=0)
Iext_leak = P[0]
P = fmin(search_current_for_freq, 17.0, 
         args=(targetF,G_noleak,spks_noleak), disp = 0, full_output=0)
Iext_no_leak = P[0]

print('''The current necessary to evoke {0}Hz 
firing is {1:.3} and {2:.3} mA for the leaky and non-leaky 
models respectively.'''.format(int(targetF),Iext_leak*b.mA, Iext_no_leak*b.mA))


# ### Compute the phase response curve
# Plots figure to compare phase response curves of the leaky and non-leaky models. Persistent re-init and removal of parameters slows down computations, could improve by performing simulations on population of neurons to compute PRC in parallel, but keeping like this for simplicity.

def compute_prc_point(model,ref,phase=0.5,exportV=False):
    '''
    Function to compute the spike advance in response to a stimulus.
    '''
    b.reinit()
    G_leak.Iext = Iext_leak*b.mA
    G_noleak.Iext = Iext_no_leak*b.mA
    Ipert = 100*b.pA
    if exportV:
        V = b.StateMonitor(model,'V',record=True)
    spks = b.SpikeMonitor(model,record=True)
    # Stimulation
    stim_time = [(0,ref[0] + phase*(np.diff(ref))*b.second)]
    perturbation=b.SpikeGeneratorGroup(1,stim_time)
    connection=b.Connection(perturbation,model,'V')
    connection[0,0]=1*b.mV
    b.run(tstop)
    del perturbation,connection
    #print ref,spks[0],spks[0][4]
    spk_advance =  spks[0][4]-ref[0]
    if exportV:
        tmp = V[0].copy()/b.mV
        del V
        return spk_advance, tmp
    
    return spk_advance

# This is the part that takes longer. Not optimized.
b.reinit() # So that we go back to zero (there must be a better way..)

G_leak.Iext = Iext_leak*b.mA
G_noleak.Iext = Iext_no_leak*b.mA

# Run it once to find the reference spiketimes
spks_leak = b.SpikeMonitor(G_leak,record=True)
spks_noleak = b.SpikeMonitor(G_noleak,record=True)

V_leak = b.StateMonitor(G_leak,'V',record=True)
V_noleak = b.StateMonitor(G_noleak,'V',record=True)

tstop = 0.25 * b.second
b.run(tstop)

ref_leak = spks_leak.spiketimes[0][3:5]
ref_noleak = spks_noleak.spiketimes[0][3:5]
# For plotting (later)
time = V_leak.times.copy()
Vref_leak = V_leak[0].copy()/b.mV
Vref_noleak = V_noleak[0].copy()/b.mV
del V_leak,V_noleak,spks_leak,spks_noleak

[spk_advance,tmpV_leak]=compute_prc_point(G_leak,ref_leak,exportV=True)
[spk_advance,tmpV_noleak]=compute_prc_point(G_noleak,ref_noleak,exportV=True)

# Phase response computation
phase = np.linspace(0,1,50) # Change resolution of the phase response curve
spk_advance_leak = np.zeros(phase.shape)
spk_advance_noleak = np.zeros(phase.shape)

for ii,phi in enumerate(phase):
    spk_advance_leak[ii]=compute_prc_point(G_leak,ref_leak,phi)
    spk_advance_noleak[ii]=compute_prc_point(G_noleak,ref_noleak,phi)


# Redefine the model groups to include more neurons
# Network simulation
b.reinit()

N = 5000
# Neuron groups for leaky and non-leaky integrate and fire
G_leak=b.NeuronGroup(N, model=eqs_leak, reset=Vr, threshold=Vt)
G_noleak=b.NeuronGroup(N, model=eqs_noleak, reset=Vr, threshold=Vt)

G_leak.Iext = Iext_leak*b.mA
G_noleak.Iext = Iext_no_leak*b.mA

# Initialize equal sampling of phase
G_noleak.V = np.linspace(Vr,Vt,N)

fTh = lambda x:(Iext_leak*b.mV)-((Iext_leak*b.mV)-Vt)/(np.exp(-x/tau))
x = -np.linspace(-1./(targetF*b.Hz),0*b.ms,N+1)[1:]
G_leak.V = fTh(x)

stim_time = []
np.random.seed(seed=9192037)
for ii in (200+400*np.cumsum(abs(b.rand(100))))*b.ms:#np.linspace(200,800,10):
    stim_time.append((0,ii))

perturbation = b.SpikeGeneratorGroup(1,stim_time)
connection_leak = b.Connection(perturbation,G_leak)
connection_noleak = b.Connection(perturbation,G_noleak)
for ii in range(N):
    connection_leak[0,ii] = 0.3*b.mV
    connection_noleak[0,ii] = 0.3*b.mV

spks_leak = b.SpikeMonitor(G_leak,record=True)
spks_noleak = b.SpikeMonitor(G_noleak,record=True)

b.run(5*b.second)

all_spks_leak = np.hstack([spks_leak[ii] for ii in range(N)])
all_spks_noleak = np.hstack([spks_noleak[ii] for ii in range(N)])
edges = np.arange(0,5,0.01)
bins_leak,edges = np.histogram(all_spks_leak,edges,density=False)
bins_noleak,edges = np.histogram(all_spks_noleak,edges,density=False)

# Gather the entire figure and print it
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['font.sans-serif'] = 'arial'
rcParams['font.size'] = 8
fig = plt.figure(figsize = (6.92,5.92))
ax=list()
ax.append(fig.add_axes([0.1,0.77,0.35,0.18]))
ax.append(fig.add_axes([0.58,0.77,0.35,0.18]))
ax.append(fig.add_axes([0.07,0.54,0.4,0.15]))
ax.append(fig.add_axes([0.52,0.54,0.4,0.15]))
ax.append(fig.add_axes([0.1,0.3,0.85,0.15]))
ax.append(fig.add_axes([0.1,0.1,0.85,0.15]))
fig.text(0.05,0.95,'A',fontsize=10,verticalalignment='bottom',
         horizontalalignment='right')
fig.text(0.50,0.95,'B',fontsize=10,verticalalignment='bottom',
         horizontalalignment='right')
fig.text(0.05,0.69,'C',fontsize=10,verticalalignment='bottom',
         horizontalalignment='right')
fig.text(0.06,0.63,'non-leaky',fontsize=9,verticalalignment='center',
         horizontalalignment='right',rotation=90)
#fig.text(0.5,0.69,'D',fontsize=10,verticalalignment='bottom',
#         horizontalalignment='right')
fig.text(0.51,0.63,'leaky',fontsize=9,verticalalignment='center',
         horizontalalignment='right',rotation=90)
fig.text(0.05,0.48,'D',fontsize=10,verticalalignment='bottom',
         horizontalalignment='right')
fig.text(0.15,0.45,'non-leaky',fontsize=9,verticalalignment='bottom',
         horizontalalignment='left',rotation=0)
fig.text(0.15,0.25,'leaky',fontsize=9,verticalalignment='bottom',
         horizontalalignment='left',rotation=0)

try: # In case this was not ran move along.
    # Voltage trajectories
    ax[0].plot(time/b.mV,Vref_leak,color='black')
    ax[0].plot(time/b.mV,Vref_noleak,'-',color='gray')
    ax[0].plot(time/b.mV,tmpV_leak,color='r')
    ax[0].plot(time/b.mV,tmpV_noleak,'-',color=[.5,0,0])
    ax[0].set_xlim((ref_leak+np.array([-0.005,+0.005]))/b.mV)
    ax[0].set_xlabel('time (ms)')
    ax[0].set_ylabel('Voltage (mV)')

    # Phase response curves for each model
    ax[1].plot(phase,1-(spk_advance_leak/np.diff(ref_leak)),color='black')
    ax[1].plot(phase,1-(spk_advance_noleak/np.diff(ref_noleak)),color='gray')
    ax[1].axis('tight')
    ax[1].plot([0.5,0.5],ax[1].get_ylim(),'--',color='red')
    ax[1].set_xlabel('phase ($\phi$)')
    ax[1].set_ylabel('phase advance ($\Delta\phi$)')
except:
    pass
# Pick 10 random neurons
NN = 15
tmin=0.15
tmax = 1.5
for ii in stim_time:
    if (ii[1]>tmin) & (ii[1] < tmax):
        ax[2].plot([0,0]+ii[1],[0,NN],'-',color='red',lw=0.7)
for ii,i in enumerate(np.linspace(0, N-1, num=NN).astype('int32')):
    sp = spks_noleak[i][(spks_noleak[i]>tmin) & (spks_noleak[i]<tmax)]
    X = np.tile(sp,(2,1))
    Y = np.transpose(np.tile(np.transpose([ii,ii+0.6]),(len(sp),1))+0)
    ax[2].plot(X,Y,color='black',lw=1)
ax[2].axis('tight')

for ii in stim_time:
    if (ii[1]>tmin) & (ii[1] < tmax):
        ax[3].plot([0,0]+ii[1],[0,NN],'-',color='red',lw=0.7)
for ii,i in enumerate(np.linspace(0, N-1, num=NN).astype('int32')):
    sp = spks_leak[i][(spks_leak[i]>tmin) & (spks_leak[i]<tmax)]
    X = np.tile(sp,(2,1))
    Y = np.transpose(np.tile(np.transpose([ii,ii+0.6]),(len(sp),1))+0)
    ax[3].plot(X,Y,color='black',lw=1)
ax[3].axis('tight')

# Fix axes properties 
ax[-1].fill_between(edges[:-1]+np.diff(edges)/2,bins_leak/(N*(np.diff(edges[:2]))),where=bins_leak>=0, 
                 interpolate=False,color='black',edgecolor=None)
plt.hold(True)
ax[-2].fill_between(edges[:-1]+np.diff(edges)/2,bins_noleak/(N*(np.diff(edges[:2]))),where=bins_noleak>=0, 
                 interpolate=False,color='black',edgecolor=None)#align='center',width=min(np.diff(edges)),edgecolor=None,

ax[-1].set_xlabel('time (s)')
for a in ax[4:6]:
    a.set_xlim([0,max(edges)])
    a.set_ylim([0,90])
    a.set_ylabel('Population Rate (Hz)')
    for ii in stim_time:
        a.plot([0,0]+ii[1],plt.gca().get_ylim(),'--',color='red')
ax[-2].xaxis.set_visible(False)
for a in ax:
    a.set_axisbelow(True)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['left'].set_visible(True)
    a.spines['bottom'].set_visible(True)
    a.yaxis.set_ticks_position('left')
    a.xaxis.set_ticks_position('bottom')
    for line in a.xaxis.get_ticklines():
        line.set_markeredgewidth(1)
    for line in a.yaxis.get_ticklines():
        line.set_markeredgewidth(1)

for a in ax[2:4]:
    a.spines['left'].set_visible(False)
    a.yaxis.set_visible(False)
    a.set_xlabel('time (s)')

figurepath = 'lif_prc_study'
fig.savefig('{0}.pdf'.format(figurepath))

