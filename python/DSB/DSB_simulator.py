from neuron import h
import numpy as np
import pylab as p
import time
import sys
import os
import h5py as h5
import itertools as it
from DS2M0Purk import DS2M0Purk

def save_dict(fid, group, data):
    for key,value in data.iteritems():
        if isinstance(value, dict):
            new_group = fid.create_group(group.name + '/' + key)
            save_dict(fid, new_group, value)
        elif type(value) in (int,float,tuple,str):
            group.attrs.create(key,value)
        else:
            group.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=9)

def save_h5_file(filename, **kwargs):
    with h5.File(filename, 'w') as fid:
        save_dict(fid, fid, kwargs)

def make_output_filename(prefix='', extension='.out'):
    filename = prefix
    if prefix != '' and prefix[-1] != '_':
        filename = filename + '_'
    now = time.localtime(time.time())
    filename = filename + '%d%02d%02d-%02d%02d%02d' % \
        (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    if extension[0] != '.':
        extension = '.' + extension
    suffix = ''
    k = 0
    while os.path.exists(filename + suffix + extension):
        k = k+1
        suffix = '_%d' % k
    return filename + suffix + extension

def somatic_current_injection(cell, amplitude=3., tbefore=100., tstim=500., tafter=100., celsius=37.):
    Vrest = -68.
    print('Inserting stimulus...')
    stim = h.IClamp(cell.soma(0.5))
    stim.delay = tbefore
    stim.dur = tstim
    stim.amp = float(amplitude)
    print('Setting up recorders...')
    rec = {}
    for lbl in 't','vsoma','vdend','spikes':
        rec[lbl] = h.Vector()
    rec['t'].record(h._ref_t)
    rec['vsoma'].record(cell.soma(0.5)._ref_v)
    rec['vdend'].record(cell.dendrites[1511](0.5)._ref_v)
    apc = h.APCount(cell.soma(0.5))
    apc.record(rec['spikes'])
    print('Setting up the simulation...')
    h.load_file('stdrun.hoc')
    h.t = 0.
    h.dt = 0.02
    h.v_init = Vrest
    h.celsius = celsius
    h.tstop = tbefore + tstim + tafter
    print('Running model...')
    h.finitialize(h.v_init)
    h.run()
    return np.array(rec['t'])*1e-3,np.array(rec['vsoma']),np.array(rec['vdend']),np.array(rec['spikes'])*1e-3

def synaptic_activation(cell, exc_rate, inh_rate, tstop=5000., seed=None, celsius=37.):
    if not cell.with_synapses:
        raise Exception('No synapses present')

    cell.add_offset_current()
    
    if seed is None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    print('Computing the activation times of the synapses...')
    if exc_rate > 0:
        nspikes = (tstop/1e3)*exc_rate
        for syn in cell.synapses['granule_cells']:
            isi = -np.log(np.random.uniform(size=nspikes)) / exc_rate * 1e3
            cell.set_presynaptic_spike_times(syn, np.cumsum(isi))
    if inh_rate > 0:
        nspikes = (tstop/1e3)*inh_rate
        for syn in cell.synapses['stellate_cells']:
            isi = -np.log(np.random.uniform(size=nspikes)) / inh_rate * 1e3
            cell.set_presynaptic_spike_times(syn, np.cumsum(isi))

    print('Setting up recorders...')
    rec = {}
    for lbl in 't','vsoma','vdend','spikes':
        rec[lbl] = h.Vector()
    rec['t'].record(h._ref_t)
    rec['vsoma'].record(cell.soma(0.5)._ref_v)
    rec['vdend'].record(cell.dendrites[1511](0.5)._ref_v)
    apc = h.APCount(cell.soma(0.5))
    apc.record(rec['spikes'])
    print('Setting up the simulation...')
    h.load_file('stdrun.hoc')
    h.t = 0.
    h.dt = 0.02
    h.v_init = -69.
    h.celsius = celsius
    h.tstop = tstop

    print('Running model...')
    h.finitialize(h.v_init)
    h.run()

    return np.array(rec['t'])*1e-3,np.array(rec['vsoma']),np.array(rec['vdend']),np.array(rec['spikes'])*1e-3

def compute_PRC(cell, firing_rate, exc_rate=None, inh_rate=None, seed=None, which_isi=0, ttran=3000., tstop=5000., ntrials=100, pulse_amp=0.5, pulse_dur=0.5, celsius=37., jittered_isi=False):

    if cell.with_synapses: # PRC with synaptic activation
        cell.add_offset_current()
        np.random.seed(seed)
        print('Computing the activation times of the synapses...')
        presyn_spike_times = {}
        if exc_rate > 0:
            nspikes = (tstop/1e3)*exc_rate
            presyn_spike_times['exc'] = np.zeros((len(cell.synapses['granule_cells']),nspikes))
            for i,syn in enumerate(cell.synapses['granule_cells']):
                isi = -np.log(np.random.uniform(size=nspikes)) / exc_rate * 1e3
                cell.set_presynaptic_spike_times(syn, np.cumsum(isi))
                presyn_spike_times['exc'][i,:] = np.cumsum(isi)
        if inh_rate > 0:
            nspikes = (tstop/1e3)*inh_rate
            presyn_spike_times['inh'] = np.zeros((len(cell.synapses['stellate_cells']),nspikes))
            for i,syn in enumerate(cell.synapses['stellate_cells']):
                isi = -np.log(np.random.uniform(size=nspikes)) / inh_rate * 1e3
                cell.set_presynaptic_spike_times(syn, np.cumsum(isi))
                presyn_spike_times['inh'][i,:] = np.cumsum(isi)
    else:
        print('Not adding any stimulation to the cell.')

    pulse = h.IClamp(cell.soma(0.5))
    pulse.delay = 1e8
    pulse.dur = pulse_dur
    pulse.amp = pulse_amp

    print('Setting up recorders...')
    rec = {}
    for lbl in 't','vsoma','spikes':
        rec[lbl] = h.Vector()
    rec['t'].record(h._ref_t)
    rec['vsoma'].record(cell.soma(0.5)._ref_v)
    apc = h.APCount(cell.soma(0.5))
    apc.record(rec['spikes'])

    print('Setting up the simulation...')
    h.load_file('stdrun.hoc')
    h.t = 0.
    h.dt = 0.02
    h.v_init = -69.
    h.celsius = celsius
    h.tstop = ttran + 100

    h.finitialize(h.v_init)

    print('Running the model...')
    sys.stdout.flush()
    h.run()

    found = False
    while h.tstop <= tstop:
        t = np.array(rec['t'])
        v = np.array(rec['vsoma'])
        spikes = np.array(rec['spikes'])
        spikes = spikes[spikes>ttran]
        isi = np.diff(spikes)
        if firing_rate <= 0:
            firing_rate = 1000./isi[0]
        try:
            idx = np.where((1000./isi >= firing_rate) & (1000./isi <= firing_rate+1.))[0][which_isi]
            found = True
            break
        except:
            print('No spikes @ %g Hz in the first %g ms. Continuing...' % (firing_rate,h.tstop))
            h.tstop += 100.
            h.continuerun(h.tstop)

    if not found:
        print('Unable to find the right ISI in the first %g ms.' % h.tstop)
        print np.sort(1000./isi)
        #ax = p.subplot(211)
        #p.plot(t,v,'k')
        #p.subplot(212, sharex=ax)
        #p.plot(spikes[1:],1000./isi,'ko')
        #p.show()
        sys.exit(0)

    tbefore = 2.
    tafter = 5.
    t0 = spikes[idx]
    t1 = spikes[idx+1]
    idx = int(t0/h.dt) + np.argmax(v[int(t0/h.dt):int((t0+2)/h.dt)])
    t0 = t[idx]
    idx = int(t1/h.dt) + np.argmax(v[int(t1/h.dt):int((t1+2)/h.dt)])
    t1 = t[idx]
    print('Good ISI found between %g and %g ms.' % (t0,t1))

    if cell.with_synapses and jittered_isi:
        print('Adding all other objects...')
        rs = np.random.RandomState(int(time.time()))
        for j in range(1,ntrials):
            np.random.seed(seed)
            for k,syn in enumerate(it.chain(cell.synapses['granule_cells'],cell.synapses['stellate_cells'])):
                s,st,nc = cell.insert_synapse(syn['sec'], syn['tau'], syn['E'], 0.)
                if syn in cell.synapses['granule_cells']:
                    isi = -np.log(np.random.uniform(size=len(syn['spike_times'][0]))) / exc_rate * 1e3
                else:
                    isi = -np.log(np.random.uniform(size=len(syn['spike_times'][0]))) / inh_rate * 1e3
                spks = np.cumsum(isi)            
                idx, = np.where((spks>t0) & (spks<t1))
                if len(idx) > 0:
                    isi[idx] += rs.uniform(size=len(idx))
                spks = np.cumsum(isi)
                vec = h.Vector(spks)
                st.play(vec)
                syn['syn'].append(s)
                syn['stim'].append(st)
                syn['conn'].append(nc)
                syn['spike_times'].append(vec)

    print('Running the model again...')
    sys.stdout.flush()
    rec['t'].resize(0)
    rec['vsoma'].resize(0)
    rec['spikes'].resize(0)
    apc.n = 0
    h.t = 0

    h.tstop = t0 - tbefore
    h.finitialize(h.v_init)
    h.run()

    print('Saving the state...')
    ss = h.SaveState()
    ss.save()

    h.dt = 0.005
    nsamples = np.round((t1-t0+tbefore+tafter) / h.dt)
    V = np.zeros((ntrials,nsamples))
    spike_times = np.nan + np.zeros((ntrials,2))
    perturbation_times = np.zeros(ntrials)
    for i in range(ntrials):
        sys.stdout.write('\rTrial [%02d/%02d] ' % (i+1,ntrials))
        sys.stdout.flush()
        ss.restore()
        rec['t'].resize(0)
        rec['vsoma'].resize(0)
        rec['spikes'].resize(0)
        apc.n = 0
        if cell.with_synapses and jittered_isi:
            for syn in it.chain(cell.synapses['granule_cells'],cell.synapses['stellate_cells']):
                for j in range(ntrials):
                    syn['conn'][j].weight[0] = 0
                syn['conn'][i].weight[0] = syn['w']
        if i > 0:
            pulse.delay = t0 + i*(t1+5-t0)/ntrials
        perturbation_times[i] = pulse.delay
        h.continuerun(t1+tafter)
        idx, = np.where(np.array(rec['spikes']) > t0-tbefore)
        try:
            spike_times[i,:] = np.array(rec['spikes'])[idx[0]:idx[0]+2]
        except:
            pass
        V[i,:] = np.array(rec['vsoma'])[:nsamples]
    t = np.array(rec['t'])[idx[0]:idx[0]+nsamples]

    sys.stdout.write('\n')

    #p.figure()
    #for i in range(ntrials):
    #    p.plot(t,V[i,:])
    #p.show()

    if cell.with_synapses:
        return t,V,perturbation_times,spike_times,presyn_spike_times,1000./(t1-t0)
    return t,V,perturbation_times,spike_times,1000./(t1-t0)

def usage():
    print('')
    print('This script can be used to compute the PRC of the DeSchutter-Bower Purkinje cell model.')
    print('')
    print('Usage:')
    print('')
    print('   %s iclamp [amplitude]  Simulate the injection of a constant current step lasting 10 seconds.' % os.path.basename(sys.argv[0]))
    print('                          The default amplitude is 0.1 nA.')
    print('   %s synapses            Simulate the activation of synapses across the dendritic tree, lasting 10 seconds.' % os.path.basename(sys.argv[0]))
    print('   %s PRC_syn             Compute the PRC during synaptic activation.' % os.path.basename(sys.argv[0]))
    print('   %s PRC_iclamp          Compute the PRC during current clamp stimulation.' % os.path.basename(sys.argv[0]))
    print('   %s [-h|--help]         Print this help message and exit.' % os.path.basename(sys.argv[0]))
    print('')
    print('Author: Daniele Linaro - danielelinaro@gmail.com')
    print('')

def main():
    if sys.argv[1] in ('-h','--help'):
        usage()
        sys.exit(0)
    elif sys.argv[1] == 'iclamp':
        cell = DS2M0Purk(g_granule=None,g_stellate=None) # no synapses
        # Somatic current injection
        try:
            amplitude = float(sys.argv[2])
        except:
            amplitude = 0.1
        tbefore = 0
        tstim = 10000.
        tafter = 0.
        celsius = 28.
        t,vsoma,vdend,spikes = somatic_current_injection(cell, amplitude, tbefore, tstim, tafter, celsius)
        tbefore *= 1e-3
        tstim *= 1e-3
        print('Firing rate: {0} Hz.'.format(len(np.intersect1d(np.where(spikes>=tbefore+1)[0],\
                                                                   np.where(spikes<=tbefore+tstim)[0]))/(tstim-1)))
    elif sys.argv[1] == 'synapses':
        # Dendritic synapse activation
        cell = DS2M0Purk(g_granule=0.7e-3, g_stellate=[7000.,1400.])
        t,vsoma,vdend,spikes = synaptic_activation(cell,exc_rate=35.,inh_rate=2, tstop=10000., seed=5061983)
        save_h5_file(make_output_filename('synaptic_activation_','.h5'), dt=t[1]-t[0], V=vsoma)
    elif sys.argv[1] == 'PRC_syn':
        # Computation of the PRC
        g_granule = 0.7e-3
        g_stellate = [7000.,1400.]
        cell = DS2M0Purk('PM9',g_granule, g_stellate)
        exc_rate = 35.
        inh_rate = 2.
        try:
            firing_rate = float(sys.argv[2])
        except:
            firing_rate = 85.
        print('Target firing rate: %.1f Hz.' % firing_rate)
        try:
            pulse_amp = float(sys.argv[3])
        except:
            pulse_amp = 0.5
        print('Pulse amplitude: %.2f nA.' % pulse_amp)
        which_isi = 0
        ttran = 5000.
        tstop = 10000.
        ntrials = 50
        seed = 1416394853 # int(time.time())
        pulse_dur = 0.5
        temperature = 37.
        t,V,perturbation_times,spike_times,presyn_spike_times,actual_firing_rate = \
            compute_PRC(cell=cell, firing_rate=firing_rate, exc_rate=exc_rate, inh_rate=inh_rate, seed=seed, which_isi=which_isi, ttran=ttran, \
                            tstop=tstop, ntrials=ntrials, pulse_amp=pulse_amp, pulse_dur=pulse_dur, celsius=temperature, jittered_isi=False)
        save_h5_file(make_output_filename('prc_','.h5'),dt=t[1]-t[0],V=V,perturbation_times=perturbation_times,
                     spike_times=spike_times,exc_rate=exc_rate,inh_rate=inh_rate,firing_rate=actual_firing_rate,
                     which_isi=which_isi,ttran=ttran,tstop=tstop,ntrials=ntrials,g_granule=g_granule,g_stellate=g_stellate,
                     presyn_spike_times=presyn_spike_times,seed=seed,pulse_amp=pulse_amp,pulse_dur=pulse_dur, temperature=temperature)
    elif sys.argv[1] == 'PRC_iclamp':
        # Computation of the PRC
        cell = DS2M0Purk('PM10',None,None)
        try:
            firing_rate = float(sys.argv[2])
        except:
            firing_rate = 85.
        print('Target firing rate: %.1f Hz.' % firing_rate)
        try:
            offset = float(sys.argv[3])
        except:
            offset = 0.2
        print('DC offset: %.2f nA.' % offset)
        try:
            pulse_amp = float(sys.argv[4])
        except:
            pulse_amp = 0.5
        print('Pulse amplitude: %.2f nA.' % pulse_amp)
        cell.add_offset_current(offset)
        which_isi = 0
        ttran = 3000.
        tstop = 3100.
        ntrials = 50
        pulse_dur = 0.5
        t,V,perturbation_times,spike_times,actual_firing_rate = \
            compute_PRC(cell, firing_rate=firing_rate, exc_rate=None, inh_rate=None, seed=None, which_isi=which_isi, \
                            ttran=ttran, tstop=tstop, ntrials=ntrials, pulse_amp=pulse_amp, pulse_dur=pulse_dur, celsius=28)
        save_h5_file(make_output_filename('prc_','.h5'),dt=t[1]-t[0],V=V,perturbation_times=perturbation_times,
                     spike_times=spike_times,firing_rate=actual_firing_rate,
                     which_isi=which_isi,ttran=ttran,tstop=tstop,ntrials=ntrials,
                     pulse_amp=pulse_amp,pulse_dur=pulse_dur)

    try:
        print np.sort(1./np.diff(spikes))
        p.figure()
        p.plot(t,vsoma,'k',label='Soma')
        p.plot(t,vdend,'r',label='Dendrite')
        p.xlabel('Time (s)')
        p.ylabel('Membrane voltage (mV)')
        p.legend(loc='best')
        p.figure()
        p.plot(spikes[1:],1./np.diff(spikes),'ko')
        p.xlabel('Time (s)')
        p.ylabel('1/ISI (s^-1)')
        p.show()
    except:
        pass

if __name__ == '__main__':
    main()
