#!/usr/bin/env python

from neuron import h
import numpy as np
import numpy.random as rnd
import ConfigParser as cp
import sys
import os
import tables as tbl
import time

DEBUG = False
default_configuration_file = 'simulator.cfg'

class SimulationDetails(tbl.IsDescription):
    neuron     = tbl.StringCol(2)
    L          = tbl.Float64Col()
    diam       = tbl.Float64Col()
    Rin        = tbl.Float64Col()
    T          = tbl.Float64Col()
    Tdelay     = tbl.Float64Col()
    dt         = tbl.Float64Col()
    Ibase      = tbl.Float64Col()
    Iperturb   = tbl.Float64Col()
    durperturb = tbl.Float64Col()
    noisemu    = tbl.Float64Col()
    noisesigma = tbl.Float64Col()
    noisetau   = tbl.Float64Col()
    with_pid   = tbl.Int32Col()

def savePRCData(filename, neuron_type, L, diam, Rin, T, Tdelay, dt, Ibase, Iperturb, \
                    durperturb, noise_mu, noise_sigma, noise_tau, tspikes, tperturb, with_pid):
    h5file = tbl.openFile(filename, mode='w', title='Simulations for PRC calculation')
    table = h5file.createTable(h5file.root, 'Details', SimulationDetails, 'Simulation info')
    details = table.row
    details['neuron'] = neuron_type
    details['L'] = L
    details['diam'] = diam
    details['Rin'] = Rin
    details['T'] = T
    details['Tdelay'] = Tdelay
    details['dt'] = dt
    details['Ibase'] = Ibase
    details['Iperturb'] = Iperturb
    details['durperturb'] = durperturb
    details['with_pid'] = with_pid
    details['noisemu'] = noise_mu
    details['noisesigma'] = noise_sigma
    details['noisesigma'] = noise_tau
    details.append()
    group = h5file.createGroup(h5file.root, 'Data', 'Spikes and perturbations times')
    h5file.createArray(group, 'spikes', tspikes, 'Spikes times')
    h5file.createArray(group, 'perturb', tperturb, 'Perturbation times')
    h5file.close()

def makeOutputFilename(prefix='', extension='.out'):
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

def usage():
    print('')
    print('This script can be used to compute the PRC of a Purkinje cell or its CV in the presence')
    print('of channel noise.')
    print('')
    print('Usage:')
    print('')
    print('   %s prc [options] [configuration file]' % os.path.basename(sys.argv[0]))
    print('   %s cv [options] length diameter' % os.path.basename(sys.argv[0]))
    print('   %s [-h|--help]         print this help message and exit.' % os.path.basename(sys.argv[0]))
    print('')
    print('If "prc" is specified as a first argument, the script will compute a PRC, using the options')
    print('contained in the (optional) configuration file passed as an argument. Alternatively, the script')
    print('will look for a file called %s.' % default_configuration_file)
    print('Additionally, the following options are accepted:')
    print('')
    print('       -o,--output        specify the path of the file where the results will be saved.')
    print('          --with-pid      use a PID controller to clamp the frequency and deliver the stimulation pulses.')
    print('')
    print('If "cv" is specified as a first argument, the script will simulate a model containing channel')
    print('noise for 3 seconds and will print the coefficient of variation of the inter-spike intervals.')
    print('As an example, a length of 60 um and a diameter of 50 um lead to a CV of approximately 0.1, while')
    print('a length of 110 um and a diameter of 100 um lead to a CV of approximately 0.05.')
    print('Additionally, the following options are accepted:')
    print('')
    print('       -d,--duration      specify the duration of the simulation (in ms, default 5000).')
    print('       -t,--transient     specify the duration of the transient (in ms, default 1000).')
    print('       -f,--firing-rate   specify the firing rate of the cell (in Hz, default 30).')
    print('          --dt            specify the timestep (in ms, default 0.001).')
    print('')
    print('Author: Daniele Linaro - danielelinaro@gmail.com')
    print('')

def parseArgs():
    import getopt

    if len(sys.argv) == 1:
        print('Type %s -h for help on how to use this program.' % os.path.basename(sys.argv[0]))
        sys.exit(1)

    if sys.argv[1] in ('-h','--help'):
        usage()
        sys.exit(0)

    if not sys.argv[1].lower() in ('prc','cv'):
        print('Mode must be either "prc" or "cv".')
        sys.exit(1)

    options = {'mode': sys.argv[1].lower()}
    if options['mode'] == 'prc':
        options['config_file'] = default_configuration_file
        options['output_file'] = makeOutputFilename('prc_', '.h5')
        options['use_pid'] = False
        opts,args = getopt.getopt(sys.argv[2:], 'o:', ['with-pid','output='])
        for o,a in opts:
            if o in ('-o','--output'):
                options['output_file'] = a
            elif o == '--with-pid':
                options['use_pid'] = True
        if len(args) == 1:
            options['config_file'] = args[0]
        elif len(args) > 1:
            print('You can specify only one configuration file.')
            sys.exit(1)
    else:
        opts,args = getopt.getopt(sys.argv[2:], 'd:t:f:', ['duration=','transient=','dt=','firing-rate='])
        options['duration'] = 5000
        options['transient'] = 1000
        options['dt'] = 0.001
        options['firing_rate'] = 30
        for o,a in opts:
            if o in ('-d','--duration'):
                options['duration'] = float(a)
            elif o in ('-t','--transient'):
                options['transient'] = float(a)
            elif o == '--dt':
                options['dt'] = float(a)
            elif o in ('-f','--firing-rate'):
                options['firing_rate'] = float(a)
        if len(args) != 2:
            print('You must specify length and diameter of the cell.')
            sys.exit(0)
        options['length'] = float(args[0])
        options['diameter'] = float(args[1])
    return options

def makeKR(L=20, diam=20, type='deterministic'):
    sec = h.Section()
    if type == 'deterministic':
        suffix = ''
    elif type == 'stochastic':
        suffix = '_cn'
    else:
        raise Exception('KR: no such mechanism.')        
    sec.insert('naRsg' + suffix)
    sec.insert('kpkj' + suffix)
    sec.insert('kpkj2' + suffix)
    sec.insert('kpkjslow' + suffix)
    sec.insert('bkpkj' + suffix)
    sec.insert('hpkj' + suffix)
    sec.insert('cadiff')
    sec.insert('cap')
    sec.insert('lkpkj')
    if type == 'stochastic':
        import numpy.random as rnd
        rnd.seed(int(time.time()))
        for mech in sec(0.5):
            if 'cn' in mech.name():
                mech.seed = int(rnd.uniform() * 100000)
                print('%s>> seed: %ld' % (mech.name().split('_')[0],mech.seed))
    sec.L = L
    sec.diam = diam
    sec.ena = 60
    sec.ek = -88
    return sec

def computeInputResistance(segment, Irange, dur, delay, dt=0.005, plot=False):
    if plot:
        import pylab as p
    stim = makeIclamp(segment, dur, 0, delay)
    rec = makeRecorders(segment, {'v': '_ref_v'})
    ap = h.APCount(segment)
    ap.thresh = -20
    spks = h.Vector()
    ap.record(spks)
    I = []
    V = []
    h.load_file('stdrun.hoc')
    h.dt = dt
    h.celsius = 36
    h.tstop = dur+delay*2
    if plot:
        p.figure()
        p.subplot(1,2,1)
    for k,i in enumerate(np.arange(Irange[0],Irange[1],Irange[2])):
        spks.clear()
        ap.n = 0
        stim.amp = i
        h.run()
        spike_times = np.array(spks)
        if len(np.intersect1d(np.nonzero(spike_times>delay)[0], np.nonzero(spike_times<delay+dur)[0])) == 0:
            t = np.array(rec['t'])
            v = np.array(rec['v'])
            idx = np.intersect1d(np.nonzero(t > delay+0.75*dur)[0], np.nonzero(t < delay+dur)[0])
            I.append(i)
            V.append(np.mean(v[idx]))
        else:
            print('The neuron emitted spikes at I = %g pA' % (stim.amp*1e3))
        if plot:
            p.plot(1e-3*t,v)
    V = np.array(V)*1e-3
    I = np.array(I)*1e-9
    poly = np.polyfit(I,V,1)
    if plot:
        ymin,ymax = p.ylim()
        p.plot([1e-3*(delay+0.75*dur),1e-3*(delay+0.75*dur)],[ymin,ymax],'r--')
        p.plot([1e-3*(delay+dur),1e-3*(delay+dur)],[ymin,ymax],'r--')
        p.xlabel('t (s)')
        p.ylabel('V (mV)')
        p.box(True)
        p.grid(False)
        p.subplot(1,2,2)
        x = np.linspace(I[0],I[-1],100)
        y = np.polyval(poly,x)
        p.plot(1e12*x,1e3*y,'k--')
        p.plot(1e12*I,1e3*V,'bo')
        p.xlabel('I (pA)')
        p.ylabel('V (mV)')
        p.show()
    return poly[0]
    
def optimizeF(segment, F, ftol=0.1, dur=5000, dt=0.025, amp=[0,0.2], delay=200, maxiter=50):
    from sys import stdout
    f = F + 2*ftol
    iter = 0
    stim = makeIclamp(segment, dur, amp[1], delay)
    spks = h.Vector()
    apc = h.APCount(segment)
    apc.thresh = -20
    apc.record(spks)
    rec = makeRecorders(segment, {'v': '_ref_v'})
    print('\nStarting frequency optimization: target is F = %.2f.' % F)

    h.load_file('stdrun.hoc')
    h.dt = dt
    h.celsius = 36
    h.tstop = dur+2*delay
    h.run()

    f = float(apc.n)/(dur*1e-3)
    if f < F:
        print('[00] !! Increase maximal current !!')
        raise Exception('Required frequency out of current bounds')
    else:
        print('[00] I = %.4f -> F = %.4f Hz.' % (stim.amp, f))

    while abs(F - f) > ftol and iter < maxiter:
        iter = iter+1
        stim.amp = (amp[0]+amp[1])/2
        stdout.write('[%02d] I = %.4f ' % (iter, stim.amp))
        spks = h.Vector()
        apc.n = 0
        apc.record(spks)
        h.t = 0
        h.run()
        if len(spks) == 0:
            amp[0] = stim.amp
            stdout.write('no spikes.\n')
            stdout.flush()
            continue
        f = float(apc.n) / (dur*1e-3)
        stdout.write('-> F = %.4f Hz.\n' % f)
        stdout.flush()
        if f > F:
            amp[1] = stim.amp
        else:
            amp[0] = stim.amp
    I = stim.amp
    del apc
    del stim
    del spks
    return f,I

def makeIclamp(segment, dur, amp, delay=0):
    stim = h.IClamp(segment)
    stim.delay = delay
    stim.dur = dur
    stim.amp = amp
    return stim

def makeNoisyIclamp(segment, dur, dt, mu, sigma, tau, delay=0, seed=int(time.time())):
    np.random.seed(seed)
    if abs(tau) < 1e-12:
        nsteps = int(np.ceil((dur)/dt)) + 1
        I = mu + sigma*np.random.normal(size=nsteps)
    else:
        nsteps = int(np.ceil((dur)/dt)) + 1
        coeff = np.exp(-dt/tau)
        I = (1-np.exp(-dt/tau))*mu + sigma * np.sqrt(2*dt/tau) * np.random.normal(size=nsteps)
        I[0] = mu
        for i in range(1,nsteps):
            I[i] = I[i] + coeff*I[i-1]
    vec = h.Vector(I)
    stim = h.IClamp(segment)
    stim.dur = dur
    stim.delay = delay
    vec.play(stim._ref_amp,dt)
    return stim,vec

def makeRecorders(segment, labels, rec=None):
    if rec is None:
        rec = {'t': h.Vector()}
        rec['t'].record(h._ref_t)
    for k,v in labels.items():
        rec[k] = h.Vector()
        rec[k].record(getattr(segment, v))
    return rec

def computeCV(L,diam,dur,ttran,dt,firing_rate):
    n = makeKR(L,diam,'deterministic')
    n.push()
    ref_area = 50*50*np.pi
    interval = [-0.1,0.4*h.area(0.5)/ref_area]
    F = {}
    F['expected'],I0 = optimizeF(n(0.5),firing_rate,amp=interval)
    n = makeKR(L,diam,'stochastic')
    stim = makeIclamp(n(0.5),dur,I0,0)
    spks = h.Vector()
    apc = h.APCount(n(0.5))
    apc.thresh = -20
    apc.record(spks)
    rec = makeRecorders(n(0.5),{'v':'_ref_v'})
    h.load_file('stdrun.hoc')
    h.celsius = 36
    h.dt = dt
    h.tstop = dur
    h.run()
    spks = np.array(spks)
    isi = np.diff(spks[spks>=ttran])
    F['measured'] = (len(isi)+1)/(dur-ttran)*1e3
    return np.std(isi)/F['measured'],F,I0

def main():
    options = parseArgs()

    if options['mode'] == 'cv':
        CV,F,I0 = computeCV(options['length'],options['diameter'],\
                                options['duration'],options['transient'],options['dt'],options['firing_rate'])
        print(('Length = %g um.\nDiameter = %g um.\nExpected firing rate = %g Hz.\n' + \
                   'Measured firing rate = %g Hz.\nI0 = %g nA.\nCV = %g.') % \
                  (options['length'],options['diameter'],F['expected'],F['measured'],I0,CV))
        with open('CVs.txt','a') as fid:
            fid.write('%10.4f %10.4f %9.1f %9.1f %8.4f %8.4f %8.4f %8.4f %10.5f\n' % \
                          (options['length'],options['diameter'],\
                               options['duration'],options['transient'],options['dt'],\
                               F['expected'],F['measured'],I0,CV))
        sys.exit(0)

    if not os.path.exists(options['config_file']):
        print('%s: no such file.' % options['config_file'])
        sys.exit(1)
    fid = open(options['config_file'],'r')
    config = cp.ConfigParser()
    config.readfp(fid)
    fid.close()

    neuron_type = config.get('Neuron','type')
    prop = {'L': config.getfloat('Neuron','length'),
            'diam': config.getfloat('Neuron','diameter')}

    neuron_mode = 'deterministic'
    if neuron_type[-2:] == 'cn':
        neuron_mode = 'stochastic'
        neuron_type = neuron_type[:2]

    if neuron_type == 'KR':
        n = makeKR(prop['L'],prop['diam'],'deterministic')
        Rin = computeInputResistance(n(0.5), [-0.3,-0.12,0.04], 2000, 200, h.dt, False)*1e-6
        if not options['use_pid']:
            F,I0 = optimizeF(n(0.5),config.getfloat('FiringRate','target_rate'),
                             amp=[float(s) for s in config.get('FiringRate','current_range').split(',')])
        del n
        soma = makeKR(prop['L'],prop['diam'],neuron_mode)
    else:
        print('Unknown neuron type: ' + neuron_type + '. Aborting.')
        sys.exit(1)

    ntrials = config.getint('Simulation','trials')
    Tdelay = config.getfloat('Simulation','Tdelay')

    if not options['use_pid']:
        stimulation_period = config.getfloat('Simulation','T')
    else:
        target_rate = config.getfloat('FiringRate','target_rate')
        spikes_per_perturb = 6
        stimulation_period = 1000. * spikes_per_perturb / target_rate

    Ttotal = Tdelay + ntrials * stimulation_period
    perturbation = {'dur': config.getfloat('Perturbation','duration'), \
                        'amp': config.getfloat('Perturbation','amplitude')}

    try:
        dt = config.getfloat('Simulation','dt')
    except cp.NoOptionError:
        dt = h.dt

    if neuron_mode == 'deterministic':
        noise_mu = config.getfloat('Noise','mean')
        noise_sigma = config.getfloat('Noise','stddev')
        noise_tau = config.getfloat('Noise','tau')
        noise_stim,noise_vec = makeNoisyIclamp(soma(0.5), Ttotal, dt, noise_mu, noise_sigma, noise_tau)
    else:
        noise_mu = 0.
        noise_sigma = 0.
        noise_tau = -1.

    if not options['use_pid']:
        base = makeIclamp(soma(0.5), Ttotal, I0, 0)
        perturb = makeIclamp(soma(0.5), perturbation['dur'], perturbation['amp'], 0)                                     
    else:
        perturb = h.SobolPulses(soma(0.5))
        perturb.delay = Tdelay
        perturb.dur = perturbation['dur']
        perturb.amp = perturbation['amp']
        perturb.F = target_rate
        perturb.spkCount = spikes_per_perturb
        perturb.gp = 0.001
        perturb.gi = 0.1
        nc = [h.NetCon(soma(0.5)._ref_v,perturb,sec=soma), h.NetCon(perturb._ref_i,None)]
        nc[0].delay = 0
        nc[1].delay = 0
        nc[0].threshold = -20
        nc[1].threshold = 0.9 * config.getfloat('Perturbation','amplitude')
        perturb_times = h.Vector()
        nc[1].record(perturb_times)

    apc = h.APCount(soma(0.5))
    apc.thresh = -20
    data = {'spks': h.Vector(), 'perturb': np.zeros(ntrials)}
    apc.record(data['spks'])

    if DEBUG:
        rec = makeRecorders(soma(0.5), {'v':'_ref_v'})
        if neuron_mode == 'deterministic':
            rec = makeRecorders(noise_stim, {'i':'_ref_i'}, rec)
        if options['use_pid']:
            rec = makeRecorders(perturb, {'pid':'_ref_i'}, rec)

    h.load_file('stdrun.hoc')
    h.celsius = 36
    h.dt = dt

    if options['use_pid']:
        h.tstop = Ttotal
        h.run()
        data['perturb'] = np.array(perturb_times)
    else:
        # ``Transient''
        h.tstop = Tdelay
        h.run()
        for k in range(ntrials):
            tperturb = stimulation_period/3 + rnd.uniform(0,stimulation_period/3)
            perturb.delay = h.t + tperturb
            data['perturb'][k] = perturb.delay
            tstop = h.t + stimulation_period
            h.continuerun(tstop)
            sys.stdout.write('\rt = %.4f s [%.0f%%]' % (h.t/1000, round(h.t/Ttotal*100)))
            sys.stdout.flush()
        sys.stdout.write('\n')

    if DEBUG:
        import pylab as p
        p.plot(rec['t'],rec['v'],'k')
        for tp in data['perturb']:
            p.plot([tp,tp],[-80,40],'r--')
        if neuron_mode == 'deterministic':
            p.plot(rec['t'],rec['i'],'g')
        if options['use_pid']:
            p.plot(rec['t'],rec['pid'],'m')
        p.show()

    if not options['use_pid']:
        baseline_amplitude = base.amp
    else:
        baseline_amplitude = 0

    savePRCData(options['output_file'], neuron_type+neuron_mode, soma.L, soma.diam, Rin, stimulation_period, Tdelay, h.dt,
                baseline_amplitude, perturb.amp, perturb.dur, noise_mu, noise_sigma, noise_tau,
                np.array(data['spks']), data['perturb'], options['use_pid'])

if __name__ == '__main__':
    main()
