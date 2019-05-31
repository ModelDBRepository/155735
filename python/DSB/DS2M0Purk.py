#!/usr/bin/env python

# De Schutter model conductances in De Schutter-Rapp cell
# This Python implementation is a translation of the HOC version
# converted from GENESIS by Jenny Davie, Arnd Roth, Volker Steuber,
# Erik De Schutter & Michael Hausser, dated 28.8.2004.

# Author: Daniele Linaro - October 2014

from neuron import h
import numpy as np

class DS2M0Purk:
    def __init__(self, model='PM9', g_granule=0.7e-3, g_stellate=[7000,1400]):
        if not model.upper() in ('PM9','PM10'):
            raise Exception("model must be either 'PM9' or 'PM10'")
        self.with_synapses = False
        self._model = model.upper()
        self._init_morphology()
        self._insert_passive_mechanisms()
        self._insert_active_mechanisms()
        self._soma.push() # make the soma the currently accessed section
        if not g_granule is None and not g_stellate is None:
            self._add_synapses(g_granule, g_stellate)

    def add_offset_current(self, amp=-0.01):
        # an offset current to prevent the cell from spiking
        self._offset = h.IClamp(self._soma(0.5))
        self._offset.delay = 0
        self._offset.amp = amp
        self._offset.dur = 1e8

    def _init_morphology(self):
        h('xopen("Purk2M0.nrn")')
        self._soma = h.soma
        self._dendrites = h.dend
        self._nsections = len(self._dendrites) + 1 # the number of sections in the dendrites, plus the soma
        self._main_dendrites = []
        for sec in h.md:
            self._main_dendrites.append(sec)
        self._thick_dendrites = []
        for sec in h.td:
            self._thick_dendrites.append(sec)
        self._thin_dendrites = []
        self._shell_depth = 0.2 # um
        self._main_dendrites_areas = []
        self._thick_dendrites_areas = []
        self._thin_dendrites_areas = []
        for i,sec in enumerate(self._dendrites):
            area = h.area(0.5,sec)
            if sec in self._main_dendrites:
                self._main_dendrites_areas.append(area)
            elif sec in self._thick_dendrites:
                self._thick_dendrites_areas.append(area)
            else:
                self._thin_dendrites.append(sec)
                self._thin_dendrites_areas.append(area)

    def _insert_passive_mechanisms(self):
        self._membrane_properties = {
            'capacitance': 1.64, # uF cm^-2
            'dendritic_resistance': 30000.,  # ohm cm^2
            'somatic_resistance': 10000., # ohm cm^2
            'axial_resistance': 250. # ohm cm
            }
        self._spine_density = 13
        self._spine_area = 1.33
        for sec in h.allsec():
            sec.insert('Leak')
            sec(0.5).Leak.el = -80.
            sec.Ra = self._membrane_properties['axial_resistance']
            sec.cm = 1.0 * self._membrane_properties['capacitance']
        for sec in self._dendrites:
            sec(0.5).Leak.gl = 1./self._membrane_properties['dendritic_resistance']
        for sec in self._main_dendrites:
            sec(0.5).Leak.gl = 1./self._membrane_properties['dendritic_resistance']
        self._soma(0.5).Leak.gl = 1./self._membrane_properties['somatic_resistance']
        for sec in self._dendrites:
            self._add_spines(sec)

    def _insert_active_mechanisms(self):
        ##### somatic ion channels
        soma = self._soma
        soma.insert('NaF')
        soma(0.5).NaF.gnabar = 7.5
        soma.insert('NaP')
        soma(0.5).NaP.gnabar = 0.001 # default is correct value for soma
        #soma.insert('CaP')
        #soma(0.5).CaP.gcabar = 0.0 # no P type Ca Ch in soma
        soma.insert('CaT')
        soma(0.5).CaT.gcabar = 0.0005 # default is correct value for soma and dend
        soma.insert('Kh1')
        soma(0.5).Kh1.gkbar = 0.0003 # default is correct value for soma, Ih uses only K ions here
        soma.insert('Kh2')
        soma(0.5).Kh2.gkbar = 0.0003 # default is correct value for soma, Ih uses only K ions here
        soma.insert('Kdr')
        soma(0.5).Kdr.gkbar = 0.6 # default is correct value for soma
        soma.insert('KMnew2')
        soma(0.5).KMnew2.gkbar = 0.00004 # default is correct value for soma
        soma.insert('KA')
        soma(0.5).KA.gkbar = 0.015 # default is correct value for soma
        #soma.insert('KC')
        #soma(0.5).KC.gkbar = 0.0 # no BK Ch in soma
        #soma.insert('K2')
        #soma(0.5).K2.gkbar = 0.0 # no KCa Ch in soma
        if self._model == 'PM10':
            soma(0.5).NaF.gnabar = 7.5
            soma(0.5).Kdr.gkbar = 0.9
            soma(0.5).KMnew2.gkbar = 0.00014
        soma.insert('cad')
        # GENESIS calculates shell area as (outer circle area - inner circle area)
        # our calcium diffusion had used depth * outer circumference
        soma(0.5).cad.depth = self._shell_depth - self._shell_depth**2/soma.diam
        soma.cai = 4e-5
        soma.cao = 2.4
        soma.eca = 12.5 * np.log(self._soma.cao/self._soma.cai)
        h.ion_style('ca_ion', 1, 1, 0, 0, 0, sec=soma) # was ('ca_ion', 1, 1, 0, 1, 0)
        soma.ena = 45 # mV Na+ reversal potential
        soma.ek = -85 # mV K+ reversal potential
        
        ##### dendritic ion channels
        # gmax as in DeSchutter 1994 PM9 'rest of dendrite'
        for sec in self._dendrites:
            #sec.insert('NaF')
            #sec(0.5).NaF.gnabar = 0.
            #sec.insert('NaP')
            #sec(0.5).NaP.gnabar = 0.
            sec.insert('CaP')
            sec(0.5).CaP.gcabar = 0.0045
            sec.insert('CaT')
            sec(0.5).CaT.gcabar = 0.0005
            #sec.insert('Kh1')
            #sec(0.5).Kh1.gkbar = 0.
            #sec.insert('Kh2')
            #sec(0.5).Kh2.gkbar = 0.
            #sec.insert('Kdr')
            #sec(0.5).Kdr.gkbar = 0.
            sec.insert('KMnew2')
            sec(0.5).KMnew2.gkbar = 0.000013
            #sec.insert('KA')
            #sec(0.5).KA.gkbar = 0.
            sec.insert('KC')
            sec(0.5).KC.gkbar = 0.08
            sec.insert('K2')
            sec(0.5).K2.gkbar = 0.00039
            sec.insert('cad')
            # GENESIS calculates shell area as (outer circle area - inner circle area)
            # our calcium diffusion had used depth * outer circumference
            sec(0.5).cad.depth = self._shell_depth - self._shell_depth**2/sec.diam
            sec.cai = 4e-5
            sec.cao = 2.4
            h.ion_style('ca_ion', 1, 1, 1, 1, 0, sec=sec)
            sec.ek = -85 # mV K+ reversal potential

        # add (the same) conductances to main dend, the spine necks and spine heads
        for sec in self._main_dendrites:
            sec.insert('Kdr')
            sec.insert('KA')
            if self._model == 'PM9':
                sec(0.5).CaP.gcabar = 0.0045
                sec(0.5).Kdr.gkbar = 0.06
                sec(0.5).KMnew2.gkbar = 0.00001
            else: # PM10
                sec(0.5).CaP.gcabar = 0.004
                sec(0.5).Kdr.gkbar = 0.09
                sec(0.5).KMnew2.gkbar = 0.00004
            sec(0.5).CaT.gcabar = 0.0005
            sec(0.5).KA.gkbar = 0.002
            sec(0.5).KC.gkbar = 0.08
            sec(0.5).K2.gkbar = 0.00039
            sec(0.5).cad.depth = self._shell_depth - self._shell_depth**2/sec.diam
            sec.cai = 4e-5
            sec.cao = 2.4
            h.ion_style('ca_ion', 1, 1, 1, 1, 0, sec=sec) # was ('ca_ion', 1, 1, 0, 1, 0)
            sec.ek = -85 # mV K+ reversal potential
            
    def _spine_correction(self, sec):
        max_diam = 0
        for seg in sec:
            if seg.diam > max_diam:
                max_diam = seg.diam
        if max_diam <= 3.17: # spine correction only for thin dendrites
            area = 0
            for seg in sec:
                area = area + seg.area()
            corr = (sec.L * self._spine_area * self._spine_density + area) / area
            if not sec in self._thin_dendrites:
                print('Adding %s to the thin dendrites.' % h.secname(sec=sec))
                self._thin_dendrites.append(sec)
                self._thin_dendrites_areas.append(area)
        else:
            corr = 1.
        return corr
    
    def _add_spines(self, sec):
        corr = self._spine_correction(sec)
        sec(0.5).gl_Leak = corr / self._membrane_properties['dendritic_resistance']
        sec.cm = corr * self._membrane_properties['capacitance']

    def _distance_from_soma(self, sec):
        return self._distance(self._soma, sec)

    def _distance(self, origin, end, x=0.5):
        h.distance(sec=origin)
        return h.distance(x, sec=end)

    def _add_synapses(self,
                      g_granule = 0.7e-3, # [uS/cm2]
                      g_stellate = [7000.,1400.] # [us/cm2]
                      ):
        self.with_synapses = True
        print('Adding synapses:')
        # we model only excitatory synapses coming from granule cells and
        # inhibitory synapses coming from stellate and basket cells.
        self._synapses = {'granule_cells': [], 'stellate_cells': [], 'basket_cells': []}
        # one granule cell synapse per spiny dendritic compartment
        for sec in self.spiny_dendrites:
            s,st,nc = self.insert_synapse(sec, [0.5,1.2], 0., g_granule)
            self._synapses['granule_cells'].append({'syn': [s], 'stim': [st], 'conn': [nc], 'spike_times': [],
                                                    'w': g_granule, 'sec': sec, 'E': 0.,  'tau': [0.5,1.2]})
        print('   total conductance of granule cell synapses on the spiny dendrites: %g uS.' % \
                  (g_granule*len(self.spiny_dendrites)))
        # one stellate cell synapse per spiny dendritic compartment
        g_tot = 0
        for sec in self.spiny_dendrites:
            g = h.area(0.5,sec=sec) * 1e-8 * g_stellate[0]
            g_tot += g
            s,st,nc = self.insert_synapse(sec, [0.9,26.5], -80., g)
            self._synapses['stellate_cells'].append({'syn': [s], 'stim': [st], 'conn': [nc], 'spike_times': [],
                                                     'w': g, 'sec': sec, 'E': -80., 'tau': [0.9,26.5]})
        print('   total conductance of stellate cell synapses on the spiny dendrites: %g uS.' % g_tot)
        # two stellate cell synapses per smooth dendritic compartment
        g_tot = 0
        for sec in self.smooth_dendrites:
            for i in range(2):
                g = h.area(0.5,sec=sec) * 1e-8 * g_stellate[1]
                g_tot += g
                s,st,nc = self.insert_synapse(sec, [0.9,26.5], -80., g)
                self._synapses['stellate_cells'].append({'syn': [s], 'stim': [st], 'conn': [nc], 'spike_times': [],
                                                         'w': g, 'sec': sec, 'E': -80., 'tau': [0.9,26.5]})
        print('   total conductance of stellate cell synapses on the smooth dendrites: %g uS.' % g_tot)
        g_basket = 100 # [uS/cm2]
        g_tot = 0
        # some on the soma
        while g_tot < 0.139:
            g = h.area(0.5,sec=self.soma) * 1e-8 * g_basket
            g_tot += g
            s,st,nc = self.insert_synapse(self.soma, [0.9,26.5], -80., g)
            self._synapses['basket_cells'].append({'syn': [s], 'stim': [st], 'conn': [nc], 'spike_times': [],
                                                   'w': g, 'sec': sec, 'E': -80., 'tau': [0.9,26.5]})
        somatic_contacts = len(self._synapses['basket_cells'])
        print('   total conductance of basket cell synapses on the soma: %g uS (%d contacts).' % \
                  (g_tot,somatic_contacts))
        # the remaining on the main dendrite
        g_basket = 50 # [uS/cm2]
        g_tot = 0
        while g_tot < 0.047:
            for sec in self.main_dendrites:
                g = h.area(0.5,sec=self.soma) * 1e-8 * g_basket
                g_tot += g
                s,st,nc = self.insert_synapse(sec, [0.9,26.5], -80., g)
                self._synapses['basket_cells'].append({'syn': [s], 'stim': [st], 'conn': [nc], 'spike_times': [],
                                                       'w': g, 'sec': sec, 'E': -80., 'tau': [0.9,26.5]})
                if g_tot >= 0.047:
                    break
        print('   total conductance of basket cell synapses on the main dendrite: %g uS (%d contacts).' % \
                  (g_tot,len(self._synapses['basket_cells'])-somatic_contacts))
        # print the total numbers
        print('   number of granule cells synapses: %d.' % len(self._synapses['granule_cells']))
        print('   number of stellate cells synapses: %d.' % len(self._synapses['stellate_cells']))
        print('   number of basket cells synapses: %d.' % len(self._synapses['basket_cells']))

    def insert_synapse(self, sec, tau, E, w=0.0001, delay=0.):
        # the synapse
        s = h.Exp2Syn(sec(0.5))
        s.tau1 = tau[0]
        s.tau2 = tau[1]
        s.e = E
        st,nc = self.make_stim_and_conn(s, w, delay)
        return s,st,nc        

    def make_stim_and_conn(self, syn, w, delay=0.):
        # the vecstim
        st = h.VecStim()
        # the netcon
        nc = h.NetCon(st, syn)
        nc.weight[0] = w
        nc.delay = delay
        return st,nc

    def set_presynaptic_spike_times(self, synapse, spike_times):
        synapse['spike_times'].append(h.Vector(spike_times))
        synapse['stim'][0].play(synapse['spike_times'][-1])

    def pick_section(self, group, areas):
        tmp = np.cumsum(areas)
        return group[np.where(tmp > np.random.uniform(0,tmp[-1]))[0][0]]

    @property
    def soma(self):
        return self._soma
    
    @property
    def dendrites(self):
        return self._dendrites

    @property
    def main_dendrites(self):
        return self._main_dendrites

    @property
    def thick_dendrites(self):
        return self._thick_dendrites

    @property
    def smooth_dendrites(self):
        return self._thick_dendrites

    @property
    def thin_dendrites(self):
        return self._thin_dendrites
    
    @property
    def spiny_dendrites(self):
        return self._thin_dendrites

    @property
    def synapses(self):
        return self._synapses

