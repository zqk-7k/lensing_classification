import numpy as np, pandas as pd, bilby, logging
bilby.core.utils.logger.setLevel(logging.ERROR)
S=pd.read_csv('/root/autodl-tmp/apjs_rev/data_ET/SIS_data_0228/lensed_source_samples.csv').iloc[:2500].sample(40,random_state=1)
out=[]
for fmin,dur in [(20,24),(5,128)]:
    wfg=bilby.gw.WaveformGenerator(duration=dur,sampling_frequency=2048,frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant='IMRPhenomXPHM',reference_frequency=10.,minimum_frequency=fmin))
    ifo=bilby.gw.detector.InterferometerList(['ET'])[0]; ifo.minimum_frequency=fmin
    ifo.set_strain_data_from_zero_noise(sampling_frequency=2048,duration=dur,start_time=0)
    s=[]
    for _,r in S.iterrows():
        p=r.to_dict(); p['geocent_time']=dur-2
        h=wfg.frequency_domain_strain(p); s.append(np.sqrt(abs(np.real(ifo.optimal_snr_squared(ifo.get_detector_response(h,p))))))
    out.append(np.array(s))
g=out[1]/out[0]; print('ET1 arm S/N gain from 5 Hz vs 20 Hz: median %.2f p10 %.2f p90 %.2f'%(np.median(g),*np.percentile(g,[10,90])))
