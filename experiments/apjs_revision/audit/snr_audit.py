import numpy as np, pandas as pd, sys, os
R='/root/autodl-tmp/qkzhang'
def q(a): a=np.asarray(a,float).ravel(); return "med %.1f  p10 %.1f  p90 %.1f  min %.1f  max %.1f  frac<8 %.3f  frac>50 %.3f n=%d"%(np.median(a),*np.percentile(a,[10,90]),a.min(),a.max(),(a<8).mean(),(a>50).mean(),a.size)
for v in ['0222','0228']:
  for fam in ['SIS','PM']:
    d=f'{R}/{fam}_data_{v}'
    s1=np.load(f'{d}/{fam}_optimal_SNR_1.npy'); s2=np.load(f'{d}/{fam}_optimal_SNR_2.npy')
    print(v,fam,'shape',s1.shape, np.load(f'{d}/{fam}_data_strain_1.npy',mmap_mode='r').shape)
    print('  image1', q(s1)); print('  image2', q(s2)); print('  rho_min', q(np.minimum(s1,s2)))
    t1=np.load(f'{d}/test_{fam}_optimal_SNR_1.npy'); print('  test1 ', q(t1), np.load(f'{d}/test_{fam}_data_strain_1.npy',mmap_mode='r').shape)
    L=pd.read_csv(f'{d}/lens.csv'); P=pd.read_csv(f'{d}/lens_params.csv')
    print('  lens cols',list(L.columns),list(P.columns))
    print('  mu0',q(L.iloc[:,0]),'\n  |mu1|',q(np.abs(L.iloc[:,1])))
    print('  t_d[s] med %.3g p10 %.3g p90 %.3g'%(np.median(L['t_d']),*np.percentile(L['t_d'],[10,90])))
    if 'y' in P: print('  y min %.3f max %.3f'%(P.y.min(),P.y.max()))
    S=pd.read_csv(f'{d}/lensed_source_samples.csv'); print('  src cols',list(S.columns)[:6], len(S))
  u=f'{R}/Unlensed_data_{v}'; su=np.load(f'{u}/unlensed_optimal_SNR.npy'); print(v,'unlensed',q(su), np.load(f'{u}/unlensed_data_strain.npy',mmap_mode='r').shape)
