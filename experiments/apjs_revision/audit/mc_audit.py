import numpy as np, pandas as pd
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
rng=np.random.default_rng(0)
def mc(m1,m2): return (m1*m2)**0.6/(m1+m2)**0.2
def report(name, mcd):
    i=rng.integers(0,len(mcd),200000); j=rng.integers(0,len(mcd),200000); k=i!=j
    r=np.abs(mcd[i[k]]-mcd[j[k]])/np.minimum(mcd[i[k]],mcd[j[k]])
    print(f'{name}: n={len(mcd)} median Mc_det {np.median(mcd):.1f}; unrelated pairs with |dMc|/Mc < 1%: {np.mean(r<0.01):.4f}, <3%: {np.mean(r<0.03):.4f}, <10%: {np.mean(r<0.10):.3f}')
S=pd.read_csv('/root/autodl-tmp/qkzhang/SIS_data_0228/lensed_source_samples.csv').iloc[:2500]
zgrid=np.linspace(0.001,12,20000); dlg=cosmo.luminosity_distance(zgrid).value
z=np.interp(S.luminosity_distance.values,dlg,zgrid)
report('original 0228 SIS', mc(S.mass_1_source.values,S.mass_2_source.values)*(1+z))
for cfg in ['ET','O4']:
    p=f'/root/autodl-tmp/apjs_rev/data_{cfg}/SIS_data_0228/source_samples.csv'
    T=pd.read_csv(p); cols=list(T.columns)
    if 'mass_1' in cols: m1,m2=T.mass_1.values,T.mass_2.values
    else:
        zz=T['zs'].values if 'zs' in cols else np.interp(T.luminosity_distance.values,dlg,zgrid)
        m1,m2=T.mass_1_source.values*(1+zz),T.mass_2_source.values*(1+zz)
    report(f'ler {cfg} 0228', mc(m1,m2))
