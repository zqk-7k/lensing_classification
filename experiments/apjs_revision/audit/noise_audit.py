import numpy as np, pandas as pd
R='/root/autodl-tmp/qkzhang'
W=8192
def noise(path_d, path_h, rows):
    d=np.load(path_d,mmap_mode='r'); h=np.load(path_h,mmap_mode='r')
    return np.stack([np.asarray(d[r,-W:]-h[r,-W:]) for r in rows])
def corr(a,b): a=a-a.mean(1,keepdims=True); b=b-b.mean(1,keepdims=True); return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
rows=list(range(0,2500,250))
N={}
for v in ['0222','0228']:
  for f in ['SIS','PM']:
    for k in [1,2]:
      N[(v,f,k)]=noise(f'{R}/{f}_data_{v}/{f}_data_strain_{k}.npy',f'{R}/{f}_data_{v}/{f}_h_strain_{k}.npy',rows)
  N[(v,'U',0)]=noise(f'{R}/Unlensed_data_{v}/unlensed_data_strain.npy',f'{R}/Unlensed_data_{v}/unlensed_h_strain.npy',rows)
  N[(v,'T',1)]=noise(f'{R}/SIS_data_{v}/test_SIS_data_strain_1.npy',f'{R}/SIS_data_{v}/test_SIS_h_strain_1.npy',rows)
print('noise std (row0):',{str(k):round(float(x[0].std()),3) for k,x in N.items()})
keys=list(N)
print('max |corr| between same-row noise across datasets (10 rows each):')
for i,a in enumerate(keys):
  for b in keys[i+1:]:
    c=np.abs(corr(N[a],N[b])).max()
    flag='  <-- IDENTICAL/CORRELATED' if c>0.05 else ''
    if c>0.05 or (a[0]!=b[0] and a[1]==b[1] and a[2]==b[2]) or (a[0]==b[0] and a[1]==b[1]): print(f'  {a} vs {b}: {c:.4f}{flag}')
# shifted-row check: same dataset, row i vs row i+1
for k in [('0228','SIS',1),('0228','U',0)]:
  x=N[k]; print('  adjacent-row within',k, np.abs(corr(x[:-1],x[1:])).max())
for v in ['0222','0228']:
  s=pd.read_csv(f'{R}/SIS_data_{v}/source_samples.csv')
  print(v,'m1_src',s.mass_1_source.min().round(1),s.mass_1_source.max().round(1),'frac m2>m1',(s.mass_2_source>s.mass_1_source).mean().round(3),'dL max',s.luminosity_distance.max().round(0))
