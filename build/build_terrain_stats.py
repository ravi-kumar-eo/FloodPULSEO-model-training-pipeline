"""Per-acquisition terrain summary, computed once so the split search can use it cheaply."""
import numpy as np, pandas as pd, rasterio
from pathlib import Path
PD=Path('/media/ravi/Jupyter/flood_vit/pipeline/data/patches/patches_tif')
p=pd.read_csv('metadata/splits/patches.csv',low_memory=False,keep_default_na=False,na_values=[''],
              usecols=['folder_name','patch_number'])
rows=[]
for ev,g in p.groupby('folder_name'):
    k=g.sample(min(10,len(g)),random_state=0)
    hand=[];elev=[]
    for r in k.itertuples():
        st=PD/ev.split('_')[0]/ev/f'patch_{int(r.patch_number):04d}_input_80m.tif'
        try:
            with rasterio.open(st) as f: a=np.transpose(f.read(),(1,2,0)).astype(np.float32)
        except Exception: continue
        a[a>=9999]=np.nan
        hand.append(np.nanmedian(a[...,4])); elev.append(np.nanmedian(a[...,0]))
    hand=[x for x in hand if x==x]; elev=[x for x in elev if x==x]
    if hand:
        rows.append(dict(folder_name=ev, hand_med=float(np.median(hand)),
                         flat_frac=float(np.mean(np.array(hand)<2.0)),
                         elev_med=float(np.median(elev)) if elev else np.nan))
T=pd.DataFrame(rows)
T.to_csv('metadata/acquisition_terrain.csv',index=False)
print(f'wrote metadata/acquisition_terrain.csv  ({len(T)} of {p.folder_name.nunique()} acquisitions)')
print(T[['hand_med','flat_frac','elev_med']].describe().round(2).to_string())
