"""IoU of the T0 networks by HydroBASINS Level-5 basin of the test patches (patch centre in the basin; basins of
rq2_basins.gpkg; a coastal patch whose centre falls outside the simplified outlines takes the nearest basin, median
1.3 km and at most 30 km away, 25 Oct 2026): (a) world map of the basins coloured by the micro IoU of ViT-T0 pooled over the basin's scored
cells, mean of seeds 1-3, with (b) Europe, (c) Southeast Asia and (d) the Middle East and Pakistan enlarged (Ravi, 25 Oct
2026). Writes basin_iou.csv and basin_activation_iou.csv, which fig_basin_iou_share.py plots.
    python fig_basin_iou.py <out dir> [unet_t0|vit_t0] [png name]     (default vit_t0, fig_basin_iou.png)
"""
import re, sys
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer
from scipy.stats import spearmanr

sys.path.insert(0, '/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/scripts')
import regime_common as rg
import rq1_common as rc
from rq1_common import ps

C5 = Path('/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/data/conditions')
SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ARCH = sys.argv[2] if len(sys.argv) > 2 else 'vit_t0'                 # network whose basin IoU colours the maps
PNG = sys.argv[3] if len(sys.argv) > 3 else 'fig_basin_iou.png'
NAME = {'unet_t0': 'U-Net-T0', 'vit_t0': 'ViT-T0'}[ARCH]

t = pd.read_csv(C5 / 'patches_test.csv.gz')
s = pd.read_csv(SPLIT, usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_miny', 'bounds_maxx', 'bounds_maxy', 'cohort'])
t = t.merge(s[s.cohort == 'test'], on=['folder_name', 'patch_number'], how='left', validate='1:1')
x, y = (t.bounds_minx + t.bounds_maxx) / 2, (t.bounds_miny + t.bounds_maxy) / 2
lon, lat = np.empty(len(t)), np.empty(len(t))
for crs, idx in t.groupby('crs').groups.items():
    lon[idx], lat[idx] = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(x[idx].values, y[idx].values)
B = gpd.read_file(C5 / 'rq2_basins.gpkg', layer='basins')
j = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326), B[['pfaf5', 'geometry']], how='left', predicate='within')
t['pfaf5'] = j[~j.index.duplicated()].pfaf5.values
out = t.pfaf5.isna().to_numpy()                                    # 2,328 coastal patches outside the outlines
P = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon[out], lat[out]), crs=4326).to_crs(6933)
near = gpd.sjoin_nearest(P, B[['pfaf5', 'geometry']].to_crs(6933), how='left', distance_col='d')
t.loc[out, 'pfaf5'] = near[~near.index.duplicated()].pfaf5.values
t['activation'] = t.folder_name.str.extract(r'^(EMSR\d+)')[0]
t = t.dropna(subset=['pfaf5'])

def iou(d, a):
    v = []
    for sd in (1, 2, 3):
        tp, fp, fn = d[f'{a}_s{sd}_tp'].sum(), d[f'{a}_s{sd}_fp'].sum(), d[f'{a}_s{sd}_fn'].sum()
        v.append(tp / (tp + fp + fn) if tp + fp + fn else np.nan)
    return np.nanmean(v)

rows = []
for k, d in t.groupby('pfaf5'):
    rows.append(dict(pfaf5=k, activations=d.activation.nunique(), products=d.folder_name.nunique(),
                     flood=d.flood.sum(), scored=d.scored.sum(), unet_t0=iou(d, 'unet_t0'), vit_t0=iou(d, 'vit_t0')))
G = pd.DataFrame(rows); G['share'] = G.flood / G.scored
G = G[G.flood > 0]
per = []
for (k, a), d in t.groupby(['pfaf5', 'activation']):
    if d.flood.sum() > 0:
        per.append(dict(pfaf5=k, activation=a, flood=d.flood.sum(), vit_t0=iou(d, 'vit_t0'), unet_t0=iou(d, 'unet_t0')))
Pa = pd.DataFrame(per); multi = Pa.groupby('pfaf5').filter(lambda g: len(g) >= 2)
G.to_csv(C5 / 'basin_iou.csv', index=False); Pa.to_csv(C5 / 'basin_activation_iou.csv', index=False)

rc.style()
import cartopy.crs as ccrs, cartopy.feature as cfeature
PC = ccrs.PlateCarree()
W = ps.TEXT_WIDTH_IN
bounds = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 1.0]
cmap = ListedColormap(['#DCE8F7', '#B5CFF0', '#7FAEE6', '#4A88D8', '#1F5FB8', '#0B3C8C']); norm = BoundaryNorm(bounds, cmap.N)
g = B.merge(G, on='pfaf5', how='inner')
nofl = B[~B.pfaf5.isin(G.pfaf5) & B.pfaf5.isin(t.pfaf5)]
# zoom areas: Europe and the two regions with the most flood-positive basins after it (Ravi, 25 Oct 2026)
ZOOMS = [('b', 'Europe', [-11, 31, 35, 71], ccrs.LambertAzimuthalEqualArea(10, 53)),
         ('c', 'Southeast Asia', [99, 112, 7, 26], ccrs.LambertAzimuthalEqualArea(105.5, 16.5)),
         ('d', 'Middle East and Pakistan', [40, 75, 8, 38], ccrs.LambertAzimuthalEqualArea(57.5, 23))]


def paint(ax, lw):
    for _, r in g.iterrows():
        ax.add_geometries([r.geometry], crs=PC, facecolor=cmap(norm(r[ARCH])), edgecolor='black', linewidth=lw, zorder=4)
    ax.add_geometries(nofl.geometry, crs=PC, facecolor='#BDBDBD', edgecolor='black', linewidth=lw, zorder=4)


def regional(fig, rect, extent, proj):
    ax = fig.add_axes(rect, projection=proj); ax.set_extent(extent, crs=PC)
    ne = cfeature.NaturalEarthFeature
    ax.add_feature(ne('physical', 'ocean', '50m', facecolor='#D6E4F0', edgecolor='none'), zorder=0)
    ax.add_feature(ne('physical', 'land', '50m', facecolor='#F3F1EC', edgecolor='none'), zorder=1)
    ax.add_feature(ne('physical', 'coastline', '50m', facecolor='none', edgecolor='#A9B6C3', linewidth=0.3), zorder=3)
    ax.add_feature(ne('cultural', 'admin_0_boundary_lines_land', '50m', facecolor='none', edgecolor='#CBC6BD', linewidth=0.3), zorder=3)
    ax.spines['geo'].set_linewidth(0.5)
    return ax


def aspect(extent, proj):                                           # height / width of a zoom area in its projection
    f = plt.figure(); a = f.add_subplot(projection=proj); a.set_extent(extent, crs=PC)
    x0, x1, y0, y1 = a.get_extent(crs=proj); plt.close(f); return (y1 - y0) / (x1 - x0)


gap, title, cbar = 0.10, 0.22, 0.55
asp = [aspect(e, p) for _, _, e, p in ZOOMS]
h2 = (W - gap * (len(ZOOMS) - 1)) / sum(1 / a for a in asp)        # common height of the zoom panels
h1 = W * 0.505                                                      # Robinson world map
H = title + h1 + title + h2 + cbar
fig = plt.figure(figsize=(W, H))
ax = rg.world_axes(fig, [0, (H - title - h1) / H, 1, h1 / H])
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='#F3F1EC', edgecolor='none'), zorder=1)
paint(ax, 0.15)
for k, name, (x0, x1, y0, y1), _ in ZOOMS:                          # outline of each zoom area on the world map
    xs = [x0] * 20 + list(np.linspace(x0, x1, 20)) + [x1] * 20 + list(np.linspace(x1, x0, 20))
    ys = list(np.linspace(y0, y1, 20)) + [y1] * 20 + list(np.linspace(y1, y0, 20)) + [y0] * 20
    ax.plot(xs, ys, transform=PC, color='#333333', lw=0.6, zorder=5)
    ax.text(x0, y1 + 1.5, k, transform=PC, fontsize=ps.FONT_SIZE - 1, ha='left', va='bottom', zorder=6)
rc.panel_title(ax, f'(a) {NAME} IoU by Level-5 basin of the test patches')
x = 0.0
for (k, name, ext, proj), a in zip(ZOOMS, asp):
    w = h2 / a
    zx = regional(fig, [x / W, cbar / H, w / W, h2 / H], ext, proj)
    paint(zx, 0.3)
    rc.panel_title(zx, f'({k}) {name}')
    x += w + gap
cax = fig.add_axes([0.2, 0.2 / H, 0.6, 0.09 / H])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', ticks=bounds)
cb.ax.tick_params(labelsize=ps.FONT_SIZE - 1.5); cb.set_label(f'IoU of {NAME} (grey: no labelled flood)', fontsize=ps.FONT_SIZE - 1.5)
fig.savefig(OUT / PNG, dpi=300, facecolor='white', bbox_inches='tight'); plt.close(fig)
rng = multi.groupby('pfaf5').vit_t0.agg(lambda v: v.max() - v.min())
print(f'flood-positive basins {len(G)}, with >=2 activations {multi.pfaf5.nunique()}; IoU range within a basin: median {rng.median():.3f}, max {rng.max():.3f}')
print('Spearman share~IoU:', {a: round(spearmanr(G.share, G[a]).correlation, 2) for a in ('unet_t0', 'vit_t0')})

# ---- numbers for the text of Section 5.2.1
import cartopy.io.shapereader as shp
cn = gpd.read_file(shp.natural_earth(resolution='50m', category='cultural', name='admin_0_countries'))[['NAME', 'geometry']]
gg = B.merge(G, on='pfaf5', how='inner')
cen = gpd.GeoDataFrame(gg[['pfaf5', 'regime', 'vit_t0', 'unet_t0', 'activations', 'share']], geometry=gg.to_crs(6933).centroid.to_crs(4326))
cen = gpd.sjoin_nearest(cen.to_crs(3857), cn.to_crs(3857), how='left')
cen = cen[~cen.index.duplicated()]
print(f'test patches {len(t):,} in {t.pfaf5.nunique()} basins; flood-positive {len(G)}; single activation {int((G.activations == 1).sum())}')
good = cen[cen.vit_t0 >= 0.3].sort_values('vit_t0', ascending=False)
print('ViT-T0 IoU >= 0.3:', len(good)); print(good[['pfaf5', 'NAME', 'regime', 'activations', 'vit_t0', 'unet_t0', 'share']].round(3).to_string(index=False))
print('basins below 0.1 (ViT-T0):', int((G.vit_t0 < 0.1).sum()), 'of', len(G))
print('max IoU of basins with flooded share < 1%:', G[G.share < 0.01][['unet_t0', 'vit_t0']].max().round(3).to_dict())
eu = cen[cen.NAME.isin(['France','Germany','Italy','Spain','Portugal','Greece','Norway','Sweden','Finland','Ireland','United Kingdom','Belgium','Netherlands','Slovenia','Croatia','Bosnia and Herz.','Serbia','Romania','Bulgaria','Austria','Poland','Czechia','Slovakia','Hungary','Albania','North Macedonia','Montenegro','Denmark','Switzerland','Ukraine','Moldova','Lithuania','Latvia','Estonia'])]
print('European flood-positive basins', len(eu), 'ViT-T0 IoU < 0.1:', int((eu.vit_t0 < 0.1).sum()))
print('multi-flood basins:', multi.pfaf5.nunique(), 'range median %.3f max %.3f' % (rng.median(), rng.max()), 'basin of max', rng.idxmax(),
      'n floods there', int((multi.pfaf5 == rng.idxmax()).sum()), 'min/max IoU', round(multi[multi.pfaf5 == rng.idxmax()].vit_t0.min(), 3), round(multi[multi.pfaf5 == rng.idxmax()].vit_t0.max(), 3))
