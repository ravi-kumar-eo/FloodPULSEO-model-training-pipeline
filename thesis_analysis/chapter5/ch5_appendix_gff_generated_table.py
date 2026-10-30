"""Appendix A.7 table: IoU of GFF U-TAE, GFF MaxViT U-Net and all ten U-Net and ViT configurations at each of the 30 sites
whose labels GFF generated (Ravi, 30 Oct 2026: the dot plot of these sites, the former Figure A.11, could not be read).
Same layout as the table of the 21 expert-labelled sites: sites named and ordered as there, a dagger on the sites that
map the same floods as test mapping products, configurations as the mean of seeds 1-3.
Data: data/baseline/sites.csv (ch5_baseline_data.py). Writes data/baseline/persite_generated_rows.{tex,md}.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_appendix_gff_generated_table.py
"""
import baseline_common as bc

COLS = ['gff_utae', 'gff_maxvit'] + [f'{f}_{w}' for f in ('unet', 'vit') for w in ('t0', 't3', 't7', 't15', 't30')]


def main():
    d = bc.read('generated')
    order = bc.site_order(d)
    piv = d.pivot_table(index='site', columns='group', values='IoU')[COLS].loc[order.site]
    names = [n + ('†' if te else '') for n, te in zip(bc.site_names(order), order.our_test_event)]
    tex, md = [], []
    for name, (_, r) in zip(names, piv.iterrows()):
        v = [f'{x:.3f}' for x in r]
        tex.append(' & '.join([name] + v) + ' \\\\')
        md.append('| ' + ' | '.join([name] + v) + ' |')
    (bc.D / 'persite_generated_rows.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    (bc.D / 'persite_generated_rows.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(len(md), 'sites')


if __name__ == '__main__':
    main()
