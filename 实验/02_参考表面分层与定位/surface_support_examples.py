"""Small actual-image examples and source-backed support statistics after the batch."""
import os,json,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE/'结果'
os.environ['MPLCONFIGDIR']=str(HERE/'缓存绘图')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def main():
    frozen=json.loads((HERE/'固定参数.json').read_text());lam=frozen['lambda'];suffix='l001' if lam==.01 else 'l003'
    rows=[]
    for out in sorted(ROOT.glob('scan*'),key=lambda p:int(p.name[4:])):
        files=list(out.glob('支持统计_*.json'))
        if not files:continue
        stats=[]
        for p in files:
            data=json.loads(p.read_text())
            if data and suffix in data[0].get('lambdas',{}):stats=data;break
        if not stats:continue
        n=sum(s['pixels'] for s in stats);nm=sum(s['original_mask_pixels'] for s in stats)
        q=[s['lambdas'][suffix] for s in stats];used=sum(t['used'] for t in q)
        fs=[s['fixed_support'] for s in stats];fu=sum(t['used'] for t in fs)
        row=dict(scan=int(out.name[4:]),scope='探索' if int(out.name[4:]) in frozen['exploration_scans'] else '验证',
            views=len(stats),pixels=n,original_mask_pixels=nm,
            missing_reference_fraction=sum(s['missing_reference'] for s in stats)/n,
            missing_reference_in_mask_fraction=sum(s['missing_reference_in_mask'] for s in stats)/nm,
            fixed_empty_fraction=sum(s['fixed_empty'] for s in stats)/n,
            insufficient_support_fraction=sum(t['insufficient_support'] for t in q)/n,
            bad_records=sum(s['bad_records'] for s in stats),overflow_retries=sum(s['overflow_retries'] for s in stats),
            used_support_pixels=used,mean_interval_span_voxel=sum(t['span_voxel_sum'] for t in q)/used if used else None,
            mean_contribution_retained=sum(t['retained_contribution_sum'] for t in q)/used if used else None,
            mean_member_count=sum(t['member_count_sum'] for t in q)/used if used else None,
            mean_mean_median_distance_voxel=sum(t['mean_median_distance_voxel_sum'] for t in q)/used if used else None,
            fixed_mean_contribution_retained=sum(t['retained_contribution_sum'] for t in fs)/fu if fu else None,
            fixed_mean_member_span_voxel=sum(t['span_voxel_sum'] for t in fs)/fu if fu else None,
            full_depth_seconds=sum(s['seconds'] for s in stats))
        rows.append(row)
    with (HERE/'支持统计汇总.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Representative known exploration + heldout scenes, retain opposing reference changes.
    for scan in [24,105,40,118]:
        out=ROOT/f'scan{scan}';files=list(out.glob('0000_支持实例_*.json'))
        if not files:continue
        cases=json.loads(files[0].read_text());source=np.asarray(Image.open(out/'0000_图像.png'))
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        axes[0,0].imshow(source);axes[0,0].set_title(f'scan{scan} actual original image / view 0000')
        for i,c in enumerate(cases):
            x,y=c['x'],c['y'];axes[0,0].plot(x,y,'r+',ms=7);axes[0,0].text(x+5,y,str(i+1),c='red',fontsize=9)
            ax=axes.flat[i+1];dw=np.asarray(c['depth_weight'],float).reshape(-1,2);d,w=dw[:,0],dw[:,1]
            ref=c['reference'];scale=c['scale_z'];k=c['lambdas'].index(lam);q=np.array(c['statistics'][6+9*k:6+9*(k+1)])
            ax.scatter(d,w,s=12,c='.6',label='all original proposals')
            if ref is not None:
                ax.axvline(ref,c='#168b4b',lw=1.5,label='reference')
                ax.axvspan(ref-4*scale,ref+4*scale,color='#d5b05b',alpha=.15,label='fixed 4-voxel band')
            if ref is not None and q[0]==0 and q[3]>0:
                member=(d>=q[5])&(d<=q[6]);assert int(member.sum())==int(q[3])
                mu=float(d[member]@w[member]/w[member].sum());idx=np.argsort(d[member],kind='stable');ds=d[member][idx];ws=w[member][idx]
                md=float(ds[np.searchsorted(np.cumsum(ws),ws.sum()/2)])
                np.testing.assert_allclose(mu,c['values']['reference_support_mean_'+suffix],rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(md,c['values']['reference_support_wmedian_'+suffix],rtol=2e-5,atol=2e-6)
                ax.scatter(d[member],w[member],s=17,c='#2780aa',label='selected support')
                ax.axvspan(q[5],q[6],color='#2780aa',alpha=.12)
                ax.axvline(mu,c='#e17816',lw=1.2,label='same-support mean');ax.axvline(md,c='#a8458b',ls=':',lw=1.2,label='same-support weighted median')
                info=f'span {q[1]:.2f} vox; mass {q[2]*100:.1f}%; n {int(q[3])}'
            else:info='fallback to original expected'
            ax.axvline(c['expected'],c='black',ls='--',lw=.8,label='original expected')
            ax.set_title(f'{i+1} {c["label"]} ({x},{y})\n{info}',fontsize=9)
            ax.set_xlabel('Original camera z / model units');ax.set_ylabel('Original T * alpha');ax.grid(alpha=.15)
        for i in range(len(cases)+1,len(axes.flat)):axes.flat[i].axis('off')
        handles,labels=axes.flat[1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
        fig.suptitle(f'Reference-guided support, lambda={lam}; selected by reference differences, NOT GT accuracy examples',fontsize=11)
        fig.savefig(HERE/f'scan{scan}_支持与定位实例.png',dpi=150);plt.close(fig)
    print('STATISTICS_SCANS',len(rows));print('STATISTICS',rows)

if __name__=='__main__':main()
