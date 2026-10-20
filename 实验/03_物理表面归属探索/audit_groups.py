"""Audit depth connectivity against provisional mesh correspondences.

Labels below are constructed from reference distances and normal agreement;
they are not independent Gaussian ground truth or unbiased scene statistics.
"""
import json,argparse
from pathlib import Path
import numpy as np

def main(root):
    report={'warning':'diagnostic samples; provisional nearest-reference labels; not GT membership','scans':{}}
    for scan in sorted(root.glob('scan*')):
        sums={k:dict(two_supported=0,inseparable_depth_gaps=0,mixed_fixed_1pct=0,
                    fragmented_fixed_1pct=0,front_not_largest=0) for k in [1,2,4]}
        examples=[];rays=hits=multi=0
        for vp in sorted(scan.glob('*/表面归属.json')):
            cs=json.loads(vp.read_text());f=np.load(vp.parent/'射线支持.npz')
            for j,c in enumerate(cs):
                rays+=1;hits+=bool(c['reference_depths']);multi+=len(c['reference_depths'])>1
                if len(c['reference_depths'])<2 or f'ray_{j}_audit' not in f:continue
                r=f[f'ray_{j}_records'];a=f[f'ray_{j}_audit'];d=r[:,0];w=r[:,1]
                lab=a[:,0].astype(int);order=np.argsort(d,kind='stable');ds=d[order]
                groups=np.cumsum(np.r_[True,ds[1:]/ds[:-1]-1>.01])-1
                grp=np.empty(len(d),int);grp[order]=groups
                for k,s in sums.items():
                    # This angle gate is an audit check, not a proposed method.
                    keep=(a[:,1]<=k*.002)&(a[:,4]>.9)
                    labels=np.unique(lab[keep])
                    if len(labels)<2:continue
                    s['two_supported']+=1
                    gin=max(np.diff(np.sort(np.log(d[keep&(lab==v)]))).max(initial=0) for v in labels)
                    sub=np.flatnonzero(keep);so=sub[np.argsort(d[sub])]
                    diff=lab[so][1:]!=lab[so][:-1]
                    gout=np.diff(np.log(d[so]))[diff].min(initial=np.inf)
                    mixed=any(len(np.unique(lab[keep&(grp==g)]))>1 for g in np.unique(grp))
                    frag=any(len(np.unique(grp[keep&(lab==v)]))>1 for v in labels)
                    masses={int(v):float(w[keep&(lab==v)].sum()) for v in labels}
                    loses=(0 in masses and max(masses,key=masses.get)!=0)
                    s['inseparable_depth_gaps']+=int(gin>=gout)
                    s['mixed_fixed_1pct']+=int(mixed)
                    s['fragmented_fixed_1pct']+=int(frag)
                    s['front_not_largest']+=int(loses)
                    if k==1 and (mixed or gin>=gout or loses):
                        examples.append(dict(view=vp.parent.name,x=c['x'],y=c['y'],
                            source=c['sample_source'],gamma_in=float(gin),gamma_out=float(gout),
                            mixed=bool(mixed),masses=masses))
        report['scans'][scan.name]=dict(rays=rays,reference_hits=hits,multiple_intersections=multi,
                                      bands_normal_voxel=sums,examples=examples[:12])
    (root/'分组审计.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
