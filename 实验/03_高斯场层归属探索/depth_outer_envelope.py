"""Exact rational outer envelope over six continuously covered direction faces."""
from collections import deque
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import json,subprocess,time

ZERO=F(0); ONE=F(1); DELTA=F(1,100); THRESHOLD=F(4,5); SCALE=1<<64; BUDGET=1365

def sqrt_bounds(x):
    assert x>=0
    k=isqrt((x.numerator*SCALE*SCALE)//x.denominator)
    low=F(k,SCALE); high=low if low*low==x else F(k+1,SCALE)
    assert low*low<=x<=high*high
    return low,high

def dot_range(box,vec):
    terms=[(min(a*v,b*v),max(a*v,b*v)) for (a,b),v in zip(box,vec)]
    return sum((a for a,b in terms),ZERO),sum((b for a,b in terms),ZERO)

def merge(intervals):
    intervals=sorted(intervals,key=lambda p:p[0]);out=[]
    for a,b in intervals:
        if out and (out[-1][1] is None or a<=out[-1][1]):
            if out[-1][1] is not None:out[-1]=(out[-1][0],None if b is None else max(out[-1][1],b))
        else:out.append((a,b))
    return out

def mass_union(intervals,weights):
    # Every finite endpoint is closed, including singleton intervals. None is +infinity.
    events=sorted(set(x for a,b in intervals for x in (a,b) if x is not None))
    pieces=[]
    for i,x in enumerate(events):
        mass=sum((w for (a,b),w in zip(intervals,weights) if a<=x and (b is None or x<=b)),ZERO)
        if mass>=THRESHOLD:pieces.append((x,x))
        if i+1<len(events):
            y=events[i+1];mid=(x+y)/2
            m=sum((w for (a,b),w in zip(intervals,weights) if a<=mid and (b is None or mid<=b)),ZERO)
            if m>=THRESHOLD:pieces.append((x,y))
        else:
            m=sum((w for (a,b),w in zip(intervals,weights) if b is None and a<=x),ZERO)
            if m>=THRESHOLD:pieces.append((x,None))
    return merge(pieces)

def evaluate(box,points,weights,c,r):
    norm2=sum((max(abs(a),abs(b))**2 for a,b in box),ZERO);_,normhi=sqrt_bounds(norm2); margin=DELTA*normhi
    offsets=[]
    for point in points:
        a,b=dot_range(box,point);offsets.append((a-margin,b+margin))
    if not mass_union(offsets,weights):return []
    dl,dh=dot_range(box,r)
    if dl<=0<=dh:return [(ZERO,None)]
    intervals=[];active=[]
    for point,w in zip(points,weights):
        al,ah=dot_range(box,[v-u for v,u in zip(point,c)]);al-=margin;ah+=margin
        vals=[a/d for a in (al,ah) for d in (dl,dh)];lo,hi=min(vals),max(vals)
        if hi>0:
            intervals.append((max(ZERO,lo),hi));active.append(w)
    return mass_union(intervals,active) if intervals else []

def render(intervals):return [dict(lo=str(a),lo_open=(a==0),hi='inf' if b is None else str(b),hi_open=(b is None)) for a,b in intervals if b is None or b>0]
def contains(intervals,z):return any(a<=z and (b is None or z<=b) for a,b in intervals)

def solve(points,weights,c,r):
    roots=[]
    for axis in range(3):
        for sign in (-1,1):
            box=[(-ONE,ONE) for _ in range(3)];box[axis]=(F(sign),F(sign));roots.append(tuple(box))
    leaves={};queue=deque();processed=0;pruned=0;counter=0
    for box in roots:
        u=evaluate(box,points,weights,c,r);processed+=1
        if u:leaves[counter]=(box,u);queue.append(counter);counter+=1
        else:pruned+=1
    while queue and processed+2<=BUDGET:
        key=queue.popleft();box,_=leaves.pop(key)
        axis=max(range(3),key=lambda j:(box[j][1]-box[j][0],-j));lo,hi=box[axis];mid=(lo+hi)/2
        for bounds in ((lo,mid),(mid,hi)):
            child=list(box);child[axis]=bounds;child=tuple(child);u=evaluate(child,points,weights,c,r);processed+=1
            if u:leaves[counter]=(child,u);queue.append(counter);counter+=1
            else:pruned+=1
    envelope=merge([piece for _,u in leaves.values() for piece in u]);return envelope,dict(processed=processed,pruned=pruned,retained_leaves=len(leaves),queued=len(queue),unbounded_unexcluded=any(b is None for a,b in envelope))

def main():
    t=time.monotonic();out=Path(__file__).parent/'结果/20261029_深度外包络/envelope.json';out.parent.mkdir(parents=True,exist_ok=True)
    pts1=[(x,y,F(2)) for x in (F(-1),F(-1,2),F(1,2),F(1)) for y in (F(-1),ZERO,ONE)]
    cases=[('single_plane',pts1,[F(1,12)]*12,(ZERO,ZERO,ZERO),(ZERO,ZERO,ONE),[F(2)]),('collinear',[(F(-1),ZERO,ZERO),(ONE,ZERO,ZERO)],[F(1,2)]*2,(ZERO,ZERO,ZERO),(ZERO,ZERO,ONE),[F(1,10),ONE,F(2),F(10)]),('two_planes',[(F(-1),ZERO,ZERO),(ONE,ZERO,ZERO),(ZERO,ONE,ZERO),(ZERO,ZERO,ONE)],[F(3,10),F(3,10),F(1,5),F(1,5)],(ZERO,F(-1),F(-2)),(ZERO,ONE,ONE),[ONE,F(2)])]
    results=[]
    for name,points,weights,c,r,witness in cases:
        assert sum(weights)==1
        env,counts=solve(points,weights,c,r);kept=[contains(env,z) for z in witness]
        results.append(dict(case=name,counts=counts,outer_envelope=render(env),witnesses=[dict(z=str(z),retained=ok) for z,ok in zip(witness,kept)],single_plane_budget_failure=(name=='single_plane' and (not env or any(b is None for a,b in env)))))
    bad=any(not w['retained'] for case in results for w in case['witnesses'])
    report=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),delta='1/100',eta='1/5',sqrt_grid='1/18446744073709551616',budget_per_case=BUDGET,case2_proof='Plane y=0 contains both points with weight 1 and the entire positive z ray; every z>0 is feasible.',cases=results,algorithm_error=bad,elapsed_seconds=time.monotonic()-t,limitations='Outer envelope, not inner witness or proof of unboundedness. Fixed synthetic cases only; no real data, efficiency or method validation.')
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report,ensure_ascii=False));raise SystemExit(1 if bad else 0)
if __name__=='__main__':main()
