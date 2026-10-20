"""Same-camera raw/post mesh pictures; visualization does not alter evaluation."""
import sys
sys.dont_write_bytecode=True
import os
import json
from pathlib import Path
import numpy as np
import open3d as o3d
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'缓存绘图')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def compare(scan,methods):
    model=ROOT/'outtest'/f'scan{scan}'
    cams=json.loads((model/'cameras.json').read_text())
    cam=next(c for c in cams if c['img_name']=='0000')
    rotation=np.asarray(cam['rotation']);center=np.asarray(cam['position'])
    # CPU painter rendering avoids an EGL dependency; use all triangles.
    fig,axes=plt.subplots(len(methods),2,figsize=(10,len(methods)*3.5),squeeze=False)
    for row,method in enumerate(methods):
        for col,post in enumerate([False,True]):
            path=(model/('recon_post.ply' if post else 'recon.ply') if method=='expected'
                  else HERE/'结果'/f'scan{scan}'/method/('recon_post_layered.ply' if post else 'recon_layered.ply'))
            mesh=o3d.io.read_triangle_mesh(str(path));mesh.compute_triangle_normals()
            xyz=(np.asarray(mesh.vertices)-center)@rotation
            faces=np.asarray(mesh.triangles)
            valid=(xyz[faces,2]>0).all(axis=1)
            faces=faces[valid]
            order=np.argsort(xyz[faces,2].mean(axis=1))[::-1]
            # Intrinsics from camera metadata, rescaled only for display.
            uv=np.column_stack([xyz[:,0]/xyz[:,2]*cam['fx']/cam['width']+.5,
                                xyz[:,1]/xyz[:,2]*cam['fy']/cam['height']+.5])
            normals=np.asarray(mesh.triangle_normals)[valid]
            light=np.abs(normals@(-rotation[:,2]))*.7+.3
            colors=np.asarray(mesh.vertex_colors)
            colors=colors[faces].mean(axis=1) if len(colors) else np.ones((len(faces),3))*.65
            colors=np.clip(colors*light[:,None],0,1)
            ax=axes[row,col]
            ax.add_collection(PolyCollection(uv[faces][order],facecolors=colors[order],
                                            edgecolors='none',antialiaseds=False))
            ax.set_xlim(0,1);ax.set_ylim(1,0);ax.set_aspect(cam['height']/cam['width'])
            ax.axis('off');ax.set_title(f'scan{scan} {method} '+('post' if post else 'raw'),fontsize=10)
            print('PICTURE',scan,method,col,flush=True)
    fig.tight_layout();fig.savefig(HERE/f'scan{scan}_raw_post.png',dpi=140);plt.close(fig)

if __name__=='__main__':compare(int(sys.argv[1]),sys.argv[2:])
