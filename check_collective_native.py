"""GPU analytic smoke test after review/deployment; does not use DTU/GT/model files."""
import math
from argparse import Namespace
import numpy as np
import torch
from scene import GaussianModel
from scene.cameras import Camera
from collective_readout import render_collective_half


def main():
    model=GaussianModel(3)
    model._xyz=torch.tensor([[0.,0.,2.]],device='cuda')
    model._scaling=torch.full((1,3),math.log(.2),device='cuda')
    model._rotation=torch.tensor([[1.,0.,0.,0.]],device='cuda')
    model._opacity=torch.tensor([[math.log(.8/.2)]],device='cuda')
    model._features_dc=torch.zeros(1,1,3,device='cuda')
    model._features_rest=torch.zeros(1,15,3,device='cuda')
    model.filter_3D=torch.zeros(1,1,device='cuda')
    camera=Camera(colmap_id=0,R=np.eye(3),T=np.zeros(3),FoVx=math.pi/3,FoVy=math.pi/3,
        image=torch.zeros(3,9,9),gt_alpha_mask=None,image_name='synthetic',uid=0)
    with torch.no_grad():
        z,status,audit=render_collective_half(camera,model,Namespace(debug=False,
            compute_cov3D_python=False,convert_SHs_python=False),torch.ones(3,device='cuda'),0.,batch_pixels=81)
    expected=2-.2*math.sqrt(2*math.log(.8/.5))
    measured=float(z[0,4,4])
    if int(status[4,4])!=1 or abs(measured-expected)>5e-4:
        raise RuntimeError(f'native/reference mismatch: {measured} vs {expected}')
    print(dict(center_depth=measured,analytic_depth=expected,absolute_difference=abs(measured-expected),audit=audit))


if __name__=='__main__':main()
