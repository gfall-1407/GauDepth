# RaDe-GS: Rasterizing Depth in Gaussian Splatting

### RaDe-GS: Rasterizing Depth in Gaussian Splatting
Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, Ping Tan

[Project page](https://baowenz.github.io/radegs/)
![Teaser image](assets/teaser.png)
## News!
- **The paper has been accepted for publication in ACM Transactions on Graphics (TOG)!**
- **We incorporate the multi-view regularization from PGSR.**

## 1. Installation
### Clone this repository.
```
git clone https://github.com/HKUST-SAIL/RaDe-GS.git --recursive
```

### Create an environment
```
conda create -n radegs python=3.12
conda activate radegs
```

### Install pytorch and other dependencies.
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### Install submodules
```
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn/ --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

# tetra-nerf for Marching Tetrahedra
conda install conda-forge::cgal
pip install submodules/tetra_triangulation/ --no-build-isolation
```
---

## 2. Data Preparation

### DTU
We train on the **preprocessed DTU dataset** from **2DGS**:  
https://surfsplatting.github.io/

For geometry evaluation, download the **official DTU point clouds** and place them under:
```text
dtu_eval/Offical_DTU_Dataset
```
DTU dataset page: https://roboimagedata.compute.dtu.dk/?page_id=36

### Tanks and Temples (TnT)
Please follow [PGSR](https://github.com/zju3dv/PGSR) to preprocess the TnT dataset. For evaluation, download the **GT point clouds**, **camera poses**, **alignments**, and **crop files** from:  
https://www.tanksandtemples.org/download/

Expected structure:
```text
GT_TNT_dataset/
  Barn/
    images/
      000001.jpg
      000002.jpg
      ...
    sparse/
      0/
        ...
    Barn.json
    Barn.ply
    Barn_COLMAP_SfM.log
    Barn_trans.txt
  Caterpillar/
    ...
```

### Objaverse
For depth and normal evaluation, we render multi-view images from Objaverse assets and export the corresponding ground-truth depth maps and surface normal maps. The rendered dataset can be downloaded from [this link](https://huggingface.co/datasets/BaowenZ/objaverse-multiview-renderings).

---

## 3. Training & Evaluation

Below are example commands for training, mesh extraction, rendering, and evaluation.

### DTU

```bash
# Training
python train.py -s <path_to_dtu> -m <output_dir> -r 2 --use_decoupled_appearance 3

# Mesh extraction
python mesh_extract.py -m <output_dir>

# Evaluation
python evaluate_dtu_mesh.py -m <output_dir>
```

### Tanks and Temples (TnT)

```bash
# Training
python train.py -s <path_to_preprocessed_tnt> -m <output_dir> -r 2 --use_decoupled_appearance 3

# Mesh extraction
python mesh_extract_tnt.py -m <output_dir>

# Evaluation
python eval_tnt/run.py \
  --dataset-dir <path_to_gt_tnt> \
  --traj-path <path_to_COLMAP_SfM.log> \
  --ply-path <output_dir>/recon_post.ply \
  --out-dir <output_dir>/mesh
```

### Novel View Synthesis

```bash
# Training
python train.py -s <path_to_dataset> -m <output_dir> --eval

# Rendering
python render.py -m <output_dir>

# Evaluation
python metrics.py -m <output_dir>
```

### Objaverse
```bash
# Training
python train.py -s <path_to_dataset> -m <output_dir> --eval

# Evaluation
python geometry_metric.py -m <output_dir>
```

### Per-view depth point clouds

After training a scene, the following command renders all training views and
exports both the mean (`expected_depth`) and median depth for every view:

```bash
python render_depth_pointcloud.py -m <output_dir>
```

The default output directory is
`<output_dir>/depth_pointcloud/iteration_<loaded_iteration>/`. It contains
lossless raw depth maps (`depth_maps/<mean|median>/*_<mean|median>_depth.npy`),
colorized preview images
(`depth_visualization/<mean|median>/*_<mean|median>_depth_visualization.png`),
and one colored PLY point cloud per view
(`point_clouds/<mean|median>/*_<mean|median>_depth_point_cloud.ply`). The PLY
coordinates are in the corresponding camera coordinate system, not the world
coordinate system. Camera intrinsics and file mappings are recorded in
`metadata.json`.

---

# 4. Viewer
Current viewer in this repository is very similar to the original Gaussian Splatting viewer, with minor updates for newer library versions and for loading 3D Gaussian models.
You can build and use it in the same way as [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).


# 5. Acknowledgements

This project is built upon the original implementation of 3D Gaussian Splatting (3DGS):
https://github.com/graphdeco-inria/gaussian-splatting.

We integrate components and ideas from several recent works, including the filtering strategy from [Mip-Splatting](https://github.com/autonomousvision/mip-splatting), and regularization terms from [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and [PGSR](https://github.com/zju3dv/PGSR).

We also incorporate the densification strategy proposed in [GOF](https://github.com/autonomousvision/gaussian-opacity-fields), and adopt decoupled appearance modeling practices inspired by 3DGS, GOF, and PGSR.

For geometric evaluation, we use the DTU and Tanks and Temples evaluation toolboxes from DTUeval-python
(https://github.com/jzhangbs/DTUeval-python) and the TanksAndTemples Python evaluation scripts
(https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation), respectively.

We thank the authors of these projects for making their code publicly available.
