"""Explicit identity records for separate read-only feasibility probes."""
import hashlib
from pathlib import Path


MODEL_IDENTITIES = {
    'scan40_original7000': {
        'scene': 'scan40',
        'ply_sha256': '51f5f92bd6015ae98509b58b5aa293d110f4b4f13b8217916fdea0a819d84187',
        'source': 'G6 original scan40 PLY independently checked 2026-11-04',
        'expected_rows': None,
        'colmap_sha256': {},
    },
    'scan110_original7000': {
        'scene': 'scan110',
        'ply_sha256': '733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7',
        'source': 'G5 native capture faa9294 metadata.json; runtime verifies PLY and COLMAP hashes',
        'expected_rows': 142892,
        'colmap_sha256': {
            'cameras.bin': '9dd012bebe2dea0514af3f784dbb8fff23354fd7e9476ae6f9c668c40162d06b',
            'images.bin': '433bda84b16ae7961619442c5bc74c70ddbb090e38ff61eaf601ff3719bb2794',
        },
    },
}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def verify_inputs(input_ply, source_path, model_id):
    identity=MODEL_IDENTITIES[model_id]
    source=Path(source_path).resolve()
    if source.name!=identity['scene']:
        raise ValueError('source directory does not match explicit model scene')
    actual_ply=sha256(input_ply)
    if actual_ply!=identity['ply_sha256']:
        raise ValueError('PLY SHA mismatch for '+model_id)
    colmap={name:sha256(source/'sparse'/'0'/name) for name in ('cameras.bin','images.bin')}
    if any(colmap[name]!=expected for name,expected in identity['colmap_sha256'].items()):
        raise ValueError('COLMAP SHA mismatch for '+model_id)
    return dict(model_id=model_id,scene=identity['scene'],input_ply=str(Path(input_ply).resolve()),
                input_sha256=actual_ply,source_path=str(source),colmap_sha256=colmap,
                expected_rows=identity['expected_rows'],identity_source=identity['source'])
