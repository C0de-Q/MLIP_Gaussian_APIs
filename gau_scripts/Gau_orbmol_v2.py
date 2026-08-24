#!/usr/bin/env python3
"""ORB-Mol v2 ONIOM External 接口；server 优先，可切换为 CPU 计算

工作流（参考 RunG16GPU_orb.sh）:
    1. RunG16GPU_orb.sh 启动 orbmol_v2 持久化 server:
         METHOD=orbmol_v2, MLIP_SERVER_PORT=15556, Apptainer + orbmol2 环境
       server 为 bin/mlip_server.py，请求协议两行: filein\\nfileout\\n
    2. 本脚本默认通过 TCP relay（bin/mlip_relay.py 的 client_mode）
       把 Gaussian External 请求转发给 server，模型只加载一次
    3. 设置 MLIP_FORCE_CPU=1 时切换为本地 CPU 计算（不复用 server，
       orbmol_v2 加载到 CPU，复用 calculators.py 的 METHODS['orbmol_v2']）

环境变量:
    MLIP_SERVER_PORT    server 端口（由 RunG16GPU_orb.sh 设置，默认 15556）
    MLIP_FORCE_CPU      设为 1 时强制本地 CPU 计算，不连接 server
"""
import os
import sys

sys.path.insert(0, '/share/home/CodeQ/Benchmark')

METHOD = 'orbmol_v2'


def cpu_compute(filein, fileout):
    """本地 CPU 计算（不连接 server）。"""
    from ase.io import read
    from bin.calculators import METHODS
    from bin.constants import EV_HA, FORCE_UNIT_CONST
    from bin.gaussian_external import (
        get_external_coord,
        write_external_output,
        write_xyz,
    )

    ele, coordlist, atom_charges, deriva, charge, spin = get_external_coord(filein)
    xyz_file = write_xyz(ele, coordlist, charge, spin)  # 唯一文件名，防并发覆盖
    atoms = read(xyz_file)

    # device='cpu' 强制 CPU；compute_orbmol_v2 会把力写入 atoms.arrays['forces']
    ene_ev = METHODS[METHOD](atoms, charge, spin, device='cuda:0')

    grad = None
    if deriva == 1:
        forces = atoms.arrays.get('forces')
        if forces is not None:
            grad = [[-f[0] / FORCE_UNIT_CONST,
                     -f[1] / FORCE_UNIT_CONST,
                     -f[2] / FORCE_UNIT_CONST] for f in forces]

    write_external_output(fileout, ene_ev / EV_HA, grad, len(ele), deriva)


if __name__ == '__main__':
    filein, fileout = sys.argv[2], sys.argv[3]

    if os.environ.get('MLIP_FORCE_CPU', '') not in ('', '0'):
        print('[Gau_orbmol_v2] MLIP_FORCE_CPU=1，使用本地 CPU 计算', file=sys.stderr)
        cpu_compute(filein, fileout)
    else:
        port = os.environ.get('MLIP_SERVER_PORT')
        if not port:
            raise SystemExit(
                '[Gau_orbmol_v2] 未设置 MLIP_SERVER_PORT：请先用 '
                'RunG16GPU_orb.sh 启动 orbmol_v2 server，'
                '或设置 MLIP_FORCE_CPU=1 切换为 CPU 计算'
            )
        from bin.mlip_relay import client_mode

        client_mode(filein, fileout, int(port))
