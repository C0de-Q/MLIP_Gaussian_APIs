#!/usr/bin/env python3
"""
mlip_server.py — 持久化 MLIP GPU server for Gaussian ONIOM 计算

在 Apptainer 容器内运行，启动时加载模型一次，通过 TCP 接收推理请求。
避免每步优化重复加载模型，大幅加速 ONIOM 几何优化。

支持的模型（通过 METHOD 环境变量选择）:
    METHOD=mace_omol   → MACE-OMol extra-large
    METHOD=mace_off24  → MACE-OFF24 medium (默认)
    METHOD=mace_polar  → MACE-POLAR-1-M
    METHOD=orbmol_v2   → ORB-Mol v2

环境变量:
    MLIP_SERVER_PORT   — 监听端口 (默认 15556)
    MLIP_SERVER_HOST   — 监听地址 (默认 127.0.0.1)
    METHOD             — 模型选择
    MLIP_XYZ_OUT       — 退出时保存最终结构的 XYZ 文件路径

合并自: server_MLIP/Mace_Omol_server.py, Mace_off_server.py, mlip_server_v1.py
"""

import socket
import sys
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from ase import Atoms
from ase.io import write as ase_write

# ═══════════════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════════════

EV_HA = 27.211386245
FORCE_UNIT_CONST = 51.42208619
BOHR_TO_ANGSTROM = 0.52917721067

# 元素符号 (1-indexed)
_Z_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe',
    27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se',
    35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
    43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce',
    59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy',
    67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
    75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
    83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn',
}

# 模型配置: {METHOD: {"model_path": ..., "calculator": ..., "needs_info": bool}}
_MODEL_BASE = '/share/home/CodeQ/.mlatom/models'
MODEL_CONFIGS = {
    'mace_omol': {
        'model_path': f'{_MODEL_BASE}/MACE-omol-0-extra-large-1024.model',
        'from': 'mace.calculators',
        'import': 'mace_omol',
        'needs_info': True,
    },
    'mace_off24': {
        'model_path': f'{_MODEL_BASE}/MACE-OFF24_medium.model',
        'from': 'mace.calculators',
        'import': 'mace_off',
        'needs_info': True,
    },
    'mace_polar': {
        'model_path': f'{_MODEL_BASE}/MACE-POLAR-1-M.model',
        'from': 'mace.calculators',
        'import': 'mace_polar',
        'needs_info': True,
    },
    'orbmol_v2': {
        'type': 'orbmol_v2',
        'from': 'orb_models.forcefield.pretrained',
        'import': 'orbmol_v2',
        'needs_info': True,
    },
}



# ═══════════════════════════════════════════════════════════════════════
#  Gaussian External 接口解析
# ═══════════════════════════════════════════════════════════════════════

def parse_gau_external(filein):
    """解析 Gaussian External 接口文件。

    Returns:
        eles: np.ndarray (natoms,) int32
        coords: np.ndarray (natoms, 3) float64 (Å)
        deriva: int
        charge: int
        spin: int
    """
    with open(filein, 'r') as f:
        natoms, deriva, charge, spin = [int(s) for s in f.readline().split()]
        eles = np.empty(natoms, dtype=np.int32)
        coords = np.empty((natoms, 3), dtype=np.float64)
        for i in range(natoms):
            parts = f.readline().split()
            eles[i] = int(parts[0])
            coords[i, 0] = float(parts[1]) * BOHR_TO_ANGSTROM
            coords[i, 1] = float(parts[2]) * BOHR_TO_ANGSTROM
            coords[i, 2] = float(parts[3]) * BOHR_TO_ANGSTROM
    return eles, coords, deriva, charge, spin


def write_gau_output(fileout, energy_ev, forces_ev_ang, deriva, natoms):
    """写入 Gaussian External 期望的输出格式。

    Args:
        energy_ev:     能量 (eV)
        forces_ev_ang: 力 (eV/Å) 或 None
        deriva:        0 或 1
        natoms:        原子数
    """
    ene_ha = energy_ev / EV_HA
    with open(fileout, 'w') as f:
        f.write(f'{ene_ha:20.12E}{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}\n')
        if deriva == 0 or forces_ev_ang is None:
            for _ in range(natoms):
                f.write(f'{0.0:20.12E}{0.0:20.12E}{0.0:20.12E}\n')
        else:
            conv = -1.0 / FORCE_UNIT_CONST
            for i in range(natoms):
                f.write(f'{forces_ev_ang[i, 0] * conv:20.12E}'
                        f'{forces_ev_ang[i, 1] * conv:20.12E}'
                        f'{forces_ev_ang[i, 2] * conv:20.12E}\n')


# ═══════════════════════════════════════════════════════════════════════
#  MLIP Server
# ═══════════════════════════════════════════════════════════════════════

class MLIPServer:
    """持久化 MLIP 推理 server。"""

    def __init__(self, method='mace_off24', host='127.0.0.1', port=15556,
                 xyz_out=None):
        self.method = method
        self.host = host
        self.port = port
        self.xyz_out = xyz_out
        self.model_lock = threading.Lock()
        self.calculator = None
        self._is_orbmol_v2 = False
        self._model = None
        self._atoms_adapter = None
        # 记录最后一次计算结果，用于退出时保存结构
        self.last_atoms = None
        self.last_energy = None
        self.last_forces = None
        self._shutdown_flag = threading.Event()
        self._load_model()
        #self._warmup()

    def _load_model(self):
        """加载模型到 GPU。"""
        cfg = MODEL_CONFIGS[self.method]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device
        print(f'[MLIP Server] Loading {self.method} model on {device}...', flush=True)
        mod = __import__(cfg['from'], fromlist=[cfg['import']])
        cls_or_fn = getattr(mod, cfg['import'])

        if cfg.get('type') == 'orbmol_v2':
            self._model, self._atoms_adapter = cls_or_fn(device=device)
            self._is_orbmol_v2 = True
        else:
            self.calculator = cls_or_fn(model=cfg['model_path'], device=device)
        print('[MLIP Server] Model loaded.', flush=True)

    def _warmup(self):
        """预热 CUDA JIT / kernel cache。"""
        try:
            print('[MLIP Server] Warming up model...', flush=True)
            atoms = Atoms(numbers=[6, 8], positions=[[0., 0., 0.], [1.2, 0., 0.]])
            cfg = MODEL_CONFIGS[self.method]
            if cfg.get('needs_info'):
                atoms.info["charge"] = 0
                atoms.info["spin"] = 1

            if self._is_orbmol_v2:
                graph = self._atoms_adapter.from_ase_atoms(atoms).to(self._device)
                result = self._model.predict(graph, split=False, compute_forces=True)
                # Verify energy/forces exist
                _ = float(result["energy"].cpu().detach())
                _ = result.get("forces")
                del atoms, graph, result
            else:
                if 'mace_polar' in self.method:
                    atoms.info["external_field"] = [0.0, 0.0, 0.0]
                atoms.calc = self.calculator
                atoms.get_potential_energy()
                atoms.get_forces()
                del atoms
            print('[MLIP Server] Warmup complete.', flush=True)
        except Exception as e:
            print(f'[MLIP Server] Warmup warning: {e}', flush=True)

    def compute(self, filein, fileout):
        """处理一次 Gaussian External 请求。"""
        t_start = time.time()
        eles, coords, deriva, charge, spin = parse_gau_external(filein)

        # 直接从数组构建 ASE Atoms（无中间 XYZ 文件）
        symbols = [_Z_SYMBOLS[int(z)] for z in eles]
        atoms = Atoms(symbols=symbols, positions=coords)
        cfg = MODEL_CONFIGS[self.method]
        if cfg.get('needs_info'):
            atoms.info["charge"] = charge
            atoms.info["spin"] = spin

        # 加锁推理
        with self.model_lock:
            if self._is_orbmol_v2:
                # orbmol_v2 通过 autograd 计算 forces，不能在 no_grad() 下运行
                atoms.info["charge"] = int(charge)
                atoms.info["spin"] = int(spin)
                graph = self._atoms_adapter.from_ase_atoms(atoms).to(self._device)
                result = self._model.predict(graph, split=False, compute_forces=True)
                energy = float(result["energy"].cpu().detach())
                if deriva == 1 and "forces" in result:
                    forces = result["forces"].cpu().detach().numpy()
                else:
                    forces = None
                del graph, result
            else:
                with torch.no_grad():
                    if 'mace_polar' in self.method:
                        atoms.info["external_field"] = [0.0, 0.0, 0.0]
                    atoms.calc = self.calculator
                    energy = atoms.get_potential_energy()
                    forces = None if deriva == 0 else atoms.get_forces()

        elapsed = time.time() - t_start
        print(f'[MLIP Server] Step  natoms={len(eles)}  E={energy:.8f} eV  '
              f'grad={deriva}  time={elapsed:.3f}s', flush=True)

        write_gau_output(fileout, energy, forces, deriva, len(eles))

        # 缓存最后状态
        if self.last_atoms is not None:
            del self.last_atoms
        self.last_atoms = atoms.copy()
        self.last_energy = energy
        self.last_forces = forces

        # 释放 GPU 内存
        del atoms, forces
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def handle_client(self, conn):
        """处理单个 TCP 客户端连接。"""
        try:
            # 读取 filein 路径
            data = b''
            while b'\n' not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            filein = data.split(b'\n', 1)[0].decode().strip()
            remaining = data.split(b'\n', 1)[1]

            # 读取 fileout 路径
            data = remaining
            while b'\n' not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            fileout = data.split(b'\n', 1)[0].decode().strip()

            self.compute(filein, fileout)
            conn.sendall(b'OK\n')
        except Exception as e:
            try:
                conn.sendall(('ERROR: %s\n' % str(e)).encode())
            except Exception:
                pass
        finally:
            conn.close()

    def _save_final_xyz(self):
        """退出时保存最后一步的结构到 XYZ 文件。"""
        if self.last_atoms is None or self.xyz_out is None:
            return
        try:
            info = self.last_atoms.info
            info['energy_eV'] = self.last_energy
            if self.last_forces is not None:
                self.last_atoms.arrays['forces'] = self.last_forces
            ase_write(self.xyz_out, self.last_atoms)
            print(f'[MLIP Server] Final structure saved to {self.xyz_out}', flush=True)
        except Exception as e:
            print(f'[MLIP Server] Failed to save XYZ: {e}', flush=True)

    def run(self):
        """启动 server 主循环。"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(8)
        self.sock.settimeout(1.0)
        print(f'[MLIP Server] Listening on {self.host}:{self.port}', flush=True)

        with ThreadPoolExecutor(max_workers=4) as executor:
            while not self._shutdown_flag.is_set():
                try:
                    conn, addr = self.sock.accept()
                    executor.submit(self.handle_client, conn)
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self._shutdown_flag.is_set():
                        print(f'[MLIP Server] Accept error: {e}', flush=True)

            executor.shutdown(wait=True)

        self._save_final_xyz()
        self.sock.close()
        print('[MLIP Server] Shut down.', flush=True)

    def shutdown(self):
        """信号驱动的优雅关闭。"""
        self._shutdown_flag.set()


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

_server_ref = None


def main():
    global _server_ref
    method = os.environ.get('METHOD', 'mace_off24')
    port = int(os.environ.get('MLIP_SERVER_PORT', 15556))
    host = os.environ.get('MLIP_SERVER_HOST', '127.0.0.1')
    xyz_out = os.environ.get('MLIP_XYZ_OUT', None)

    if method not in MODEL_CONFIGS:
        print(f'[MLIP Server] Unknown METHOD={method}. Available: {list(MODEL_CONFIGS.keys())}',
              flush=True)
        sys.exit(1)

    server = MLIPServer(method=method, host=host, port=port, xyz_out=xyz_out)
    _server_ref = server

    def _handle_signal(signum, frame):
        print('[MLIP Server] Received shutdown signal.', flush=True)
        server.shutdown()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    server.run()


if __name__ == '__main__':
    main()
