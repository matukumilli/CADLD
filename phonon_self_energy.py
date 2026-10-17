import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linear_sum_assignment

# Periodic table atomic masses in AMU
MASS_TABLE = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.868, "Cd": 112.41,
    "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50,
    "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49,
    "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Th": 232.04,
    "Pa": 231.04, "U": 238.03
}

# Sublattice compatibility groups
ANIONS = {'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'N', 'P', 'As', 'Sb'}

def are_compatible(elem1, elem2, enable_compat=True):
    if not enable_compat:
        return True
    if elem1 == elem2:
        return True
    if elem1 in ANIONS and elem2 in ANIONS:
        return True
    if elem1 not in ANIONS and elem2 not in ANIONS:
        return True
    return False

def parse_poscar(poscar_path):
    with open(poscar_path, 'r') as f:
        lines = [line.strip() for line in f]
    
    scale = float(lines[1])
    cell = []
    for i in range(2, 5):
        cell.append([float(x) for x in lines[i].split()])
    cell = np.array(cell) * scale
    
    species_names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    
    ref_species = []
    for name, cnt in zip(species_names, counts):
        ref_species.extend([name] * cnt)
        
    coord_type_line = lines[7]
    is_direct = coord_type_line.lower().startswith('d')
    start_line = 8
    
    if coord_type_line.lower().startswith('s'):
        coord_type_line = lines[8]
        is_direct = coord_type_line.lower().startswith('d')
        start_line = 9
        
    coords = []
    for i in range(start_line, start_line + sum(counts)):
        parts = lines[i].split()
        coords.append([float(x) for x in parts[:3]])
    coords = np.array(coords)
    
    if is_direct:
        coords_frac = coords % 1.0
        coords_cart = np.dot(coords_frac, cell)
    else:
        coords_cart = coords * scale
        inv_cell = np.linalg.inv(cell)
        coords_frac = np.dot(coords_cart, inv_cell) % 1.0
        
    return cell, ref_species, coords_cart, coords_frac

def parse_tdep_forceconstants(filename):
    """
    Parses TDEP's outfile.forceconstants file.
    Robustly handles trailing text comments on numeric lines.
    """
    print(f"Parsing TDEP force constants from {filename}...")
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    idx = 0
    na = int(lines[idx].split()[0])
    idx += 1
    
    cutoff = float(lines[idx].split()[0])
    idx += 1
    
    pairs = []
    for a1 in range(1, na + 1):
        n_pairs = int(lines[idx].split()[0])
        idx += 1
        
        for p in range(n_pairs):
            a2 = int(lines[idx].split()[0])
            idx += 1
            
            translation = np.array([float(x) for x in lines[idx].split()[:3]])
            idx += 1
            
            fc_matrix = []
            for j in range(3):
                fc_matrix.append([float(x) for x in lines[idx].split()[:3]])
                idx += 1
            fc_matrix = np.array(fc_matrix)
            
            pairs.append({
                "atom1": a1,
                "atom2": a2,
                "translation": translation,
                "fc": fc_matrix
            })
            
    polar_flag = int(lines[idx].split()[0])
    idx += 1
    
    polar_info = {"is_polar": False}
    if polar_flag != 0:
        polar_info["is_polar"] = True
        polar_info["correction_type"] = polar_flag
        
        eps = []
        for i in range(3):
            eps.append([float(x) for x in lines[idx].split()[:3]])
            idx += 1
        polar_info["eps"] = np.array(eps)
        
        polar_info["lambda"] = float(lines[idx].split()[0])
        idx += 1
        
        nx_Z = int(lines[idx].split()[0])
        idx += 1
        
        x_Z = []
        for i in range(nx_Z):
            x_Z.append(float(lines[idx].split()[0]))
            idx += 1
        polar_info["x_Z"] = np.array(x_Z)
        
        coeff_Z = []
        for i in range(na * 9):
            coeff_Z.append([float(x) for x in lines[idx].split()])
            idx += 1
        polar_info["coeff_Z"] = np.array(coeff_Z)
        
    return na, cutoff, pairs, polar_info

def parse_vasprun_xml(xml_path):
    symbols = []
    trajectory = []
    
    context = ET.iterparse(xml_path, events=("start", "end"))
    
    in_atominfo = False
    in_atoms_array = False
    in_set = False
    
    initial_cell = None
    current_cell = None
    current_positions = None
    current_forces = None
    
    try:
        for event, elem in context:
            tag = elem.tag
            
            if event == "start":
                if tag == "atominfo":
                    in_atominfo = True
                elif in_atominfo and tag == "array" and elem.get("name") == "atoms":
                    in_atoms_array = True
                elif in_atoms_array and tag == "set":
                    in_set = True
                    
            elif event == "end":
                if tag == "atominfo":
                    in_atominfo = False
                elif tag == "array" and elem.get("name") == "atoms":
                    in_atoms_array = False
                elif in_atoms_array and tag == "set":
                    in_set = False
                    for rc in elem.findall("rc"):
                        c_nodes = rc.findall("c")
                        if len(c_nodes) >= 1:
                            symbols.append(c_nodes[0].text.strip())
                            
                elif tag == "structure":
                    name = elem.get("name")
                    if name in ("initialpos", "primitive_cell"):
                        crystal = elem.find("crystal")
                        if crystal is not None:
                            basis_node = crystal.find(".//varray[@name='basis']")
                            if basis_node is not None:
                                basis = []
                                for v in basis_node.findall("v"):
                                    basis.append([float(x) for x in v.text.split()])
                                initial_cell = np.array(basis)
                        elem.clear()
                    elif name is None:
                        crystal = elem.find("crystal")
                        if crystal is not None:
                            basis_node = crystal.find(".//varray[@name='basis']")
                            if basis_node is not None:
                                basis = []
                                for v in basis_node.findall("v"):
                                    basis.append([float(x) for x in v.text.split()])
                                current_cell = np.array(basis)
                                
                        pos_node = elem.find(".//varray[@name='positions']")
                        if pos_node is not None:
                            positions = []
                            for v in pos_node.findall("v"):
                                positions.append([float(x) for x in v.text.split()])
                            current_positions = np.array(positions)
                        elem.clear()
                        
                elif tag == "varray" and elem.get("name") == "forces":
                    forces = []
                    for v in elem.findall("v"):
                        forces.append([float(x) for x in v.text.split()])
                    current_forces = np.array(forces)
                    elem.clear()
                    
                if current_positions is not None and current_forces is not None:
                    cell_to_use = current_cell
                    if cell_to_use is None:
                        cell_to_use = trajectory[-1]["cell"] if len(trajectory) > 0 else initial_cell
                            
                    if cell_to_use is not None:
                        trajectory.append({
                            "cell": cell_to_use,
                            "positions": current_positions,
                            "forces": current_forces
                        })
                    current_cell = None
                    current_positions = None
                    current_forces = None
                    
                if tag in ("energy", "time", "eigenvalues", "projected", "parameters"):
                    elem.clear()
                    
    except ET.ParseError:
        print("\n[Warning] vasprun.xml parsing ended prematurely. Processing frames parsed so far.")
        
    # Force default values as requested by the user: POTIM = 0.5 fs, NBLOCK = 10
    potim = 0.5
    nblock = 10
        
    return symbols, trajectory, potim, nblock

def expand_reference_to_supercell(ref_coords_frac, ref_species, ref_cell, M):
    super_cell = np.dot(M, ref_cell)
    inv_M = np.linalg.inv(M)
    
    vertices = []
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                vertices.append(np.dot(M, [x, y, z]))
    vertices = np.array(vertices)
    min_t = np.floor(np.min(vertices, axis=0)).astype(int) - 1
    max_t = np.ceil(np.max(vertices, axis=0)).astype(int) + 1
    
    valid_translations = []
    for i in range(min_t[0], max_t[0] + 1):
        for j in range(min_t[1], max_t[1] + 1):
            for k in range(min_t[2], max_t[2] + 1):
                t = np.array([i, j, k])
                t_frac = np.dot(inv_M, t)
                if np.all(t_frac >= -1e-5) and np.all(t_frac < 1.0 - 1e-5):
                    valid_translations.append(t)
                    
    num_replicas = int(round(np.abs(np.linalg.det(M))))
    if len(valid_translations) != num_replicas:
        valid_translations = []
        n1, n2, n3 = int(round(M[0,0])), int(round(M[1,1])), int(round(M[2,2]))
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    valid_translations.append(np.array([i, j, k]))
                    
    super_coords_frac = []
    super_species = []
    
    for t in valid_translations:
        for s_frac, spec in zip(ref_coords_frac, ref_species):
            s_super = np.dot(inv_M, s_frac + t) % 1.0
            super_coords_frac.append(s_super)
            super_species.append(spec)
            
    super_coords_frac = np.array(super_coords_frac)
    super_coords_cart = np.dot(super_coords_frac, super_cell)
    
    return super_coords_cart, super_coords_frac, super_species, super_cell, valid_translations

def align_and_map(
    ref_coords_frac, ref_species, ref_cell,
    snap_coords_frac, snap_species, snap_cell,
    dist_cutoff=1.5, max_iter=3, enable_compat=True
):
    s_ref = ref_coords_frac % 1.0
    s_snap = snap_coords_frac % 1.0
    
    shift = np.zeros(3)
    
    for iteration in range(max_iter):
        s_snap_shifted = (s_snap - shift) % 1.0
        diff = s_snap_shifted[:, np.newaxis, :] - s_ref[np.newaxis, :, :]
        diff = diff - np.round(diff)
        
        diff_cart = np.dot(diff, ref_cell)
        cost_matrix = np.sum(diff_cart**2, axis=-1)
        
        if enable_compat:
            for s_idx, s_spec in enumerate(snap_species):
                for r_idx, r_spec in enumerate(ref_species):
                    if not are_compatible(s_spec, r_spec, enable_compat):
                        cost_matrix[s_idx, r_idx] = 1e6
                        
        snap_ind, ref_ind = linear_sum_assignment(cost_matrix)
        
        valid_diffs = []
        for s_idx, r_idx in zip(snap_ind, ref_ind):
            d = np.sqrt(cost_matrix[s_idx, r_idx])
            if d <= dist_cutoff:
                valid_diffs.append(diff[s_idx, r_idx])
                
        if len(valid_diffs) == 0:
            break
            
        mean_diff = np.mean(valid_diffs, axis=0)
        shift += mean_diff
        shift = shift % 1.0
        
        if np.linalg.norm(np.dot(mean_diff, ref_cell)) < 1e-4:
            break
            
    s_snap_shifted = (s_snap - shift) % 1.0
    diff = s_snap_shifted[:, np.newaxis, :] - s_ref[np.newaxis, :, :]
    diff = diff - np.round(diff)
    diff_cart = np.dot(diff, ref_cell)
    cost_matrix = np.sum(diff_cart**2, axis=-1)
    
    if enable_compat:
        for s_idx, s_spec in enumerate(snap_species):
            for r_idx, r_spec in enumerate(ref_species):
                if not are_compatible(s_spec, r_spec, enable_compat):
                    cost_matrix[s_idx, r_idx] = 1e6
                    
    snap_ind, ref_ind = linear_sum_assignment(cost_matrix)
    
    mapping = {}
    for j in range(len(ref_coords_frac)):
        mapping[j] = {
            "occupied": False,
            "mapped_atom_idx": None,
            "species": "Vacancy",
            "distance": None
        }
        
    for s_idx, r_idx in zip(snap_ind, ref_ind):
        dist = np.sqrt(cost_matrix[s_idx, r_idx])
        if dist <= dist_cutoff:
            mapping[r_idx] = {
                "occupied": True,
                "mapped_atom_idx": int(s_idx),
                "species": str(snap_species[s_idx]),
                "distance": float(dist)
            }
            
    return mapping

def compute_dynamical_matrix(q_frac, pairs, na, masses, ref_cell):
    D = np.zeros((3 * na, 3 * na), dtype=complex)
    
    for pair in pairs:
        a1 = pair["atom1"] - 1
        a2 = pair["atom2"] - 1
        translation = pair["translation"]
        fc = pair["fc"]
        
        phase = 2 * np.pi * np.dot(q_frac, translation)
        factor = np.exp(1j * phase) / np.sqrt(masses[a1] * masses[a2])
        
        D[a1*3 : a1*3 + 3, a2*3 : a2*3 + 3] += fc * factor
                   
    return D

def solve_harmonic_phonons(q_frac, pairs, na, masses, ref_cell):
    D = compute_dynamical_matrix(q_frac, pairs, na, masses, ref_cell)
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    
    freqs = np.zeros(len(eigenvalues))
    for i, val in enumerate(eigenvalues):
        if val >= 0:
            freqs[i] = 15.633304 * np.sqrt(val)
        else:
            freqs[i] = -15.633304 * np.sqrt(-val)
            
    return freqs, eigenvectors

def compute_correlation_fft(x, y):
    N = len(x)
    n_fft = 2**int(np.ceil(np.log2(2 * N - 1)))
    X = np.fft.fft(x, n=n_fft)
    Y = np.fft.fft(y, n=n_fft)
    S = X * np.conj(Y)
    r = np.fft.ifft(S).real[:N]
    lags = np.arange(N)
    normalization = N - lags
    return r / normalization

def unwrap_trajectory(trajectory):
    N_frames = len(trajectory)
    N_atoms = trajectory[0]["positions"].shape[0]
    
    unwrapped_cart = np.zeros((N_frames, N_atoms, 3))
    first_cell = trajectory[0]["cell"]
    first_frac = trajectory[0]["positions"] % 1.0
    unwrapped_cart[0] = np.dot(first_frac, first_cell)
    
    prev_frac = first_frac.copy()
    current_unwrapped_frac = first_frac.copy()
    
    for f_idx in range(1, N_frames):
        curr_frac = trajectory[f_idx]["positions"] % 1.0
        curr_cell = trajectory[f_idx]["cell"]
        df = curr_frac - prev_frac
        df = df - np.round(df)
        current_unwrapped_frac += df
        unwrapped_cart[f_idx] = np.dot(current_unwrapped_frac, curr_cell)
        prev_frac = curr_frac.copy()
        
    return unwrapped_cart

def compute_msd_fft(pos_series):
    N = len(pos_series)
    msd = np.zeros(N)
    
    for d in range(3):
        x = pos_series[:, d]
        n_fft = 2**int(np.ceil(np.log2(2 * N - 1)))
        X = np.fft.fft(x, n=n_fft)
        S = X * np.conj(X)
        r = np.fft.ifft(S).real[:N]
        
        sq = x**2
        cumsum_sq = np.concatenate(([0], np.cumsum(sq)))
        lags = np.arange(N)
        S1 = (cumsum_sq[N - lags] - cumsum_sq[0]) + (cumsum_sq[N] - cumsum_sq[lags])
        msd += (S1 - 2 * r) / (N - lags)
        
    return msd

def double_sided_fft(C_t, dt):
    N_lag = len(C_t)
    N_double = 2 * N_lag - 1
    C_double = np.zeros(N_double, dtype=complex)
    zero_idx = N_lag - 1
    
    C_double[zero_idx] = C_t[0]
    for t in range(1, N_lag):
        C_double[zero_idx + t] = C_t[t]
        C_double[zero_idx - t] = np.conj(C_t[t])
        
    C_double_shifted = np.fft.ifftshift(C_double)
    Sigma_w = np.fft.fft(C_double_shifted) * dt
    
    freqs = np.fft.fftfreq(N_double, d=dt)
    sorted_idx = np.argsort(freqs)
    
    return freqs[sorted_idx], Sigma_w[sorted_idx]

def hilbert_transform_kk(gamma):
    N = len(gamma)
    G = np.fft.ifft(gamma)
    sign = np.zeros(N)
    sign[1:N//2] = 1.0
    sign[N//2+1:] = -1.0
    
    H_operator = -1j * sign
    delta = np.fft.fft(G * H_operator).real
    return delta

def generate_qpath(q_bounds, n_points=30, ref_cell=np.eye(3)):
    q_points = []
    q_distances = []
    curr_dist = 0.0
    
    rec_cell = 2 * np.pi * np.linalg.inv(ref_cell).T
    
    for i in range(len(q_bounds) - 1):
        q_start = np.array(q_bounds[i])
        q_end = np.array(q_bounds[i+1])
        
        for step in range(n_points):
            if i < len(q_bounds) - 2 and step == n_points - 1:
                continue
            t = step / (n_points - 1)
            q = q_start + t * (q_end - q_start)
            q_points.append(q)
            
            if len(q_points) > 1:
                dq = q_points[-1] - q_points[-2]
                dq_cart = np.dot(dq, rec_cell)
                curr_dist += np.linalg.norm(dq_cart)
            q_distances.append(curr_dist)
            
    return np.array(q_points), np.array(q_distances)

def find_fwhm(w_grid, A, idx_max):
    half_max = A[idx_max] * 0.5
    
    # Left half-max
    idx_left = 0
    for i in range(idx_max, -1, -1):
        if A[i] < half_max:
            idx_left = i
            break
    if idx_left < idx_max:
        A_diff = A[idx_left+1] - A[idx_left]
        if A_diff > 1e-8:
            w_left = w_grid[idx_left] + (half_max - A[idx_left]) / A_diff * (w_grid[idx_left+1] - w_grid[idx_left])
        else:
            w_left = w_grid[idx_left]
    else:
        w_left = w_grid[0]
        
    # Right half-max
    idx_right = len(w_grid) - 1
    for i in range(idx_max, len(w_grid)):
        if A[i] < half_max:
            idx_right = i
            break
    if idx_right > idx_max:
        A_diff = A[idx_right-1] - A[idx_right]
        if A_diff > 1e-8:
            w_right = w_grid[idx_right-1] + (A[idx_right-1] - half_max) / A_diff * (w_grid[idx_right] - w_grid[idx_right-1])
        else:
            w_right = w_grid[idx_right]
    else:
        w_right = w_grid[-1]
        
    return w_right - w_left

def main():
    parser = argparse.ArgumentParser(description="Phonon Dispersion and Self-Energy Correction from TDEP forceconstants and MD force-force correlation.")
    parser.add_argument("--poscar", type=str, default="POSCAR", help="Path to reference unit cell POSCAR.")
    parser.add_argument("--vasprun", type=str, default="vasprun.xml", help="Path to vasprun.xml trajectory.")
    parser.add_argument("--forceconstants", type=str, default="outfile.forceconstants", help="TDEP second order force constants file.")
    parser.add_argument("--cutoff", type=float, default=1.5, help="Distance mapping cutoff (in Å).")
    parser.add_argument("--temp", type=float, default=300.0, help="Simulation temperature (in K).")
    parser.add_argument("--qpath", type=str, default="0,0,0 0.5,0,0 0.333,0.333,0 0,0,0", 
                        help="High symmetry q-path boundary points (fractional reciprocal), e.g. '0,0,0 0.5,0,0'")
    parser.add_argument("--npoints", type=int, default=15, help="Number of q-points per segment.")
    parser.add_argument("--damping", type=float, default=500.0, help="Gaussian damping time for forces (in fs).")
    parser.add_argument("--max-lag-pct", type=float, default=50.0, help="Maximum lag time as percentage of trajectory length.")
    parser.add_argument("--broadening", type=float, default=0.1, help="Instrumental/artificial peak broadening (in THz) to resolve sharp modes.")
    parser.add_argument("--no-compat", action="store_true", help="Disable sublattice compatibility constraints.")
    parser.add_argument("--select-species", type=str, default=None, 
                        help="Filter projection to include only contributions from these species (space-separated, e.g. 'Ag' or 'Te S').")
    parser.add_argument("--select-sites", type=str, default=None,
                        help="Filter projection to include only contributions from these specific 0-based site indices (space-separated, e.g. '0 1 12').")
    parser.add_argument("--scale-self-energy", type=float, default=1.0,
                        help="Anharmonic self-energy scaling factor (0.0 to 1.0) to study the renormalization path.")
    parser.add_argument("--labels", type=str, default="Gamma M K Gamma",
                        help="Space-separated boundary q-point labels.")
    parser.add_argument("--min-broadening", type=float, default=1.0,
                        help="Minimum broadening cap (in cm-1) to resolve sharp modes on the discrete grid.")
    parser.add_argument("--wgrid-points", type=int, default=2000,
                        help="Number of points in the frequency grid.")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory.")
    
    args = parser.parse_args()
    
    # 1. Parse POSCAR and build masses
    print(f"Reading unit cell POSCAR from {args.poscar}...")
    ref_cell, ref_species, ref_coords_cart, ref_coords_frac = parse_poscar(args.poscar)
    na = len(ref_species)
    masses = []
    for s in ref_species:
        if s in MASS_TABLE:
            masses.append(MASS_TABLE[s])
        else:
            print(f"[Warning] Element '{s}' not in mass table. Defaulting to 1.0 AMU.")
            masses.append(1.0)
    masses = np.array(masses)
    
    # 2. Parse force constants
    fc_na, fc_cutoff, pairs, polar_info = parse_tdep_forceconstants(args.forceconstants)
    if fc_na != na:
        print(f"[Error] Atoms in forceconstants ({fc_na}) and POSCAR ({na}) do not match!")
        return
        
    # 3. Generate Q-path
    q_bounds = []
    for q_str in args.qpath.split():
        parts = [float(x) for x in q_str.split(',')]
        if len(parts) == 3:
            q_bounds.append(parts)
    q_path, q_distances = generate_qpath(q_bounds, n_points=args.npoints, ref_cell=ref_cell)
    N_q = len(q_path)
    print(f"Generated Q-path with {N_q} points.")
    
    # 4. Parse Trajectory
    print(f"Parsing trajectory from {args.vasprun}...")
    snap_species, trajectory, potim, nblock = parse_vasprun_xml(args.vasprun)
    N_frames = len(trajectory)
    N_snap = len(snap_species)
    dt = potim * nblock
    print(f"  Parsed {N_frames} frames with {N_snap} atoms (dt = {dt:.4f} fs).")
    
    if N_frames == 0:
        print("[Error] No trajectory frames parsed. Exiting.")
        return
    print(np.shape(trajectory)) 
    trajectory = trajectory[:4900]   
    print(np.shape(trajectory))     
    N_frames = len(trajectory)             
    # Determine supercell expansion matrix M
    first_snap_cell = trajectory[0]["cell"]
    M = np.round(np.dot(first_snap_cell, np.linalg.inv(ref_cell)))
    print(f"Detected supercell expansion matrix M:\n{M.astype(int)}")
    
    # Expand reference cell
    super_coords_cart, super_coords_frac, super_species, super_cell, valid_translations = expand_reference_to_supercell(
        ref_coords_frac, ref_species, ref_cell, M
    )
    N_super = len(super_species)
    N_rep = len(valid_translations)
    print(f"Expanded supercell has {N_super} sites ({N_rep} replicas).")
    
    # 5. Map snapshots to supercell sites and store site forces & displacements
    print("Mapping trajectory atoms to supercell sites...")
    site_forces = np.zeros((N_frames, N_super, 3))
    site_disps = np.zeros((N_frames, N_super, 3))
    
    for f_idx, frame in enumerate(trajectory):
        if (f_idx + 1) % 100 == 0 or (f_idx + 1) == N_frames:
            print(f"  Processing frame {f_idx + 1}/{N_frames}...")
            
        mapping = align_and_map(
            super_coords_frac, super_species, super_cell,
            frame["positions"], snap_species, frame["cell"],
            dist_cutoff=args.cutoff, enable_compat=not args.no_compat
        )
        
        for r_idx in range(N_super):
            m = mapping[r_idx]
            if m["occupied"]:
                s_idx = m["mapped_atom_idx"]
                diff_frac = frame["positions"][s_idx] - super_coords_frac[r_idx]
                diff_frac = diff_frac - np.round(diff_frac)
                diff_cart = np.dot(diff_frac, super_cell)
                
                site_forces[f_idx, r_idx, :] = frame["forces"][s_idx, :]
                site_disps[f_idx, r_idx, :] = diff_cart
            else:
                site_forces[f_idx, r_idx, :] = 0.0
                site_disps[f_idx, r_idx, :] = 0.0
                
    # 6. Calculate Mean Squared Displacements (MSD)
    print("\nCalculating Mean Squared Displacements (MSD)...")
    unwrapped_cart = unwrap_trajectory(trajectory)
    unique_elements = sorted(list(set(snap_species)))
    msd_results = {}
    
    total_msd = np.zeros(N_frames)
    for i in range(N_snap):
        total_msd += compute_msd_fft(unwrapped_cart[:, i, :])
    total_msd /= N_snap
    msd_results["Total"] = total_msd
    
    for elem in unique_elements:
        indices = [i for i, s in enumerate(snap_species) if s == elem]
        if len(indices) > 0:
            elem_msd = np.zeros(N_frames)
            for idx in indices:
                elem_msd += compute_msd_fft(unwrapped_cart[:, idx, :])
            elem_msd /= len(indices)
            msd_results[elem] = elem_msd
            print(f"  Calculated MSD for species: {elem} ({len(indices)} atoms)")
            
    os.makedirs(args.out_dir, exist_ok=True)
    msd_file = os.path.join(args.out_dir, "MSD.dat")
    with open(msd_file, 'w') as f:
        f.write("# Mean Squared Displacement (MSD) in Å^2 as a function of time\n")
        header_cols = ["Time(fs)", "Total"] + unique_elements
        f.write("# " + " ".join([f"{col:<14}" for col in header_cols]) + "\n")
        for t in range(N_frames):
            t_fs = t * dt
            line = f"{t_fs:14.6f} {msd_results['Total'][t]:14.8e} " + " ".join([f"{msd_results[elem][t]:14.8e}" for elem in unique_elements]) + "\n"
            f.write(line)
    print(f"  Saved Mean Squared Displacements to {msd_file}")
                
    # Max lag setup
    max_lag = int((args.max_lag_pct / 100.0) * N_frames)
    max_lag = max(1, min(max_lag, N_frames))
    t_lag = np.arange(max_lag) * dt
    window = np.exp(-(t_lag / args.damping)**2) if args.damping > 0 else np.ones(max_lag)
    
    # Scale factor for frequency-squared self-energy: 0.2444002 / (2 * kB * T * N_rep)
    kB = 8.617333262e-5
    scaling_factor_sq = 0.2444002 / (2 * kB * args.temp * N_rep)
    
    # Output arrays in THz
    harmonic_freqs = np.zeros((N_q, 3 * na))
    dressed_freqs = np.zeros((N_q, 3 * na))
    linewidths = np.zeros((N_q, 3 * na))
    
    # Frequency grid from 0 to 15 THz (will be written to file as cm^-1 with 2000 points for high resolution)
    w_grid = np.linspace(0.0, 15.0, args.wgrid_points)
    spectral_function = np.zeros((N_q, len(w_grid)))
    
    print("\nSolving Phonons and Self-Energy Corrections along Q-path...")
    
    # Setup species projection mask
    selected_indices = None
    if args.select_sites:
        try:
            selected_indices = [int(x) for x in args.select_sites.split()]
            selected_indices = [idx for idx in selected_indices if 0 <= idx < na]
            if len(selected_indices) == 0:
                print(f"[Warning] No valid site indices found in '{args.select_sites}'. Projecting all.")
                selected_indices = list(range(na))
            else:
                print(f"Filtering projection to specific sites: {selected_indices}")
        except ValueError:
            print(f"[Error] Invalid format for --select-sites. Must be space-separated integers. Projecting all.")
            selected_indices = list(range(na))
    elif args.select_species:
        selected_species_list = args.select_species.split()
        selected_indices = [i for i, s in enumerate(ref_species) if s in selected_species_list]
        if len(selected_indices) == 0:
            print(f"[Warning] No atoms of species '{args.select_species}' found. Projecting all.")
            selected_indices = list(range(na))
        else:
            print(f"Filtering projection to species: {selected_species_list} (sites: {selected_indices})")
    else:
        selected_indices = list(range(na))
        
    mask = np.zeros(3 * na, dtype=bool)
    for idx in selected_indices:
        mask[idx * 3 : idx * 3 + 3] = True

    # Setup high-symmetry q-points mapping for pair-resolved output
    vertex_idx_list = [i * (args.npoints - 1) for i in range(len(q_bounds))]
    labels_list = args.labels.split() if args.labels else []
    if not labels_list or len(labels_list) != len(q_bounds):
        labels_list = [f"q_{q_bounds[i][0]:.3f}_{q_bounds[i][1]:.3f}_{q_bounds[i][2]:.3f}" for i in range(len(q_bounds))]
        
    unique_species = sorted(list(set(ref_species)))
    unique_pairs = []
    for i in range(len(unique_species)):
        for j in range(i, len(unique_species)):
            unique_pairs.append((unique_species[i], unique_species[j]))
    N_unique = len(unique_pairs)
        
    for q_idx, q in enumerate(q_path):
        h_freqs, eigenvectors = solve_harmonic_phonons(q, pairs, na, masses, ref_cell)
        harmonic_freqs[q_idx, :] = h_freqs
        
        phase = -2j * np.pi * np.dot(valid_translations, q)
        exp_phase = np.exp(phase)
        
        F_q_t = np.zeros((3 * na, N_frames), dtype=complex)
        u_q_t = np.zeros((3 * na, N_frames), dtype=complex)
        for kappa in range(na):
            for alpha in range(3):
                site_slice_f = site_forces[:, kappa::na, alpha]
                site_slice_u = site_disps[:, kappa::na, alpha]
                
                F_q_t[kappa * 3 + alpha, :] = np.dot(site_slice_f, exp_phase)
                u_q_t[kappa * 3 + alpha, :] = np.dot(site_slice_u, exp_phase)
                
        masses_sqrt = np.repeat(np.sqrt(masses), 3)
        F_q_t_weighted = F_q_t / masses_sqrt[:, np.newaxis]
        u_q_t_weighted = u_q_t * masses_sqrt[:, np.newaxis]
        
        # Project components using the species mask
        F_proj_t = np.dot(eigenvectors[mask, :].conj().T, F_q_t_weighted[mask, :])
        Q_proj_t = np.dot(eigenvectors[mask, :].conj().T, u_q_t_weighted[mask, :])
        
        # If this is a high-symmetry q-point, precalculate species-specific projections
        is_vertex = q_idx in vertex_idx_list
        if is_vertex:
            A_q_XY = {pair: np.zeros(len(w_grid)) for pair in unique_pairs}
            F_proj_t_spec = {}
            Q_proj_t_spec = {}
            for X in unique_species:
                mask_X = np.zeros(3 * na, dtype=bool)
                indices_X = [idx for idx, s in enumerate(ref_species) if s == X]
                for idx in indices_X:
                    mask_X[idx * 3 : idx * 3 + 3] = True
                F_proj_t_spec[X] = np.dot(eigenvectors[mask_X, :].conj().T, F_q_t_weighted[mask_X, :])
                Q_proj_t_spec[X] = np.dot(eigenvectors[mask_X, :].conj().T, u_q_t_weighted[mask_X, :])
                
        q_spectral_density = np.zeros((3 * na, len(w_grid)))
        
        for lamb in range(3 * na):
            w0 = h_freqs[lamb]
            w0_abs = np.abs(w0)
            w0_sq = w0_abs**2
            
            lambda_0 = np.sign(w0) * (w0_abs / 15.633304)**2
            F_proj_t_anh = F_proj_t[lamb, :] + lambda_0 * Q_proj_t[lamb, :]
            
            C_t = compute_correlation_fft(F_proj_t_anh, F_proj_t_anh)[:max_lag]
            C_t_windowed = C_t * window
            
            freqs_w, Sigma_w = double_sided_fft(C_t_windowed, dt)
            freqs_thz = freqs_w * 1000.0
            
            gamma_sq_full = np.abs(freqs_thz) * Sigma_w.real * scaling_factor_sq
            delta_sq_full = hilbert_transform_kk(gamma_sq_full)
            
            gamma_sq_w = np.interp(w_grid, freqs_thz, gamma_sq_full) * args.scale_self_energy
            delta_sq_w = np.interp(w_grid, freqs_thz, delta_sq_full) * args.scale_self_energy
            
            # Add instrumental/artificial broadening (in THz) to resolve sharp modes on the discrete grid
            # We enforce a frequency-dependent minimum cap (min_broadening_sq) corresponding to a minimum FWHM of 1.0 cm^-1
            # to guarantee that extremely sharp peaks never fall between grid points and disappear.
            min_broadening_sq = 2.0 * np.maximum(w_grid, 0.1) * (args.min_broadening / 33.35641)
            gamma_sq_w_total = gamma_sq_w + 2.0 * w_grid * args.broadening
            gamma_sq_w_total = np.maximum(gamma_sq_w_total, min_broadening_sq)
            
            num = (1.0 / np.pi) * gamma_sq_w_total
            denom = (w_grid**2 - w0_sq - delta_sq_w)**2 + gamma_sq_w_total**2
            
            with np.errstate(divide='ignore', invalid='ignore'):
                A_lamb = np.where(denom > 0, num / denom, 0.0)
            A_lamb = np.nan_to_num(A_lamb)
            q_spectral_density[lamb, :] = A_lamb
            
            # If this is a high-symmetry q-point, compute pair-resolved spectral contributions
            if is_vertex:
                for (X, Y) in unique_pairs:
                    F_proj_t_anh_X = F_proj_t_spec[X][lamb, :] + lambda_0 * Q_proj_t_spec[X][lamb, :]
                    F_proj_t_anh_Y = F_proj_t_spec[Y][lamb, :] + lambda_0 * Q_proj_t_spec[Y][lamb, :]
                    
                    C_t_XY = 0.5 * (compute_correlation_fft(F_proj_t_anh_X, F_proj_t_anh_Y) + 
                                    compute_correlation_fft(F_proj_t_anh_Y, F_proj_t_anh_X))[:max_lag]
                    C_t_XY_windowed = C_t_XY * window
                    
                    freqs_w, Sigma_w_XY = double_sided_fft(C_t_XY_windowed, dt)
                    gamma_sq_XY = np.abs(freqs_thz) * Sigma_w_XY.real * scaling_factor_sq
                    gamma_sq_w_XY = np.interp(w_grid, freqs_thz, gamma_sq_XY) * args.scale_self_energy
                    
                    # Distribute the instrumental broadening equally among unique pairs
                    gamma_sq_w_XY_total = (gamma_sq_w_XY if X == Y else 2.0 * gamma_sq_w_XY) + (gamma_sq_w_total - gamma_sq_w) / N_unique
                    
                    A_lamb_XY = (1.0 / np.pi) * gamma_sq_w_XY_total / denom
                    A_q_XY[(X, Y)] += np.nan_to_num(A_lamb_XY)
            
            idx_max = np.argmax(A_lamb)
            w_dressed = w_grid[idx_max]
            fwhm = find_fwhm(w_grid, A_lamb, idx_max)
            
            dressed_freqs[q_idx, lamb] = w_dressed
            linewidths[q_idx, lamb] = fwhm
            
        spectral_function[q_idx, :] = np.sum(q_spectral_density, axis=0)
        
        # Write pair files if we are at a high-symmetry q-point
        if is_vertex:
            v_idx = vertex_idx_list.index(q_idx)
            label = labels_list[v_idx]
            to_cm = 33.35641
            for (X, Y) in unique_pairs:
                filename = os.path.join(args.out_dir, f"pair_spec_{label}_{X}-{Y}.dat")
                with open(filename, 'w') as out_f:
                    out_f.write(f"# Pair-resolved spectral function A_{{{X}-{Y}}}(w) at high symmetry point {label}\n")
                    out_f.write(f"# Wavevector q = {q}\n")
                    out_f.write(f"# Columns: Frequency(cm-1) Spectral_Intensity(cm)\n")
                    for w_idx, w_cm in enumerate(w_grid * to_cm):
                        out_f.write(f"{w_cm:14.6f} {A_q_XY[(X, Y)][w_idx] / to_cm:14.8e}\n")
                print(f"  Saved pair-resolved spectral function to {filename}")
        
    # Write output files converting THz to cm^-1
    # 1 THz = 33.35641 cm^-1
    to_cm = 33.35641
    
    # 1. Harmonic Dispersion
    harm_file = os.path.join(args.out_dir, "phonon_dispersion_harmonic.dat")
    with open(harm_file, 'w') as f:
        f.write("# Harmonic phonon dispersion from TDEP second order force constants\n")
        f.write(f"# Columns: q_dist q_h q_k q_l " + " ".join([f"w0_{i}(cm-1)" for i in range(3*na)]) + "\n")
        for q_idx in range(N_q):
            q = q_path[q_idx]
            q_dist = q_distances[q_idx]
            freq_str = " ".join([f"{x * to_cm:12.6f}" for x in harmonic_freqs[q_idx]])
            f.write(f"{q_dist:12.6f} {q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f} {freq_str}\n")
    print(f"Saved harmonic phonon dispersion in cm-1 to {harm_file}")
            
    # 2. Dressed Dispersion (Dyson solved)
    dressed_file = os.path.join(args.out_dir, "phonon_dispersion_dressed.dat")
    with open(dressed_file, 'w') as f:
        f.write("# Dressed phonon dispersion solving Dyson equation with MD force-force correlation self-energy\n")
        f.write(f"# Columns: q_dist q_h q_k q_l " + " ".join([f"w_dressed_{i}(cm-1) FWHM_{i}(cm-1)" for i in range(3*na)]) + "\n")
        for q_idx in range(N_q):
            q = q_path[q_idx]
            q_dist = q_distances[q_idx]
            dressed_vals = []
            for lamb in range(3*na):
                dressed_vals.append(f"{dressed_freqs[q_idx, lamb] * to_cm:12.6f} {linewidths[q_idx, lamb] * to_cm:12.6f}")
            dressed_str = " ".join(dressed_vals)
            f.write(f"{q_dist:12.6f} {q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f} {dressed_str}\n")
    print(f"Saved self-energy corrected dispersion in cm-1 to {dressed_file}")
            
    # 3. Spectral function A(q, w) heatmap data
    # Convert frequency grid to cm^-1, and A(q, w) values to 1/cm^-1 to preserve normalization (integral A dw = 1)
    w_grid_cm = w_grid * to_cm
    spectral_function_cm = spectral_function / to_cm
    
    spec_file = os.path.join(args.out_dir, "spectral_function_A_q_w.dat")
    with open(spec_file, 'w') as f:
        f.write("# Interacting phonon spectral function A(q, omega) heatmap data\n")
        f.write(f"# Columns: q_dist q_h q_k q_l " + " ".join([f"A({w:.3f}_cm-1)" for w in w_grid_cm]) + "\n")
        for q_idx in range(N_q):
            q = q_path[q_idx]
            q_dist = q_distances[q_idx]
            spec_str = " ".join([f"{x:14.6e}" for x in spectral_function_cm[q_idx]])
            f.write(f"{q_dist:12.6f} {q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f} {spec_str}\n")
    print(f"Saved interacting spectral function heatmap data in cm-1 to {spec_file}")
    
    print("\nPhonon dispersion and self-energy correction calculation completed successfully!")

if __name__ == "__main__":
    main()
