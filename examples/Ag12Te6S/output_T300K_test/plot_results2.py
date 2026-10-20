import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
#python3.8 plot_results2.py --out-dir "../output_T300K_test/" --labels "Gamma M K Gamma A"

def detect_high_symmetry_points(q_coords, q_distances):
    """
    Automatically detects indices of high-symmetry points where the q-path bends.
    """
    N_q = len(q_coords)
    boundary_indices = [0]
    
    dq = np.diff(q_coords, axis=0)
    norms = np.linalg.norm(dq, axis=1)
    norms[norms == 0] = 1e-8
    dq_norm = dq / norms[:, np.newaxis]
    
    for i in range(1, len(dq_norm)):
        dot = np.dot(dq_norm[i-1], dq_norm[i])
        if dot < 0.999:
            boundary_indices.append(i)
            
    boundary_indices.append(N_q - 1)
    return boundary_indices

def plot_msd(msd_file, out_dir):
    """
    Plots the Mean Squared Displacement (MSD) as a function of time.
    """
    if not os.path.exists(msd_file):
        print(f"[Warning] {msd_file} not found. Skipping MSD plot.")
        return
        
    print(f"Plotting MSD from {msd_file}...")
    data = np.loadtxt(msd_file)
    
    with open(msd_file, 'r') as f:
        f.readline()  # Skip first comment line
        header = f.readline()
        if "Columns" in header or header.strip().startswith("#"):
            columns = header.replace("#", "").split()
        else:
            columns = ["Time", "Total", "Ag", "S", "Te"]
            
    time = data[:, 0]
    
    plt.figure(figsize=(7, 5))
    for col_idx in range(1, data.shape[1]):
        label = columns[col_idx] if col_idx < len(columns) else f"Col {col_idx}"
        linewidth = 2.5 if label in ("Total", "Ag", "Total(fs)") else 1.5
        plt.plot(time, data[:, col_idx], label=label, linewidth=linewidth)
        
    plt.xlabel("Time (fs)", fontsize=12)
    plt.ylabel("Mean Squared Displacement ($\AA^2$)", fontsize=12)
    plt.title("Mean Squared Displacement (MSD)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, "plot_MSD.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved MSD plot to {plot_path}")

def plot_phonon_dispersion(harm_file, dressed_file, labels_str, out_dir):
    """
    Plots the comparison between bare harmonic and self-energy dressed phonon bands in cm-1 or THz.
    """
    if not os.path.exists(harm_file) or not os.path.exists(dressed_file):
        print(f"[Warning] Phonon files not found. Skipping dispersion plot.")
        return
        
    print(f"Plotting phonon dispersions from {harm_file} and {dressed_file}...")
    harm_data = np.loadtxt(harm_file)
    dressed_data = np.loadtxt(dressed_file)
    
    q_dist = harm_data[:, 0]
    q_coords = harm_data[:, 1:4]
    
    harm_bands = harm_data[:, 4:]
    N_modes = harm_bands.shape[1]
    
    dressed_bands = dressed_data[:, 4::2]
    linewidths = dressed_data[:, 5::2]
    
    # Check units from file header
    unit_label = "THz"
    with open(harm_file, 'r') as f:
        f.readline()
        header = f.readline()
        if "cm-1" in header:
            unit_label = "cm$^{-1}$"
            
    boundary_idx = detect_high_symmetry_points(q_coords, q_dist)
    boundary_dists = q_dist[boundary_idx]
    
    if labels_str:
        labels = labels_str.split()
    else:
        labels = []
        for idx in boundary_idx:
            q = q_coords[idx]
            labels.append(f"({q[0]:.2f},{q[1]:.2f},{q[2]:.2f})")
    labels = [r"$\Gamma$" if l.lower() in ("gamma", "g", "0,0,0") else l for l in labels]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for m in range(N_modes):
        ax.plot(q_dist, harm_bands[:, m], color="grey", linestyle="--", linewidth=1.0, 
                label="Harmonic" if m == 0 else "")
        
    for m in range(N_modes):
        w_dressed = dressed_bands[:, m]
        fwhm = linewidths[:, m]
        
        ax.plot(q_dist, w_dressed, color="crimson", linestyle=":", linewidth=1.5,
                label="Dressed" if m == 0 else "")
        ax.fill_between(q_dist, w_dressed - fwhm/2, w_dressed + fwhm/2, 
                        color="crimson", alpha=0.1, edgecolor="none")
        
    for bd in boundary_dists:
        ax.axvline(x=bd, color="black", linestyle="-", alpha=0.3)
        
    ax.set_xticks(boundary_dists)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(q_dist[0], q_dist[-1])
    ax.set_ylabel(f"Frequency ({unit_label})", fontsize=13)
    ax.set_title("Phonon Dispersion: Bare vs. Dyson-Dressed", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    
    y_min = min(np.min(harm_bands), np.min(dressed_bands))
    y_max = max(np.max(harm_bands), np.max(dressed_bands))
    if "cm" in unit_label:
        ax.set_ylim(max(y_min - 10.0, -100.0), y_max + 20.0)
    else:
        ax.set_ylim(max(y_min - 0.5, -2.0), y_max + 1.0)
        
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "plot_phonon_dispersion.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved dispersion plot to {plot_path}")

def plot_spectral_function(spec_file, harm_file, labels_str, out_dir, use_log=False, vmax=None):
    """
    Plots the interacting spectral function A(q, omega) as a 2D intensity heatmap.
    """
    if not os.path.exists(spec_file) or not os.path.exists(harm_file):
        print(f"[Warning] Spectral function or harmonic file not found. Skipping heatmap plot.")
        return
        
    print(f"Plotting spectral function heatmap from {spec_file}...")
    spec_data = np.loadtxt(spec_file)
    harm_data = np.loadtxt(harm_file)
    
    q_dist = spec_data[:, 0]
    q_coords = spec_data[:, 1:4]
    
    A_q_w = spec_data[:, 4:].T
    harm_bands = harm_data[:, 4:]
    N_modes = harm_bands.shape[1]
    
    with open(spec_file, 'r') as f:
        header_cols = None
        for line in f:
            if "Columns" in line:
                header_cols = line.replace("# Columns:", "").replace("#", "").split()
                break
        if header_cols is None:
            print("[Error] Could not find header columns in spectral function file!")
            return
            
    w_grid = []
    unit_label = "THz"
    for col in header_cols[4:]:
        val_str = col.replace("A(", "").replace("_THz)", "").replace("_cm-1)", "")
        w_grid.append(float(val_str))
        if "cm-1" in col:
            unit_label = "cm$^{-1}$"
    w_grid = np.array(w_grid)
    
    boundary_idx = detect_high_symmetry_points(q_coords, q_dist)
    boundary_dists = q_dist[boundary_idx]
    
    if labels_str:
        labels = labels_str.split()
    else:
        labels = []
        for idx in boundary_idx:
            q = q_coords[idx]
            labels.append(f"({q[0]:.2f},{q[1]:.2f},{q[2]:.2f})")
    labels = [r"$\Gamma$" if l.lower() in ("gamma", "g", "0,0,0") else l for l in labels]
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    Q, W = np.meshgrid(q_dist, w_grid)
    
    # Scale options
    if use_log:
        A_plot = np.log10(np.maximum(A_q_w, 1e-5))
        cbar_label = "Spectral Intensity $\log_{10}(A(\mathbf{q}, \omega))$"
    else:
        A_plot = np.power(A_q_w, 0.6)
        cbar_label = "Spectral Intensity $A(\mathbf{q}, \omega)^{0.6}$ (arb. units)"
        
    if vmax is not None:
        vmax_val = np.log10(vmax) if use_log else vmax
        cax = ax.pcolormesh(Q, W, A_plot, cmap="CMRmap", shading="gouraud", vmax=vmax_val)
    else:
        cax = ax.pcolormesh(Q, W, A_plot, cmap="CMRmap", shading="gouraud")
    
    #for m in range(N_modes):
    #    ax.plot(q_dist, harm_bands[:, m], color="cyan", linestyle="-", linewidth=0.5, alpha=0.3,
    #            label="Harmonic Bands" if m == 0 else "")
        
    for bd in boundary_dists:
        ax.axvline(x=bd, color="white", linestyle="-", alpha=0.3)
        
    ax.set_xticks(boundary_dists)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(q_dist[0], q_dist[-1])
    ax.set_ylim(0.0, 300)#np.max(w_grid))
    
    ax.set_ylabel(f"Frequency ({unit_label})", fontsize=13)
    ax.set_title("Phonon Spectral Function $A(\mathbf{q}, \omega)$ (Heatmap)", fontsize=14, fontweight="bold")
    
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "plot_spectral_function_jet.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved spectral function heatmap plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Plotting script for Phonon Self-Energy and MSD results.")
    parser.add_argument("--out-dir", type=str, default="output", help="Directory where data files are stored.")
    parser.add_argument("--labels", type=str, default="Gamma M K Gamma", help="Space-separated boundary q-point labels.")
    parser.add_argument("--log", action="store_true", help="Plot spectral function using logarithmic base-10 scale.")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for the colorbar (in raw intensity units, e.g. 1.0).")
    
    args = parser.parse_args()
    
    msd_file = os.path.join(args.out_dir, "MSD.dat")
    harm_file = os.path.join(args.out_dir, "phonon_dispersion_harmonic.dat")
    dressed_file = os.path.join(args.out_dir, "phonon_dispersion_dressed.dat")
    spec_file = os.path.join(args.out_dir, "spectral_function_A_q_w.dat")
    
    print(f"Starting plotting analysis in directory: {args.out_dir}\n")
    
    plot_msd(msd_file, args.out_dir)
    plot_phonon_dispersion(harm_file, dressed_file, args.labels, args.out_dir)
    plot_spectral_function(spec_file, harm_file, args.labels, args.out_dir, use_log=args.log, vmax=args.vmax)
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
