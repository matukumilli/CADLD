import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
#python3.8 plot-spec_Q.py
SAgAg = np.loadtxt('pair_spec_Gamma_Ag-Ag.dat')
STeTe = np.loadtxt('pair_spec_Gamma_Te-Te.dat')
SSS = np.loadtxt('pair_spec_Gamma_S-S.dat')

SAgTe = np.loadtxt('pair_spec_Gamma_Ag-Te.dat')
SAgS = np.loadtxt('pair_spec_Gamma_Ag-S.dat')
SSTe = np.loadtxt('pair_spec_Gamma_S-Te.dat')

linewidth=2
plt.figure(figsize=(7, 5))
plt.plot(SAgAg[:,0], SAgAg[:, 1], color='blue', label="Ag-Ag", linewidth=1.0)
plt.plot(STeTe[:,0], STeTe[:, 1], color='crimson', label="Te-Te", linewidth=1.0)
plt.plot(SSS[:,0], SSS[:, 1], color='orange', label="S-S", linewidth=1.0)

plt.plot(SAgTe[:,0], SAgTe[:, 1], '--', color='dodgerblue', label="Ag-Te", linewidth=1.0)
plt.plot(SAgS[:,0], SAgS[:, 1], '--', color='green', label="Ag-S", linewidth=1.0)
plt.plot(SSTe[:,0], SSTe[:, 1], '--', color='yellow', label="S-Te", linewidth=1.0)

plt.xlim(0,300)
plt.xlabel("$\omega$ (cm$^{-1}$)", fontsize=12)
plt.ylabel("$A(q,\omega)$ (arb. units)", fontsize=12)
plt.title("Decomposed phonon spectral functions at $\Gamma$", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

out_dir='./'
plot_path = os.path.join(out_dir, "plot_ph-spectra_Gamma.png")
plt.savefig(plot_path, dpi=300)
plt.close()

SAgAg = np.loadtxt('pair_spec_M_Ag-Ag.dat')
STeTe = np.loadtxt('pair_spec_M_Te-Te.dat')
SSS = np.loadtxt('pair_spec_M_S-S.dat')

SAgTe = np.loadtxt('pair_spec_M_Ag-Te.dat')
SAgS = np.loadtxt('pair_spec_M_Ag-S.dat')
SSTe = np.loadtxt('pair_spec_M_S-Te.dat')

linewidth=2
plt.figure(figsize=(7, 5))
plt.plot(SAgAg[:,0], SAgAg[:, 1], color='blue', label="Ag-Ag", linewidth=1.0)
plt.plot(STeTe[:,0], STeTe[:, 1], color='crimson', label="Te-Te", linewidth=1.0)
plt.plot(SSS[:,0], SSS[:, 1], color='orange', label="S-S", linewidth=1.0)

plt.plot(SAgTe[:,0], SAgTe[:, 1], '--', color='dodgerblue', label="Ag-Te", linewidth=1.0)
plt.plot(SAgS[:,0], SAgS[:, 1], '--', color='green', label="Ag-S", linewidth=1.0)
plt.plot(SSTe[:,0], SSTe[:, 1], '--', color='yellow', label="S-Te", linewidth=1.0)

plt.xlim(0,300)
plt.xlabel("$\omega$ (cm$^{-1}$)", fontsize=12)
plt.ylabel("$A(q,\omega)$ (arb. units)", fontsize=12)
plt.title("Decomposed phonon spectral functions at $M$", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

out_dir='./'
plot_path = os.path.join(out_dir, "plot_ph-spectra_M.png")
plt.savefig(plot_path, dpi=300)
plt.close()

SAgAg = np.loadtxt('pair_spec_K_Ag-Ag.dat')
STeTe = np.loadtxt('pair_spec_K_Te-Te.dat')
SSS = np.loadtxt('pair_spec_K_S-S.dat')

SAgTe = np.loadtxt('pair_spec_K_Ag-Te.dat')
SAgS = np.loadtxt('pair_spec_K_Ag-S.dat')
SSTe = np.loadtxt('pair_spec_K_S-Te.dat')

linewidth=2
plt.figure(figsize=(7, 5))
plt.plot(SAgAg[:,0], SAgAg[:, 1], color='blue', label="Ag-Ag", linewidth=1.0)
plt.plot(STeTe[:,0], STeTe[:, 1], color='crimson', label="Te-Te", linewidth=1.0)
plt.plot(SSS[:,0], SSS[:, 1], color='orange', label="S-S", linewidth=1.0)

plt.plot(SAgTe[:,0], SAgTe[:, 1], '--', color='dodgerblue', label="Ag-Te", linewidth=1.0)
plt.plot(SAgS[:,0], SAgS[:, 1], '--', color='green', label="Ag-S", linewidth=1.0)
plt.plot(SSTe[:,0], SSTe[:, 1], '--', color='yellow', label="S-Te", linewidth=1.0)

plt.xlim(0,300)
plt.xlabel("$\omega$ (cm$^{-1}$)", fontsize=12)
plt.ylabel("$A(q,\omega)$ (arb. units)", fontsize=12)
plt.title("Decomposed phonon spectral functions at $K$", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

out_dir='./'
plot_path = os.path.join(out_dir, "plot_ph-spectra_K.png")
plt.savefig(plot_path, dpi=300)
plt.close()
