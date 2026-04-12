#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import subprocess
from sys import stdout

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb

from openmm import app
import openmm as mm
from openmm import unit
from pdbfixer import PDBFixer

import mdtraj as md
import pytraj as pt
from pytraj import matrix

# ---------- Inputs / parameters ----------
temperature = float(os.getenv("temperature", "298"))           # Kelvin
pressure = float(os.getenv("pressure", "1"))                   # bar
timestep = float(os.getenv("timestep", "2"))                   # femtoseconds
force_constant = float(os.getenv("forceConstant", "1000"))     # kJ/mol/nm^2
minimization_steps = int(os.getenv("minimizationSteps", "10000"))
equilibration_time = float(os.getenv("equilibrationTime", "0.2"))  # ns
production_time = float(os.getenv("productionTime", "2"))          # ns
equil_traj_freq = int(os.getenv("equilTrajFreq", "1000"))
prod_traj_freq = int(os.getenv("prodTrajFreq", "1000"))
padding_nm = float(os.getenv("paddingNm", "1.0"))              # solvation padding
ionic_strength = float(os.getenv("ionicStrength", "0.15"))     # molar
ph = float(os.getenv("pH", "7.0"))

equilibration_steps = int(equilibration_time * 1_000_000 / timestep)
production_steps = int(production_time * 1_000_000 / timestep)

os.makedirs('out', exist_ok=True)
os.makedirs('out/analysis', exist_ok=True)

# ---------- Locate input protein ----------
candidates = sorted(glob.glob('inputs/*.pdb') + glob.glob('inputs/*.cif'))
if not candidates:
    print("ERROR: no .pdb or .cif file found in inputs/")
    sys.exit(1)
input_protein = candidates[0]
print(f"Input protein: {input_protein}")

# ---------- Prepare system with PDBFixer ----------
print("Running PDBFixer...")
fixer = PDBFixer(filename=input_protein)
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.removeHeterogens(keepWater=False)
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(ph)

with open('out/protein_fixed.pdb', 'w') as f:
    app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

# ---------- Parameterize with tleap (Amber ff14SB + TIP3P) ----------
print("Building topology with tleap...")
tleap_in = f"""source leaprc.protein.ff14SB
source leaprc.water.tip3p
mol = loadpdb out/protein_fixed.pdb
solvateBox mol TIP3PBOX {padding_nm * 10.0}
addIonsRand mol Na+ 0
addIonsRand mol Cl- 0
saveAmberParm mol out/system.prmtop out/system.inpcrd
savepdb mol out/system.pdb
quit
"""
with open('out/tleap.in', 'w') as f:
    f.write(tleap_in)
subprocess.run(['tleap', '-f', 'out/tleap.in'], check=True)

# ---------- Load into OpenMM ----------
prmtop = app.AmberPrmtopFile('out/system.prmtop')
inpcrd = app.AmberInpcrdFile('out/system.inpcrd')

system = prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0*unit.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
    ewaldErrorTolerance=5e-4,
)

nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
nb.setUseDispersionCorrection(True)
nb.setUseSwitchingFunction(True)
nb.setSwitchingDistance(0.9*unit.nanometer)

baro = mm.MonteCarloBarostat(pressure*unit.bar, temperature*unit.kelvin, 25)
system.addForce(baro)

# Positional restraints on protein heavy atoms
k0 = force_constant
rest = mm.CustomExternalForce('0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2')
rest.addGlobalParameter('k', k0)
rest.addPerParticleParameter('x0')
rest.addPerParticleParameter('y0')
rest.addPerParticleParameter('z0')

exclude_res = {'WAT', 'HOH', 'Na+', 'Cl-', 'K+', 'K', 'MG', 'MG2', 'CA', 'CA2'}
for i, atom in enumerate(prmtop.topology.atoms()):
    el = atom.element.symbol if atom.element is not None else ''
    if el != 'H' and atom.residue.name not in exclude_res:
        xyz = inpcrd.positions[i].value_in_unit(unit.nanometer)
        rest.addParticle(i, xyz)
system.addForce(rest)

integrator = mm.LangevinIntegrator(
    temperature*unit.kelvin,
    1.0/unit.picoseconds,
    timestep*unit.femtoseconds,
)
integrator.setConstraintTolerance(1e-5)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

reduced_k = force_constant / 10.0

# ---------- Minimization ----------
print('Minimizing...')
simulation.minimizeEnergy(
    tolerance=10 * unit.kilojoules_per_mole / unit.nanometer,
    maxIterations=minimization_steps,
)

# ---------- Equilibration (half full restraints, half reduced) ----------
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)
simulation.currentStep = 0
simulation.reporters.clear()
simulation.reporters.append(app.XTCReporter(
    'out/traj_equil.xtc', equil_traj_freq, enforcePeriodicBox=True
))
simulation.reporters.append(app.StateDataReporter(
    stdout, reportInterval=equil_traj_freq, step=True,
    potentialEnergy=True, temperature=True, progress=True,
    remainingTime=True, totalSteps=equilibration_steps, separator='\t'
))

half1 = equilibration_steps // 2
half2 = equilibration_steps - half1
simulation.step(half1)
simulation.context.setParameter('k', reduced_k)
simulation.step(half2)

# ---------- Production (no restraints) ----------
print('Running Production...')
simulation.context.setParameter('k', 0.0)
simulation.currentStep = 0
simulation.reporters.clear()
simulation.reporters.append(app.XTCReporter(
    'out/traj_prod_raw.xtc', prod_traj_freq, enforcePeriodicBox=True
))
simulation.reporters.append(app.StateDataReporter(
    stdout, reportInterval=prod_traj_freq, step=True,
    potentialEnergy=True, temperature=True, progress=True,
    remainingTime=True, totalSteps=production_steps, separator='\t'
))
simulation.step(production_steps)
print('Production complete.')

# ---------- PBC correction, alignment, downsample ----------
print('Post-processing trajectory...')
topology = md.load_prmtop('out/system.prmtop')
ca_indices = topology.select('protein and name CA')
solute_indices = topology.select('protein')

traj_dt = int(os.getenv('analysisStride', '10'))

if os.path.exists('out/traj_equil.xtc'):
    te = md.load('out/traj_equil.xtc', top='out/system.prmtop')
    if te.n_frames > 0:
        te.make_molecules_whole(inplace=True)
        te.image_molecules(inplace=True)
        if len(ca_indices) > 0:
            te.superpose(te, frame=0, atom_indices=ca_indices)
        te.center_coordinates()
        te.save_xtc('out/traj_equil.xtc')
    del te

tp = md.load('out/traj_prod_raw.xtc', top='out/system.prmtop')
tp.make_molecules_whole(inplace=True)
tp.image_molecules(inplace=True)
if len(ca_indices) > 0:
    tp.superpose(tp, frame=0, atom_indices=ca_indices)
tp.center_coordinates()
tp.save_xtc('out/traj_prod_full.xtc')

# first frame + solute-only topology
tp[0].save_pdb('out/first_frame.pdb')
tp[0].atom_slice(solute_indices).save_pdb('out/topology_no_water.pdb')

# stripped (no water) full-length trajectory for downstream use
tp.atom_slice(solute_indices).save_xtc('out/traj_prod_no_water.xtc')

# downsampled analysis trajectory
tp_ds = tp[::traj_dt]
tp_ds.save_xtc('out/traj_prod.xtc')
del tp, tp_ds
os.remove('out/traj_prod_raw.xtc')

# ---------- Analysis ----------
top = 'out/system.prmtop'
trajfile = 'out/traj_prod.xtc'
outdir = 'out/analysis'
frame_ps = prod_traj_freq * timestep / 1000.0 * traj_dt

t = pt.iterload(trajfile, top)
time_ns = np.arange(len(t)) * (frame_ps / 1000.0)

# RMSD
try:
    rmsd = pt.rmsd(t, ref=0, mask='@CA')
    pd.DataFrame({'time_ns': time_ns, 'rmsd_A': rmsd}).to_csv(f'{outdir}/rmsd_ca.csv', index=False)
    plt.figure()
    plt.plot(time_ns, rmsd, alpha=0.9, lw=1.0)
    plt.title('Protein Cα RMSD vs Time\n(superposed to frame 0 on Cα; units in Å)')
    plt.xlabel('Time (ns)'); plt.ylabel('RMSD (Å)')
    plt.tight_layout(); plt.savefig(f'{outdir}/rmsd_ca.png', dpi=600)

    plt.figure()
    sb.kdeplot(rmsd, color="blue", fill=True, alpha=0.25, linewidth=0.8)
    plt.title('Distribution of Protein Cα RMSD across time\n(superposed to frame 0 on Cα; units in Å)')
    plt.xlabel('RMSD (Å)'); plt.yticks([]); plt.ylabel('')
    for spine in ['top', 'right', 'left']:
        plt.gca().spines[spine].set_visible(False)
    plt.tight_layout(); plt.savefig(f'{outdir}/rmsd_dist.png', dpi=600)
    print("RMSD analysis completed")
except Exception as e:
    print(f"Error in RMSD analysis: {e}")

# Radius of gyration
try:
    rg = pt.radgyr(t, mask='@CA')
    pd.DataFrame({'time_ns': time_ns, 'rg_A': rg}).to_csv(f'{outdir}/radius_gyration.csv', index=False)
    plt.figure()
    plt.plot(time_ns, rg, alpha=0.9, color='green', lw=1.0)
    plt.title('Protein Cα Radius of Gyration vs Time\n(superposed to frame 0 on Cα; units in Å)')
    plt.xlabel('Time (ns)'); plt.ylabel('Radius of gyration (Å)')
    plt.tight_layout(); plt.savefig(f'{outdir}/radius_gyration.png', dpi=600)

    plt.figure()
    sb.kdeplot(rg, color="green", fill=True, alpha=0.25, linewidth=0.8)
    plt.title('Distribution of Protein Cα Radius of Gyration across time\n(superposed to frame 0 on Cα; units in Å)')
    plt.xlabel('Radius of gyration (Å)'); plt.yticks([]); plt.ylabel('')
    for spine in ['top', 'right', 'left']:
        plt.gca().spines[spine].set_visible(False)
    plt.tight_layout(); plt.savefig(f'{outdir}/radius_gyration_dist.png', dpi=600)
    print("Radius of gyration analysis completed")
except Exception as e:
    print(f"Error in radius of gyration analysis: {e}")

# RMSF
try:
    rmsf = pt.rmsf(t, '@CA')
    topology_pt = pt.load_topology(top)
    residue_numbers = [topology_pt.atom(int(idx)).resid + 1 for idx in rmsf[:, 0]]
    pd.DataFrame({'residue_number': residue_numbers, 'RMSF_A': rmsf[:, 1]}).to_csv(f'{outdir}/rmsf_ca.csv', index=False)
    plt.figure()
    plt.plot(residue_numbers, rmsf[:, 1], color='red', lw=1.0)
    plt.title('Per-Residue RMSF (Cα)\n(aligned on Cα; units in Å)')
    plt.xlabel('Residue index'); plt.ylabel('RMSF (Å)')
    plt.xlim(0, len(rmsf[:-1]))
    plt.tight_layout(); plt.savefig(f'{outdir}/rmsf_ca.png', dpi=600)
    print("RMSF analysis completed")
except Exception as e:
    print(f"Error in RMSF analysis: {e}")

# 2D RMSD
try:
    step = int(os.getenv('stepSize', '5'))
    mat2d = pt.pairwise_rmsd(t[::step], mask='@CA')
    pd.DataFrame(mat2d).to_csv(f'{outdir}/rmsd2d.csv', index=False)
    plt.figure()
    plt.imshow(mat2d, cmap='PRGn', origin='lower', interpolation='bicubic', aspect='auto')
    plt.title('2D RMSD of Protein Cα\n(entry [i,j] = RMSD between frames i and j; Å)')
    plt.xlabel('Frame index'); plt.ylabel('Frame index')
    cbar = plt.colorbar(); cbar.set_label('RMSD (Å)')
    plt.tight_layout(); plt.savefig(f'{outdir}/2D_rmsd.png', dpi=600)
    print("2D RMSD analysis completed")
except Exception as e:
    print(f"Error in 2D RMSD analysis: {e}")

# Cross-correlation
try:
    t_ca = pt.align(t, mask='@CA', ref=0)
    cc_mat = matrix.correl(t_ca, '@CA')
    pd.DataFrame(cc_mat).to_csv(f'{outdir}/cross_correlation.csv', index=False)
    plt.figure()
    plt.imshow(cc_mat, cmap='PiYG_r', vmin=-1, vmax=1, interpolation='bicubic',
               origin='lower', aspect='auto')
    plt.title('Residue-wise Motion Cross-Correlation (Cα)\nPearson CC_ij, aligned on Cα; −1 (anti) … +1 (corr)')
    plt.xlabel('Residue index'); plt.ylabel('Residue index')
    cbar = plt.colorbar(); cbar.set_label('Correlation $CC_{ij}$')
    plt.tight_layout(); plt.savefig(f'{outdir}/cross_correlation.png', dpi=600)
    print("Cross-correlation analysis completed")
except Exception as e:
    print(f"Error in cross-correlation analysis: {e}")

print(f"All done. Figures and CSVs in: {outdir}")
