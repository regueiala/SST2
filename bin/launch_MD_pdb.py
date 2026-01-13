import argparse
import numpy as np
import time
import os
import sys
import logging
import pandas as pd

from openmm.app import PDBFile, PDBxFile, ForceField, Simulation
from openmm import LangevinMiddleIntegrator, unit, Platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

from SST2.st import ST, run_st
import SST2.tools as tools
import pdb_numpy

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add sys.sdout as handler
logger.addHandler(logging.StreamHandler(sys.stdout))


def parser_input():
    # Parse arguments :
    parser = argparse.ArgumentParser(
        description="Simulate a peptide starting from a linear conformation."
    )
    parser.add_argument(
        "-pdb",
        action="store",
        dest="pdb",
        help="Input PDB file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        action="store",
        dest="name",
        help="Output file name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-dir",
        action="store",
        dest="out_dir",
        help="Output directory for intermediate files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-pad",
        action="store",
        dest="pad",
        help="Box padding, default=1.5 nm",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "-eq_time_expl",
        action="store",
        dest="eq_time_expl",
        help="Explicit Solvent Equilibration time, default=10 (ns)",
        type=float,
        default=10,
    )
    parser.add_argument(
        "-time",
        action="store",
        dest="time",
        help="Simulation time, default=10.000 (ns)",
        type=float,
        default=10000,
    )
    parser.add_argument(
        "-hmr",
        action="store",
        dest="hmr",
        help="Hydrogen mass repartition, default=3.0 a.m.u.",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "-k_rest",
        action="store",
        dest="k_rest",
        help="Calpha restraint k, default=100.0 KJ.mol-1.nm-2",
        type=float,
        default=100.0,
    )
    return parser


if __name__ == "__main__":
    my_parser = parser_input()
    args = my_parser.parse_args()
    logger.info(args)

    OUT_PATH = args.out_dir
    name = args.name

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    tools.prepare_pdb(args.pdb, f"{OUT_PATH}/{name}_fixed.cif", pH=7.0, overwrite=False)

    # forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    forcefield_files = ["amber14-all.xml", "amber14/tip3p.xml"]
    forcefield = ForceField(*forcefield_files)

    tools.create_water_box(
        f"{OUT_PATH}/{name}_fixed.cif",
        f"{OUT_PATH}/{name}_water.cif",
        pad=args.pad,
        forcefield=forcefield,
        overwrite=False,
    )

    ###########################
    ### BASIC EQUILIBRATION ###
    ###########################

    dt = 4 * unit.femtosecond
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picoseconds
    hydrogenMass = args.hmr * unit.amu
    rigidWater = True
    ewaldErrorTolerance = 0.0005
    nsteps = int(np.ceil(args.eq_time_expl * unit.nanoseconds / dt))

    cif = PDBxFile(f"{OUT_PATH}/{name}_water.cif")
    PDBFile.writeFile(
        cif.topology, cif.positions, open(f"{OUT_PATH}/{name}_water.pdb", "w"), True
    )

    integrator = LangevinMiddleIntegrator(temperature, friction, dt)

    system = tools.create_sim_system(
        cif,
        forcefield=forcefield,
        temp=temperature,
        h_mass=args.hmr,
        base_force_group=1,
    )

    # Add position restraints on CA atoms
    CA_indices = [int(i.index) for i in cif.topology.atoms() if i.name in ["CA"]]

    logger.info("- Add position restraints")

    restraint = tools.add_pos_restr(system, CA_indices, cif, k_rest=args.k_rest)

    # Simulation Options
    platform = Platform.getPlatformByName("CUDA")
    # platform = Platform.getPlatformByName('OpenCL')
    platformProperties = {"Precision": "single"}

    simulation = Simulation(
        cif.topology, system, integrator, platform, platformProperties
    )
    simulation.context.setPositions(cif.positions)

    logger.info(f"- Minimize system")

    tools.minimize(
        simulation,
        f"{OUT_PATH}/{name}_em_water.cif",
        cif.topology,
        maxIterations=10000,
        overwrite=False,
    )

    simulation.context.setVelocitiesToTemperature(temperature)

    save_step_log = 10000
    save_step_dcd = 10000
    tot_steps = int(np.ceil(args.eq_time_expl * unit.nanoseconds / dt))

    logger.info(f"- Launch equilibration")
    tools.simulate(
        simulation,
        cif.topology,
        tot_steps=tot_steps,
        dt=dt,
        generic_name=f"{OUT_PATH}/{name}_explicit_equi",
        save_step_log=save_step_log,
        save_step_dcd=save_step_dcd,
    )

    simulation.context.setParameter("k", 0)

    logger.info(f"- Launch production")
    tot_steps = int(np.ceil(args.time * unit.nanoseconds / dt))

    tools.simulate(
        simulation,
        cif.topology,
        tot_steps=tot_steps,
        dt=dt,
        generic_name=f"{OUT_PATH}/{name}_explicit_prod",
        save_step_log=save_step_log,
        save_step_dcd=save_step_dcd,
    )
