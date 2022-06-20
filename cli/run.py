from pathlib import Path
import logging
import os

import openmm.unit as unit
import openmm.app as app
import openmm as openmm

from paprika.io import load_restraints
from paprika.restraints.restraints import create_window_list

logger = logging.getLogger("run")


def run(args):
    host_guest_restraints = load_restraints(filepath="windows/restraints.json")
    guest_restraints = load_restraints(filepath="windows/guest_restraints.json")

    window_list = create_window_list(guest_restraints)

    for window in window_list:
        folder = os.path.join('windows', window)

        if Path(f'{folder}/minimized.pdb').exists():
            continue

        logging.info(f"Running minimization in window {window}...")
    
        # Load XML and input coordinates
        with open(os.path.join(folder, 'system.xml'), 'r') as file:
            system = openmm.XmlSerializer.deserialize(file.read())
        coords = app.PDBFile(os.path.join(folder, 'system.pdb'))
    
        # Integrator
        integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtoseconds)
    
        # Simulation Object
        simulation = app.Simulation(coords.topology, system, integrator)
        simulation.context.setPositions(coords.positions)

        # Minimize Energy
        simulation.minimizeEnergy(tolerance=1.0*unit.kilojoules_per_mole, maxIterations=10000)

        # Save final coordinates
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(os.path.join(folder, 'minimized.pdb'), 'w') as file:
            app.PDBFile.writeFile(simulation.topology, positions, file, keepIds=True)

    # equilibration

    for window in window_list:
        folder = os.path.join('windows', window)

        if Path(f'{folder}/equilibration.pdb').exists():
            continue

        logging.info(f"Running equilibration in window {window}...")

        # Load XML and input coordinates
        with open(os.path.join(folder, 'system.xml'), 'r') as file:
            system = openmm.XmlSerializer.deserialize(file.read())
        coords = app.PDBFile(os.path.join(folder, 'minimized.pdb'))
    
        # Integrator
        integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)

        # Reporters
        state_reporter = app.StateDataReporter(
            os.path.join(folder, 'equilibration.log'),
            500,
            step=True,
            kineticEnergy=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
        )
    
        # Simulation Object
        simulation = app.Simulation(coords.topology, system, integrator)
        simulation.context.setPositions(coords.positions)
        simulation.reporters.append(state_reporter)
    
        # MD steps
        simulation.step(1 / 0.002 * 1000) # 1ns
    
        # Save final coordinates
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(os.path.join(folder, 'equilibration.pdb'), 'w') as file:
            app.PDBFile.writeFile(simulation.topology, positions, file, keepIds=True)

    # production

    for window in window_list:
        folder = os.path.join('windows', window)

        if Path(f'{folder}/production.pdb').exists():
            continue

        logging.info(f"Running production in window {window}...")

        # Load XML and input coordinates
        with open(os.path.join(folder, 'system.xml'), 'r') as file:
            system = openmm.XmlSerializer.deserialize(file.read())
        coords = app.PDBFile(os.path.join(folder, 'equilibration.pdb'))
    
        # Integrator
        integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)

        # Reporters
        dcd_reporter = app.DCDReporter(os.path.join(folder, 'production.dcd'), 500)
        state_reporter = app.StateDataReporter(
            os.path.join(folder, 'production.log'),
            500,
            step=True,
            kineticEnergy=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
        )
    
        # Simulation Object
        simulation = app.Simulation(coords.topology, system, integrator)
        simulation.context.setPositions(coords.positions)
        simulation.reporters.append(dcd_reporter)
        simulation.reporters.append(state_reporter)
    
        # MD steps
        simulation.step(args.ns / 0.004 * 1000)
    
        # Save final coordinates
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(os.path.join(folder, 'production.pdb'), 'w') as file:
            app.PDBFile.writeFile(simulation.topology, positions, file, keepIds=True)

