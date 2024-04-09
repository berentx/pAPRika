from pathlib import Path
import logging
import os
import shutil

import openmm.unit as unit
import openmm.app as app
import openmm as openmm

from paprika.io import load_restraints
from paprika.restraints.restraints import create_window_list

logger = logging.getLogger("run")

def run_equilibration(folder, args, enforcePBC=True):
    if Path(f'{folder}/equilibration.rst').exists():
        return

    temp = 300 * unit.kelvin

    logger.info(f"Running minimization in window {folder}...")
    print(f"Running minimization in window {folder}...")

    # Load XML and input coordinates
    with open(os.path.join(folder, 'system.xml'), 'r') as file:
        system = openmm.XmlSerializer.deserialize(file.read())
    coords = app.PDBFile(os.path.join(folder, 'system.pdb'))

    if not args.implicit:
        system.addForce(openmm.AndersenThermostat(temp, 1/unit.picosecond))
        system.addForce(openmm.MonteCarloBarostat(1*unit.bar, temp))

    # Integrator
    integrator = openmm.LangevinMiddleIntegrator(temp, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)

    # Simulation Object
    simulation = app.Simulation(coords.topology, system, integrator)
    simulation.context.setPositions(coords.positions)

    # Minimize Energy
    simulation.minimizeEnergy(tolerance=1.0*unit.kilojoules_per_mole, maxIterations=20000)

    # Save final coordinates
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(os.path.join(folder, 'minimized.pdb'), 'w') as file:
        app.PDBFile.writeFile(simulation.topology, positions, file, keepIds=True)


    logger.info(f"Running equilibration in window {folder}...")
    print(f"Running equilibration in window {folder}...")

    # Reporters
    state_reporter = app.StateDataReporter(
        os.path.join(folder, 'equilibration.log'),
        5000,
        step=True,
        kineticEnergy=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
    )

    # Simulation Object
    simulation.context.setVelocitiesToTemperature(temp)
    simulation.reporters.append(state_reporter)

    # MD steps
    simulation.step(int(0.1 / 0.002 * 10000 + 0.5)) # 100ps

    # Save final coordinates
    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=enforcePBC)
    with open(os.path.join(folder, 'equilibration.pdb'), 'w') as file:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), file, keepIds=True)
    with open(os.path.join(folder, 'equilibration.rst'), 'w') as f:
        f.write(openmm.XmlSerializer.serialize(state))
    assert (Path(folder)/'equilibration.rst').exists()


def run_production(folder, args, enforcePBC=True):
    prefix = 'production'
    rstfile = 'equilibration.rst'

    if Path(f'{folder}/{prefix}.pdb').exists():
        if args.extend:
            n_cycle = len(list(Path(folder).glob(f'{prefix}.*.pdb')))
            prefix = f'production.{n_cycle+1}'
            if n_cycle > 0:
                rstfile = f'production.{n_cycle}.rst'
            else:
                rstfile = f'production.rst'
        else:
            return

    temp = 300 * unit.kelvin

    logger.info(f"Running production in window {folder}...")
    print(f"Running production in window {folder}...")

    # Load XML and input coordinates
    with open(os.path.join(folder, 'system.xml'), 'r') as file:
        system = openmm.XmlSerializer.deserialize(file.read())
    coords = app.PDBFile(os.path.join(folder, 'minimized.pdb'))

    if not args.implicit:
        system.addForce(openmm.AndersenThermostat(temp, 1/unit.picosecond))
        system.addForce(openmm.MonteCarloBarostat(1*unit.bar, temp))
        timestep = 4.0 * unit.femtoseconds

    else:
        timestep = 2.0 * unit.femtoseconds

    # Integrator
    integrator = openmm.LangevinIntegrator(temp, 1.0 / unit.picoseconds, timestep)

    # Reporters
    dcd_reporter = app.DCDReporter(os.path.join(folder, f'{prefix}.dcd'), 2500)
    state_reporter = app.StateDataReporter(
        os.path.join(folder, f'{prefix}.log'),
        1000,
        step=True,
        kineticEnergy=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
    )

    # Simulation Object
    simulation = app.Simulation(coords.topology, system, integrator)
    simulation.context.setPositions(coords.positions)
    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(state_reporter)
    with open(os.path.join(folder, rstfile), 'r') as f:
        simulation.context.setState(openmm.XmlSerializer.deserialize(f.read()))

    # MD steps
    simulation.step(int(args.ns * unit.nanoseconds / timestep + 0.5))

    # Save final coordinates
    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=enforcePBC)
    with open(os.path.join(folder, f'{prefix}.pdb'), 'w') as file:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), file, keepIds=True)
    with open(os.path.join(folder, f'{prefix}.rst'), 'w') as f:
        f.write(openmm.XmlSerializer.serialize(state))


def run(args):
    window_list = None

    try:
        host_guest_restraints = load_restraints(filepath="windows/restraints.json")
        guest_restraints = load_restraints(filepath="windows/guest_restraints.json")

        window_list = create_window_list(guest_restraints)

    except:
        pass

    enforcePBC = True
    if args.implicit:
        enforcePBC = False

    if args.equilibrate:
        for i, window in enumerate(window_list):
            folder = Path(os.path.join('windows', window))
            if i > 0:
                prev = window_list[i-1]
                prev_folder = Path(os.path.join('windows', prev))
                shutil.copy(prev_folder/'equilibration.pdb', folder/'system.pdb')
            run_equilibration(folder, args, enforcePBC)

    elif args.final:
        for i, window in enumerate(window_list):
            folder = Path(os.path.join('windows', window))
            pdb = folder/'production.pdb'
            if pdb.exists():
                continue

            prev = window_list[i-1]
            prev_folder = Path(os.path.join('windows', prev))
            for p in folder.glob("equilibration.*"):
                p.unlink()
            shutil.copy(prev_folder/'production.pdb', folder/'system.pdb')
            run_equilibration(folder, args, enforcePBC)
            run_production(folder, args, enforcePBC)

    elif window_list is not None:
        for window in window_list:
            folder = os.path.join('windows', window)
    
            if args.window != 'all' and args.window != window:
                continue
    
            run_equilibration(folder, args, enforcePBC)
            run_production(folder, args, enforcePBC)
    
    else:
        folder = Path('.')
        run_equilibration(folder, args, enforcePBC)
        run_production(folder, args, enforcePBC)

