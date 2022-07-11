from pathlib import Path
import logging
import os

def run(args):
    import openmm.unit as unit
    import openmm.app as app
    import openmm as openmm
    
    from paprika.io import load_restraints
    from paprika.restraints.restraints import create_window_list
    
    logger = logging.getLogger("run")


    window_list = [d for d in Path('windows').glob("*") if d.is_dir()]

    enforcePBC = True
    if args.implicit:
        enforcePBC = False

    for window in window_list:
        folder = window

        if Path(f'{folder}/equilibration.rst').exists():
            continue

        logger.info(f"Running minimization in window {window}...")
        print(f"Running minimization in window {window}...")
    
        # Load XML and input coordinates
        with open(os.path.join(folder, 'system.xml'), 'r') as file:
            system = openmm.XmlSerializer.deserialize(file.read())
        coords = app.PDBFile(os.path.join(folder, 'system.pdb'))

        if not args.implicit:
            system.addForce(openmm.AndersenThermostat(300*unit.kelvin, 1/unit.picosecond))
            system.addForce(openmm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))

        # Integrator
        integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
    
        # Simulation Object
        simulation = app.Simulation(coords.topology, system, integrator)
        simulation.context.setPositions(coords.positions)

        # Minimize Energy
        simulation.minimizeEnergy(tolerance=1.0*unit.kilojoules_per_mole, maxIterations=10000)

        # Save final coordinates
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(os.path.join(folder, 'minimized.pdb'), 'w') as file:
            app.PDBFile.writeFile(simulation.topology, positions, file, keepIds=True)


        logger.info(f"Running equilibration in window {window}...")
        print(f"Running equilibration in window {window}...")

        # Reporters
        state_reporter = app.StateDataReporter(
            os.path.join(folder, 'equilibration.log'),
            1000,
            step=True,
            kineticEnergy=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
        )
    
        # Simulation Object
        simulation.context.setVelocitiesToTemperature(300)
        simulation.reporters.append(state_reporter)
    
        # MD steps
        simulation.step(int(0.1 / 0.002 * 1000 + 0.5)) # 100ps
    
        # Save final coordinates
        state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=enforcePBC)
        with open(os.path.join(folder, 'equilibration.pdb'), 'w') as file:
            app.PDBFile.writeFile(simulation.topology, state.getPositions(), file, keepIds=True)
        with open(os.path.join(folder, 'equilibration.rst'), 'w') as f:
            f.write(openmm.XmlSerializer.serialize(state))
        assert (Path(folder)/'equilibration.rst').exists()


    # production
    for window in window_list:
        folder = window

        if Path(f'{folder}/production.pdb').exists():
            continue

        logger.info(f"Running production in window {window}...")
        print(f"Running production in window {window}...")

        # Load XML and input coordinates
        with open(os.path.join(folder, 'system.xml'), 'r') as file:
            system = openmm.XmlSerializer.deserialize(file.read())
        coords = app.PDBFile(os.path.join(folder, 'minimized.pdb'))

        if not args.implicit:
            system.addForce(openmm.AndersenThermostat(300*unit.kelvin, 1/unit.picosecond))
            system.addForce(openmm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
            timestep = 4.0 * unit.femtoseconds

        else:
            timestep = 2.0 * unit.femtoseconds

        # Integrator
        integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, timestep)

        # Reporters
        dcd_reporter = app.DCDReporter(os.path.join(folder, 'production.dcd'), 5000)
        state_reporter = app.StateDataReporter(
            os.path.join(folder, 'production.log'),
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
        with open(os.path.join(folder, 'equilibration.rst'), 'r') as f:
            simulation.context.setState(openmm.XmlSerializer.deserialize(f.read()))
    
        # MD steps
        simulation.step(int(args.ns * unit.nanoseconds / timestep + 0.5))
    
        # Save final coordinates
        state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=enforcePBC)
        with open(os.path.join(folder, 'production.pdb'), 'w') as file:
            app.PDBFile.writeFile(simulation.topology, state.getPositions(), file, keepIds=True)
        with open(os.path.join(folder, 'production.rst'), 'w') as f:
            f.write(openmm.XmlSerializer.serialize(state))

