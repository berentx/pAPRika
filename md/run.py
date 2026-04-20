import math
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
    temperature = args.temp


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
            system.addForce(openmm.AndersenThermostat(temperature*unit.kelvin, 1/unit.picosecond))
            system.addForce(openmm.MonteCarloBarostat(1*unit.bar, temperature*unit.kelvin))

        # setup restraints
        restraint = openmm.CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        system.addForce(restraint)
        restraint.addGlobalParameter('k', 100.0*unit.kilojoules_per_mole/(unit.nanometer**2))
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')

        for atom in coords.topology.atoms():
            if atom.residue.name in ['MOL', 'LIG'] and atom.element.symbol != 'H':
                restraint.addParticle(atom.index, coords.positions[atom.index])

        # Integrator
        integrator = openmm.LangevinMiddleIntegrator(temperature*unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)

        # Simulation Object
        simulation = app.Simulation(coords.topology, system, integrator)
        simulation.context.setPositions(coords.positions)

        # Minimize Energy
        simulation.minimizeEnergy(tolerance=1.0*unit.kilojoules_per_mole/unit.nanometer, maxIterations=10000)

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
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.reporters.append(state_reporter)


        # MD steps
        simulation.step(int(1.0 / 0.002 * 1000 + 0.5)) # 1ns

        # Save final coordinates
        state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=enforcePBC)
        with open(os.path.join(folder, 'equilibration.pdb'), 'w') as file:
            app.PDBFile.writeFile(simulation.topology, state.getPositions(), file, keepIds=True)
        with open(os.path.join(folder, 'equilibration.rst'), 'w') as f:
            f.write(openmm.XmlSerializer.serialize(state))
        assert (Path(folder)/'equilibration.rst').exists()


    # production
    chunk_ns = args.chunk
    n_chunks = math.ceil(args.ns / chunk_ns)

    for window in window_list:
        folder = window

        for chunk_idx in range(1, n_chunks + 1):
            this_chunk_ns = min(chunk_ns, args.ns - (chunk_idx - 1) * chunk_ns)
            prefix = f'production.{chunk_idx:03d}'

            if Path(f'{folder}/{prefix}.rst').exists():
                continue

            rstfile = 'equilibration.rst' if chunk_idx == 1 else f'production.{chunk_idx-1:03d}.rst'
            if not Path(f'{folder}/{rstfile}').exists():
                logger.warning(f"Restart file {rstfile} not found in {folder}, stopping at chunk {chunk_idx}")
                break

            logger.info(f"Running production chunk {chunk_idx}/{n_chunks} in window {window}...")
            print(f"Running production chunk {chunk_idx}/{n_chunks} in window {window}...")

            # Load XML and input coordinates
            with open(os.path.join(folder, 'system.xml'), 'r') as file:
                system = openmm.XmlSerializer.deserialize(file.read())
            coords = app.PDBFile(os.path.join(folder, 'minimized.pdb'))

            if not args.implicit:
                system.addForce(openmm.AndersenThermostat(temperature*unit.kelvin, 1/unit.picosecond))
                system.addForce(openmm.MonteCarloBarostat(1*unit.bar, temperature*unit.kelvin))
                timestep = 4.0 * unit.femtoseconds
            else:
                timestep = 4.0 * unit.femtoseconds

            # Integrator
            integrator = openmm.LangevinIntegrator(temperature*unit.kelvin, 1.0 / unit.picoseconds, timestep)

            # Reporters
            dcd_reporter = app.DCDReporter(os.path.join(folder, f'{prefix}.dcd'), args.dcd_freq)
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
            simulation.step(int(this_chunk_ns * unit.nanoseconds / timestep + 0.5))

            # Save final coordinates
            state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=enforcePBC)
            with open(os.path.join(folder, f'{prefix}.pdb'), 'w') as file:
                app.PDBFile.writeFile(simulation.topology, state.getPositions(), file, keepIds=True)
            with open(os.path.join(folder, f'{prefix}.rst'), 'w') as f:
                f.write(openmm.XmlSerializer.serialize(state))
