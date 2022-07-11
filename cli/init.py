import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np
import openmm.unit as unit
import openmm.app as app
import openmm as openmm
import parmed as pmd
from rdkit import Chem

from paprika.build import align
from paprika.build import dummy
from paprika.build.system.utils import PBCBox
from paprika import restraints
from paprika.io import save_restraints
from paprika.restraints.restraints import create_window_list
from paprika.restraints.amber import amber_restraint_line
from paprika.restraints.utils import parse_window
from paprika.restraints.openmm import apply_positional_restraints, apply_dat_restraint

from restraints import *


logger = logging.getLogger("init")


def antechamber(input, format, output_path, pl=-1, overwrite=False):
    output_mol2 = f'{input.stem}.gaff2.mol2'
    output_frcmod = f'{input.stem}.frcmod'
    input_path = input.resolve()

    cwd = os.getcwd()
    if not output_path.exists():
        output_path.mkdir(parents=True)

    os.chdir(output_path)

    if not Path(output_mol2).exists() or overwrite:
        cmd = ['antechamber', '-fi', 'mol2', '-fo', 'mol2', '-i', str(input_path),
               '-o', f'{output_mol2}', '-c', 'bcc', '-s', '2', '-at', 'gaff2']
        if pl > 0:
            cmd += ['-pl', f'{pl:d}']
        os.system(' '.join(cmd))
        assert Path(output_mol2).exists()

    if not Path(output_frcmod).exists() or overwrite:
        cmd = ['parmchk2', '-i', str(output_mol2), '-f', 'mol2', '-o', str(output_frcmod),
               '-s', 'gaff2']
        os.system(' '.join(cmd))
        assert Path(output_frcmod).exists()

    os.chdir(cwd)


def build(info, complex_pdb, system_path, system_top, system_rst, system_pdb, dummy=False):
    from paprika.build.system import TLeap

    system = TLeap()
    system.output_path = system_path
    system.pbc_type = None
    system.neutralize = False

    host_dir = Path(info['host']['par_path']).resolve()
    host_name = info['host']['name']
    
    system.template_lines = [
        "source leaprc.gaff2",
        "source leaprc.lipid21",
    ]

    system.template_lines += [
        f"loadamberparams {host_dir}/{host_name}.frcmod",
        f"{host_name.upper()} = loadmol2 {host_dir}/{host_name}.gaff2.mol2",
    ]

    if info['guest']['par_path'] != 'None':
        guest_dir = Path(info['guest']['par_path']).resolve()
        guest_name = info['guest']['name']
        system.template_lines += [
            f"loadamberparams {guest_dir}/{guest_name}.frcmod",
            f"{guest_name.upper()} = loadmol2 {guest_dir}/{guest_name}.gaff2.mol2",
        ]

    if dummy:
        system.template_lines += [
            "loadamberparams dummy.frcmod",
            "DM1 = loadmol2 dm1.mol2",
            "DM2 = loadmol2 dm2.mol2",
            "DM3 = loadmol2 dm3.mol2",
        ]

    system.template_lines += [
        f"model = loadpdb {str(complex_pdb.resolve()).strip()}",
        "check model",
        f"savepdb model {system_pdb.name}",
        f"saveamberparm model {system_top.name} {system_rst.name}"
    ]

    system.build(clean_files=False)
    assert Path(system_top).exists()
    assert Path(system_rst).exists()
    assert Path(system_pdb).exists()


def solvate(info, complex_pdb, complex_dir, system_path, system_prefix, num_waters, ion_conc, dummy=True):
    from paprika.build.system import TLeap

    system = TLeap()
    system.output_path = system_path
    system.pbc_type = PBCBox.rectangular
    system.neutralize = True
    system.output_prefix = system_prefix
    system.target_waters = num_waters
    #system.set_water_model("tip3p", model_type="force-balance")
    system.set_water_model("tip3p")

    n_ions = int(ion_conc / 55.0 * system.target_waters)
    system.add_ions = ["Na+", n_ions, "Cl-", n_ions]

    host_dir = Path(info['host']['par_path']).resolve()
    host_name = info['host']['name']
    guest_dir = Path(info['guest']['par_path']).resolve()
    guest_name = info['guest']['name']
    complex_dir = Path(complex_dir).resolve()
    complex_pdb = Path(complex_pdb)
    
    system.template_lines = [
        "source leaprc.gaff2",
        "source leaprc.lipid21",
        #"source leaprc.water.tip3p",
    ]

    system.template_lines += [
        f"loadamberparams {host_dir}/{host_name}.frcmod",
        f"{host_name.upper()} = loadmol2 {host_dir}/{host_name}.gaff2.mol2",
    ]

    if info['guest']['par_path'] != 'None':
        guest_dir = Path(info['guest']['par_path']).resolve()
        guest_name = info['guest']['name']
        system.template_lines += [
            f"loadamberparams {guest_dir}/{guest_name}.frcmod",
            f"{guest_name.upper()} = loadmol2 {guest_dir}/{guest_name}.gaff2.mol2",
        ]

    if dummy:
        system.template_lines += [
            f"loadamberparams {complex_dir}/dummy.frcmod",
            f"DM1 = loadmol2 {complex_dir}/dm1.mol2",
            f"DM2 = loadmol2 {complex_dir}/dm2.mol2",
            f"DM3 = loadmol2 {complex_dir}/dm3.mol2",
        ]

    system.template_lines += [
        f"model = loadpdb {str(complex_pdb.resolve()).strip()}",
        "check model",
    ]

    system.build(clean_files=False)
    system.repartition_hydrogen_mass()


def init(args):
    host = Path(args.host)
    guest = Path(args.guest)
    complex = Path(args.complex)

    logger.info('preparing host parameters')

    hostname = host.stem
    host_par_path = Path(hostname)
    host_top = host_par_path/f'{hostname}.prmtop'
    if not host_top.exists():
        antechamber(host, 'mol2', host_par_path, 10, args.overwrite)

    host_mol = Chem.MolFromMol2File(str(host))

    p_monomer = Chem.MolFromSmarts("O1C(O)CCCC1")
    monomers = host_mol.GetSubstructMatches(p_monomer)
    assert len(monomers) >= 6 & len(monomers) <= 8


    logger.info('preparing guest parameters')

    guestname = guest.stem
    if guestname.upper() == 'CHL':
        guest_par_path = None
        guest_top = None
    else:
        guest_par_path = Path(guestname)
        guest_top = guest_par_path/f'{guestname}.prmtop'
        if not guest_top.exists():
            antechamber(guest, 'mol2', guest_par_path, overwrite=args.overwrite)

    info = {
        'host': {
            'name': hostname,
            'file': str(host),
            'par_path': str(host_par_path),
            'top': str(host_top),
        },
        'guest': {
            'name': guestname,
            'file': str(guest),
            'par_path': str(guest_par_path),
            'top': str(guest_top),
        },
    }

    logger.info('build complex system')
    
    system_path = Path('complex')
    system_rst = system_path/'vac.rst7'
    system_top = system_path/'vac.prmtop'
    system_pdb = system_path/'vac.pdb'
    if not system_top.exists():
        build(info, complex, system_path, system_top, system_rst, system_pdb)

    info['system'] = {
        'top': str(system_top),
        'pdb': str(system_pdb),
        'rst': str(system_rst),
    }

    logger.info('align complex')

    G1 = f":{guestname.upper()}@{args.g1}"
    G2 = f":{guestname.upper()}@{args.g2}"

    structure = pmd.load_file('complex/vac.prmtop', 'complex/vac.rst7', structure=True)

    aligned_structure = align.zalign(structure, G1, G2)
    aligned_structure.save(str(system_path/"aligned.prmtop"), overwrite=True)
    aligned_structure.save(str(system_path/"aligned.rst7"), overwrite=True)

    logger.info('add dummy atoms')

    structure = pmd.load_file("complex/aligned.prmtop",
                              "complex/aligned.rst7",
                              structure=True)

    d0 = args.d0
    structure = dummy.add_dummy(structure, residue_name="DM1", z=d0)
    structure = dummy.add_dummy(structure, residue_name="DM2", z=d0-3)
    structure = dummy.add_dummy(structure, residue_name="DM3", z=d0-5.2, y=2.2)

    structure.save("complex/aligned_with_dummy.prmtop", overwrite=True)
    structure.save("complex/aligned_with_dummy.rst7", overwrite=True)
    structure.save("complex/aligned_with_dummy.pdb", overwrite=True)

    dummy.write_dummy_frcmod(filepath="complex/dummy.frcmod")
    dummy.write_dummy_mol2(residue_name="DM1", filepath="complex/dm1.mol2")
    dummy.write_dummy_mol2(residue_name="DM2", filepath="complex/dm2.mol2")
    dummy.write_dummy_mol2(residue_name="DM3", filepath="complex/dm3.mol2")

    logger.info('build simulation system with dummy atoms')

    complex = Path('complex/aligned_with_dummy.pdb')
    system_rst = system_path/'apr.rst7'
    system_top = system_path/'apr.prmtop'
    system_pdb = system_path/'apr.pdb'
    if not system_top.exists():
        build(info, complex, system_path, system_top, system_rst, system_pdb, dummy=True)

    logger.info("setup restraints")

    attach_string = "0.00 0.40 0.80 1.60 2.40 4.00 5.50 8.65 11.80 18.10 24.40 37.00 49.60 74.80 100.00"
    attach_fractions = [float(i) / 100 for i in attach_string.split()]

    initial_distance = args.r0
    final_distance = args.r1 
    offset = args.offset
    k_dist = args.k_dist
    pull_distances = np.arange(initial_distance, final_distance+offset, offset)

    release_string = "0.00 0.40 0.80 2.40 5.50 11.80 24.40 49.60 74.80 100.00"
    release_fractions = [float(i) / 100 for i in release_string.split()][::-1]
    #release_fractions = attach_fractions[::-1]
    windows = [len(attach_fractions), len(pull_distances), len(release_fractions)]
    print(f"There are {windows} windows in this attach-pull-release calculation.")

    # compute host anchor atoms

    structure = pmd.load_file("complex/apr.prmtop", "complex/apr.rst7")
    structure.save("complex/apr_pmd.pdb", overwrite=True)
    host_mol = Chem.MolFromPDBFile(str(system_path/"apr_pmd.pdb"), removeHs=False)
    p_monomer = Chem.MolFromSmarts("O1C(O)CCCC1")
    monomers = host_mol.GetSubstructMatches(p_monomer)

    if args.h1 and args.h2 and args.h3:
        H1 = f":{hostname.upper()}@{args.h1}"
        H2 = f":{hostname.upper()}@{args.h2}"
        H3 = f":{hostname.upper()}@{args.h3}"
    else:
        H1 = f":{hostname.upper()}@{monomers[0][1]+1}"
        H2 = f":{hostname.upper()}@{monomers[2][1]+1}"
        H3 = f":{hostname.upper()}@{monomers[-2][1]+1}"

    print(H1, H2, H3)

    D1 = ":DM1"
    D2 = ":DM2"
    D3 = ":DM3"

    static_restraints = setup_static_restraints(structure, windows, H1, H2, H3, D1, D2, D3, G1, G2)

    host_restraints = []

    for i, m in enumerate(monomers):
        O5 = f":{hostname.upper()}@{monomers[i][0]+1}"
        C1 = f":{hostname.upper()}@{monomers[i][1]+1}"
        O1 = f":{hostname.upper()}@{monomers[i][2]+1}"

        o1 = host_mol.GetAtomWithIdx(monomers[i][2])
        neighbor = [n.GetIdx() for n in o1.GetNeighbors() if n.GetIdx() not in monomers[i]].pop()
        for j in range(len(monomers)):
            if i == j: continue
            if neighbor in monomers[j]:
                break

        C4n = f":{hostname.upper()}@{monomers[j][5]+1}"
        C5n = f":{hostname.upper()}@{monomers[j][6]+1}"

        print(O5, C1, O1, C4n, C5n)

        r = restraints.DAT_restraint()
        r.mask1 = O5
        r.mask2 = C1
        r.mask3 = O1
        r.mask4 = C4n
        r.topology = structure
        r.auto_apr = True
        r.continuous_apr = True
        r.amber_index = False
        
        r.attach["target"] = 108.7                          # Degree
        r.attach["fraction_list"] = attach_fractions
        r.attach["fc_final"] = 6.0                          # kcal/mol/Radian**2
        
        r.pull["target_final"] = 108.7                      # Degrees
        r.pull["num_windows"] = windows[1]
    
        r.release["fc_final"] = 6.0                         # kcal/mol/Radian**2
        r.release["target"] = 108.7
        r.release["fraction_list"] = release_fractions
 
        r.initialize()
        host_restraints.append(r)

        r = restraints.DAT_restraint()
        r.mask1 = C1
        r.mask2 = O1
        r.mask3 = C4n
        r.mask4 = C5n
        r.topology = structure
        r.auto_apr = True
        r.continuous_apr = True
        r.amber_index = False
        
        r.attach["target"] = -112.5                         # Degree
        r.attach["fraction_list"] = attach_fractions
        r.attach["fc_final"] = 6.0                          # kcal/mol/Radian**2

        r.pull["target_final"] = -112.5                     # Degrees
        r.pull["num_windows"] = windows[1]
    
        r.release["fc_final"] = 6.0                         # kcal/mol/Radian**2
        r.release["target"] = -112.5
        r.release["fraction_list"] = release_fractions
        
        r.initialize()
        host_restraints.append(r)


    guest_restraints = []

    r = restraints.DAT_restraint()
    r.mask1 = D1
    r.mask2 = G1
    r.topology = structure
    r.auto_apr = True
    r.continuous_apr = True
    r.amber_index = False
    
    r.attach["target"] = initial_distance               # Angstroms
    r.attach["fraction_list"] = attach_fractions
    r.attach["fc_final"] = k_dist                       # kcal/mol/Angstroms**2
    
    r.pull["fc"] = k_dist                               # kcal/mol/Angstroms**2
    r.pull["target_final"] = final_distance             # Angstroms
    r.pull["num_windows"] = windows[1]

    r.release["target"] = final_distance
    r.release["fraction_list"] = [1.0] * len(release_fractions)
    r.release["fc_final"] = k_dist
    
    r.initialize()
    guest_restraints.append(r)

    r = restraints.DAT_restraint()
    r.mask1 = D2
    r.mask2 = D1
    r.mask3 = G1
    r.topology = structure
    r.auto_apr = True
    r.continuous_apr = True
    r.amber_index = False
    
    r.attach["target"] = 180.0                          # Degrees
    r.attach["fraction_list"] = attach_fractions
    r.attach["fc_final"] = 100.0                        # kcal/mol/radian**2
    
    r.pull["target_final"] = 180.0                      # Degrees
    r.pull["num_windows"] = windows[1]

    r.release["target"] = 180.0
    r.release["fraction_list"] = [1.0] * len(release_fractions)
    r.release["fc_final"] = 100.0
    
    r.initialize()
    guest_restraints.append(r)

    r = restraints.DAT_restraint()
    r.mask1 = D1
    r.mask2 = G1
    r.mask3 = G2
    r.topology = structure
    r.auto_apr = True
    r.continuous_apr = True
    r.amber_index = False
    
    r.attach["target"] = 180.0                          # Degrees
    r.attach["fraction_list"] = attach_fractions
    r.attach["fc_final"] = 100.0                        # kcal/mol/radian**2
    
    r.pull["target_final"] = 180.0                      # Degrees
    r.pull["num_windows"] = windows[1]

    r.release["target"] = 180.0
    r.release["fraction_list"] = [1.0] * len(release_fractions)
    r.release["fc_final"] = 100.0
    
    r.initialize()
    guest_restraints.append(r)

    window_list = create_window_list(guest_restraints)
    for window in window_list:
        if not os.path.isdir(f"windows/{window}"):
            os.makedirs(f"windows/{window}")

    host_guest_restraints = (static_restraints + host_restraints + guest_restraints)
    for window in window_list:
        folder = Path(f'windows/{window}')
        with open(folder/"disang.rest", "w") as file:
            for restraint in host_guest_restraints:
                string = amber_restraint_line(restraint, window)
                if string is not None:
                    file.write(string)

    for window in window_list:
        folder = Path(f'windows/{window}')
        window_number, phase = parse_window(window)

        if window[0] == "a":
            shutil.copy("complex/apr.prmtop", folder/"apr.prmtop")
            shutil.copy("complex/apr.rst7", folder/"apr.rst7")

        elif window[0] == "p":
            structure = pmd.load_file("complex/apr.prmtop", "complex/apr.rst7", structure = True)
            target_difference = guest_restraints[0].phase['pull']['targets'][int(window[1:])].magnitude + d0

            for atom in structure.atoms:
                if atom.residue.name == guestname.upper():
                    atom.xz += target_difference
            structure.save(str(folder/"apr.prmtop"), overwrite=True)
            structure.save(str(folder/"apr.rst7"), overwrite=True)

        elif window[0] == "r":
            structure = pmd.load_file("complex/apr.prmtop", "complex/apr.rst7", structure = True)
            target_difference = final_distance + d0

            for atom in structure.atoms:
                if atom.residue.name == guestname.upper():
                    atom.xz += target_difference
            structure.save(str(folder/"apr.prmtop"), overwrite=True)
            structure.save(str(folder/"apr.rst7"), overwrite=True)

        prefix = 'apr'

        # solvate window
        if not args.implicit:
            print(f"Solvating system in window {window}.")
            prefix = 'apr-solvated'
            structure = pmd.load_file(str(folder/"apr.prmtop"), str(folder/"apr.rst7"))
            structure.save(str(folder/"apr.pdb"), overwrite=True)
            solvate(info, str(folder/"apr.pdb"), 'complex', folder, prefix, num_waters=args.nwater, ion_conc=(args.conc/1000.0))

        # Load Amber
        prmtop = app.AmberPrmtopFile(str(folder/f'{prefix}.prmtop'))
        inpcrd = app.AmberInpcrdFile(str(folder/f'{prefix}.rst7'))

        # Create PDB file
        with open(folder/'system.pdb', 'w') as file:
            app.PDBFile.writeFile(prmtop.topology, inpcrd.positions, file, keepIds=True)

        # Create an OpenMM system from the Amber topology
        if not args.implicit:
            for line in open(folder/'system.pdb'):
                if line.startswith("CRYST1"):
                    boxx, boxy, boxz = list(map(float, line.split()[1:4]))
                    break

            system = prmtop.createSystem(
                nonbondedMethod=app.PME,
                nonbondedCutoff=9.0*unit.angstroms,
                constraints=app.HBonds,
            )

        else:
            system = prmtop.createSystem(
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                implicitSolvent=app.HCT,
            )

        # Apply positional restraints on the dummy atoms
        apply_positional_restraints(str(folder/'system.pdb'), system, force_group=15)

        # Apply host static restraints
        for restraint in static_restraints:
            apply_dat_restraint(system, restraint, phase, window_number, force_group=10)

        # Apply guest restraints
        for restraint in guest_restraints:
            apply_dat_restraint(system, restraint, phase, window_number, force_group=11)

        # Apply host restraints
        for restraint in host_restraints:
            apply_dat_restraint(system, restraint, phase, window_number, force_group=12)

        # Save OpenMM system to XML file
        system_xml = openmm.XmlSerializer.serialize(system)
        with open(folder/'system.xml', 'w') as file:
            file.write(system_xml)

    save_restraints(host_guest_restraints, filepath="windows/restraints.json")
    save_restraints(host_restraints, filepath="windows/host_restraints.json")
    save_restraints(guest_restraints, filepath="windows/guest_restraints.json")
      

