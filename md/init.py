import json
import logging
import os
from pathlib import Path
import shutil

def antechamber(input, format, output_path, pl=-1, overwrite=False):
    output_mol2 = f'{input.stem}.gaff2.mol2'
    output_frcmod = f'{input.stem}.frcmod'
    input_path = input.resolve()

    cwd = os.getcwd()
    if not output_path.exists():
        output_path.mkdir(parents=True)

    os.chdir(output_path)

    if not Path(output_mol2).exists() or overwrite:
        cmd = ['antechamber', '-fi', format, '-fo', 'mol2', '-i', str(input_path),
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


def build(info, system_path, system_top, system_rst, system_pdb, dummy=False):
    from paprika.build.system import TLeap

    system = TLeap()
    system.output_path = system_path
    system.pbc_type = None
    system.neutralize = False

    system.template_lines = [
        "source leaprc.gaff2",
        "source leaprc.lipid21",
    ]

    if 'host' in info:
        host_dir = Path(info['host']['par_path']).resolve()
        host_name = info['host']['name']
    
        system.template_lines += [
            f"loadamberparams {host_dir}/{host_name}.frcmod",
            f"MOL = loadmol2 {host_dir}/{host_name}.gaff2.mol2",
        ]

    if 'guest' in info:
        guest_dir = Path(info['guest']['par_path']).resolve()
        guest_name = info['guest']['name']

        if info['guest']['par_path'] != 'None':
            system.template_lines += [
                f"loadamberparams {guest_dir}/{guest_name}.frcmod",
                f"LIG = loadmol2 {guest_dir}/{guest_name}.gaff2.mol2",
            ]
        else:
            guest_file = Path(info['guest']['file']).resolve()
            ext = guest_file.suffix[1:]
            system.template_lines += [
                f"LIG = load{ext} {guest_file}",
            ]

    if 'complex' in info:
        complex_file = Path(info['complex']['file']).resolve()
        ext = complex_file.suffix[1:]
        system.template_lines += [
            f"model = load{ext} {complex_file}",
        ]
    else:
        if args.host and args.guest:
            system.template_lines += [ f"model = combine {{ host guest }}", ]
        elif args.host:
            system.template_lines += [ f"model = host", ]
        else:
            system.template_lines += [ f"model = guest", ]

    system.template_lines += [
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
    from paprika.build.system.utils import PBCBox

    system = TLeap()
    system.output_path = system_path
    system.pbc_type = PBCBox.cubic
    system.neutralize = True
    system.output_prefix = system_prefix
    system.target_waters = num_waters
    #system.set_water_model("tip3p", model_type="force-balance")
    system.set_water_model("tip3p")

    n_ions = int(ion_conc / 55.0 * system.target_waters)
    system.add_ions = ["Na+", n_ions, "Cl-", n_ions]
    complex_dir = Path(complex_dir).resolve()
    complex_pdb = Path(complex_pdb)
    
    system.template_lines = [
        "source leaprc.gaff2",
        "source leaprc.lipid21",
        #"source leaprc.water.tip3p",
    ]

    if 'host' in info:
        host_dir = Path(info['host']['par_path']).resolve()
        host_name = info['host']['name']
        system.template_lines += [
            f"loadamberparams {host_dir}/{host_name}.frcmod",
            f"MOL = loadmol2 {host_dir}/{host_name}.gaff2.mol2",
        ]

    if 'guest' in info and info['guest']['par_path'] != 'None':
        guest_dir = Path(info['guest']['par_path']).resolve()
        guest_name = info['guest']['name']
        system.template_lines += [
            f"loadamberparams {guest_dir}/{guest_name}.frcmod",
            f"VTR = loadmol2 {guest_dir}/{guest_name}.gaff2.mol2",
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
    import numpy as np
    import openmm.unit as unit
    import openmm.app as app
    import openmm as openmm
    import parmed as pmd
    from rdkit import Chem
    
    from paprika.build import align
    from paprika.build import dummy
    from paprika import restraints
    from paprika.io import save_restraints
    from paprika.restraints.restraints import create_window_list
    from paprika.restraints.amber import amber_restraint_line
    from paprika.restraints.utils import parse_window
    from paprika.restraints.openmm import apply_positional_restraints, apply_dat_restraint
    from restraints import setup_static_restraints
    
    logger = logging.getLogger("init")


    info = {}

    if args.host:
        logger.info('preparing host parameters')
        host = Path(args.host)
    
        hostname = host.stem
        host_par_path = host.parent/'gaff2'
        host_top = host_par_path/f'{hostname}.prmtop'
        if not host_top.exists():
            antechamber(host, host.suffix[1:], host_par_path, 10, args.overwrite)
    
        info['host'] = {
            'name': hostname,
            'file': str(host),
            'par_path': str(host_par_path),
            'top': str(host_top),
        }

    if args.guest:
        logger.info('preparing guest parameters')
        guest = Path(args.guest)

        guestname = guest.stem
        if guestname.upper() == 'CHL':
            guest_par_path = None
            guest_top = None
        else:
            guest_par_path = guest.parent/'gaff2'
            guest_top = guest_par_path/f'{guestname}.prmtop'
            if not guest_top.exists():
                antechamber(guest, guest.suffix[1:], guest_par_path, overwrite=args.overwrite)

        info['guest'] = {
            'name': guestname,
            'file': str(guest),
            'par_path': str(guest_par_path),
            'top': str(guest_top),
        }

    if args.complex:
        logger.info('loading complex')
        complex = Path(args.complex)

        info['complex'] = {
            'file': str(args.complex),
        }


    if not args.host and not args.guest:
        raise ValueError('At least host or guest parameter is required')

    logger.info('build complex system')
    
    system_path = Path('complex')
    system_rst = system_path/'vac.rst7'
    system_top = system_path/'vac.prmtop'
    system_pdb = system_path/'vac.pdb'
    if not system_top.exists():
        build(info, system_path, system_top, system_rst, system_pdb)

    info['system'] = {
        'top': str(system_top),
        'pdb': str(system_pdb),
        'rst': str(system_rst),
    }

    structure = pmd.load_file('complex/vac.prmtop', 'complex/vac.rst7', structure=True)

    aligned_structure = align.align_principal_axes(structure)
    aligned_structure.save(str(system_path/"aligned.prmtop"), overwrite=True)
    aligned_structure.save(str(system_path/"aligned.rst7"), overwrite=True)
    aligned_structure.save(str(system_path/"aligned.pdb"), overwrite=True)
    
    sysnr = 1
    sysdir = Path(f"windows/{sysnr}")
    sysdir.mkdir(exist_ok=True, parents=True)

    folder = sysdir
    shutil.copy("complex/aligned.prmtop", folder/"system.prmtop")
    shutil.copy("complex/aligned.rst7", folder/"system.rst7")

    prefix = 'system'

    # solvate window
    if not args.implicit:
        print(f"Solvating system in window {sysnr}.")
        prefix = 'system-solvated'
        structure = pmd.load_file(str(folder/"system.prmtop"), str(folder/"system.rst7"))
        structure.save(str(folder/"system.pdb"), overwrite=True)
        solvate(info, str(folder/"system.pdb"), 'complex', folder, prefix,
                num_waters=args.nwater, ion_conc=(args.conc/1000.0), dummy=False)

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

    # Save OpenMM system to XML file
    system_xml = openmm.XmlSerializer.serialize(system)
    with open(folder/'system.xml', 'w') as file:
        file.write(system_xml)

