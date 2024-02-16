import copy
import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFMCS import BondCompare
from rdkit.Chem.rdFMCS import FindMCS, MCSParameters
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


logger = logging.getLogger("init")

def parse_config(args):
    config = yaml.load(open(args.config).read(), Loader=Loader)
    if 'host' in config:
        args.host = config['host']
    if 'guest' in config:
        args.guest = config['guest']
    if 'complex' in config:
        args.complex = config['complex']

def parse_mol2_atomnames(mol2file):
    in_atomblock = False
    atomnames = []
    for line in open(mol2file):
        if line.startswith('@<TRIPOS>ATOM'):
            in_atomblock = True
            continue
        if line.startswith('@<TRIPOS>BOND'):
            in_atomblock = False
            continue
        if in_atomblock:
            name = line.split()[1]
            atomnames.append(name)
    return atomnames

def rename_complex_pdb(pdbfile, host_atomnames, guest_atomnames):
    atomnames = host_atomnames + guest_atomnames
    resname = ['MOL'] * len(host_atomnames) + ['LIG'] * len(guest_atomnames)
    resnr = ['1'] * len(host_atomnames) + ['2'] * len(guest_atomnames)
    index = 0
    lines = []
    for line in open(pdbfile):
        if line.startswith('HETATM') or line.startswith('ATOM'):
            line = f'ATOM  ' + line[6:12] + f'{atomnames[index]:^5s}' + f'{resname[index]:4s}' + line[21:23] + f'{resnr[index]:>3s}' + line[27:]
            index += 1
        lines.append(line)
    open(pdbfile, 'w').write(''.join(lines))

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

        if format == 'sdf':
            m = Chem.MolFromMolFile(str(input_path))
        elif format == 'mol2':
            m = Chem.MolFromMol2File(str(input_path))
        charge = Chem.GetFormalCharge(m)
        if charge != 0:
            cmd += ['-nc', f'{charge:d}']
        print(cmd)

        os.system(' '.join(cmd))
        assert Path(output_mol2).exists()

    if not Path(output_frcmod).exists() or overwrite:
        cmd = ['parmchk2', '-i', str(output_mol2), '-f', 'mol2', '-o', str(output_frcmod),
               '-s', 'gaff2']
        os.system(' '.join(cmd))
        assert Path(output_frcmod).exists()

    os.chdir(cwd)


def duplicate_structures(original, copies):
    n_copies = len(copies.residues) / len(original.residues)
    n_atoms = len(original.atoms)
    coor_max = np.max(original.coordinates, axis=0)
    coor_min = np.min(original.coordinates, axis=0)
    radius = np.linalg.norm( coor_max - coor_min ) / 2
    dist_min = radius * 1.4
    dist_max = radius * 1.6
    centers = np.array([np.mean(original.coordinates, axis=0)])
    n_trial = 0
    max_trial = 200
    coordinates = np.array(copies.coordinates)
    while len(centers) < n_copies:
        center_index = np.random.choice(len(centers), 1)[0]
        center = centers[center_index]
        is_new_point_found = False

        for j in range(50):
            dd = np.random.rand(3)
            dr = (dd[2] + 1.0) * radius
            dtheta = dd[0] * np.pi
            dphi = dd[1] * 2 * np.pi
            dx = dr * np.sin(dtheta) * np.cos(dphi)
            dy = dr * np.sin(dtheta) * np.sin(dphi)
            dz = dr * np.cos(dtheta)
            dt = np.array((dx, dy, dz))
            new_center = center + dt
            dist = np.linalg.norm(centers - new_center, axis=1)
            is_new_point_found = all(dist > dist_min) and any(dist < dist_max)
            if is_new_point_found:
                break
        
        if is_new_point_found:
            atom_index = center_index * n_atoms
            old_coords = coordinates[atom_index:atom_index+n_atoms]

            atom_index = len(centers) * n_atoms
            coordinates[atom_index:atom_index+n_atoms] = old_coords + dt
            centers = np.array(list(centers) + [new_center])
            n_trial = 0
        
        else:
            n_trial += 1
        
        assert n_trial < max_trial, len(centers)

    copies.coordinates = coordinates
    

def build(info, complex_pdb, system_path, system_top, system_rst, system_pdb, dummy=False, gaff='gaff2'):
    from paprika.build.system import TLeap

    if not system_path.exists():
        system_path.mkdir()

    has_host = 'host' in info
    has_guest = 'guest' in info
    has_complex = 'complex' in info

    system = TLeap()
    system.output_path = system_path
    system.pbc_type = None
    system.neutralize = False

    system.template_lines = [
        f"source leaprc.{gaff}",
        "source leaprc.lipid21",
        "set default PBRadii mbondi2",
    ]

    if has_host:
        host_dir = Path(info['host']['par_path']).resolve()
        host_name = info['host']['name']
        host_mol2_file = f'{host_dir}/{host_name}.{gaff}.mol2'
        host_atomnames = parse_mol2_atomnames(host_mol2_file)
    
        system.template_lines += [
            f"loadamberparams {host_dir}/{host_name}.frcmod",
            f"MOL = loadmol2 {host_mol2_file}",
            f"saveamberparm MOL receptor.prmtop receptor.rst7",
        ]

    if has_guest:
        guest_dir = Path(info['guest']['par_path']).resolve()
        guest_name = info['guest']['name']
        guest_mol2_file = f'{guest_dir}/{guest_name}.{gaff}.mol2'
        guest_atomnames = parse_mol2_atomnames(guest_mol2_file)

        if info['guest']['par_path'] != 'None':
            system.template_lines += [
                f"loadamberparams {guest_dir}/{guest_name}.frcmod",
                f"LIG = loadmol2 {guest_mol2_file}",
                f"saveamberparm LIG ligand.prmtop ligand.rst7",
            ]
        else:
            guest_file = Path(info['guest']['file']).resolve()
            ext = guest_file.suffix[1:]
            system.template_lines += [
                f"LIG = load{ext} {guest_file}",
                f"saveamberparm LIG ligand.prmtop ligand.rst7",
            ]

    if has_complex:
        #complex_file = Path(info['complex']['file']).resolve()
        #ext = complex_file.suffix[1:]

        ## copy complex file
        #shutil.copy(complex_file, system_path/complex_file.name)
        #complex_file = (system_path/complex_file.name).resolve()
        ##rename_complex_pdb(complex_file, host_atomnames, guest_atomnames)

        system.template_lines += [
            f"model = loadpdb {str(complex_pdb.resolve()).strip()}",
        ]

    else:
        if has_host and has_guest:
            system.template_lines += [ f"model = combine {{ MOL LIG }}", ]
        elif has_host:
            system.template_lines += [ f"model = combine {{ MOL }}", ]
        else:
            system.template_lines += [ f"model = combine {{ LIG }}", ]

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
        "set default PBRadii mbondi2",
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
            f"LIG = loadmol2 {guest_dir}/{guest_name}.gaff2.mol2",
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

def match_host_atomnames(host, guest, complex, matched_complex_pdb, gaff='gaff2'):
    host_m = Chem.MolFromMolFile(str(host), removeHs=False)
    guest_m = Chem.MolFromMolFile(str(guest), removeHs=False)
    complex_m = Chem.MolFromPDBFile(str(complex), removeHs=False)
    complex_host, complex_guest = Chem.GetMolFrags(complex_m, asMols=True)

    ref = host_m
    target = complex_host
    gaff_mol2 = host.parent/gaff/f'{host.stem}.{gaff}.mol2'

    ps = MCSParameters()
    mcs = FindMCS([ref, target], bondCompare=ps.BondTyper.CompareAny)
    ref_aids = ref.GetSubstructMatch(mcs.queryMol)
    target_aids = target.GetSubstructMatch(mcs.queryMol)
    if len(target_aids) < len(target.GetAtoms()):
        print("some atom names are not matched; expand MCS for matching any bond order")
        mcs = FindMCS([ref, target], bondCompare=BondCompare.CompareAny)
        ref_aids = ref.GetSubstructMatch(mcs.queryMol)
        target_aids = target.GetSubstructMatch(mcs.queryMol)
        
    atomnames = parse_mol2_atomnames(gaff_mol2)
    for aid1, aid2 in zip(ref_aids, target_aids):
        a1 = ref.GetAtomWithIdx(aid1)
        a2 = target.GetAtomWithIdx(aid2)
        atomname = atomnames[aid1]
        a2.GetPDBResidueInfo().SetName('{:>4}'.format('{:<3}'.format(atomname)))

    ref = guest_m
    target = complex_guest
    gaff_mol2 = guest.parent/gaff/f'{guest.stem}.{gaff}.mol2'

    mcs = FindMCS([ref, target], bondCompare=ps.BondTyper.CompareAny)
    ref_aids = ref.GetSubstructMatch(mcs.queryMol)
    target_aids = target.GetSubstructMatch(mcs.queryMol)
    atomnames = parse_mol2_atomnames(gaff_mol2)
    for aid1, aid2 in zip(ref_aids, target_aids):
        a1 = ref.GetAtomWithIdx(aid1)
        a2 = target.GetAtomWithIdx(aid2)
        atomname = atomnames[aid1]
        a2.GetPDBResidueInfo().SetName('{:>4}'.format('{:<3}'.format(atomname)))

    complex = Chem.CombineMols(complex_host, complex_guest)
    Chem.MolToPDBFile(complex, str(matched_complex_pdb))

def init(args):
    import openmm.unit as unit
    import openmm.app as app
    import openmm as openmm
    import parmed as pmd
    from paprika.build import align
    
    info = {}

    if args.config:
        parse_config(args)

    if args.host:
        host = Path(args.host)
        logger.info('preparing host parameters')
    
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
        if guestname.upper() == 'CHL' and 0:
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
        system_path.mkdir(exist_ok=True)
        if args.complex:
            matched_complex_pdb = system_path/'complex.pdb'
            match_host_atomnames(host, guest, complex, matched_complex_pdb, gaff='gaff2')
        elif args.host:
            matched_complex_pdb = host_par_path/'sqm.pdb'
        elif args.guest:
            matched_complex_pdb = guest_par_path/'sqm.pdb'
        build(info, matched_complex_pdb, system_path, system_top, system_rst, system_pdb, gaff='gaff2')

    info['system'] = {
        'top': str(system_top),
        'pdb': str(system_pdb),
        'rst': str(system_rst),
    }

    structure = pmd.load_file('complex/vac.prmtop', 'complex/vac.rst7', structure=True)

    aligned_structure = align.align_principal_axes(structure)
    aligned_prmtop = system_path/"aligned.prmtop"
    aligned_rst7 = system_path/"aligned.rst7"
    aligned_pdb = system_path/"aligned.pdb"

    aligned_structure.save(str(aligned_prmtop), overwrite=True)
    aligned_structure.save(str(aligned_rst7), overwrite=True)
    aligned_structure.save(str(aligned_pdb), overwrite=True)
    
    sysnr = 1
    sysdir = Path(f"windows/{sysnr}")
    sysdir.mkdir(exist_ok=True, parents=True)

    folder = sysdir
    if args.copy == 1:    
        shutil.copy(aligned_prmtop, folder/"system.prmtop")
        shutil.copy(aligned_rst7, folder/"system.rst7")
    else:
        original = pmd.load_file(str(aligned_prmtop), str(aligned_rst7))
        copies = original * args.copy
        duplicate_structures(original, copies)

        copies.save(str(folder/"system.prmtop"), overwrite=True)
        copies.save(str(folder/"system.rst7"), overwrite=True)
        copies.save(str(folder/"system.pdb"), overwrite=True)

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

