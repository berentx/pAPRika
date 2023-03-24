import json
import logging
import os
from pathlib import Path
import shutil
import sys
import itertools

import numpy as np
import openmm.unit as unit
import openmm.app as app
import openmm as openmm
import parmed as pmd
from rdkit import Chem
from rdkit.Chem.rdFMCS import FindMCS
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from paprika.build import align
from paprika.build import dummy
from paprika.build.system.utils import PBCBox
from paprika import restraints
from paprika.io import save_restraints
from paprika.restraints.restraints import create_window_list
from paprika.restraints.amber import amber_restraint_line
from paprika.restraints.utils import parse_window
from paprika.restraints.openmm import apply_positional_restraints, apply_dat_restraint

import parmed as pmd
from rdkit import Chem
import seekr2.modules.common_base as base
import seekr2.modules.common_cv as common_cv
import seekr2.modules.common_prepare as common_prepare
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from tqdm import tqdm


logger = logging.getLogger("init")


def pdb2pqr(input, input_pdb, output, gaff='gaff2'):
    input_mol2 = f'{input.stem}.{gaff}.mol2'
    input_path = input.resolve().parent/gaff
    output_path = output.parent

    cwd = os.getcwd()
    if not output_path.exists():
        output_path.mkdir(parents=True)

    os.chdir(output_path)

    if not Path(output).exists() or overwrite:
        cmd = ['sed', '-i', '-e', "'s/ATOM  /HETATM/g'", f'{input_pdb.name}']
        os.system(' '.join(cmd))
        cmd = ['pdb2pqr30', '--assign-only', '--ligand', str(input_path/input_mol2), f'{input_pdb.name}', f'{output.name}']
        #print(' '.join(cmd))
        os.system(' '.join(cmd))
        assert Path(output.name).exists()

    os.chdir(cwd)


def build(info, complex_pdb, system_path, system_top, system_rst, system_pdb, dummy=False, gaff='gaff2'):
    from paprika.build.system import TLeap

    system = TLeap()
    system.output_path = system_path
    system.pbc_type = None
    system.neutralize = False

    host_dir = Path(info['host']['par_path']).resolve()
    host_name = info['host']['name']
    
    system.template_lines = [
        f"source leaprc.{gaff}",
        "source leaprc.lipid21",
    ]

    system.template_lines += [
        f"loadamberparams {host_dir}/{host_name}.frcmod",
        f"MOL = loadmol2 {host_dir}/{host_name}.{gaff}.mol2",
        f"saveamberparm MOL receptor.prmtop receptor.rst7",
        f"savepdb MOL receptor.pdb",
    ]

    if info['guest']['par_path'] != 'None':
        guest_dir = Path(info['guest']['par_path']).resolve()
        guest_name = info['guest']['name']
        system.template_lines += [
            f"loadamberparams {guest_dir}/{guest_name}.frcmod",
            f"LIG = loadmol2 {guest_dir}/{guest_name}.{gaff}.mol2",
            f"saveamberparm LIG ligand.prmtop ligand.rst7",
            f"savepdb LIG ligand.pdb",
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


def solvate(info, complex_pdb, complex_dir, system_path, system_prefix, num_waters, ion_conc, dummy=True, gaff='gaff2'):
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
        f"source leaprc.{gaff}",
        "source leaprc.lipid21",
        #"source leaprc.water.tip3p",
    ]

    system.template_lines += [
        f"loadamberparams {host_dir}/{host_name}.frcmod",
        f"MOL = loadmol2 {host_dir}/{host_name}.{gaff}.mol2",
    ]

    if info['guest']['par_path'] != 'None':
        guest_dir = Path(info['guest']['par_path']).resolve()
        guest_name = info['guest']['name']
        system.template_lines += [
            f"loadamberparams {guest_dir}/{guest_name}.frcmod",
            f"LIG = loadmol2 {guest_dir}/{guest_name}.{gaff}.mol2",
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


def parse_config(args):
    config = yaml.load(open(args.config).read(), Loader=Loader)
    args.host = config['host']
    args.guest = config['guest']
    args.complex = config['complex']
    args.g1 = config['anchor']['g1']
    args.g2 = config['anchor']['g2']


def parse_mol2_atomnames(mol2file):
    _in = False
    atomnames = []
    for line in open(mol2file):
        if line.startswith("@<TRIPOS>ATOM"):
            _in = True
            continue
        if _in:
            if line.startswith('@'):
                break
            atomname = line.split()[1]
            atomnames.append(atomname)
    return atomnames


def match_host_atomnames(host, guest, complex, matched_complex_pdb, gaff='gaff2'):
    host_m = Chem.MolFromMolFile(str(host), removeHs=False)
    guest_m = Chem.MolFromMolFile(str(guest), removeHs=False)
    complex_m = Chem.MolFromPDBFile(str(complex), removeHs=False)
    complex_host, complex_guest = Chem.GetMolFrags(complex_m, asMols=True)

    ref = host_m
    target = complex_host
    gaff_mol2 = host.parent/gaff/f'{host.stem}.{gaff}.mol2'

    mcs = FindMCS([ref, target])
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

    mcs = FindMCS([ref, target])
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
    if args.config:
        parse_config(args)

    host = Path(args.host)
    guest = Path(args.guest)
    complex = Path(args.complex)
    gaff = args.gaff

    logger.info('preparing host parameters')

    hostname = host.stem
    host_par_path = host.parent/gaff
    host_pqr = host_par_path/f'{hostname}.pqr'
    host_format = host.suffix[1:]
    if not host_pqr.exists():
        pdb2pqr(host, Path('complex/receptor.pdb'), Path('complex/receptor.pqr'))

    if host_format == 'mol2':
        host_mol = Chem.MolFromMol2File(str(host), removeHs=False)
    elif host_format == 'sdf':
        host_mol = Chem.MolFromMolFile(str(host), removeHs=False)

    p_monomer = Chem.MolFromSmarts("O1C(O)CCCC1")
    monomers = host_mol.GetSubstructMatches(p_monomer)
    assert len(monomers) >= 6 & len(monomers) <= 8


    logger.info('preparing guest parameters')

    guestname = guest.stem
    if guestname.upper() == 'CHL1':
        guest_par_path = None
        guest_pqr = None
    else:
        guest_par_path = guest.parent/gaff
        guest_pqr = guest_par_path/f'{guestname}.pqr'
        if not guest_pqr.exists():
            pdb2pqr(guest, Path('complex/ligand.pdb'), Path('complex/ligand.pqr'))

    info = {
        'host': {
            'name': hostname,
            'file': str(host),
            'par_path': str(host_par_path),
            'pqr': str(host_pqr),
        },
        'guest': {
            'name': guestname,
            'file': str(guest),
            'par_path': str(guest_par_path),
            'pqr': str(guest_pqr),
        },
    }

    logger.info('build complex system')

    model_input = common_prepare.Model_input()
    model_input.root_directory = './seekr2'
    calculation_type = 'mmvt'

    if calculation_type == "mmvt":
        model_input.calculation_type = "mmvt"
        model_input.calculation_settings = common_prepare.MMVT_input_settings()
        model_input.calculation_settings.md_output_interval  =     5000
        model_input.calculation_settings.md_steps_per_anchor = 25000000

    elif calculation_type == "elber":
        model_input.calculation_type = "elber"
        model_input.calculation_settings = common_prepare.Elber_input_settings()
        model_input.calculation_settings.temperature_equil_progression = [310.0, 320.0]
        model_input.calculation_settings.num_temperature_equil_steps = 234
        model_input.calculation_settings.num_umbrella_stage_steps = 40000
        model_input.calculation_settings.fwd_rev_interval = 400
        model_input.calculation_settings.rev_output_interval = 100
        model_input.calculation_settings.fwd_output_interval = 100

    else:
        raise Exception("Option not available: %s", calculation_type)

    model_input.temperature = 300
    model_input.run_minimization = False
    model_input.hydrogenMass = 4.0
    model_input.ensemble = 'npt'
    model_input.pressure = 1.0
    model_input.timestep = 0.004
    model_input.md_program = 'openmm'
    model_input.integrator_type = 'langevin'
    #model_input.openmm_settings = base.Openmm_settings()
    #model_input.openmm_settings.langevin_integrator = base.Langevin_integrator_settings()
    #model_input.openmm_settings.langevin_integrator.timestep = 0.004
    #model_input.openmm_settings.langevin_integrator.target_temperature = 300
    #model_input.openmm_settings.barostat = base.Barostat_settings_openmm()
    #model_input.openmm_settings.barostat.target_temperature = 300

    if guestname.upper() == 'CHL1':
        guest_mask = ":CHL"
    else:
        guest_mask = ":LIG"

    if args.guest_resname and args.guest_resnr:
        guest_mask = f":{args.guest_resnr}&:{args.guest_resname}"
    elif args.guest_resname:
        guest_mask = f":{args.guest_resname}"
    elif args.guest_resnr:
        guest_mask = f":{args.guest_resnr}"

    G1 = f"{guest_mask}&@{args.g1}"
    G2 = f"{guest_mask}&@{args.g2}"

    print(G1, G2)


    structure = pmd.load_file('complex/complex.pdb', structure=True)
    selection = structure.view[f"{G1},{G2}"].atoms
    offset = len(structure.view[f":MOL"].atoms)

    cv_input1 = common_cv.Spherical_cv_input()
    cv_input1.group1 = list(itertools.chain.from_iterable(monomers))
    cv_input1.group2 = list([a.idx for a in selection])
    cv_input1.bd_group1 = list(itertools.chain.from_iterable(monomers))
    cv_input1.bd_group2 = [a.idx-offset for a in selection]
    cv_input1.input_anchors = []

    trajfiles = [str(_) for _ in Path('windows').glob('p*/production.dcd')]
    u = mda.Universe('windows/p001/apr-solvated.pdb', trajfiles)
    r_com = np.empty((u.trajectory.n_frames, 3), dtype=np.float)
    l_com = np.empty((u.trajectory.n_frames, 3), dtype=np.float)
    host_indices = [f'index {i}' for i in cv_input1.group1]
    guest_indices = [f'index {i}' for i in cv_input1.group2]
    r = u.select_atoms(' or '.join(host_indices))
    l = u.select_atoms(' or '.join(guest_indices))
    for i, ts in enumerate(tqdm(u.trajectory)):
        r_com[i, :] = r.center_of_mass()
        l_com[i, :] = l.center_of_mass()
    dist = np.linalg.norm(r_com - l_com, axis=1)

    structure = pmd.load_file("complex/aligned.prmtop", "complex/aligned.rst7", structure = True)
    a1 = structure[cv_input1.group1].coordinates
    a2 = structure[cv_input1.group2].coordinates
    d0 = np.linalg.norm(np.mean(a1, axis=0) - np.mean(a2, axis=0))

    lower = d0 - 1
    upper = d0
    dx = 2

    for i in range(1, 14):
        folder = Path(f'seekr2/inputs')
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f'p{i:03}'

        frames = np.argwhere((dist > lower) & (dist < upper))
        while len(frames) == 0:
            upper += 1
            frames = np.argwhere((dist > lower) & (dist < upper))

        center = ( upper + lower ) / 2
        index = np.argmin(np.abs(dist[frames] - upper))
        solute = u.select_atoms('resname MOL or resname LIG')
        with mda.Writer(str(folder/f'system.pdb')) as w:
            for ts in u.trajectory[frames[index]]:
                print(i, lower, upper, dist[frames[index]])
                w.write(solute.atoms)

        solvate(info, str(folder/f"system.pdb"), 'complex', folder, prefix, num_waters=args.nwater,
                ion_conc=(args.conc/1000.0), gaff=gaff)

        structure = pmd.load_file(str(folder/f"{prefix}.prmtop"), str(folder/f"{prefix}.rst7"), structure = True)

        input_anchor = common_cv.Spherical_cv_anchor()
        input_anchor.radius = upper / 10.
        input_anchor.starting_amber_params = base.Amber_params()
        input_anchor.starting_amber_params.prmtop_filename = f"seekr2/inputs/{prefix}.prmtop"
        input_anchor.starting_amber_params.inpcrd_filename = f"seekr2/inputs/{prefix}.rst7"
        input_anchor.starting_amber_params.pdb_coordinates_filename = f"seekr2/inputs/{prefix}.pdb"
        input_anchor.bound_state = False
        cv_input1.input_anchors.append(input_anchor)

        lower = upper
        upper = upper + dx

    cv_input1.input_anchors[0].bound_state = True
    cv_input1.input_anchors[-1].bulk_anchor = True
    
    model_input.cv_inputs = [cv_input1]
    
    model_input.browndye_settings_input = common_prepare.Browndye_settings_input()
    model_input.browndye_settings_input.receptor_pqr_filename = f"complex/receptor.pqr"
    model_input.browndye_settings_input.ligand_pqr_filename = f"complex/ligand.pqr"
    model_input.browndye_settings_input.apbs_grid_spacing = 0.5
    #model_input.browndye_settings_input.receptor_indices = list(range(len(structure.view[':MOL'].atoms)))
    #model_input.browndye_settings_input.ligand_indices = list(range(len(structure.view[':LIG'].atoms)))
    model_input.browndye_settings_input.binary_directory = ""
    
    ion1 = base.Ion()
    ion1.radius = 1.2
    ion1.charge = -1.0
    ion1.conc = 0.15
    ion2 = base.Ion()
    ion2.radius = 0.9
    ion2.charge = 1.0
    ion2.conc = 0.15
    model_input.browndye_settings_input.ions = [ion1, ion2]
    model_input.browndye_settings_input.num_bd_milestone_steps = 1000
    model_input.browndye_settings_input.num_b_surface_steps = 10000
    model_input.browndye_settings_input.num_b_surface_trajectories = 10000
    model_input.browndye_settings_input.n_threads = 1
    
    model_input.serialize('seekr2.xml')
