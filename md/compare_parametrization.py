#!/usr/bin/env python
"""
Compare fragment-based vs. direct GAFF2 parametrization of beta-cyclodextrin.

Usage:
    python md/compare_parametrization.py --mol /path/to/BCD.sdf --workdir ./compare_work
"""

import argparse
import os
import subprocess

import numpy as np
import parmed
from rdkit.Chem import AllChem as Chem

from paprika.build.system.fragmenter import CyclodextrinFragmenter


def run_direct(mol_sdf, out_mol2, out_frcmod, residue_name="MGO", atom_type="gaff2", work_dir="./"):
    """Run antechamber directly on the full molecule (slow baseline)."""
    mol = Chem.MolFromMolFile(mol_sdf, removeHs=False)
    net_charge = Chem.GetFormalCharge(mol)

    subprocess.run(
        [
            "antechamber",
            "-fi", "sdf",
            "-fo", "mol2",
            "-i", os.path.abspath(mol_sdf),
            "-o", os.path.abspath(out_mol2),
            "-c", "bcc",
            "-at", atom_type,
            "-nc", str(net_charge),
            "-rn", residue_name,
            "-pf", "y",
        ],
        check=True,
        cwd=work_dir,
    )
    subprocess.run(
        [
            "parmchk2",
            "-i", os.path.abspath(out_mol2),
            "-f", "mol2",
            "-o", os.path.abspath(out_frcmod),
            "-s", atom_type,
        ],
        check=True,
        cwd=work_dir,
    )


def run_fragmented(mol_sdf, out_mol2, out_frcmod, residue_name="MGO", atom_type="gaff2", work_dir="./"):
    """Run fragment-based parametrization via CyclodextrinFragmenter.
    Returns the fragmenter after parametrization (fragments already computed)."""
    mol = Chem.MolFromMolFile(mol_sdf, removeHs=False)
    fragmenter = CyclodextrinFragmenter(mol)
    fragmenter.parametrize(
        output_mol2=out_mol2,
        output_frcmod=out_frcmod,
        residue_name=residue_name,
        atom_type=atom_type,
        charge_method="bcc",
        work_dir=work_dir,
    )
    return fragmenter


def write_debug_fragments(fragmenter, work_dir):
    """Write each unique capped fragment to SDF and print a summary table."""
    print("\n=== Fragment Debug ===")
    print(f"  Total monomers : {len(fragmenter._capped_monomers)}")
    print(f"  Unique types   : {len(set(fragmenter._capped_monomers_index))}")
    print()
    print(f"  {'Monomer':>7}  {'FragIdx':>7}  {'Atoms':>5}  {'Heavy':>5}  SMILES")

    seen = set()
    for monomer_idx, (frag_idx, mol) in enumerate(
        zip(fragmenter._capped_monomers_index, fragmenter._capped_monomers)
    ):
        n_atoms = mol.GetNumAtoms()
        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        smiles = Chem.MolToSmiles(mol)
        print(f"  {monomer_idx:>7}  {frag_idx:>7}  {n_atoms:>5}  {n_heavy:>5}  {smiles}")

        if frag_idx not in seen:
            seen.add(frag_idx)
            sdf_path = os.path.join(work_dir, f"debug_frag{frag_idx}.sdf")
            mol_h = Chem.AddHs(mol)
            Chem.EmbedMultipleConfs(mol_h)
            writer = Chem.SDWriter(sdf_path)
            writer.write(mol_h, confId=0)
            writer.close()
            print(f"           -> written to {sdf_path}")

    print(f"\n  orig_indices coverage: {sum(len(x) for x in fragmenter._capped_monomers_atom_indices)}"
          f" / {fragmenter.input_molecule.GetNumAtoms()} atoms mapped")


def parse_frcmod(path):
    """Parse an frcmod file into a dict of section -> list of parameter lines."""
    sections = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            stripped = line.strip()
            if stripped in ("MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"):
                current = stripped
                sections[current] = []
            elif current and stripped:
                sections[current].append(line)
    return sections


def compare_charges(frag_mol2, direct_mol2):
    print("\n=== Charge Comparison ===")
    frag = parmed.load_file(frag_mol2, structure=True)
    direct = parmed.load_file(direct_mol2, structure=True)

    if len(frag.atoms) != len(direct.atoms):
        print(f"  WARNING: atom count mismatch — frag={len(frag.atoms)}, direct={len(direct.atoms)}")
        return

    frag_charges = np.array([a.charge for a in frag.atoms])
    direct_charges = np.array([a.charge for a in direct.atoms])
    delta = frag_charges - direct_charges

    print(f"  Atoms:  {len(frag.atoms)}")
    print(f"  Sum (frag):   {frag_charges.sum():.4f} e")
    print(f"  Sum (direct): {direct_charges.sum():.4f} e")
    print(f"  MAE:   {np.mean(np.abs(delta)):.4f} e")
    print(f"  RMSE:  {np.sqrt(np.mean(delta**2)):.4f} e")
    print(f"  Max |delta|: {np.max(np.abs(delta)):.4f} e  (atom {np.argmax(np.abs(delta))})")

    # Print worst 10
    worst = np.argsort(np.abs(delta))[::-1][:10]
    print("\n  Top 10 largest charge differences:")
    print(f"  {'Idx':>5}  {'Type(frag)':>12}  {'Type(direct)':>12}  {'q(frag)':>10}  {'q(direct)':>10}  {'delta':>10}")
    for i in worst:
        print(
            f"  {i:>5}  {frag.atoms[i].type:>12}  {direct.atoms[i].type:>12}"
            f"  {frag_charges[i]:>10.4f}  {direct_charges[i]:>10.4f}  {delta[i]:>10.4f}"
        )


def compare_atom_types(frag_mol2, direct_mol2):
    print("\n=== Atom Type Comparison ===")
    frag = parmed.load_file(frag_mol2, structure=True)
    direct = parmed.load_file(direct_mol2, structure=True)

    if len(frag.atoms) != len(direct.atoms):
        return

    mismatches = [
        (i, frag.atoms[i].type, direct.atoms[i].type)
        for i in range(len(frag.atoms))
        if frag.atoms[i].type != direct.atoms[i].type
    ]
    if mismatches:
        print(f"  {len(mismatches)} atom type mismatches:")
        for i, ft, dt in mismatches:
            print(f"    atom {i:>5}: frag={ft!r:>10}  direct={dt!r}")
    else:
        print("  All atom types match.")


def compare_frcmod(frag_frcmod, direct_frcmod):
    print("\n=== frcmod Comparison ===")
    frag_params = parse_frcmod(frag_frcmod)
    direct_params = parse_frcmod(direct_frcmod)

    all_sections = set(frag_params) | set(direct_params)
    any_diff = False
    for section in sorted(all_sections):
        f_set = set(frag_params.get(section, []))
        d_set = set(direct_params.get(section, []))
        only_frag = f_set - d_set
        only_direct = d_set - f_set
        if only_frag or only_direct:
            any_diff = True
            print(f"\n  [{section}]")
            for p in sorted(only_frag):
                print(f"    frag only:   {p}")
            for p in sorted(only_direct):
                print(f"    direct only: {p}")
        else:
            print(f"  [{section}]: identical ({len(f_set)} entries)")

    if not any_diff:
        print("  frcmod files are identical.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol", default="/home/sunhwan/work/beren/compounds/BCD/1_model/BCD.sdf",
                        help="Input SDF file with 3D coordinates and hydrogens")
    parser.add_argument("--workdir", default="./compare_work", help="Working directory for temp files")
    parser.add_argument("--residue-name", default="MGO")
    parser.add_argument("--atom-type", default="gaff2")
    parser.add_argument("--skip-direct", action="store_true",
                        help="Skip direct antechamber (very slow for large molecules)")
    parser.add_argument("--debug", action="store_true",
                        help="Write each unique capped fragment to SDF and print summary")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.mol))[0]
    frag_mol2 = os.path.join(args.workdir, f"{stem}.frag.mol2")
    frag_frcmod = os.path.join(args.workdir, f"{stem}.frag.frcmod")
    direct_mol2 = os.path.join(args.workdir, f"{stem}.direct.mol2")
    direct_frcmod = os.path.join(args.workdir, f"{stem}.direct.frcmod")

    print(f"Input: {args.mol}")
    print(f"Work dir: {args.workdir}")

    # --- Fragmented ---
    if not os.path.exists(frag_mol2):
        print("\n[1/2] Running fragment-based parametrization...")
        fragmenter = run_fragmented(
            args.mol, frag_mol2, frag_frcmod,
            residue_name=args.residue_name,
            atom_type=args.atom_type,
            work_dir=args.workdir,
        )
        print("  Done.")
    else:
        print(f"\n[1/2] Using existing {frag_mol2}")
        fragmenter = None

    if args.debug:
        if fragmenter is None:
            # Re-run fragmentation without parametrization just to get fragments
            mol = Chem.MolFromMolFile(args.mol, removeHs=False)
            fragmenter = CyclodextrinFragmenter(mol)
            fragmenter._get_capped_monomer()
        write_debug_fragments(fragmenter, args.workdir)

    # --- Direct ---
    if not args.skip_direct:
        if not os.path.exists(direct_mol2):
            print("\n[2/2] Running direct parametrization (slow)...")
            run_direct(
                args.mol, direct_mol2, direct_frcmod,
                residue_name=args.residue_name,
                atom_type=args.atom_type,
                work_dir=args.workdir,
            )
            print("  Done.")
        else:
            print(f"\n[2/2] Using existing {direct_mol2}")

        compare_charges(frag_mol2, direct_mol2)
        compare_atom_types(frag_mol2, direct_mol2)
        compare_frcmod(frag_frcmod, direct_frcmod)
    else:
        print("\n[2/2] Skipping direct parametrization (--skip-direct)")
        print("\nFragment mol2 written to:", frag_mol2)
        print("Fragment frcmod written to:", frag_frcmod)


if __name__ == "__main__":
    main()
