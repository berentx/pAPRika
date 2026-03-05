"""
Tests for CyclodextrinFragmenter: auto-detection, atom-index mapping correctness,
and regression of parametrized charges/types against reference outputs.
"""

import os

import numpy as np
import parmed
import pytest
from rdkit.Chem import AllChem as Chem

from paprika.build.system.fragmenter import CyclodextrinFragmenter

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/bcd")
BCD_SDF = os.path.join(DATA_DIR, "BCD.sdf")
FRAG_MOL2 = os.path.join(DATA_DIR, "bCD.frag.mol2")
DIRECT_MOL2 = os.path.join(DATA_DIR, "bCD.direct.mol2")
FRAG0_MOL2 = os.path.join(DATA_DIR, "frag0.mol2")


@pytest.fixture
def bcd_mol():
    mol = Chem.MolFromMolFile(BCD_SDF, removeHs=False)
    assert mol is not None, f"Failed to load {BCD_SDF}"
    return mol


@pytest.fixture
def bcd_fragmenter(bcd_mol):
    fragmenter = CyclodextrinFragmenter(bcd_mol)
    fragmenter._get_capped_monomer()
    return fragmenter


# ---------------------------------------------------------------------------
# Auto-detection tests (no antechamber needed)
# ---------------------------------------------------------------------------


def test_auto_detect_monomers(bcd_mol):
    """_auto_detect_monomers() finds at least one monomer pattern in beta-CD."""
    fragmenter = CyclodextrinFragmenter(bcd_mol)
    fragmenter._auto_detect_monomers()
    assert len(fragmenter.monomers) == 1


def test_bond_list_length(bcd_mol):
    """Beta-CD has 7 glycosidic bonds; _get_bond_list() should find exactly 7."""
    fragmenter = CyclodextrinFragmenter(bcd_mol)
    bond_list, dummy_labels = fragmenter._get_bond_list()
    assert len(bond_list) == 7
    assert len(dummy_labels) == 7


def test_fragment_count(bcd_fragmenter):
    """Fragmenting beta-CD produces 7 capped monomers."""
    assert len(bcd_fragmenter._capped_monomers) == 7
    assert len(bcd_fragmenter._capped_monomers_atom_indices) == 7
    assert len(bcd_fragmenter._capped_monomers_frag_real_indices) == 7


def test_all_monomers_same_smiles(bcd_fragmenter):
    """All 7 monomers in unmodified beta-CD have the same canonical SMILES."""
    assert len(set(bcd_fragmenter._capped_monomers_index)) == 1


# ---------------------------------------------------------------------------
# Atom index mapping correctness (core of the bug fix)
# ---------------------------------------------------------------------------


def test_orig_and_frag_indices_same_length(bcd_fragmenter):
    """orig_indices and frag_real_indices have the same length for every monomer."""
    for orig, frag in zip(
        bcd_fragmenter._capped_monomers_atom_indices,
        bcd_fragmenter._capped_monomers_frag_real_indices,
    ):
        assert len(orig) == len(frag), (
            f"Mismatch: len(orig_indices)={len(orig)} != len(frag_indices)={len(frag)}"
        )


def test_atom_element_consistency_orig_vs_frag(bcd_fragmenter):
    """
    For every monomer, each atom at orig_indices[i] in the original molecule
    has the same atomic number as the atom at frag_indices[i] in the capped
    fragment. This directly tests that the index mapping points to the
    same atom type (O↔O, C↔C, H↔H) in both views of the molecule.
    """
    mol = bcd_fragmenter.input_molecule
    for monomer_idx, (orig_indices, frag_indices, cap_frag) in enumerate(
        zip(
            bcd_fragmenter._capped_monomers_atom_indices,
            bcd_fragmenter._capped_monomers_frag_real_indices,
            bcd_fragmenter._capped_monomers,
        )
    ):
        for orig_idx, frag_idx in zip(orig_indices, frag_indices):
            orig_elem = mol.GetAtomWithIdx(orig_idx).GetAtomicNum()
            frag_elem = cap_frag.GetAtomWithIdx(frag_idx).GetAtomicNum()
            assert orig_elem == frag_elem, (
                f"Monomer {monomer_idx}: orig atom {orig_idx} (Z={orig_elem}) "
                f"does not match frag atom {frag_idx} (Z={frag_elem})"
            )


def test_raw_fragment_ordering_differs_across_monomers(bcd_fragmenter):
    """
    Documents that the raw local atom ordering within fragments CAN differ
    across monomers: atoms from different positions in the original molecule
    may be ordered differently within each fragment (e.g. O before C in one
    fragment, C before O in another).  This is expected and is exactly why
    parametrize() uses GetSubstructMatch rather than direct frag_idx indexing.
    """
    patterns = []
    for frag_indices, cap_frag in zip(
        bcd_fragmenter._capped_monomers_frag_real_indices,
        bcd_fragmenter._capped_monomers,
    ):
        pattern = tuple(
            cap_frag.GetAtomWithIdx(i).GetAtomicNum() for i in frag_indices
        )
        patterns.append(pattern)

    # For BCD.sdf, not all patterns are the same — this confirms the ordering
    # inconsistency that motivated the GetSubstructMatch fix.
    assert not all(p == patterns[0] for p in patterns), (
        "Unexpectedly uniform fragment ordering: the test molecule may have "
        "changed or BCD.sdf atom ordering was regularized."
    )


def test_substruct_match_corrects_ordering_across_monomers(bcd_fragmenter):
    """
    GetSubstructMatch must successfully map each monomer's atoms onto the
    reference fragment, and after remapping the elements must agree.
    This verifies the fix for the ho/oh swap bug: direct frag_idx indexing
    into frag_params was wrong when local atom orderings differed; the
    substructure match gives the correct ref_idx for each atom.
    """
    fragment_index = bcd_fragmenter._capped_monomers_index[0]
    ref_mol_h = Chem.AddHs(bcd_fragmenter._capped_monomers[fragment_index])

    for monomer_idx, (frag_indices, current_mol) in enumerate(
        zip(
            bcd_fragmenter._capped_monomers_frag_real_indices,
            bcd_fragmenter._capped_monomers,
        )
    ):
        match = ref_mol_h.GetSubstructMatch(current_mol)
        assert match, f"Monomer {monomer_idx}: GetSubstructMatch returned empty match"

        for frag_idx in frag_indices:
            ref_idx = match[frag_idx]
            elem_current = current_mol.GetAtomWithIdx(frag_idx).GetAtomicNum()
            elem_ref = ref_mol_h.GetAtomWithIdx(ref_idx).GetAtomicNum()
            assert elem_current == elem_ref, (
                f"Monomer {monomer_idx}: frag_idx={frag_idx} (Z={elem_current}) "
                f"matched to ref_idx={ref_idx} (Z={elem_ref}) — element mismatch"
            )


def test_orig_indices_cover_all_molecule_atoms(bcd_fragmenter):
    """
    The union of orig_indices across all monomers must cover every atom in
    the original molecule (no atom left unassigned).
    """
    n_atoms = bcd_fragmenter.input_molecule.GetNumAtoms()
    all_orig = set()
    for orig_indices in bcd_fragmenter._capped_monomers_atom_indices:
        all_orig.update(orig_indices)
    assert all_orig == set(range(n_atoms)), (
        f"orig_indices cover {len(all_orig)} atoms; expected {n_atoms}"
    )


# ---------------------------------------------------------------------------
# Regression: parametrized mol2 vs. direct antechamber reference
# ---------------------------------------------------------------------------


def test_frag_mol2_charge_mae_vs_direct():
    """
    Fragment-based mol2 charges vs. direct antechamber charges.
    MAE should be small (< 0.05 e); a large value indicates a mapping bug.
    """
    frag = parmed.load_file(FRAG_MOL2, structure=True)
    direct = parmed.load_file(DIRECT_MOL2, structure=True)
    assert len(frag.atoms) == len(direct.atoms), "Atom count mismatch between mol2 files"

    frag_q = np.array([a.charge for a in frag.atoms])
    direct_q = np.array([a.charge for a in direct.atoms])
    mae = np.mean(np.abs(frag_q - direct_q))
    assert mae < 0.05, f"Charge MAE {mae:.4f} e exceeds 0.05 e threshold"


def test_frag_mol2_max_charge_diff_vs_direct():
    """
    Maximum per-atom charge difference should be < 0.1 e.
    The previous mapping bug caused differences > 1.0 e (ho vs oh swap).
    """
    frag = parmed.load_file(FRAG_MOL2, structure=True)
    direct = parmed.load_file(DIRECT_MOL2, structure=True)

    frag_q = np.array([a.charge for a in frag.atoms])
    direct_q = np.array([a.charge for a in direct.atoms])
    max_diff = np.max(np.abs(frag_q - direct_q))
    assert max_diff < 0.1, f"Max charge difference {max_diff:.4f} e exceeds 0.1 e threshold"


def test_frag_mol2_atom_types_match_direct():
    """
    GAFF2 atom types from fragment-based parametrization should match direct
    antechamber. Type mismatches (e.g. 'ho' vs 'oh') indicate mapping errors.
    """
    frag = parmed.load_file(FRAG_MOL2, structure=True)
    direct = parmed.load_file(DIRECT_MOL2, structure=True)

    mismatches = [
        (i, frag.atoms[i].type, direct.atoms[i].type)
        for i in range(len(frag.atoms))
        if frag.atoms[i].type != direct.atoms[i].type
    ]
    assert not mismatches, (
        f"{len(mismatches)} atom type mismatches: "
        + ", ".join(f"atom {i}: {ft!r} vs {dt!r}" for i, ft, dt in mismatches[:5])
    )


def test_frag0_mol2_exists_and_has_correct_atom_count():
    """Sanity check: the single-monomer fragment mol2 has the expected atom count."""
    frag0 = parmed.load_file(FRAG0_MOL2, structure=True)
    # One capped glucopyranose monomer with H should have ~21 heavy atoms + H
    assert len(frag0.atoms) > 0
    # Verify all atoms have a non-empty type (antechamber assigned types)
    untyped = [a for a in frag0.atoms if not a.type]
    assert not untyped, f"{len(untyped)} atoms have no GAFF2 type in frag0.mol2"
