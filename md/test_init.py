import textwrap
import pytest
from init import get_mol2block


MOL2_TEMPLATE = textwrap.dedent("""\
    @<TRIPOS>MOLECULE
    MOL
    @<TRIPOS>ATOM
    {atoms}
    @<TRIPOS>BOND
""")

# Each atom line: id name x y z sybyl_type subst_id subst_name charge
# Positions 50-61 are the element field that get_mol2block rewrites.
def make_atom_line(idx, name, sybyl_type):
    # Build a line that is at least 61 chars; element field at [50:61] is placeholder
    base = f"{idx:>7} {name:<8} 0.000     0.000     0.000     {sybyl_type:<8} 1  MOL       0.0000"
    # Pad/truncate so positions 50-61 are well-defined
    base = base.ljust(72)
    return base


def make_mol2(atoms):
    """atoms: list of (idx, name, sybyl_type)"""
    atom_lines = "\n".join(make_atom_line(*a) for a in atoms)
    return MOL2_TEMPLATE.format(atoms=atom_lines)


def get_element_from_block(block, atom_idx):
    """Extract the rewritten element field (positions 50:61, stripped) for atom line atom_idx (0-based)."""
    in_atom = False
    count = 0
    for line in block.splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if in_atom and line.startswith("@"):
            break
        if in_atom:
            if count == atom_idx:
                return line[50:61].strip()
            count += 1
    return None


def test_single_letter_element(tmp_path):
    """Standard carbon atom name C1 -> element C."""
    mol2 = make_mol2([(1, "C1", "C.3")])
    f = tmp_path / "mol.mol2"
    f.write_text(mol2)
    block = get_mol2block(str(f))
    assert get_element_from_block(block, 0) == "C"


def test_nonstandard_name_strips_to_first_char(tmp_path):
    """Atom name C1A (non-standard) -> element C, not CA."""
    mol2 = make_mol2([(1, "C1A", "C.3")])
    f = tmp_path / "mol.mol2"
    f.write_text(mol2)
    block = get_mol2block(str(f))
    assert get_element_from_block(block, 0) == "C"


def test_two_letter_element_chlorine(tmp_path):
    """Atom name Cl1 -> element Cl (two-letter preserved)."""
    mol2 = make_mol2([(1, "Cl1", "Cl")])
    f = tmp_path / "mol.mol2"
    f.write_text(mol2)
    block = get_mol2block(str(f))
    assert get_element_from_block(block, 0) == "Cl"


def test_two_letter_element_bromine(tmp_path):
    """Atom name Br2 -> element Br (two-letter preserved)."""
    mol2 = make_mol2([(1, "Br2", "Br")])
    f = tmp_path / "mol.mol2"
    f.write_text(mol2)
    block = get_mol2block(str(f))
    assert get_element_from_block(block, 0) == "Br"


def test_mixed_atoms(tmp_path):
    """Multiple atoms with varied names all resolve correctly."""
    mol2 = make_mol2([
        (1, "C1",  "C.3"),
        (2, "C1A", "C.ar"),
        (3, "N2",  "N.am"),
        (4, "O38", "O.2"),
        (5, "Cl1", "Cl"),
        (6, "Br1", "Br"),
    ])
    f = tmp_path / "mol.mol2"
    f.write_text(mol2)
    block = get_mol2block(str(f))
    expected = ["C", "C", "N", "O", "Cl", "Br"]
    for i, elem in enumerate(expected):
        assert get_element_from_block(block, i) == elem, f"atom {i} failed"


def test_non_atom_sections_unchanged(tmp_path):
    """Lines outside @<TRIPOS>ATOM are not modified."""
    mol2 = make_mol2([(1, "C1", "C.3")])
    f = tmp_path / "mol.mol2"
    f.write_text(mol2)
    block = get_mol2block(str(f))
    assert "@<TRIPOS>MOLECULE" in block
    assert "@<TRIPOS>BOND" in block
