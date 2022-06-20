import parmed as pmd
from rdkit import Chem
from paprika import restraints

def setup_static_restraints(structure, windows, H1, H2, H3, D1, D2, D3, G1, G2):
    static_restraints = []

    structure = pmd.load_file("complex/apr.prmtop",
                              "complex/apr.rst7",
                              structure = True)
    r = restraints.static_DAT_restraint(restraint_mask_list = [D1, H1],
                                        num_window_list = windows,
                                        ref_structure = structure,
                                        force_constant = 5.0,
                                        amber_index=False)
    
    static_restraints.append(r)

    r = restraints.static_DAT_restraint(restraint_mask_list = [D2, D1, H1],
                                        num_window_list = windows,
                                        ref_structure = structure,
                                        force_constant = 100.0,
                                        amber_index=False)
    
    static_restraints.append(r)

    r = restraints.static_DAT_restraint(restraint_mask_list = [D3, D2, D1, H1],
                                        num_window_list = windows,
                                        ref_structure = structure,
                                        force_constant = 100.0,
                                        amber_index=False)
    
    static_restraints.append(r)


    r = restraints.static_DAT_restraint(restraint_mask_list = [D1, H1, H2],
                                        num_window_list = windows,
                                        ref_structure = structure,
                                        force_constant = 100.0,
                                        amber_index=False)
    
    static_restraints.append(r)
    
    r = restraints.static_DAT_restraint(restraint_mask_list = [D2, D1, H1, H2],
                                        num_window_list = windows,
                                        ref_structure = structure,
                                        force_constant = 100.0,
                                        amber_index=False)
    
    static_restraints.append(r)
    
    r = restraints.static_DAT_restraint(restraint_mask_list = [D1, H1, H2, H3],
                                        num_window_list = windows,
                                        ref_structure = structure,
                                        force_constant = 100.0,
                                        amber_index=False)
    
    static_restraints.append(r)


    return static_restraints
