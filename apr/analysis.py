from pathlib import Path

import numpy as np
from paprika.io import load_restraints
from paprika import analysis as apr_analysis

def analysis(args):
    guest_restraints = load_restraints(filepath="windows/guest_restraints.json")
    host_restraints = load_restraints(filepath="windows/host_restraints.json")

    free_energy = apr_analysis.fe_calc()
    free_energy.topology = "system.pdb"
    free_energy.trajectory = "production.dcd"
    free_energy.path = 'windows'
    free_energy.restraint_list = guest_restraints + host_restraints

    #if Path("APR_simulation_data.json").exists():
    #    free_energy.collect_data_from_json("APR_simulation_data.json")
    #else:
    #    free_energy.collect_data()
    #    free_energy.save_data("APR_simulation_data.json")
    free_energy.collect_data()
    free_energy.save_data("APR_simulation_data.json")

    free_energy.methods = ['ti-block']
    free_energy.ti_matrix = "diagonal"
    free_energy.bootcycles = 1000
    free_energy.compute_free_energy()
    
    free_energy.compute_ref_state_work([
        guest_restraints[0], guest_restraints[1], None, None,
        guest_restraints[2], None
    ])

    free_energy.save_results("APR_results.json")

    binding_affinity = -1 * (
        free_energy.results["attach"]["ti-block"]["fe"] + \
        free_energy.results["pull"]["ti-block"]["fe"] + \
        free_energy.results["release"]["ti-block"]["fe"] * -1 + \
        free_energy.results["ref_state_work"]
    )
    
    sem = np.sqrt(
        free_energy.results["attach"]["ti-block"]["sem"]**2 + \
        free_energy.results["pull"]["ti-block"]["sem"]**2 + \
        free_energy.results["release"]["ti-block"]["sem"]**2
    )

    print('attach:', free_energy.results["attach"]["ti-block"]["fe"])
    print('pull:', free_energy.results["pull"]["ti-block"]["fe"])
    print('release:', free_energy.results["release"]["ti-block"]["fe"])
    print('ref:', free_energy.results["ref_state_work"])
        

    print(f"The binding affinity = {binding_affinity.magnitude:0.2f} +/- {sem.magnitude:0.2f} kcal/mol")

