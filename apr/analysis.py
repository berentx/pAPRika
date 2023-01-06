from pathlib import Path

import numpy as np
from paprika.io import load_restraints
from paprika import analysis as apr_analysis

def print_binding(results, method="ti-block"):
    fe_a = -1 * results["attach"][method]["fe"]
    sem_a = results["attach"][method]["sem"]
    fe_p = -1 * results["pull"][method]["fe"]
    sem_p = results["pull"][method]["sem"]
    fe_r = results["release"][method]["fe"]
    sem_r = results["release"][method]["sem"]
    fe_ref = -1 * results["ref_state_work"]

    binding = fe_a + fe_p + fe_r + fe_ref
    sem = np.sqrt(sem_a ** 2 + sem_p ** 2 + sem_r ** 2)

    print(f"Analysis - {method}")
    print("-" * 25)
    print(f"Attach free-energy  = {fe_a:6.2f} +/- {sem_a:0.2f} kcal/mol")
    print(f"Pull free-energy    = {fe_p:6.2f} +/- {sem_p:0.2f} kcal/mol")
    print(f"Release free-energy = {fe_r:6.2f} +/- {sem_r:0.2f} kcal/mol")
    print(f"Ref state-work      = {fe_ref:6.2f}")
    print(f"Binding free-energy = {binding:6.2f} +/- {sem:0.2f} kcal/mol\n")


def get_fe_convergence(results, method="ti-block"):
    convergence = {}

    # Free energy
    attach = results["attach"][method]["fraction_fe"]
    pull = results["pull"][method]["fraction_fe"]
    release = results["release"][method]["fraction_fe"]
    
    keys = sorted(attach.keys())
    unit = attach[keys[0]].units

    convergence["fe_a"] = -1 * np.array([np.array(attach[i]) for i in keys]) * unit
    convergence["fe_p"] = -1 * np.array([np.array(pull[i]) for i in keys]) * unit
    convergence["fe_r"] = np.array([np.array(release[i]) for i in keys]) * unit
    convergence["ref"] = -1 * results["ref_state_work"]

    convergence["fractions"] = np.array([i for i in attach])

    # Error
    attach = results["attach"][method]["fraction_sem"]
    pull = results["pull"][method]["fraction_sem"]
    release = results["release"][method]["fraction_sem"]

    convergence["sem_a"] = np.array([np.array(attach[i]) for i in keys])
    convergence["sem_p"] = np.array([np.array(pull[i]) for i in keys])
    convergence["sem_r"] = np.array([np.array(release[i]) for i in keys])

    convergence["binding"] = (
        convergence["fe_a"]
        + convergence["fe_p"]
        + convergence["fe_r"]
        + convergence["ref"]
    )
    convergence["sem"] = (
        convergence["sem_a"] ** 2
        + convergence["sem_p"] ** 2
        + convergence["sem_r"] ** 2
    ) ** 0.5

    return convergence


def create_dg_plots(restraints, results):
    attach_string = (
        "0.00 0.40 0.80 1.60 2.40 4.00 5.50 8.65 11.80 18.10 24.40 37.00 49.60 74.80 100.00"
    )
    release_string = (
        "0.00 0.40 0.80 2.40 5.50 11.80 24.40 49.60 74.80 100.00"
    )
    attach_fractions = [float(i) / 100 for i in attach_string.split()]
    release_fractions = [float(i) / 100 for i in release_string.split()]
    
    initial_distance = restraints[1].pull['target_initial']
    final_distance = restraints[1].pull['target_final']
    dr = 0.5
    pull_distances = np.arange(initial_distance, final_distance + dr, dr)
    
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.errorbar(
        attach_fractions,
        np.array(results["attach"]["ti-block"]["fe_matrix"][0, :]),
        yerr=np.array(results["attach"]["ti-block"]["sem_matrix"][0, :]),
        fmt=".-",
    )
    plt.xlabel("lambda")
    plt.ylabel(r"$\Delta G$ (kcal/mol)")
    plt.title("Attach phase")
    
    plt.subplot(1, 3, 2)
    plt.errorbar(
        pull_distances,
        np.array(results["pull"]["ti-block"]["fe_matrix"][0, :]),
        yerr=np.array(results["pull"]["ti-block"]["sem_matrix"][0, :]),
        fmt=".-",
    )
    plt.xlabel("r (Angstrom)")
    plt.ylabel(r"$\Delta G$ (kcal/mol)")
    plt.title("Pull phase")
    
    ax1 = plt.subplot(1, 3, 3)
    ax1.errorbar(
        release_fractions,
        np.array(results["release"]["ti-block"]["fe_matrix"][0, :]),
        yerr=np.array(results["release"]["ti-block"]["sem_matrix"][0, :]),
        fmt=".-",
    )
    ax1.set_xlabel("lambda")
    ax1.set_ylabel(r"$\Delta G$ (kcal/mol)")
    ax1.set_title("Release phase")
    ax1.invert_xaxis()
    plt.savefig('dg.png')


def create_convergence_plots(restraints, convergence):
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1)
    plt.errorbar(
        convergence["fractions"], convergence["fe_a"], yerr=convergence["sem_a"], fmt="o-"
    )
    plt.xlabel("fraction")
    plt.ylabel(r"$\Delta G$ (kcal/mol)")
    plt.title("Attach phase")
    
    plt.subplot(2, 3, 2)
    plt.errorbar(
        convergence["fractions"], convergence["fe_p"], yerr=convergence["sem_p"], fmt="o-"
    )
    plt.xlabel("fraction")
    plt.ylabel(r"$\Delta G$ (kcal/mol)")
    plt.title("Pull phase")
    
    ax1 = plt.subplot(2, 3, 3)
    ax1.errorbar(
        convergence["fractions"], convergence["fe_r"], yerr=convergence["sem_r"], fmt="o-"
    )
    ax1.set_xlabel("lambda")
    ax1.set_ylabel(r"$\Delta G$ (kcal/mol)")
    ax1.set_title("Release phase")
    # ax1.invert_xaxis()
    
    plt.subplot(2, 1, 2)
    plt.errorbar(
        convergence["fractions"], convergence["binding"], yerr=convergence["sem"], fmt="o-"
    )
    plt.xlabel("fraction")
    plt.ylabel(r"$\Delta G$ (kcal/mol)")
    plt.title("Binding free energy")
    plt.savefig('convergence.png')


def analysis(args):
    guest_restraints = load_restraints(filepath="windows/guest_restraints.json")
    host_restraints = load_restraints(filepath="windows/host_restraints.json")

    free_energy = apr_analysis.fe_calc()
    free_energy.topology = "system.pdb"
    free_energy.trajectory = "production*.dcd"
    free_energy.path = 'windows'
    free_energy.restraint_list = guest_restraints + host_restraints

    if Path("APR_simulation_data.json").exists():
        free_energy.collect_data_from_json("APR_simulation_data.json")
    else:
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

    print_binding(free_energy.results, method='ti-block')

    convergence = get_fe_convergence(results, method="ti-block")
    create_dg_plots(guest_restraints, free_energy.results)
    create_convergence_plots(guest_restraints, convergence)

