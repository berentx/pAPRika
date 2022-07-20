from pathlib import Path

import numpy as np

def eccentricity(points):
    from scipy.spatial.distance import euclidean
    small_latwise = np.min(points[points[:, 0] == np.min(points[:, 0])], 0)
    small_lonwise = np.min(points[points[:, 1] == np.min(points[:, 1])], 0)
    big_latwise = np.max(points[points[:, 0] == np.max(points[:, 0])], 0)
    big_lonwise = np.max(points[points[:, 1] == np.max(points[:, 1])], 0)
    distance_lat = euclidean(big_latwise, small_latwise)
    distance_lon = euclidean(big_lonwise, small_lonwise)
    if distance_lat >= distance_lon:
        major_axis_length = distance_lat
        minor_axis_length = distance_lon
    else:
        major_axis_length = distance_lon
        minor_axis_length = distance_lat
    a = major_axis_length/2
    b = minor_axis_length/2
    ecc = np.sqrt(np.square(a)-np.square(b))/a
    return ecc

def project(points):
    from scipy.linalg import svd

    p = points - np.mean(points, axis=0, keepdims=True)

    # subtract out the centroid and take the SVD
    svd = np.linalg.svd(p.T)

    # Extract the left singular vectors
    left = svd[0]
    n = left[:, -1]

    # project
    return p - np.dot(p, n)[:, np.newaxis] * n[np.newaxis,:]


def analysis(args):
    import MDAnalysis as mda
    import MDAnalysis.transformations as trans
    from MDAnalysis.analysis.density import DensityAnalysis
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
    
    import numpy as np
    from shapely.geometry import MultiPoint
    from rdkit import Chem


    dx = 32
    gridsize = 0.5
    nbin = int(dx / gridsize)
    grid = np.empty((nbin, nbin, nbin))
    count = 0
    areas = []
    ecc = []
    hbonds = []
    skip = 1

    host_mol = Chem.MolFromMolFile(str(args.host), removeHs=False)
    p_monomer = Chem.MolFromSmarts("O1C(O)CCCC1")
    monomers = host_mol.GetSubstructMatches(p_monomer)
    c6 = [set([a.GetIdx() for a in host_mol.GetAtomWithIdx(m[-1]).GetNeighbors() if a.GetSymbol() != 'H']).difference(set(m)).pop() for m in monomers]
    c2 = [set([a.GetIdx() for a in host_mol.GetAtomWithIdx(m[3]).GetNeighbors() if a.GetSymbol() != 'H']).difference(set(m)).pop() for m in monomers]
    c3 = [set([a.GetIdx() for a in host_mol.GetAtomWithIdx(m[4]).GetNeighbors() if a.GetSymbol() != 'H']).difference(set(m)).pop() for m in monomers]
    n_atoms = host_mol.GetNumAtoms()

    host_selection = f'bynum 1:{n_atoms}'
    host_backbone = ' or '.join([f'index {i}' for i in np.concatenate(monomers)])
    water_selection = "resname HOH and name O"

    traj_specs = []
    traj_dir = [d for d in Path('windows').glob('*') if d.is_dir()]

    for d in traj_dir:
        traj = {
            'system_path': str(d/'system.pdb'),
            'traj_path': str(d/'production.dcd')
        }
        traj_specs.append(traj)

    for traj in traj_specs:
        system = mda.Universe(traj['system_path'])
        ref = system.select_atoms(host_backbone)

        tr = -ref.center_of_mass()
        ref.atoms.translate(tr)

        u = mda.Universe(traj['system_path'], traj['traj_path'])
        monomer = u.select_atoms(host_backbone)
        water = u.select_atoms(water_selection)

        transforms = [
            trans.unwrap(u.atoms),
            trans.center_in_box(monomer, center='geometry'),
            trans.wrap(u.atoms, compound='residues'),
            trans.fit_rot_trans(monomer, ref, weights='mass'),
        ]

        u.trajectory.add_transformations(*transforms)
        #u.atoms.write('traj.pdb')

        # hydrophobic barrel length
        ow = u.select_atoms(water_selection)
        D = DensityAnalysis(ow, delta=0.5, xdim=dx, ydim=dx, zdim=dx, gridcenter=ref.center_of_mass())
        D.run(step=skip)
        D.results.density.convert_density('TIP3P')
        grid += D.results.density.grid
        count += 1

        # radius / shape
        primary = u.select_atoms(' or '.join([f'index {i}' for i in c6]))
        secondary = u.select_atoms(' or '.join([f'index {i}' for i in c3 + c2]))

        for ts in u.trajectory[::skip]:
            pp = project(primary.positions)[:, :2]
            sp = project(secondary.positions)[:, :2]
            areas.append((MultiPoint(pp).convex_hull.area, MultiPoint(sp).convex_hull.area))
            ecc.append((eccentricity(pp), eccentricity(sp)))

        # HB
        hydr = [a.GetIdx() for a in host_mol.GetAtoms() if a.GetSymbol() == 'H']
        acct = [a.GetIdx() for a in host_mol.GetAtoms() if a.GetSymbol() in set(('O', 'N'))]
        hbs = HBA(universe=u,
                hydrogens_sel=' or '.join([f'index {i}' for i in hydr]),
                acceptors_sel=' or '.join([f'index {i}' for i in acct]))
        hbs.run(step=skip)

        n_frames = int(len(u.trajectory)/skip)
        nhb = np.zeros(n_frames)
        for hb in hbs.results.hbonds:
            fr = int(hb[0]/skip) - 1
            nhb[fr] += 1
        hbonds.append(nhb)

    # hydrophobic barrel length
    grid /= count
    z = np.mean(grid, axis=(0,1))
    np.savetxt('water.dat', z)

    gridhalf = int(len(z)/2)
    up = np.argmax(z[:gridhalf])
    dn = np.argmax(z[gridhalf:])
    barrel_length = (gridhalf - up) + dn

    density_center = np.min(z[gridhalf-10:gridhalf+10])

    # eccentricity / radius
    avg_diameter = np.sqrt(np.mean(areas, axis=0) / np.pi) * 2
    avg_ecc = np.mean(ecc, axis=0)

    # average intra h-bonds
    avg_hbonds = np.mean(np.concatenate(hbonds))

    print(f'hydrophobic barrel length: {barrel_length * gridsize}')
    print(f'water density at center: {density_center}')
    print(f'primary diameter: {avg_diameter[0]}')
    print(f'primary eccentricity: {avg_ecc[0]}')
    print(f'secondary diameter: {avg_diameter[1]}')
    print(f'secondary eccentricity: {avg_ecc[1]}')
    print(f'average internal H-bonds: {avg_hbonds}')

