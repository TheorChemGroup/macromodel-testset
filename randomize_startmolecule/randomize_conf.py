import glob, ntpath, json, os
import ringo
import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from chemscripts.geom import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem

EXCEPTIONS = ['pdb_1KEG', 'pdb_1EHL']
NISO_JSON = 'niso_timings.json'
START_DIR = './initial_conformers'
RES_DIR = './start_conformers'

def same_element(n1_attrib, n2_attrib):
    return n1_attrib['symbol'] == n2_attrib['symbol']

def same_bondtype(e1_attrib, e2_attrib):
    return e1_attrib['type'] == e2_attrib['type']

def get_amide_bonds(sdf_name):
    mol = Molecule(sdf=sdf_name)
    graph = mol.G

    amidegroup = nx.Graph()
    amidegroup.add_node(0, symbol='C')
    amidegroup.add_node(1, symbol='N')
    amidegroup.add_node(2, symbol='O')
    amidegroup.add_node(3, symbol='H')
    amidegroup.add_node(4, symbol='C')
    amidegroup.add_edge(0, 1, type=1)
    amidegroup.add_edge(0, 2, type=2)
    amidegroup.add_edge(3, 1, type=1)
    amidegroup.add_edge(4, 1, type=1)

    # Initialize the subgraph isomorphism matcher
    matcher = isomorphism.GraphMatcher(graph, amidegroup, node_match=same_element, edge_match=same_bondtype)
    
    # Find all matches of the subgraph in the larger graph
    amide_bonds = []
    full_amide_bonds = []
    for match in matcher.subgraph_isomorphisms_iter():
        rev_match = {value: key for key, value in match.items()}
        nitrogen_idx = rev_match[1]
        carbon_idx = rev_match[0]
        oxygen_idx = rev_match[2]
        hydrogen_idx = rev_match[3]
        amide_bonds.append((carbon_idx+1, nitrogen_idx+1))
        full_amide_bonds.append((oxygen_idx+1, carbon_idx+1, nitrogen_idx+1, hydrogen_idx+1))
    return amide_bonds, full_amide_bonds

def check_limits(dih_value, lower_limit, upper_limit):
    upd_dih = dih_value
    while upd_dih < lower_limit:
        upd_dih += 360.0
    while upd_dih > upper_limit:
        upd_dih -= 360.0
    return (lower_limit < upd_dih) and (upd_dih < upper_limit), f"Limits=[{lower_limit}, {upper_limit}] Old = {dih_value} New = {upd_dih}"

def print_warnings(stage_name, molname):
    # Print important warnings if there were any
    warnings = ringo.get_status_feed(important_only=True)
    if len(warnings) > 0:
        print('---------------------')
        print(f"Please study these {stage_name} warnings carefully (mol={ntpath.basename(molname)}):")
        for item in warnings:
            print("* " + item['message'])
            if 'atoms' in item:
                print("Atoms = "+ repr(item['atoms']))
        print('---------------------')

def gen_ringo():
    with open(NISO_JSON, "r") as f:
        niso_data = json.load(f)
    
    ringo.set_radius_multiplier(0.5)
    # Performs short parallel conformation search for 3 test molecules
    for sdf_name in glob.glob(os.path.join(START_DIR, '*.sdf')):
        molname = ntpath.basename(sdf_name).replace('.sdf', '')
        if os.path.isfile(os.path.join(RES_DIR, f'{molname}.sdf')):
            continue
        if molname in EXCEPTIONS:
            continue
        print(f"Processing {sdf_name}")
        fixed_dihedrals, full_fixed_dihedrals = get_amide_bonds(sdf_name)

        # Initialize Molecule object
        mol = ringo.Molecule(sdf=sdf_name, request_free=fixed_dihedrals) # , require_best_sequence=True
        print_warnings('initialization', molname)
        dofs_list, dofs_values = mol.get_ps()
        
        custom_dof_limits = {}
        warnings = ringo.get_status_feed(important_only=False)
        manual_dihedrals = {}
        temp_pool = ringo.Confpool()
        start_mol = Molecule(sdf=sdf_name)
        xyz, sym = start_mol.as_xyz()
        temp_pool.include_from_xyz(np.array(xyz), "conf")
        for req_dih, full_req_dihedral in zip(fixed_dihedrals, full_fixed_dihedrals):
            found = False
            for i, item in enumerate(dofs_list):
                if (req_dih[0], req_dih[1]) == (item[1]+1, item[2]+1) or \
                    (req_dih[0], req_dih[1]) == (item[2]+1, item[1]+1):
                    found = True
                    assert abs(temp_pool[0].z(*[i+1 for i in item]) - dofs_values[i]) < 0.01
                    custom_dof_limits[i] = [dofs_values[i] - 10*ringo.DEG2RAD, dofs_values[i] + 10*ringo.DEG2RAD] # Normalize?
                    break
            for item in warnings:
                if item['subject'] != ringo.IK_NOT_APPLIED:
                    continue
                if req_dih[0] in item['atoms'] and req_dih[1] in item['atoms']:
                    found = True
            if not found:
                manual_dihedrals[full_req_dihedral] = temp_pool[0].z(*full_req_dihedral)
                print(f"Unable to enforce {repr(full_req_dihedral)}.")
        mol.customize_sampling(custom_dof_limits)

        # Create pool for future conformers
        p = ringo.Confpool()

        # Perform Monte-Carlo with generation time limit of 10 seconds
        rmsd_settings = 'default'
        for item in niso_data:
            if item['mol'] == ntpath.basename(sdf_name).replace('.sdf', '') and item['niso'] > 500:
                biggest_frag_atoms = mol.get_biggest_ringfrag_atoms()
                rmsd_settings = rmsd_settings = {
                        'isomorphisms': {
                            'ignore_elements': [node for node in start_mol.G.nodes if node not in biggest_frag_atoms],
                        },
                        'rmsd': {
                            'threshold': 0.2,
                            'mirror_match': True,
                        }
                    }

        mcr_kwargs = {
            'rmsd_settings': rmsd_settings,
            'nthreads': 16,
            'max_conformers': 10,
            # 'timelimit': 400,
        }
        if len(manual_dihedrals) > 0:
            def filter_function(m):
                for dihedral, value in manual_dihedrals.items():
                    if abs(m.z(*dihedral) - value) > 10.0:
                        return False
                return True
            mcr_kwargs['filter'] = filter_function
        results = ringo.run_confsearch(mol, pool=p, **mcr_kwargs)
        print(repr(results))
        print(f"Generated {len(p)} conformers of {ntpath.basename(sdf_name)}")
        print_warnings('confsearch', molname)

        # Save conformational pool to file
        # p.descr = lambda m: f"Conformer {m.idx}"
        # p.save('check.xyz')

        for m in p:
            for dof_idx, limits in custom_dof_limits.items():
                atoms = [i+1 for i in dofs_list[dof_idx]]
                dih_value = m.z(*atoms) * ringo.RAD2DEG
                okay, info = check_limits(dih_value, limits[0]*ringo.RAD2DEG, limits[1]*ringo.RAD2DEG)
                if not okay:
                    raise Exception("ERROR: " + info)
        
        if len(manual_dihedrals) > 0:
            for m in p:
                assert filter_function(m), "Filter didn't work"

        i = 0
        cur_rmsd = 0
        temp_pool.atom_symbols = p.atom_symbols
        temp_pool.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        temp_pool.generate_isomorphisms()
        while cur_rmsd <= 1.4 and i < len(p):
            temp_pool.include_from_xyz(p[i].xyz, "conf"+str(i))
            cur_rmsd, _, __ = temp_pool[0].rmsd(temp_pool[i+1])
            print(cur_rmsd)
            i += 1
            
        if cur_rmsd > 1.4:
            start_mol.from_xyz(p[i-1].xyz, p.atom_symbols)
            start_mol.save_sdf(os.path.join('start_conformers', ntpath.basename(sdf_name)))
        else:
            print(f"FAILED on {ntpath.basename(sdf_name)}")

    # Remove temporary file
    ringo.cleanup()


def gen_rdkit():
    for molname in EXCEPTIONS:
        sdf_name = os.path.join(START_DIR, f'{molname}.sdf')

        p = ringo.Confpool()
        start_mol = Molecule(sdf=sdf_name)
        xyz, sym = start_mol.as_xyz()
        p.include_from_xyz(np.array(xyz), "conf")
        p.atom_symbols = sym
        p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        p.generate_isomorphisms()

        # Parse input SDF file and generate multiple conformers
        mol = Chem.SDMolSupplier(sdf_name, removeHs=False)[0]
        assert mol is not None
        
        # Generate conformers continuously for MAX_TIME seconds
        i = 0
        while True:
            # Embed molecule with random coordinates
            AllChem.EmbedMolecule(mol) # useRandomCoords=True

            # Write molecule as XYZ file
            geom = np.zeros((mol.GetNumAtoms(), 3))
            for i in range(mol.GetNumAtoms()):
                pos = mol.GetConformer().GetAtomPosition(i)
                geom[i, 0] = pos.x
                geom[i, 1] = pos.y
                geom[i, 2] = pos.z
            p.include_from_xyz(geom, f"Conformer {i}")
            cur_rmsd, _, __ = p[0].rmsd(p[len(p)-1])
            if cur_rmsd > 1.0:
                start_mol.from_xyz(p[len(p)-1].xyz, p.atom_symbols)
                start_mol.save_sdf(os.path.join('start_conformers', ntpath.basename(sdf_name.replace('_rdkit', ''))))
                print(f"Done with {molname}")
                break
            else:
                pass
            i += 1

def final_rmsd_check():
    # for sdf_name in glob.glob(os.path.join('release_assemble', 'optimize_testset', 'opt_conformers', '*.sdf')):
    for sdf_name in glob.glob(os.path.join(RES_DIR, '*.sdf')):
        molname = ntpath.basename(sdf_name).replace('.sdf', '')
        print(f"Processing {molname}")
        
        # initial_sdf = f'./release_assemble/test_systems/{molname}.sdf'
        initial_sdf = os.path.join(START_DIR, f'{molname}.sdf')
        p = ringo.Confpool()
        initial_mol = Molecule(sdf=initial_sdf)
        xyz, sym = initial_mol.as_xyz()
        p.include_from_xyz(np.array(xyz), "initial")
        p.atom_symbols = sym
        p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        p.generate_isomorphisms()

        modif_mol = Molecule(sdf=sdf_name)
        xyz, sym = modif_mol.as_xyz()
        p.include_from_xyz(np.array(xyz), "modified")
        
        cur_rmsd, _, __ = p[0].rmsd(p[1])
        if cur_rmsd < 1.4:
            print(f"WARNING {molname}: RMSD={cur_rmsd}")

def pull_charges():
    for sdf_name in glob.glob(os.path.join(RES_DIR, '*.sdf')):
        molname = ntpath.basename(sdf_name).replace('.sdf', '')
        print(f"Processing {molname}")

        start_sdf = os.path.join(START_DIR, f'{molname}.sdf')
        mol_w_charges = Molecule(sdf=start_sdf)
        mol_new = Molecule(sdf=sdf_name)
        xyz, sym = mol_new.as_xyz()
        mol_w_charges.from_xyz(xyz, sym)
        mol_w_charges.save_sdf(sdf_name)

def optimize():
    failed_mols = []
    for sdf_name in glob.glob(os.path.join(RES_DIR, '*.sdf')):
        molname = ntpath.basename(sdf_name).replace('.sdf', '')
        print(f"Processing {molname}")
        
        ccmol = Molecule(sdf=sdf_name)
        graph = ccmol.G
        m = Chem.Mol()
        mol = Chem.EditableMol(m)
        for atom in graph.nodes:
            new_atom = Chem.Atom(graph.nodes[atom]['symbol'])
            if 'chrg' in graph.nodes[atom]:
                new_atom.SetFormalCharge(graph.nodes[atom]['chrg'])
            new_idx = mol.AddAtom(new_atom)
            assert new_idx == atom

        for edge in graph.edges:
            mol.AddBond(*edge, Chem.BondType(graph[edge[0]][edge[1]]['type']))
        
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)

        # Create a new conformer for the molecule
        conf = Chem.Conformer(mol.GetNumAtoms())
        # Set the 3D coordinates for each atom in the conformer
        for atom in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(atom, graph.nodes[atom]['xyz'])
        # Add the conformer to the molecule
        mol.AddConformer(conf)

        # Perform MMFF optimization on the molecule using the provided coordinates
        return_code = -1
        while return_code != 0:
            return_code = AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=1000) # Use confId=0 to indicate the first (and only) conformer

        geom = np.zeros((mol.GetNumAtoms(), 3))
        for i in range(mol.GetNumAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            geom[i, 0] = pos.x
            geom[i, 1] = pos.y
            geom[i, 2] = pos.z
        
        p = ringo.Confpool()
        p.include_from_xyz(geom, "conf")
        _, sym = ccmol.as_xyz()
        p.atom_symbols = sym
        p.generate_connectivity(0, mult=1.3)

        optimized_graph = p.get_connectivity()
        optimized_bonds = set(edge for edge in optimized_graph.edges)
        older_bonds = set(edge for edge in graph.edges)
        older_unique = []
        for bond in older_bonds:
            if bond not in optimized_bonds:
                older_unique.append(bond)
        optim_unique = []
        for bond in optimized_bonds:
            if bond not in older_bonds:
                optim_unique.append(bond)
        if older_bonds != optimized_bonds:
            failed_mols.append(molname)
            print(f"WARNING {molname} bonds: older_unique={older_unique} optim_unique={optim_unique}")
        
        ccmol.from_xyz(geom, sym)
        ccmol.save_sdf(sdf_name)

if __name__ == "__main__":
    gen_ringo()
    gen_rdkit()
    final_rmsd_check()
    pull_charges()
    optimize()
