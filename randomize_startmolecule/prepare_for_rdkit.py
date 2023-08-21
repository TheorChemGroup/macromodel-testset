import os, sys, glob, ntpath
import networkx as nx
from networkx.algorithms import isomorphism
from charges import CHARGES
from chemscripts.geom import Molecule

CHARGES_MOLS = ['pdb_' + key for key in CHARGES.keys()]
START_DIR = './optrandomized_conformers'
LOG_DIR = './nbo_logs/randomized_geometries'
RES_DIR = './optrandomized_conformers_fixed'

# Types of charged groups:
# 1) Nitro
# 2) 4-coordinated nitrogen
# 3) Phosphates
# 4) Carboxylates
# 5) Acetylacetone anion

def get_frags():
    nitrogenfrag = nx.Graph()
    nitrogenfrag.add_node(0, symbol='N')
    nitrogenfrag.add_node(1)
    nitrogenfrag.add_node(2)
    nitrogenfrag.add_node(3)
    nitrogenfrag.add_node(4)
    nitrogenfrag.add_edge(0, 1, type=1)
    nitrogenfrag.add_edge(0, 2, type=1)
    nitrogenfrag.add_edge(0, 3, type=1)
    nitrogenfrag.add_edge(0, 4, type=1)
    nitrogenfrag_data = {
        'name': 'nitrogen',
        'subgraph': nitrogenfrag,
        'charges': {
            0: +1
        },
        'check_valence': []
    }

    nitrogroupfrag = nx.Graph()
    nitrogroupfrag.add_node(0, symbol='N')
    nitrogroupfrag.add_node(1, symbol='O')
    nitrogroupfrag.add_node(2, symbol='O')
    nitrogroupfrag.add_edge(0, 1, type=1)
    nitrogroupfrag.add_edge(0, 2, type=2)
    nitrogroupfrag_data = {
        'name': 'nitrogroup',
        'subgraph': nitrogroupfrag,
        'charges': {
            0: +1,
            1: -1
        },
        'check_valence': []
    }

    phosphatefrag = nx.Graph()
    phosphatefrag.add_node(0, symbol='P')
    phosphatefrag.add_node(1, symbol='O')
    phosphatefrag.add_node(2, symbol='O')
    phosphatefrag.add_node(3, symbol='O')
    phosphatefrag.add_node(4)
    phosphatefrag.add_node(5)
    phosphatefrag.add_node(6)
    phosphatefrag.add_edge(0, 1)
    phosphatefrag.add_edge(0, 2)
    phosphatefrag.add_edge(0, 3)
    phosphatefrag.add_edge(0, 4)
    phosphatefrag.add_edge(4, 5)
    phosphatefrag.add_edge(3, 6)
    phosphatefrag_data = {
        'name': 'phosphate',
        'subgraph': phosphatefrag,
        'charges': { 1: -1 },
        'check_valence': [],
        'fix_bondtypes': [(0, 1, 1), (0, 2, 2)],
        'protect_atoms': [0]
    }
    
    carboxylatefrag = nx.Graph()
    carboxylatefrag.add_node(0, symbol='C')
    carboxylatefrag.add_node(1, symbol='O')
    carboxylatefrag.add_node(2, symbol='O')
    carboxylatefrag.add_edge(0, 1, type=1)
    carboxylatefrag.add_edge(0, 2, type=2)
    carboxylatefrag_data = {
        'name': 'carboxylate',
        'subgraph': carboxylatefrag,
        'charges': {
            1: -1,
        },
        'check_valence': [(1, 1)]
    }
    
    sulfoxidefrag = nx.Graph()
    sulfoxidefrag.add_node(0, symbol='S')
    sulfoxidefrag.add_node(1, symbol='O')
    sulfoxidefrag.add_node(2, symbol='O')
    sulfoxidefrag.add_edge(0, 1)
    sulfoxidefrag.add_edge(0, 2)
    sulfoxidefrag_data = {
        'name': 'sulfoxide',
        'subgraph': sulfoxidefrag,
        'charges': {},
        'check_valence': [(1, 1), (2, 1)],
        'fix_bondtypes': [(0, 1, 2), (0, 2, 2)],
        'protect_atoms': [0]
    }
    
    sulfonefrag = nx.Graph()
    sulfonefrag.add_node(0, symbol='S')
    sulfonefrag.add_node(1, symbol='O')
    sulfonefrag.add_node(2, symbol='O')
    sulfonefrag.add_node(3, symbol='O')
    sulfonefrag.add_edge(0, 1, type=1)
    sulfonefrag.add_edge(0, 2)
    sulfonefrag.add_edge(0, 3)
    sulfonefrag_data = {
        'name': 'nitrogen',
        'subgraph': sulfonefrag,
        'charges': {},
        'check_valence': [(1, 1), (2, 2), (3, 2)],
        'fix_bondtypes': [(0, 1, 2)],
        'protect_atoms': [0]
    }
    
    acacfrag = nx.Graph()
    acacfrag.add_node(0, symbol='C')
    acacfrag.add_node(1, symbol='C')
    acacfrag.add_node(2, symbol='C')
    acacfrag.add_node(3, symbol='C')
    acacfrag.add_node(4, symbol='O')
    acacfrag.add_node(5, symbol='O')
    acacfrag.add_edge(0, 1, type=1)
    acacfrag.add_edge(0, 2, type=1)
    acacfrag.add_edge(0, 3, type=1)
    acacfrag.add_edge(1, 4, type=2)
    acacfrag.add_edge(2, 5, type=2)
    acacfrag_data = {
        'name': 'nitrogen',
        'subgraph': acacfrag,
        'charges': {
            0: -1,
        },
        'check_valence': [(0, 3)]
    }

    return (nitrogenfrag_data, nitrogroupfrag_data, phosphatefrag_data, carboxylatefrag_data, acacfrag_data, sulfoxidefrag_data, sulfonefrag_data)

def same_element(n1_attrib, n2_attrib):
    if 'symbol' in n1_attrib and 'symbol' in n2_attrib:
        return n1_attrib['symbol'] == n2_attrib['symbol']
    else:
        return True

def same_bondtype(e1_attrib, e2_attrib):
    if 'type' in e1_attrib and 'type' in e2_attrib:
        return e1_attrib['type'] == e2_attrib['type']
    else:
        return True

if __name__ == "__main__":
    for sdf_name in glob.glob(os.path.join(START_DIR, '*.sdf')):
        molname = ntpath.basename(sdf_name).replace('.sdf', '')
        # if molname != "csd_KAQJER":
        #     continue

        # Parse topology from NBO3 log
        log_name = os.path.join(LOG_DIR, f"{molname}.log")
        mol = Molecule(nbo3log=log_name)
        
        # Parse coordinates from orginal SDF
        mol_sdf = Molecule(sdf=sdf_name)
        # print("SDF nodes = " + repr(sorted(list(mol_sdf.G.nodes))))
        xyz, sym = mol_sdf.as_xyz()
        mol.from_xyz(xyz, sym)
        
        # dobreak = True
        # for node in mol.G.nodes:
        #     if mol.G.nodes[node]['symbol'] == 'S':
        #         dobreak = False
        #         break
        # if dobreak:
        #     continue
        print(f"Processing {molname}")

        # Check that all bonds in SDF are present in NBO in some way
        for bondA, bondB in mol_sdf.G.edges:
            if not mol.G.has_edge(bondA, bondB):
                print(f"NBO missed the bond {bondA+1}-{bondB+1}. Adding a single bond manually.")
                mol.G.add_edge(bondA, bondB, type=1)
        for bondA, bondB in mol.G.edges:
            if not mol_sdf.G.has_edge(bondA, bondB):
                print(f"NBO found a strange bond {bondA+1}-{bondB+1}. Removing it manually.")
                mol.G.remove_edge(bondA, bondB)

        # Manual fix of NBO errors
        if molname == 'pdb_1KEG' and mol.G[29-1][38-1]['type'] == 2:
            mol.G[29-1][38-1]['type'] = 1

        fragments = get_frags()
        gr = mol.G
        for fragment_data in fragments:
            subgraph = fragment_data['subgraph']
            charges = fragment_data['charges']
            valence_data = fragment_data['check_valence']
            matcher = isomorphism.GraphMatcher(gr, subgraph, node_match=same_element, edge_match=same_bondtype)

            processed_atoms = []
            protected_atoms = []
            for match in matcher.subgraph_isomorphisms_iter():
                rev_match = {value: key for key, value in match.items()}

                proceed = True
                for sub_idx in charges.keys():
                    full_idx = rev_match[sub_idx]
                    if full_idx in processed_atoms:
                        proceed = False
                        break
                if 'protect_atoms' in fragment_data:
                    for sub_idx in fragment_data['protect_atoms']:
                        full_idx = rev_match[sub_idx]
                        if full_idx in protected_atoms:
                            proceed = False
                            break
                for sub_idx, valence in valence_data:
                    full_idx = rev_match[sub_idx]
                    if len(list(gr.neighbors(full_idx))) != valence:
                        proceed = False
                        break
                if not proceed:
                    continue
                
                # print(f"Matched {fragment_data['name']} frament")

                for sub_idx, charge in charges.items():
                    full_idx = rev_match[sub_idx]
                    assert 'chrg' not in gr.nodes[full_idx], f"Charge of atom {full_idx+1} is already set to {gr.nodes[full_idx]['chrg']}"
                    gr.nodes[full_idx]['chrg'] = charge
                    print(f"Setting the charge of {full_idx+1} to {charge}")
                    processed_atoms.append(full_idx)
                
                if 'protect_atoms' in fragment_data:
                    for sub_idx in fragment_data['protect_atoms']:
                        full_idx = rev_match[sub_idx]
                        protected_atoms.append(full_idx)
            
                if 'fix_bondtypes' in fragment_data:
                    fixbonds_data = fragment_data['fix_bondtypes']
                    for atA, atB, newtype in fixbonds_data:
                        full_idxA = rev_match[atA]
                        full_idxB = rev_match[atB]
                        gr[full_idxA][full_idxB]['type'] = newtype
                        
        
        # Cross check with total charges
        if molname in CHARGES_MOLS:
            expected_charge = CHARGES[molname.split('_')[1]]
        else:
            expected_charge = 0
        total_charge = 0
        for node in gr.nodes:
            if 'chrg' in gr.nodes[node]:
                total_charge += gr.nodes[node]['chrg']
        assert total_charge == expected_charge, f'Mismatch between expected {expected_charge} and obtained {total_charge} total charges.'

        # Save original SDF with updated topology from NBO
        new_sdf_name = os.path.join(RES_DIR, ntpath.basename(sdf_name))
        mol.save_sdf(new_sdf_name)
    
    # """
    # Check that RDKit is able to process these molecules
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    for sdf_name in glob.glob(os.path.join(RES_DIR, '*.sdf')):
        print(sdf_name)
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
        AllChem.EmbedMolecule(mol)
        sdf_string = Chem.MolToMolBlock(mol)
    # """
