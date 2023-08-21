import os, re, math
from raw_data.csd_data import CCDC_CODES, CCDC_FORMULAS
from ccdc import io
from chemscripts.geom import Molecule, Fragment
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np

RES_MOLS = './parsed_testmols'
MISMATCHED_MOLS = './sus_mols'

def parse_molecular_formula(formula):
    elements = re.findall(r'([A-Z][a-z]?)(\d+)?', formula)
    parsed_formula = {}
    for element, count in elements:
        element = element.strip()
        count = int(count) if count else 1
        parsed_formula[element] = count
    return parsed_formula

def create_largest_component_graph(graph):
    # Calculate connected components
    components = nx.connected_components(graph)

    # Find the largest component
    largest_component = max(components, key=len)

    # Create a new graph for the largest component
    largest_component_graph = graph.subgraph(largest_component).copy()

    # Copy node attributes from the original graph
    for node in largest_component_graph.nodes:
        largest_component_graph.nodes[node].update(graph.nodes[node])

    # Copy edge attributes from the original graph
    for u, v in largest_component_graph.edges:
        largest_component_graph.edges[u, v].update(graph.edges[u, v])

    return largest_component_graph

def add_hydrogen(graph, h_position, carbon_idx):
        newnode_idx = graph.number_of_nodes()
        graph.add_node(newnode_idx, symbol='H', xyz=h_position)
        graph.add_edge(newnode_idx, carbon_idx, type=1)

# Skip this direction [ 1.        ,  0.        ,  0.        ],
tet_dirs = [[-0.33333333,  0.94280992,  0.        ],
            [-0.33333333, -0.47140496, -0.81649219],
            [-0.33333333, -0.47140496,  0.81649219]]
trig_dirs = [
    [-1/2, math.sqrt(3)/2, 0.0],
    [-1/2, -math.sqrt(3)/2, 0.0],
]
syn_to_anti = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
ch_length = 1.089

def restore_Hs_at_carbon(mol, graph):
    valences = {
        'sp':  2,
        'sp2': 3,
        'sp3': 4,
    }

    new_hydrogens = []
    for node in graph.nodes:
        if graph.nodes[node]['symbol'] != 'C':
            continue
        nbt = []
        for nb in graph.neighbors(node):
            nbt.append(graph[node][nb]['type'])
        assert len(nbt) > 0, f"Atom = {node}"

        hybrid = None
        if 3 in nbt:
            hybrid = 'sp'
        elif 2 in nbt or 4 in nbt:
            hybrid = 'sp2'
        elif len(set(nbt)) == 1 and nbt[0] == 1: # Only single bonds
            hybrid = 'sp3'
        assert hybrid is not None, f"Atom = {node}"

        # Additional checks
        if hybrid == 'sp':
            assert set(nbt) == {1,3} or (set(nbt) == {3} and len(nbt) == 1), f"Atom = {node}"
        elif hybrid == 'sp2':
            assert set(nbt) == {4,1} or set(nbt) == {2,1} or (set(nbt) == {4} and len(nbt) >= 2 and len(nbt) <= 3) or (set(nbt) == {2} and len(nbt) == 1), f"Atom = {node}"
        else: # if sp3
            assert set(nbt) == {1} and len(nbt) >= 1 and len(nbt) <= 4, f"Atom = {node}"

        expected_valence = valences[hybrid]
        actual_valence = len(nbt)
        if expected_valence == actual_valence:
            continue

        if hybrid == 'sp':
            raise Exception("Triple bond!!!")
        # No need to complete Hs at triple bonds

        nbs = list(graph.neighbors(node))
        molfrag = Fragment(mol, startatom=(node+1))
        if hybrid == 'sp3' and len(nbs) > 1:
            myframe = molfrag.build_frame(node+1, nbs[0]+1, nbs[1]+1)
            expected_h_positions = []
            for dir in (tet_dirs[1], tet_dirs[2]):
                new_coords = myframe @ (np.array([x * ch_length for x in dir] + [1.0])) # This is an SE(3) representation
                expected_h_positions.append(np.array(new_coords[:3])) # Transform to normal 3D representation
            if len(nbs) == 2:
                for pos in expected_h_positions:
                    new_hydrogens.append({
                        'xyz': pos,
                        'base_carbon': node,
                    })
            elif len(nbs) == 3:
                position_differences = [np.linalg.norm(expected_position - graph.nodes[nbs[2]]['xyz']) for expected_position in expected_h_positions]
                if position_differences[0] > position_differences[1]:
                    new_hydrogens.append({
                        'xyz': expected_h_positions[0],
                        'base_carbon': node,
                    })
                else:
                    new_hydrogens.append({
                        'xyz': expected_h_positions[1],
                        'base_carbon': node,
                    })
        elif hybrid == 'sp3' and len(nbs) == 1:
            # Get some other neighbor of nbs[0] in order to have anti-orientation of dihedral
            remote_neighbor = [i for i in graph.neighbors(nbs[0]) if i != node][0]

            myframe = molfrag.build_frame(node+1, nbs[0]+1, remote_neighbor+1) @ syn_to_anti
            
            expected_h_positions = []
            for dir in tet_dirs:
                new_coords = myframe @ (np.array([x * ch_length for x in dir] + [1.0])) # This is an SE(3) representation
                expected_h_positions.append(np.array(new_coords[:3])) # Transform to normal 3D representation
            for pos in expected_h_positions:
                new_hydrogens.append({
                    'xyz': pos,
                    'base_carbon': node,
                })
            # print(f"FIXED!!!!! Atom = {node}")
        elif hybrid == 'sp2' and len(nbs) == 2:
            # Get some other neighbor of nbs[0] in order to have anti-orientation of dihedral
            myframe = molfrag.build_frame(node+1, nbs[0]+1, nbs[1]+1)

            expected_h_positions = []
            for dir in trig_dirs:
                new_coords = myframe @ (np.array([x * ch_length for x in dir] + [1.0])) # This is an SE(3) representation
                expected_h_positions.append(np.array(new_coords[:3])) # Transform to normal 3D representation
            new_hydrogens.append({
                'xyz': expected_h_positions[1],
                'base_carbon': node,
            })
            # print(f"FIXED!!!!! Atom = {node}")
        elif hybrid == 'sp2' and len(nbs) == 1:
            # Get some other neighbor of nbs[0] in order to have anti-orientation of dihedral
            remote_neighbor = [i for i in graph.neighbors(nbs[0]) if i != node][0]
            myframe = molfrag.build_frame(node+1, nbs[0]+1, remote_neighbor+1) @ syn_to_anti

            expected_h_positions = []
            for dir in trig_dirs:
                new_coords = myframe @ (np.array([x * ch_length for x in dir] + [1.0])) # This is an SE(3) representation
                expected_h_positions.append(np.array(new_coords[:3])) # Transform to normal 3D representation
            
            for pos in expected_h_positions:
                new_hydrogens.append({
                    'xyz': pos,
                    'base_carbon': node,
                })
            # print(f"FIXED!!!!! Atom = {node}")
        else:
            raise Exception('Unexpected case')

    for h_data in new_hydrogens:
        add_hydrogen(graph, h_data['xyz'], h_data['base_carbon'])

def same_element(n1_attrib, n2_attrib):
    return n1_attrib['symbol'] == n2_attrib['symbol']

def same_bondtype(e1_attrib, e2_attrib):
    return e1_attrib['type'] == e2_attrib['type']

def restore_Hs_at_amides(mol, graph):
    amidegroup = nx.Graph()
    amidegroup.add_node(0, symbol='C')
    amidegroup.add_node(1, symbol='N')
    amidegroup.add_node(2, symbol='O')
    amidegroup.add_edge(0, 1, type=1)
    amidegroup.add_edge(0, 2, type=2)

    # Initialize the subgraph isomorphism matcher
    matcher = isomorphism.GraphMatcher(graph, amidegroup, node_match=same_element, edge_match=same_bondtype)

    # Find all matches of the subgraph in the larger graph
    new_hydrogens = []
    for match in matcher.subgraph_isomorphisms_iter():
        rev_match = {value: key for key, value in match.items()}
        nitrogen_idx = rev_match[1]
        
        hydrogen_found = False
        for nb in graph.neighbors(nitrogen_idx):
            if graph.nodes[nb]['symbol'] == 'H':
                hydrogen_found = True
                break
        if hydrogen_found or len(list(graph.neighbors(nitrogen_idx))) == 3: # Normal aminoacids and proline
            continue

        # Need to add the missing hydrogen
        molfrag = Fragment(mol, startatom=(nitrogen_idx+1))
        nbs = list(graph.neighbors(nitrogen_idx))
        assert len(nbs) == 2, f"Atom = {nitrogen_idx}"
        carbon_idx = rev_match[0]

        other_nb = [nb for nb in nbs if nb != carbon_idx][0]
        myframe = molfrag.build_frame(nitrogen_idx+1, carbon_idx+1, other_nb+1)
        expected_h_positions = []
        for dir in trig_dirs:
            new_coords = myframe @ (np.array([x * ch_length for x in dir] + [1.0])) # This is an SE(3) representation
            expected_h_positions.append(np.array(new_coords[:3])) # Transform to normal 3D representation
        new_hydrogens.append({
            'xyz': expected_h_positions[1],
            'base_carbon': nitrogen_idx,
        })
        # print(f"FIXED!!!!! Atom = {nitrogen_idx}")
    
    for h_data in new_hydrogens:
        add_hydrogen(graph, h_data['xyz'], h_data['base_carbon'])


def restore_Hs_at_hydroxyls(mol, graph):
    new_hydrogens = []
    for node in graph.nodes:
        if not(graph.nodes[node]['symbol'] == 'O' and len(list(graph.neighbors(node))) == 1):
            continue
        only_nb = list(graph.neighbors(node))[0]
        if graph[node][only_nb]['type'] != 1:
            continue

        # Need to add the missing hydrogen
        molfrag = Fragment(mol, startatom=(node+1))

        # Get some other neighbor of only_nb in order to have anti-orientation of dihedral
        remote_neighbor = [i for i in graph.neighbors(only_nb) if i != node][0]
        myframe = molfrag.build_frame(node+1, only_nb+1, remote_neighbor+1) @ syn_to_anti

        dir = tet_dirs[0]
        new_coords = myframe @ (np.array([x * ch_length for x in dir] + [1.0])) # This is an SE(3) representation
        expected_h_position = np.array(new_coords[:3]) # Transform to normal 3D representation
        
        new_hydrogens.append({
            'xyz': expected_h_position,
            'base_carbon': node,
        })
    
    for h_data in new_hydrogens:
        add_hydrogen(graph, h_data['xyz'], h_data['base_carbon'])

if __name__ == "__main__":
    # Init the API
    csd_reader = io.EntryReader('CSD')
    
    # All 150-67 molecules from the SI
    for cur_code, cur_formula in zip(CCDC_CODES, CCDC_FORMULAS):
        # print(f"Mol = {cur_code}")
        cur_mol = csd_reader.molecule(cur_code)
        
        # Save using ccdc built-in writer
        sdf_name = os.path.join(RES_MOLS, f"csd_{cur_code}.sdf")
        with io.MoleculeWriter(sdf_name) as mol_writer:
            mol_writer.write(cur_mol)
        
        # Remove the '$$$$' line that crashes ChemCraft
        lines = open(sdf_name, 'r').readlines()
        new_lines = [line for line in lines if '$$$$' not in line]
        with open(sdf_name, 'w') as f:
            f.write(''.join(new_lines))
        
        # Expected composition taken from SI
        expected_composition = parse_molecular_formula(cur_formula)

        # Parse the SDF to check Brutto-formula
        mol = Molecule(sdf=sdf_name)
        # The might be water molecules etc.
        mol.G = create_largest_component_graph(mol.G)
        mol.G = nx.convert_node_labels_to_integers(mol.G)
        restore_Hs_at_carbon(mol, mol.G)
        restore_Hs_at_amides(mol, mol.G)
        restore_Hs_at_hydroxyls(mol, mol.G)

        elements = []
        for node in mol.G.nodes:
            cur_element = mol.G.nodes[node]['symbol']
            if cur_element not in elements:
                elements.append(cur_element)
        
        cur_composition = {element: 0 for element in elements}
        for node in mol.G.nodes:
            cur_element = mol.G.nodes[node]['symbol']
            cur_composition[cur_element] += 1

        if cur_composition != expected_composition:
            # print(f"{repr(cur_composition)} != {repr(expected_composition)}")
            mol.save_sdf(os.path.join(MISMATCHED_MOLS, f"{cur_code}.sdf"))
            for key in cur_composition.keys():
                if key != 'H':
                    assert cur_composition[key] == expected_composition[key]
            
            message = None
            if cur_composition['H'] > expected_composition['H']:
                message = f"{cur_composition['H'] - expected_composition['H']} extra hydrogens ({cur_composition['H']} Hs instead of {expected_composition['H']})"
            else:
                message = f"{expected_composition['H'] - cur_composition['H']} hydrogens less ({cur_composition['H']} Hs instead of {expected_composition['H']})"
            print(f"{cur_code} has {message}")
        
        mol.save_sdf(sdf_name)

