import os, glob, ntpath, shutil, re
from chemscripts.geom import Molecule, Fragment
from raw_data.pdb_data import PDB_CODES, PDB_FORMULAS

from rdkit import Chem
from rdkit.Chem import AllChem

SI_DIR = './raw_data'
RES_MOLS = './parsed_testmols'
SI_MOLNAME = 'ci5001696_si_002.txt'
BUILDMODEL_DIR = './build_model'

def apply_buildmodel(pdbfile):
    print("====================================")
    print(f"Processing {pdbfile}")
    print("====================================")
    mainwd = os.getcwd()
    temp_pdb_name = os.path.join(BUILDMODEL_DIR, ntpath.basename(pdbfile))
    shutil.copy2(pdbfile, temp_pdb_name)
    os.chdir(os.path.join(mainwd, BUILDMODEL_DIR))
    os.system(f"build_model.exe -f {ntpath.basename(pdbfile)} -lic 2_5287234088611096803.lic")
    
    resname = 'ligand.mol'
    if not os.path.isfile(resname):
        pdbname = 'protein.pdb'
        assert os.path.isfile(pdbname)
        mol = Chem.MolFromPDBFile(pdbname, removeHs=False)
        assert mol is not None
        with Chem.SDWriter(resname) as f:
            f.write(mol)
    os.chdir(mainwd)
    return os.path.join(BUILDMODEL_DIR, resname)


def parse_molecular_formula(formula):
    elements = re.findall(r'([A-Z][a-z]?)(\d+)?', formula)
    parsed_formula = {}
    for element, count in elements:
        element = element.strip()
        count = int(count) if count else 1
        parsed_formula[element] = count
    return parsed_formula


def fix_long_element(el):
    return el[0].upper() + el[1:].lower()
        

def get_failed_codes():
    failed_codes = []
    for code, formula in zip(PDB_CODES, PDB_FORMULAS):
        code = code.upper()
        sdfname = os.path.join(RES_MOLS, f'pdb_{code}.sdf')
        if not os.path.isfile(sdfname):
            print(f"{code} not found")
            failed_codes.append(code)
            continue
        
        mol = Molecule(sdf=sdfname)
        
        elements = []
        for node in mol.G.nodes:
            cur_element = mol.G.nodes[node]['symbol']
            cur_element = fix_long_element(cur_element)
            if cur_element not in elements:
                elements.append(cur_element)
        cur_composition = {element: 0 for element in elements}
        for node in mol.G.nodes:
            cur_element = mol.G.nodes[node]['symbol']
            cur_element = fix_long_element(cur_element)
            cur_composition[cur_element] += 1
        
        expected_composition = parse_molecular_formula(formula)
        for key in expected_composition.keys():
            if key not in cur_composition or (key != 'H' and cur_composition[key] != expected_composition[key]) or \
               (key == 'H' and abs(cur_composition[key] - expected_composition[key]) > 3):
                print(f"{code}: {repr(cur_composition)} vs. {repr(expected_composition)}")
                failed_codes.append(code)
                break
    return failed_codes
    
if __name__ == "__main__":
    filename = os.path.join(SI_DIR, SI_MOLNAME)
    lines = open(filename, 'r').readlines()

    curline = 0
    read = True
    cur_lines = []
    curname = lines[0].replace("\n","")
    while curline < len(lines):
        if read:
            print("READ: " + lines[curline].replace('\n', ''))
            cur_lines.append(lines[curline])

        if lines[curline] == "$$$$\n":
            assert not read
            read = True
            cur_lines = []
            if curline == len(lines) - 1:
                break
            curname = lines[curline + 1].replace("\n", "")
            print("Parsion " + curname)
        
        if lines[curline] == "M  END\n":
            res_name = os.path.join(RES_MOLS, f"pdb_{curname}.sdf")
            with open(res_name, "w") as f:
                f.write("".join(cur_lines))
            
            read = False
        curline += 1
    
    for sdfname in glob.glob(os.path.join(RES_MOLS, '*.sdf')):
        suppl = Chem.SDMolSupplier(sdfname)
        mol = suppl[0]
        assert mol is not None
        pdbname = os.path.join(RES_MOLS, ntpath.basename(sdfname).replace('.sdf', '.pdb'))
        with Chem.PDBWriter(pdbname) as f:
            f.write(mol)

    for pdbfile in glob.glob(os.path.join(RES_MOLS, '*.pdb')):
        if "pdb_1E9W" not in pdbfile and "pdb_2VYP" not in pdbfile:
            continue
        molname = os.path.join(RES_MOLS, ntpath.basename(pdbfile).replace('.pdb', '.sdf'))
        # if os.path.isfile(molname):
            # continue
        
        molname_temp = apply_buildmodel(pdbfile)
        shutil.move(molname_temp, molname)
        mollines = open(molname, 'r').readlines()
        lines_to_delete = []
        for i, line in enumerate(mollines):
            if '$$$$' in line:
                lines_to_delete.append(i)
        for i in lines_to_delete:
            del mollines[i]
        with open(molname, 'w') as f:
            f.write(''.join(mollines))
            
    failed_codes = get_failed_codes()
    print("Failed on " + repr(failed_codes))
    