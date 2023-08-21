import os, glob, shutil, ntpath
from pypdb import get_pdb_file
from raw_data.pdb_data import PDB_CODES, PDB_FORMULAS

RES_MOLS = './parsed_testmols'
BUILDMODEL_DIR = './build_model'

def apply_buildmodel(pdbfile):
    mainwd = os.getcwd()
    temp_pdb_name = os.path.join(BUILDMODEL_DIR, ntpath.basename(pdbfile))
    shutil.copy2(pdbfile, temp_pdb_name)
    os.chdir(os.path.join(mainwd, BUILDMODEL_DIR))
    os.system(f"build_model.exe -f {ntpath.basename(pdbfile)} -lic 2_5287234088611096803.lic -mode superfast")
    
    resname = 'ligand.mol'
    assert os.path.isfile(resname)
    os.chdir(mainwd)
    return os.path.join(BUILDMODEL_DIR, resname)

if __name__ == "__main__":
    for pdb_code in  ['1MRL', '1PFE', '1QZ5']:
        pdb_text = get_pdb_file(pdb_code, filetype='pdb', compression=False)
        with open(os.path.join(RES_MOLS, f"{pdb_code.upper()}.pdb"), 'w') as f:
            f.write(pdb_text)

    for pdbfile in glob.glob(os.path.join(RES_MOLS, '*.pdb')):
        molname = os.path.join(RES_MOLS, 'pdb_' + ntpath.basename(pdbfile).replace('.pdb', '.sdf'))
        if os.path.isfile(molname):
            continue
        
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
