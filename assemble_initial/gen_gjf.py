import glob, os, ntpath
from chemscripts.geom import Molecule
from chemscripts import utils
from charges import CHARGES

SDF_DIR = './parsed_testmols'
GJF_DIR = './nbo_gjfs'
LOG_DIR = './nbo_logs'

if __name__ == "__main__":
    for sdfname in glob.glob(os.path.join(SDF_DIR, '*.sdf')):
        expected_log_name = ntpath.basename(sdfname).replace('.sdf', '.log')
        if os.path.isfile(os.path.join(LOG_DIR, expected_log_name)):
            continue
        
        pdb_code = ntpath.basename(sdfname).replace('.sdf', '').replace('pdb_', '')
        if pdb_code in CHARGES:
            charge = CHARGES[pdb_code]
        else:
            charge = 0
    
        mol = Molecule(sdf=sdfname)
        xyzs, syms = mol.as_xyz()
        gjfname = os.path.join(GJF_DIR, ntpath.basename(sdfname).replace('.sdf', '.gjf'))
        utils.write_gjf(xyzs, syms, 'nbo', gjfname,
                            subs={
                                'nproc': 22,
                                'chrg': charge,
                            })
