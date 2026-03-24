import numpy as np
from pyscf import gto, scf, symm

###########################################
# 1. Drive the symmetry analyzer code
# mol - pyscf molecular object
# mf  - pyscf mean-field object
# R   - q-sc-EOM excitation eigenvector
# returns: irrep (e.g. "A1", "B1" etc.) 
#                     contributing the most
#                     to the excitation. Will
#                     print this prior to 
#                     excitation energy in 
#                     'qsc_ener'
###########################################
###########################################
##########################################
def symmetry_driver(mol,mf,R):
    # Get orbital irreps
    mo_irreps = symm.label_orb_symm(
        mol,
        mol.irrep_name,
        mol.symm_orb,
        mf.mo_coeff
    )
    
    # Convert to string labels (e.g. A1, B2, etc.)
    irrep_labels = mo_irreps
    
    nocc = mol.nelectron // 2
    nmo = len(irrep_labels)
    nvir = nmo - nocc
    
    # Decompose the R vector into R1 and R2 components 
    # for later post-processing
    r1 = R[:nocc*nvir]
    r2 = R[nocc*nvir:]
    R1 = r1.reshape((nocc,nvir))
    R2 = r2.reshape((nocc,nocc,nvir,nvir))

    #R1 = np.random.rand(nocc, nvir)
    #R2 = np.random.rand(nocc, nocc, nvir, nvir)
    weights = get_sym_weights(R1,R2,irrep_labels,nocc,nvir,mol)

    # Get string of dominant irrep
    dominant_irrep = print_sym_info(weights,mol)
    return dominant_irrep


############################################
# Irrep multiplication helpers
############################################

def multiply_irreps(mol, ir1, ir2):
    """Return product irrep"""
    return ir1 ^ ir2 #symm.direct_prod( ir1, ir2)

def multiply_many(mol, irreps):
    """Multiply multiple irreps"""
    result = irreps[0]
    for ir in irreps[1:]:
        result ^= ir#= multiply_irreps(mol, result, ir)
    return result

############################################
# 5. Accumulate symmetry weights
############################################

def get_sym_weights(R1,R2,irrep_labels,nocc,nvir,mol):
    weights = {}
    
    # --- R1 contributions ---
    for i in range(nocc):
        for a in range(nvir):
            amp = R1[i, a]
            if abs(amp) < 1e-6:
                continue
    
            ir_i = irrep_labels[i]
            ir_a = irrep_labels[nocc + a]
            print(ir_a)
            tmp_iri = symm.irrep_name2id(mol.groupname,ir_i)
            tmp_ira = symm.irrep_name2id(mol.groupname,ir_a)
            print(tmp_iri,tmp_ira)
            ir_total = multiply_irreps(mol, tmp_ira, tmp_iri)
    
            weights[ir_total] = weights.get(ir_total, 0.0) + abs(amp)**2
    
    # --- R2 contributions ---
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    amp = R2[i, j, a, b]
                    if abs(amp) < 1e-6:
                        continue
    
                    irreps = [
                        symm.irrep_name2id(mol.groupname,irrep_labels[nocc + a]),
                        symm.irrep_name2id(mol.groupname,irrep_labels[nocc + b]),
                        symm.irrep_name2id(mol.groupname,irrep_labels[i]),
                        symm.irrep_name2id(mol.groupname,irrep_labels[j])
                    ]
    
                    ir_total = multiply_many(mol, irreps)
    
                    weights[ir_total] = weights.get(ir_total, 0.0) + abs(amp)**2
   
    return weights


############################################
# 6. Normalize and print
############################################
def print_sym_info(weights,mol):
    total = sum(weights.values())
    print(weights.items())
    print("\n=== Symmetry Decomposition ===")
    for ir, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(ir)
        print(f"{symm.irrep_id2name(mol.groupname,ir):9s} : {w/total:.4f}")
    print(mol.groupname)
    dominant = max(weights, key=weights.get)
    info = symm.irrep_id2name(mol.groupname,dominant)
    print("\nDominant symmetry:", symm.irrep_id2name(mol.groupname,dominant))
    return info



############################################
# 0. Build molecule with symmetry
############################################

mol = gto.M(
    atom="O 0 0 0; O 0 0 1.1",
    basis="cc-pvdz",
    spin=2,
    symmetry='D2h'
)

mf = scf.RHF(mol)
mf.kernel()

##################################
##################################
# This info is meant to simply drive our
# symmetry routines; can/will be omitted
# after integration
#################################
mo_irreps = symm.label_orb_symm(
    mol,
    mol.irrep_name,
    mol.symm_orb,
    mf.mo_coeff
)
nocc = mol.nelectron // 2
nmo = len(mo_irreps)
nvir = nmo - nocc

R1 = np.random.rand(nocc, nvir)
R2 = np.random.rand(nocc, nocc, nvir, nvir)
R = np.concatenate([R1.ravel(), R2.ravel()])



###########################################
# 0a. Send ***ONE*** q-sc-EOM eigenvector to 
#     "symmetry_driver" at a time. Returns
#     the irrep dominating that particular 
#     excitation
##########################################
irrep_test = symmetry_driver(mol,mf,R)

print("Outside logic result:", irrep_test)

