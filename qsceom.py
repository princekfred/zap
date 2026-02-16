import vqeex 
import exc

def ee_exact(symbols, geometry, active_electrons, active_orbitals, charge,params,shots=0):
    # Build the electronic Hamiltonian
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, basis="sto-3g", method="pyscf", active_electrons=active_electrons, active_orbitals=active_orbitals, charge=charge)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    # Map excitations to the wires the UCCSD circuit will act on
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires=range(qubits)

    null_state = np.zeros(qubits,int)
    list1 = exc.inite(active_electrons,qubits)
    values =[]
    for t in range(1):
        if shots==0:
            #dev = qml.device("lightning.qubit", wires=qubits)
            dev = qml.device("lightning.qubit", wires=qubits)
        else:
            #dev = qml.device("lightning.qubit", wires=qubits,shots=shots)
            dev = qml.device("lightning.qubit", wires=qubits,shots=shots)
        #circuit for diagonal part
        @qml.qnode(dev)
        def circuit_d(params, occ,wires, s_wires, d_wires, hf_state):
            for w in occ:
                qml.X(wires=w)
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.expval(H)
        #circuit for off-diagonal part
        @qml.qnode(dev)
        def circuit_od(params, occ1, occ2,wires, s_wires, d_wires, hf_state):
            for w in occ1:
                qml.X(wires=w)
            first=-1
            for v in occ2:
                if v not in occ1:
                    if first==-1:
                        first=v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first,v])
            for v in occ1:
                if v not in occ2:
                    if first==-1:
                        first=v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first,v])
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.expval(H)
        #final M matrix
        M = np.zeros((len(list1),len(list1)))
        for i in range(len(list1)):
            for j in range(len(list1)):
                if i == j:
                    M[i,i] = circuit_d(params, list1[i], wires, s_wires, d_wires, null_state)
        #print("diagonal parts done")
        for i in range(len(list1)):
            for j in range(len(list1)):
                if i!=j:
                    Mtmp = circuit_od(params, list1[i],list1[j],wires, s_wires, d_wires, null_state)
                    M[i,j]=Mtmp-M[i,i]/2.0-M[j,j]/2.0
        #print("off diagonal terms done")
        #ERROR:not subtracting the gs energy
        eig,evec=np.linalg.eig(M)
        values.append(np.sort(eig))
        
        hf_state = np.array(range(active_electrons))  # Hartree-Fock occupied orbitals

    
    def r1r2_from_evec(eig, evec, list1, hf_state, state_idx=1, outfile="out"):
        # Sort eigenvalues and get corresponding eigenvectors
        idx = np.argsort(eig)
        eig = eig[idx]
        evec = evec[:, idx]

        vector = evec[:, state_idx]

        pr = []
        pr.append("R1/R2")
        pr.append("Excitations | Coefficients")

        for det_idx, coeff in enumerate(vector):
            det = list1[det_idx]
            holes = [hf for hf in hf_state if hf not in det]
            particles = [virt for virt in det if virt not in hf_state]
            s = ""
            for p, h in zip(particles, holes):
                    s += "; "
            s += f"{p}^, {h}"
            pr.append(f"{s}	| {coeff.item()}")

        with open(outfile, "w") as f:
            for line in pr:
                f.write(line + "\n")
        return eig, evec, vector
        # Sort eigenvalues and get corresponding eigenvectors
    eig, evec, vector = r1r2_from_evec(eig, evec, list1, hf_state, state_idx=1, outfile="out_r1_r2.txt")
    print('exact eigenvalues:\n', total_energy)
    return values

total_energy = ee_exact(symbols, geometry, active_electrons, active_orbitals, charge, params)


