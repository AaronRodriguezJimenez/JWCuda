"""
 This example illustrates the explicit construction of the Jordan-Wigner
 transformation using electron integrals and results computed from pyscf.
 As and easy-to-check case, we construct the electronic Hamiltonian in the
 Pauli basis for the H2 molecule.
 A recent paper with deeper details about this construction can be consulted:
 The Electronic Structure of the Hydrogen Molecule: A Tutorial Exercise in Classical and Quantum Computation
 https://arxiv.org/abs/2307.04290

 pyscf AND openfermion TUTORIAL

Quantum computing algorithms begin with a Hamiltonian H and a reference state |Φ⟩.
Typically, we want H as a linear combination of Paulis, and |Φ⟩ as a computational basis state.
This script describes the work involved to obtain these objects as a basic example  from "scratch"
(here we use pyscf and openfermion).
"""
import numpy            # Always useful.
import pyscf            # Knows how to do chemistry.
import openfermion      # Knows how to put chemistry on a quantum computer.

##########################################################################################
# PySCF phase: build out the molecule and obtain the molecular-orbital integrals.

""" BUILD OUT A MOLECULE in PySCF

pySCF has a very sophisticated internal representation for molecules.
The following are the main input variables for molecule definition in pySCF:\

charge - determines the number of electrons.
spin  - determines the relative number of spin "up" and spin "down" electrons.
basis - defines the functional form of the "orbitals" which a single-electron wavefunction 
        is decomposed into. These are like the s,p,d,f,... orbitals of hydrogen. 
        The most common choice for quantum information is "sto-3g"
symmetry - point-group `symmetry` of the molecule. pySCF has relatively limited support 
        for point-groups.
"""
molecule = pyscf.gto.M(
    # SET THE PHYSICS
    atom = [                        # There are many ways to specify the geometry,
        ['Li',  (0, 0, 0)],         #   including reading from a file.
        ['H',   (0, 0, 1.5)],
    ],
    charge = 0,                     # charge = num protons - num electrons
    spin = 0,                       # spin = num α electrons - num β electrons
    # SET THE MATH
    basis = "sto-3g",               # Sets the number of orbitals.
    symmetry = True,                # Makes the math nicer.
)


""" FETCH THE ATOMIC INTEGRALS

By restricting our electronic wavefunctions to a finite `basis` of orbitals,
we are able to represent our real-space Hamiltonian with a finite matrix.
Each of those matrix elements is defined by some integral over real-space
of the orbital wavefunctions.

The Hamiltonian has several components (kinetic, Coulomb attraction, Coulomb repulsion)
which can be broken down into those involving one electron, or two.
    
Zeroth order integrals - that is, there is a constant shift to the energy
    from the Coulomb repulsion of all the nuclei.

One-electron integrals - define an N⨯N matrix of one body integrals, There are two types
    of one-body integrals, an electron's Coulomb interaction with all the nuclei,
    and an electron's self-interaction (aka kinetic energy). At the level of quantum 
    computing, there is no reason to distinguish them, so we just add them together.

Two-electron integrals - define an N⨯N⨯N⨯N array of two body integrals, yhe only type of
    two-body integral comes from the electronic repulsion interaction (ERIs).
    
"""
nuc = molecule.energy_nuc()             # Nuclear repulsion.
ao_kin = molecule.intor('int1e_kin')    # Electronic kinetic energy.
ao_nuc = molecule.intor('int1e_nuc')    # Nuclear-electronic attraction.
ao_obi = ao_kin + ao_nuc                # Single-electron hamiltonian.
ao_eri = molecule.intor('int2e')        # Electonic repulsion interaction.

""" RUN HARTREE-FOCK TO FIND MOLECULAR ORBITAL BASIS
We defined the so-called ATOMIC orbitals by setting the `basis` above.

The next step is to compute the Molecular Orbitals, which are linear combinations 
of atomic orbitals forming an eigenbasis of the Fock operator.

All we need is the matrix which describes those linear combinations, so we can use it 
as a basis rotation to transform the atomic integrals into molecular integrals.

The main reason to use the molecular orbital basis is that it produces a very reasonable
and well-defined starting point for various quantum algorithms, however localized basis
sets, ot the so-called Natural Orbitals are also useful in various contexts.  

"""

scf = pyscf.scf.RHF(molecule)   # This initializes the calculation but does not run.
scf.max_cycle = 1000            # Lets optimization run longer if it needs to.
scf.run()                       # This line runs the self-consistent optimization.
U = scf.mo_coeff                # Basis transformation from atomic to molecular orbitals.

""" TRANSFORM ATOMIC INTEGRALS INTO MOLECULAR INTEGRALS

Our N⨯N and N⨯N⨯N⨯N array of integrals are currently with respect to atomic orbitals.
We just want to do a linear transformation to put them in terms of molecular orbitals.

This is really simple written out as a tensor contraction.
"""

full_obi = U.T @ ao_obi @ U                     # Rotated single-body integrals.
full_eri = pyscf.ao2mo.incore.full(ao_eri, U)   # Rotated two-body integrals.

""" IDENTIFY THE RELEVANT ORBITALS (OPTIONAL)

The full simulation may involve several orbitals.

The Hartree-Fock reference state is a single basis state in the molecular orbital basis;
each spin orbital is definitely occupied (low energy) or unoccupied (high energy).

One can reasonably expect that orbitals with particularly low energy
compared to others (aka "core" orbitals) will remain approximately occupied 
in a more exact ansatz.

Similarly, molecular orbitals with particularly high energy
may remain approximately unoccupied.

Therefore we can proceed as follows:

- Select unoccupied orbitals to leave out (simply be omitted from one and two body integrals).
- Select core orbitals that will be frozen in their occupied state
    the corresponding electrons frozen in those orbitals still 
    interact with active electrons, the integrals we retain will need to be modified accordingly.
    
This is a simple but tedious "downfolding" procedure, which openfermion will handle.
We just need to tell it which core orbital indices to `freeze`, and which orbital indices to 
treat as `active`.

The unoccupied orbitals you want to omit are simply all those orbitals not specified in 
`freeze` or `active`.

"""

freeze = [0]            # Orbitals to be traced over, modifying numbers for the rest.
active = [1,2,5]        # Orbitals to keep. All others will simply be dropped.

##########################################################################################
# OpenFermion phase: select the active space and build out a fermionic operator.

""" TRANSFORM INTEGRALS ONTO ACTIVE SPACE

This step is handled by OpenFermion

This first line is the downfolding onto the active space.

Each of the four orbitals appears in the two body integral in different ways,
so there is a difference between `h[p,q,r,s]` and `h[p,r,q,s]`.

Unfortunately, openfermion chooses to order its two body integrals differently than pyscf,
so we have to permute (`transpose`) the indices.

"""

core, obi, eri = openfermion.ops.representations.get_active_space_integrals(
    full_obi, full_eri.transpose(0,2,3,1),
    # Transpose to account for tensor index conventions between pyscf and openfermion.
    freeze, active,
)

""" BUILD OUT THE SPIN-ORBITAL TENSORS

We now need to take our molecular integrals and construct the coefficients for each one- 
and two-body spin orbital interaction appearing in the second-quantized Hamiltonian.

This is another simple but tedious antisymmetrization procedure;
openfermion will handle it.

"""

h0 = nuc + core         # Tracing out core orbitals shifts the constant term.
h1, h2 = openfermion.ops.representations.get_tensors_from_integrals(
    obi, eri,
)

""" CONSTRUCT THE FERMIONIC OPERATOR

The final step of this phase is to construct the second-quantized Hamiltonian.
This is done in the following way

"""
interop = openfermion.InteractionOperator(h0, h1, h2)
fermiop = openfermion.get_fermion_operator(interop)

##########################################################################################
# Qubit mapping phase (also OpenFermion): perform the Jordan-Wigner mapping.

""" JW-MAPPING => QUBIT HAMILTONIAN

The Jordan-Wigner mapping is the step at which we convert out fermion (second quantization) 
Hamiltonian into the corresponding Qubit Hamiltonian (expressed as a linear combination of Pauli strings).

Other mappings that can be used ar:
- The Bravyi-Kitaev mapping
- The Parity mapping

However, up to this point we will not consider them.
For more info see: 
https://quantumai.google/openfermion/tutorials/jordan_wigner_and_bravyi_kitaev_transforms

"""
hamiltonian_jw = openfermion.transforms.jordan_wigner(fermiop)
print(hamiltonian_jw)

exit()


# Check Information:
ηα, ηβ = molecule.nelec     # Total α and β electron counts.
ηα -= len(freeze)           # Account for filled core orbitals.
ηβ -= len(freeze)           #       "                   "
η = ηα + ηβ                 # Active electron count.
nspatial = len(active)      # Active spatial orbital count.
nspin = 2 * nspatial        # Active spin orbital count.

""" DESIGN THE BINARY CODE FOR THE PARITY MAPPING

The openfermion package has one-liners to perform all three of these;
however, we will also be applying two-qubit tapering in this tutorial,
so it is more convenient to use a slightly more elaborate framework
which uses the language of binary codes.

Thus, below, we construct the binary code corresponding to the JW mapping,
specifying a number of qubits two fewer than the number of spin orbitals
(anticipating the two Z2 symmetry reductions constructed in the next cell).

"""
basecode = openfermion.transforms.jordan_wigner_code(nspin - 2)
α_code = openfermion.checksum_code(nspatial, ηα & 1)
β_code = openfermion.checksum_code(nspatial, ηβ & 1)
stagger = openfermion.interleaved_code(nspin)
tapercode = stagger * (α_code + β_code)

""" CONCATENATE THE CODES AND APPLY IT TO THE FERMIONIC OPERATOR

The reason the binary code formalism is so nice is that you can string together 
multiple codes in sequence with code concatenation, which openfermion has elected to 
implement using the * operator. So qubit tapering is compatible with any other mapping technique.

"""
binarycode = tapercode * basecode
qubitop = openfermion.binary_code_transform(fermiop, binarycode)

""" ENCODE THE REFERENCE KET

The Hartree-Fock estimate of the multi-electron wavefunction for the ground state of a 
molecule with η electrons consists of a single configuration with the η lowest-energy 
spin orbitals being occupied.

In the Jordan-Wigner mapping (with openfermion ordering `αβαβ`), that yields the very intuitive 
reference state `|1..10..0⟩`.


Fortunately, the binary code is *defined* in terms of what it does to each bitstring.

The openfermion `BinaryCode` object has the `encoder` attribute, which is a linear transformation 
(represented with a sparse matrix) that acts on binary vectors.

So here, we prepare our `|1..10..0⟩` state as a binary vector, and then we simply do the matrix-vector 
multiplication (% 2 to keep it binary).

"""
reference = numpy.zeros(nspin, dtype=int)
for i in range(ηα): reference[2*i] = 1
for i in range(ηβ): reference[2*i+1] = 1
ket = (binarycode.encoder @ reference) % 2
print("Reference RHF Ket:", ket)

##########################################################################################
# Serialization phase: write the qubit operator and reference ket compactly to a file.

""" SERIALIZE IN SYMPLECTIC FORM

At this point, we have a Pauli sum and a reference state in the target qubit mapping.
All that remains is to export them for other codes, not necessarily in Python.

We will use the `numpy` data format,
    specifically an `npz` file saving four arrays:
- `ket`: the binary vector representing the bitstring reference state
- `C`: the float vector giving coefficients for each Pauli term
- `X`, `Z`: parallel integer vectors identifying the Pauli word in each Pauli term

The integers in X and Z are the so-called symplectic notation for Paulis.
Writing the integers as length-n bitstrings, the 1's identify which qubits have an X or Z.
If a qubit has both an X and a Z, it is interpreted as having a Y.
So e.g. the Pauli word `XIYZ` would have bitstrings x `1010` and z `0011`,
    so the `X` and `Z` vectors would have integers x=10 and z=3.

"""
n = len(ket)
X = []; Z = []; C = []
for actions, c in qubitop.terms.items():
    x = 0; z = 0
    for (q, σ) in actions:
        mask = 1<<(n-q-1)                       # Selects a specific qubit.
        if σ == "X" or σ == "Y": x ^= mask      # XORs the specific qubit.
        if σ == "Z" or σ == "Y": z ^= mask      #   "               "
    X.append(x); Z.append(z); C.append(c)
numpy.savez("examples/JW_example.npz", X=X, Z=Z, C=C, ket=ket)