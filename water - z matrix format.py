# --------------------------------------------------------------------
# ******************  Importing libraries ****************************
# --------------------------------------------------------------------

import timeit
import numpy as np
from matplotlib import pyplot as plt
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.algorithms import VQEUCCFactory
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

start = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)
# --------------------------------------------------------------------
# ***************************  Dictionary ****************************
# --------------------------------------------------------------------

mapper = JordanWignerMapper()
ansatz = UCCSD()
optimizer = SLSQP()
estimator = Estimator()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)
numpy_solver = NumPyMinimumEigensolver()

# --------------------------------------------------------------------
# ************  Energy Calculation For Single Point ******************
# --------------------------------------------------------------------
#   Molecule coordinates in z-matrices (compact) format
H2O = 'H; O 1 1.08; H 2 1.08 1 104.5'
driver = PySCFDriver(H2O.format(), unit=DistanceUnit.ANGSTROM, charge=0, spin=0, basis='sto3g')

full_problem = driver.run()

# ------------ Active Space Transformation ------------
#   1 HOMO - 1 LUMO
#as_transformer = ActiveSpaceTransformer(2, 2, active_orbitals=[4, 5])
#   2 HOMO - 2 LUMO
as_transformer = ActiveSpaceTransformer(4, 4, active_orbitals=[3, 4, 5, 6])

as_problem = as_transformer.transform(full_problem)

#   Hamiltonian in terms of Fermionic Operators:
fermionic_op = as_problem.hamiltonian.second_q_op()
# print(fermionic_op)

#   Hamiltonian in terms of Pauli matrices:
qubit_op = mapper.map(fermionic_op)
# print(qubit_op)


vqe_factory = VQEUCCFactory(estimator, ansatz, optimizer)
algorithm = GroundStateEigensolver(converter, vqe_factory)
es_result = algorithm.solve(as_problem)
uccsd_energy = es_result.total_energies[0]
print("Ground State Energy of O-H bonding: \n", uccsd_energy)

# --------------------------------------------------------------------
# ************ Energy Calculation Over O - H bonding *****************
# --------------------------------------------------------------------
#   Molecule coordinates in z-matrices (compact) format
H2O = 'H; O 1 1.08; H 2 {} 1 104.5'
distances = [x * 0.05 + 0.650 for x in range(40)]
energies = np.empty(len(distances))
classical_energy = []
uccsd_energy = []



for i, distance in enumerate(distances):
    driver = PySCFDriver(H2O.format(distance), basis='sto3g')
    problem = driver.run()
    as_transformer = ActiveSpaceTransformer(2, 2, active_orbitals=[4, 5])
    problem = as_transformer.transform(problem)
    #   reference energy: classical calculation:
    classical_solver = GroundStateEigensolver(converter, numpy_solver)
    classical_result = classical_solver.solve(problem)
    classical_energy += [classical_result.total_energies[0]]
    #   quantum - classical calculation:
    vqe_factory = VQEUCCFactory(estimator, ansatz, optimizer)
    algorithm = GroundStateEigensolver(converter, vqe_factory)
    uccsd_result = algorithm.solve(problem)
    uccsd_energy += [uccsd_result.total_energies[0]]

# print(uccsd_energy)
print("Reference Ground State Energy of O-H bonding over a distance: \n", min(classical_energy),'\n')
print("USSCD Ground State Energy of O-H bonding over a distance: \n", min(uccsd_energy), '\n')


stop = timeit.default_timer()
runtime = stop - start
print('\n \n \n', 'Run Time: ', runtime, 'sec', ' or ', runtime / 60, 'min', '\n')
# --------------------------------------------------------------------
# ************ Energy Calculation Over O - H bonding *****************
# --------------------------------------------------------------------
plt.plot(distances, uccsd_energy, color='red', label='UCCSD')
plt.plot(distances, classical_energy, color='blue', label='Exact Energy')
plt.grid(True, linestyle='-.', linewidth=0.5, which='major')
plt.title("Ground State Energy Curve of Water Molecule")
plt.xlabel("O-H band length")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.show()
