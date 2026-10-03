import time

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


class QuantumKernel:
    def __init__(self, qubits=4, reps=2):
        if not 2 <= qubits <= 8:
            raise ValueError('Research kernel supports 2–8 qubits.')
        self.qubits, self.reps = qubits, reps

    def states(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.qubits or not np.isfinite(values).all():
            raise ValueError('Finite input features must match the qubit count.')
        circuit = zz_feature_map(self.qubits, reps=self.reps, entanglement='linear')
        return np.asarray([Statevector.from_instruction(circuit.assign_parameters(row)).data for row in values])

    def matrix(self, left, right):
        return np.clip(np.abs(left.conj() @ right.T) ** 2, 0, 1).real


class QuantumComparison:
    def __init__(self, seed=42):
        self.seed = seed
        self.kernel = QuantumKernel()
        self.transform = make_pipeline(StandardScaler(), PCA(n_components=4, random_state=seed), MinMaxScaler(feature_range=(0, np.pi), clip=True))
        self.quantum_model = None
        self.classical_model = None
        self.training_hashes = []
        self.validation_hashes = []

    def fit(self, train_x, train_y, val_x, val_y):
        if len(train_x) > 256:
            raise ValueError('Cap both comparison models at the same 256 training examples for local statevector experiments.')
        if len(train_x) < 8 or set(train_y) != {0, 1} or set(val_y) != {0, 1}:
            raise ValueError('Both train and validation need independent normal and defective samples; at least 8 training samples are required.')
        train = self.transform.fit_transform(train_x)
        validation = self.transform.transform(val_x)
        started = time.perf_counter()
        self.train_states = self.kernel.states(train)
        train_kernel = self.kernel.matrix(self.train_states, self.train_states)
        val_kernel = self.kernel.matrix(self.kernel.states(validation), self.train_states)
        kernel_seconds = time.perf_counter() - started
        reports = {}
        for name, x, vx, kind in [('quantum', train_kernel, val_kernel, 'precomputed'), ('classical', train, validation, 'rbf')]:
            best = None
            tuning = []
            started = time.perf_counter()
            for c in [.1, 1., 10.]:
                classifier = SVC(C=c, kernel=kind, class_weight='balanced', random_state=self.seed)
                classifier.fit(x, train_y)
                score = float(balanced_accuracy_score(val_y, classifier.predict(vx)))
                tuning.append(dict(C=c, validation_balanced_accuracy=score))
                if best is None or score > best[0]:
                    best = (score, classifier, c)
            setattr(self, name + '_model', best[1])
            reports[name] = dict(selected_C=best[2], validation_balanced_accuracy=best[0], tuning=tuning,
                                 fit_and_selection_seconds=time.perf_counter() - started + (kernel_seconds if name == 'quantum' else 0))
        return dict(models=reports, train_size=len(train), validation_size=len(validation), feature_dimensions=4,
                    quantum_backend='Qiskit exact statevector on classical CPU; no quantum hardware',
                    quantum_advantage='not established; requires independent test results and uncertainty analysis')

    def predict(self, embeddings):
        transformed = self.transform.transform(embeddings)
        kernel = self.kernel.matrix(self.kernel.states(transformed), self.train_states)
        return dict(quantum=self.quantum_model.predict(kernel), classical=self.classical_model.predict(transformed),
                    quantum_margin=self.quantum_model.decision_function(kernel))
