import numpy as np
from qiskit import QuantumCircuit
import os


def draw_circuit(qc, save_path="circuits/circuit.png"):

    # crear carpeta si no existe
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # dibujar circuito
    fig = qc.draw(output="mpl")

    # guardar
    fig.savefig(save_path)

    print(f"Circuit saved to {save_path}")



def qtse(data):
    data_arr = np.asarray(data, dtype=int).ravel()

    n_qubits = 8
    qc = QuantumCircuit(n_qubits)
    a_qubits = [0, 1, 2, 3]
    t_qubits = [4, 5, 6, 7]

    for qubit in t_qubits:
        qc.h(qubit)

    for t_value, a_value in enumerate(data_arr):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

        for a_idx, abit in enumerate(a_bits):
            if abit == "1":
                qc.mcx(t_qubits, a_qubits[a_idx])

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

    return qc


def qtse_p1(data_a, data_p):
    am_arr = np.asarray(data_a, dtype=int).ravel()
    ph_arr = np.asarray(data_p, dtype=int).ravel()

    n_qubits = 12
    qc = QuantumCircuit(n_qubits)

    a_qubits = [0,1,2,3]
    p_qubits = [4,5,6,7]
    t_qubits = [8,9,10,11]

    for qubit in t_qubits:
        qc.h(qubit)

    for t_value, (a_value, p_value) in enumerate(zip(am_arr, ph_arr)):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)
        p_bits = np.binary_repr(int(p_value), width=4)

        # activar controles para seleccionar el estado |t_value>
        for i, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[i])

        # escribir amplitud
        for i, bit in enumerate(a_bits):
            if bit == "1":
                qc.mcx(t_qubits, a_qubits[i])

        # escribir fase
        for i, bit in enumerate(p_bits):
            if bit == "1":
                qc.mcx(t_qubits, p_qubits[i])

        # deshacer X de selección temporal
        for i, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[i])

    return qc


def qtse_p2(data_a,data_p):
    am_arr = np.asarray(data_a, dtype=int).ravel()
    ph_arr = np.asarray(data_p, dtype=int).ravel()

    n_qubits = 8
    qc = QuantumCircuit(n_qubits)
    a_qubits = [0, 1, 2, 3]
    t_qubits = [4, 5, 6, 7]

    for qubit in t_qubits:
        qc.h(qubit)

    for t_value, (a_value, p_value) in enumerate(zip(am_arr, ph_arr)):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)
        p_bits = np.binary_repr(int(p_value), width=4)

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

        for a_idx, abit in enumerate(a_bits):
            if abit == "1":
                qc.mcx(t_qubits, a_qubits[a_idx])
        
        for p_idx, pbit in enumerate(p_bits):
            if(pbit == "1"):
                qc.mcx(t_qubits,a_qubits[p_idx])

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

    return qc


def qtse_p2_trainable(data_a,data_p,w,b):
    am_arr = np.asarray(data_a, dtype=int).ravel()
    ph_arr = np.asarray(data_p, dtype=int).ravel()

    n_qubits = 8
    qc = QuantumCircuit(n_qubits)
    a_qubits = [0, 1, 2, 3]
    t_qubits = [4, 5, 6, 7]

    for qubit in t_qubits:
        qc.h(qubit)

    for t_value, (a_value, p_value) in enumerate(zip(am_arr, ph_arr)):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)
        p_bits = np.binary_repr(int(p_value), width=4)

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

        theta_a = (a_value/15.0)*np.pi
        theta_p = (p_value/15.0)*np.pi

        for a_idx,abit in enumerate(a_bits):
            if abit == "1":
                angle_a = w[a_idx] * theta_a + b[a_idx]
                qc.mcry(angle_a,t_qubits,a_qubits[a_idx],None)

        for p_idx,pbit in enumerate(p_bits):
            if pbit == "1":
                angle_p = w[p_idx] * theta_p + b[p_idx]
                qc.mcry(angle_p,t_qubits,a_qubits[p_idx],None)

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

    return qc

def qtse_trainable(data_a,w,b):
    am_arr = np.asarray(data_a, dtype=int).ravel()

    n_qubits = 8
    qc = QuantumCircuit(n_qubits)
    a_qubits = [0, 1, 2, 3]
    t_qubits = [4, 5, 6, 7]

    for qubit in t_qubits:
        qc.h(qubit)

    for t_value, a_value in enumerate(am_arr):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

        theta_a = (a_value/15.0)*np.pi

        for a_idx,abit in enumerate(a_bits):
            if abit == "1":
                angle_a = w[a_idx] * theta_a + b[a_idx]
                qc.mcry(angle_a,t_qubits,a_qubits[a_idx],None)

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

    return qc

def ry_rz_1(data_t,data_p):
    n_qubits = 5
    qc = QuantumCircuit(n_qubits)
    
    t_qubits =[1, 2, 3, 4]

    for qubit in t_qubits:
        qc.h(qubit)

    qc.h(0)

    for t_value, (timbre, phase) in enumerate(zip(data_t, data_p)):

        t_bits = np.binary_repr(t_value, width=4)

        # seleccionar instante temporal
        for i, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[i])

        theta_t = timbre * np.pi
        theta_p = phase * np.pi

        #qc.mcrz(theta_p, t_qubits, 0)
        qc.mcry(theta_t, t_qubits, 0, None)
        qc.mcrz(theta_p, t_qubits, 0)
        

        # deshacer selección
        for i, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[i])
    qc.h(0)

    return qc


def ry_rz_1_trainable(data_t,data_p,w, b):

    # 8 parametro entrenagarri t_0-t_7 -> 4 parametro, t_8-t_15 -> 4 parametro

    n_qubits = 5
    qc = QuantumCircuit(n_qubits)

    w_t = w[0:2]
    w_p = w[2:4]

    b_t = b[0:2]
    b_p = b[2:4]
    
    t_qubits =[1, 2, 3, 4]

    for qubit in t_qubits:
        qc.h(qubit)

    qc.h(0)

    for t_value, (timbre, phase) in enumerate(zip(data_t, data_p)):

        block = t_value // 8

        t_bits = np.binary_repr(t_value, width=4)

        # seleccionar instante temporal
        for i, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[i])

        theta_t = w_t[block] * timbre + b_t[block]
        theta_p = w_p[block] * phase + b_p[block]

        
        qc.mcry(theta_t, t_qubits, 0, None)
        qc.mcrz(theta_p, t_qubits, 0)
        

        # deshacer selección
        for i, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[i])
    qc.h(0)

    return qc


def qtse_p3(data_a,data_p):
    am_arr = np.asarray(data_a, dtype=int).ravel()
    ph_arr = np.asarray(data_p, dtype=int).ravel()

    n_qubits = 8
    qc = QuantumCircuit(n_qubits)
    a_qubits = [0, 1, 2, 3]
    t_qubits = [4, 5, 6, 7]

    for qubit in t_qubits:
        qc.h(qubit)

    for qubit in a_qubits:
        qc.h(qubit)

    for t_value, (a_value, p_value) in enumerate(zip(am_arr, ph_arr)):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

        theta_p = p_value*np.pi

        for a_idx, abit in enumerate(a_bits):
            if abit == "1":
                qc.mcx(t_qubits, a_qubits[a_idx])
                qc.mcrz(theta_p,t_qubits,a_qubits[a_idx])

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

    for qubit in a_qubits:
        qc.h(qubit)

    return qc


def qtse_p3_trainable(data_t, data_p,w,b):
    am_arr = np.asarray(data_t, dtype=int).ravel()
    ph_arr = np.asarray(data_p, dtype=float).ravel()

    n_qubits = 8
    qc = QuantumCircuit(n_qubits)
    a_qubits = [0, 1, 2, 3]
    t_qubits = [4, 5, 6, 7]

    for qubit in t_qubits:
        qc.h(qubit)

    for qubit in a_qubits:
        qc.h(qubit)

    for t_value, (a_value, p_value) in enumerate(zip(am_arr, ph_arr)):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)

        block = t_value // 4

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

        theta_p = w[block] * p_value + b[block]

        for a_idx, abit in enumerate(a_bits):
            if abit == "1":
                qc.mcx(t_qubits, a_qubits[a_idx])
                qc.mcrz(theta_p,t_qubits,a_qubits[a_idx])

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

    for qubit in a_qubits:
        qc.h(qubit)

    return qc


def build_feature_map(feature_map, data,train=False, w=None,b=None, data_p=None):
    n_qubits = len(data)

    if(train):
        if(feature_map == "qtse"):
            return qtse_trainable(data=data, w=w, b=b)
        if(feature_map == "qtse_p2"):
            return qtse_p2_trainable(data_a=data, data_p=data_p, w=w, b=b)
        if(feature_map=="ryrz_1"):
            return ry_rz_1_trainable(data_t=data, data_p=data_p,w=w,b=b)
        if(feature_map=="qtse_p3"):
            return qtse_p3_trainable(data_t=data, data_p=data_p, w=w, b=b)
        else:
            print("Ez dago kodeketarik izen horrekin")
    else:
        if(feature_map == "qtse"):
            return qtse(data=data)
        if(feature_map == "qtse_p1"):
            return qtse_p1(data_a=data, data_p=data_p)
        if(feature_map == "qtse_p2"):
            return qtse_p2(data_a=data, data_p=data_p)
        if(feature_map=="ryrz_1"):
            return ry_rz_1(data_t=data, data_p=data_p)
        if(feature_map=="qtse_p3"):
            return qtse_p3(data_a=data, data_p=data_p)
        else:
            print("Ez dago kodeketarik izen horrekin")
        
    
