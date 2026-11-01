"""McCulloch-Pitts neuron: binary threshold unit. Run: python -m ml.demo.mcculloch_pitts"""


def mp_neuron(inputs, weights, threshold):
    return int(sum(i * w for i, w in zip(inputs, weights)) >= threshold)


GATES = {"AND": ([1, 1], 2), "OR": ([1, 1], 1), "NOT": ([-1], 0)}

if __name__ == "__main__":
    for gate, (w, t) in GATES.items():
        print(gate)
        combos = [(0,), (1,)] if gate == "NOT" else [(0, 0), (0, 1), (1, 0), (1, 1)]
        for x in combos:
            print(f"  {x} -> {mp_neuron(x, w, t)}")
