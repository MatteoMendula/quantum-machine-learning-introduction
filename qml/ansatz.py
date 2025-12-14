import pennylane as qml
class Ansatz:
    def entanglementCircuit(self, weights, wires, type, gate, indexLayerStart=1, indexgate=3, lag=1):
        if type == 'circle':
            # manage circle case
            for i in range(len(wires)):
                ctrl = i
                target = (i + lag) % (len(wires))
                if gate == 'CNOT':
                    qml.CNOT((wires[ctrl], wires[target]))
                elif gate == 'CZ':
                    qml.CZ((wires[ctrl], wires[target]))
                elif gate == 'RX':
                    qml.CRX(weights[indexLayerStart][indexgate][i], (wires[ctrl], wires[target]))
                elif gate == 'RZ':
                    qml.CRZ(weights[indexLayerStart][indexgate][i], (wires[ctrl], wires[target]))
            




        
        elif type == 'single':
            # manage single
            wires_new = wires.copy()
            wires_new.reverse()
            for i in range(len(wires) -1):
                ctrl = i
                target = i + 1
                if gate == 'CNOT':
                    qml.CNOT((wires_new[ctrl], wires_new[target]))
                elif gate == 'CZ':
                    qml.CZ((wires_new[ctrl], wires_new[target]))
                elif gate == 'RX':
                    qml.CRX(weights[indexLayerStart][indexgate][i], (wires_new[ctrl], wires_new[target]))
                elif gate == 'RZ':
                    qml.CRZ(weights[indexLayerStart][indexgate][i], (wires_new[ctrl], wires_new[target]))  

        elif type == 'all':
            #manage all'''
            idxGate = indexgate
            for i in range(len(wires)):
                for j in range(len(wires)):
                    if i != j:
                        ctrl = i
                        target = j
                        if gate == 'CNOT':
                            qml.CNOT((wires[i], wires[j]))
                        elif gate == 'CZ':
                            qml.CZ((wires[i], wires[j]))
                        elif gate == 'RX':
                            qml.CRX(weights[indexLayerStart][idxGate][i], (wires[ctrl], wires[target]))
                        elif gate == 'RZ':
                            qml.CRZ(weights[indexLayerStart][idxGate][i], (wires[ctrl], wires[target])) 

                idxGate += 1 


    def superposition(self, wires):
        for i in range(len(wires)):
            qml.Hadamard(wires[i])


    def circuit_single(self, name, weights, wires):
        if name == 'strongly':
            qml.StronglyEntanglingLayers(weights, wires)
        elif name == 'basic':
            qml.BasicEntanglerLayers(weights, wires)
        
        elif name == 'circuit_9':
            for i in range(len(weights)):
                self.superposition(wires)
                self.entanglementCircuit(weights, wires, 'single', 'CZ', i, 1, lag=i % (len(wires) - 1) + 1)
                qml.AngleEmbedding(weights[i][len(wires)], wires, rotation='X')
        elif name == 'circuit_10':
            for i in range(len(weights)):
                qubitWire = wires.copy()
                qubitWire.reverse()
                self.entanglementCircuit(weights, qubitWire, 'circle', 'CZ', i, 1, lag=i % (len(wires) - 1) + 1)
                qml.AngleEmbedding(weights[i][len(wires)], wires, rotation='Y')
        elif name == 'circuit_13':
            for i in range(len(weights)): 
                qml.AngleEmbedding(weights[i][0], wires, rotation='Y')
                self.entanglementCircuit(weights, wires, 'circle', 'RZ', i, 1, lag=i % (len(wires) - 1) + 1)
        elif name == 'circuit_14':
            for i in range(len(weights)):
                qml.AngleEmbedding(weights[i][0], wires, rotation='Y')
                self.entanglementCircuit(weights, wires, 'circle', 'RX', i, 1, lag=i % (len(wires) - 1) + 1)
        elif name == 'circuit_15':
            for i in range(len(weights)):
                qml.AngleEmbedding(weights[i][0], wires, rotation='Y')
                self.entanglementCircuit(weights, wires, 'circle', 'CNOT', i, 1, lag=i % (len(wires) - 1) + 1)

        else:
            for i in range(len(weights)):
                qml.AngleEmbedding(weights[i][0], wires, rotation='X')
                qml.AngleEmbedding(weights[i][1], wires, rotation='Z')
                if name == 'circuit_2':
                    self.entanglementCircuit(weights, wires, 'single', 'CNOT', i, 2)
                elif name == 'circuit_3':
                    self.entanglementCircuit(weights, wires, 'single', 'RZ', i, 2)
                elif name == 'circuit_4':
                    self.entanglementCircuit(weights, wires, 'single', 'RX', i, 2)
                elif name == 'circuit_5':
                    self.entanglementCircuit(weights, wires, 'all', 'RZ', i, 2)
                    qml.AngleEmbedding(weights[i][2 + len(wires)], wires, rotation='X')
                    qml.AngleEmbedding(weights[i][2 + len(wires)], wires, rotation='Z')

                elif name == 'circuit_6':
                    self.entanglementCircuit(weights, wires, 'all', 'RX', i, 2)
                    qml.AngleEmbedding(weights[i][2 + len(wires)], wires, rotation='X')
                    qml.AngleEmbedding(weights[i][3 + len(wires)], wires, rotation='Z')
                elif name == 'circuit_18':
                    self.entanglementCircuit(weights, wires, 'circle', 'RZ', i, 2, lag=i % (len(wires) - 1) + 1)
                elif name == 'circuit_19':
                    self.entanglementCircuit(weights, wires, 'circle', 'RX', i, 2, lag=i % (len(wires) - 1) + 1)


                

                

    def shape_weights(self, num_wires, num_layers, nameCircuit):
        if nameCircuit in ['circuit_2', 'circuit_13', 'circuit_14']:
            return (num_layers, 2, num_wires)
        elif nameCircuit in ['circuit_3', 'circuit_4', 'circuit_18', 'circuit_19']:
            return (num_layers, 3, num_wires)
        elif nameCircuit == 'circuit_5' or nameCircuit == 'circuit_6':
            return (num_layers, 4 + num_wires, num_wires)
        
        elif nameCircuit in ['circuit_9', 'circuit_10', 'circuit_15']:
            return (num_layers, 1 + num_wires, num_wires)     
        elif nameCircuit == 'strongly':
            return (num_layers, num_wires, 3)
        elif nameCircuit == 'basic':
            return (num_layers, num_wires)
        
    
        

        
        
