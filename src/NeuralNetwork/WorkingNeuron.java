package NeuralNetwork;

import NeuralNetwork.ActivationFunctions.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public class WorkingNeuron extends Neuron {

    private List<Connection> connections = new ArrayList<>();
    private ActivationFunction activationFunction = ActivationFunction.ActivationSigmoid;
    private float smallDelta = 0;
    private float value = 0;
    private boolean valueClean = false;

    @Override
    public float getValue() {
        if(!valueClean) {
            float sum = 0;
            for (Connection c : connections) {
                sum += c.getValue();
            }

            value =  activationFunction.activation(sum);
            valueClean = true;
        }

        return value;
    }

    public void setActivationFunction(ActivationFunction af) {
        this.activationFunction = af;
    }

    public void reset() {
        smallDelta = 0;
        valueClean = false;
    }

    public void addConnection(Connection c) {
        connections.add(c);
    }

    public void deltaLearning(float epsilon) {
        float bigDeltaFactor = activationFunction.derivative(getValue()) * epsilon * smallDelta;
        for (int i = 0; i < connections.size(); i++) {
            float bigDelta = bigDeltaFactor * connections.get(i).getNeuron().getValue();
            connections.get(i).addWeight(bigDelta);
        }
    }

    public void calculateOutputDelta(float target) {
        smallDelta = target - getValue();
    }

    public void backpropagateSmallDelta() {
        for (Connection c: connections) {
            Neuron n = c.getNeuron();
            if (n instanceof WorkingNeuron) {
                WorkingNeuron wn = (WorkingNeuron)n;
                wn.smallDelta += this.smallDelta * c.getWeight();
            }
        }
    }

    public void applyBatch() {
        for (Connection c: connections) {
            c.applyBatch();
        }
    }
}
