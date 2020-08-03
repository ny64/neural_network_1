package NeuralNetwork;

public class InputNeuron extends Neuron {

    private float value = 0;

    @Override
    public float getValue() {
        return value;
    }

    public void setValue(float value) {
        this.value = value;
    }
}
