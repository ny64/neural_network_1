package NeuralNetwork.ActivationFunctions;

public class Identity implements ActivationFunction {

    @Override
    public float activation(float input) {
        return input;
    }

    @Override
    public float derivative(float input) {
        return 1;
    }
}
