package NeuralNetwork.ActivationFunctions;

public class ReLU implements ActivationFunction {
    @Override
    public float activation(float input) {
        if (input >= 0) {
            return input;
        } else {
            return 0;
        }
    }

    @Override
    public float derivative(float input) {
        if (input >= 0) {
            return 1;
        } else {
            return 0;
        }
    }
}
