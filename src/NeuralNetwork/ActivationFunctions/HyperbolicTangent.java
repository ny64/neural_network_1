package NeuralNetwork.ActivationFunctions;

public class HyperbolicTangent implements ActivationFunction {

    @Override
    public float activation(float input) {
        double epx = Math.pow(Math.E, input);
        double enx = Math.pow(Math.E, -input);

        return (float)((epx - enx) / (epx + enx));
    }

    @Override
    public float derivative(float input) {
        float tanh = activation(input);

        return 1 - tanh * tanh;
    }
}
