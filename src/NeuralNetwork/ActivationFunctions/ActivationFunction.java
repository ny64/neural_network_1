package NeuralNetwork.ActivationFunctions;

public interface ActivationFunction {
    public static Identity ActivationIdentity = new Identity();
    public static Boolean ActivationBoolean = new Boolean();
    public static Sigmoid ActivationSigmoid = new Sigmoid();
    public static HyperbolicTangent ActivationHyperbolicTangent = new HyperbolicTangent();
    public static ReLU ActivationReLU = new ReLU();

    public float activation(float input);
    public float derivative(float input);
}