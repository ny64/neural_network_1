package NeuralNetworkTest;

import NeuralNetwork.InputNeuron;
import NeuralNetwork.WorkingNeuron;
import NeuralNetwork.NeuralNetwork;

public class SingleLayerPerceptron {

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork();

        InputNeuron i1 = nn.createNewInput();
        InputNeuron i2 = nn.createNewInput();
        InputNeuron i3 = nn.createNewInput();
        InputNeuron i4 = nn.createNewInput();
        nn.createHiddenNeurons(3);
        WorkingNeuron o1 = nn.createNewOutput();

        nn.createFullMesh(
                10, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,

                10, 0, 0
        );

        i1.setValue(1);
        i2.setValue(2);
        i3.setValue(3);
        i4.setValue(4);

        System.out.println(o1.getValue());
    }
}
