package NeuralNetworkTest;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.InputNeuron;
import NeuralNetwork.WorkingNeuron;

public class MNISTLearn {

    public static List<MNISTDecoder.Digit> digits;
    public static List<MNISTDecoder.Digit> digitsTest;
    public static NeuralNetwork nn = new NeuralNetwork();
    public static InputNeuron[][] inputs = new InputNeuron[28][28];
    public static WorkingNeuron[] outputs = new WorkingNeuron[10];

    public static void main(String[] args) throws IOException {

        String samplePath = "/home/peter/Projects/py/mnist/samples/uncompressed";
        digits = MNISTDecoder.loadDataSet(samplePath + "/train-images-idx3-ubyte",
                samplePath + "/train-labels-idx1-ubyte");
        digitsTest = MNISTDecoder.loadDataSet(samplePath + "/t10k-images-idx3-ubyte",
                samplePath + "/t10k-labels-idx1-ubyte");

        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                inputs[row][col] = nn.createNewInput();
            }
        }

        for (int i = 0; i < 10; i++) {
            outputs[i] = nn.createNewOutput();
        }

        int numHiddenNeurons = 100;
        nn.createHiddenNeurons(numHiddenNeurons);

        Random rand = new Random();
        float[] weights = new float[28 * 28 * numHiddenNeurons + numHiddenNeurons * 10];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextFloat();
        }
        nn.createFullMesh(weights);

        float epsilon = 0.00005f;
        while (true) {
            test();
            for (int dig = 0; dig < digits.size(); dig++) {
                for (int row = 0; row < 28; row++) {
                    for (int col = 0; col < 28; col++) {
                        inputs[row][col].setValue(MNISTDecoder.toUnsignedByte(digits.get(dig).data[row][col]) / 255f);
                    }
                }

                float[] targets = new float[10];
                targets[digits.get(dig).label] = 1;
                nn.backpropagation(targets, epsilon);
            }

            //epsilon *= 0.9f;
        }
    }

    public static void test() {
        int correct = 0;
        int incorrect = 0;

        for (int t = 0; t < digitsTest.size(); t++) {
            nn.reset();
            for (int row = 0; row < 28; row++) {
                for (int col = 0; col < 28; col++) {
                    inputs[row][col].setValue(MNISTDecoder.toUnsignedByte(digitsTest.get(t).data[row][col]) / 255f);
                }
            }

            ProbabilityDigit[] probs = new ProbabilityDigit[10];
            for (int i = 0; i < probs.length; i++) {
                probs[i] = new ProbabilityDigit(i, outputs[i].getValue());
            }

            Arrays.sort(probs, Collections.reverseOrder());

            boolean wasCorrect = false;
            for (int i = 0; i < 1; i++) {  // Change iteration num to test best of num
                if (digitsTest.get(t).label == probs[i].DIGIT) {
                    wasCorrect = true;
                }
            }

            if (wasCorrect) {
                correct++;
            } else {
                incorrect++;
            }
        }

        float percentage = (float) correct / (float) (correct + incorrect) * 100f;
        System.out.println("Score: " + percentage + "%");
    }

    public static class ProbabilityDigit implements Comparable<ProbabilityDigit> {
        public final int DIGIT;
        public float probability;

        public ProbabilityDigit(int digit, float probability) {
            this.DIGIT = digit;
            this.probability = probability;
        }

        @Override
        public int compareTo(ProbabilityDigit o) {
            if (probability == o.probability) {
                return 0;
            } else if (probability > o.probability) {
                return 1;
            } else {
                return -1;
            }
        }
    }
}
