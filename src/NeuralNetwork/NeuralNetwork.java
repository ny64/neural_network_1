package NeuralNetwork;

import NeuralNetwork.ActivationFunctions.ActivationFunction;

import java.util.List;
import java.util.ArrayList;

public class NeuralNetwork {

    private List<InputNeuron> inputNeurons = new ArrayList<>();
    private List<WorkingNeuron> hiddenNeurons = new ArrayList<>();
    private List<WorkingNeuron> outputNeurons = new ArrayList<>();
    private int trainingSample = 0;

    public WorkingNeuron createNewOutput() {
        WorkingNeuron wn = new WorkingNeuron();
        outputNeurons.add(wn);

        return wn;
    }

    public void reset() {
        for (WorkingNeuron wn: outputNeurons) {
            wn.reset();
        }

        for (WorkingNeuron wn: hiddenNeurons) {
            wn.reset();
        }
    }

    public void createHiddenNeurons(int amount) {
        for (int i = 0; i < amount; i++) {
            WorkingNeuron wn = new WorkingNeuron();
            wn.setActivationFunction(ActivationFunction.ActivationReLU);
            hiddenNeurons.add(wn);
        }
    }

    public InputNeuron createNewInput() {
        InputNeuron inputNeuron = new InputNeuron();
        inputNeurons.add(inputNeuron);

        return inputNeuron;
    }

    public void backpropagation(float[] targets, float epsilon) {
        if (targets.length != outputNeurons.size()) {
            throw new IllegalArgumentException();
        }

        reset();

        for (int i = 0; i < targets.length; i++) {
            outputNeurons.get(i).calculateOutputDelta(targets[i]);
        }

        if (hiddenNeurons.size() > 0) {
            for (int i = 0; i < targets.length; i++) {
                outputNeurons.get(i).backpropagateSmallDelta();
            }
        }

        for (int i = 0; i < targets.length; i++) {
            outputNeurons.get(i).deltaLearning(epsilon);
        }

        for (int i = 0; i < hiddenNeurons.size(); i++) {
            hiddenNeurons.get(i).deltaLearning(epsilon);
        }

        if (trainingSample % 64 == 0) {
            for (int i = 0; i < targets.length; i++) {
                outputNeurons.get(i).applyBatch();
            }

            for (int i = 0; i < hiddenNeurons.size(); i++) {
                hiddenNeurons.get(i).applyBatch();
            }
        }

        trainingSample++;
    }

    public void createFullMesh() {
        if(hiddenNeurons.size() == 0) {
            for (WorkingNeuron outputNeuron : outputNeurons) {
                for (InputNeuron inputNeuron : inputNeurons) {
                    outputNeuron.addConnection(new Connection(inputNeuron, 0));
                }
            }
        }
        else {
            for (WorkingNeuron outputNeuron : outputNeurons) {
                for (WorkingNeuron hn : hiddenNeurons) {
                    outputNeuron.addConnection(new Connection(hn, 0));
                }
            }

            for (WorkingNeuron hiddenNeuron : hiddenNeurons) {
                for (InputNeuron inputNeuron : inputNeurons) {
                    hiddenNeuron.addConnection(new Connection(inputNeuron, 0));
                }
            }
        }
    }

    public void createFullMesh(float... weights) {
        if(hiddenNeurons.size() == 0) {
            if (weights.length != inputNeurons.size() * outputNeurons.size()) {
                throw new RuntimeException();
            }

            int index = 0;

            for (WorkingNeuron outputNeuron : outputNeurons) {
                for (InputNeuron inputNeuron : inputNeurons) {
                    outputNeuron.addConnection(new Connection(inputNeuron, weights[index++]));
                }
            }
        }
        else {
            if(weights.length != inputNeurons.size() * hiddenNeurons.size()
                    + hiddenNeurons.size() * outputNeurons.size()) {
                throw new RuntimeException();
            }

            int index = 0;

            for (WorkingNeuron hiddenNeuron : hiddenNeurons) {
                for (InputNeuron inputNeuron : inputNeurons) {
                    hiddenNeuron.addConnection(new Connection(inputNeuron, weights[index++]));
                }
            }

            for (WorkingNeuron outputNeuron : outputNeurons) {
                for (WorkingNeuron hiddenNeuron : hiddenNeurons) {
                    outputNeuron.addConnection(new Connection(hiddenNeuron, weights[index++]));
                }
            }
        }
    }
}
