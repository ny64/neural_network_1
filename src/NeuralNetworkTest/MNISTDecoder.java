package NeuralNetworkTest;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class MNISTDecoder {

    public static int toUnsignedByte(byte b) {
        return b & 0xFF;
    }

    public static List<Digit> loadDataSet(String datas, String labels) throws IOException {
        Path dataPath = Paths.get(datas);
        Path labelPath = Paths.get(labels);

        byte[] dataByte = Files.readAllBytes(dataPath);
        byte[] labelByte = Files.readAllBytes(labelPath);

        List<Digit> digits = new ArrayList<>();

        int readHeadData = 16;
        int readHeadLabel = 8;
        while (readHeadData < dataByte.length) {
            byte[][] data = new byte[28][28];
            for (int i = 0; i < 28; i++) {
                for (int k = 0; k < 28; k++) {
                    data[i][k] = dataByte[readHeadData++];
                }
            }
            int label = toUnsignedByte(labelByte[readHeadLabel++]);

            digits.add(new Digit(label, data));
        }

        return digits;
    }

    public static class Digit {
        public int label;
        public byte[][] data;

        public Digit(int label, byte[][] data) {
            this.label = label;
            this.data = data;
        }
    }
}