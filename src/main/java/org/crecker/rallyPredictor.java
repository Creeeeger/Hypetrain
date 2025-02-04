package org.crecker;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;

import static org.crecker.Main_data_handler.INDICATOR_RANGE_MAP;
import static org.crecker.Main_data_handler.frameSize;

class RallyPredictor implements AutoCloseable {
    private static LinkedList<double[]> buffer = null;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final int windowSize;
    private final int numFeatures;

    public RallyPredictor(String modelPath, int windowSize, int numFeatures) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
        buffer = new LinkedList<>();
        this.windowSize = windowSize;
        this.numFeatures = numFeatures;
    }

    public static void main(String[] args) {
        double[] features = new double[2];
        predict(features);
    }

    public static void predict(double[] features) {
        final int WINDOW_SIZE = frameSize;
        final int NUM_FEATURES = INDICATOR_RANGE_MAP.size();

        try (RallyPredictor predictor = new RallyPredictor("spike_predictor.onnx", WINDOW_SIZE, NUM_FEATURES)) {
            buffer.add(ArrayUtils.toPrimitive(Arrays.stream(features).boxed().toArray(Double[]::new)));
            Double prediction = predictor.predictSpike();

            if (prediction != null && prediction > 0.0002) {
                System.out.println("High spike probability: " + prediction);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public synchronized Double predictSpike() throws OrtException {
        if (buffer.size() < windowSize) {
            return null;
        }

        // Create 3D array [1, windowSize, numFeatures]
        double[][][] inputArray = new double[1][windowSize][numFeatures];
        int i = 0;
        for (double[] features : buffer) {
            inputArray[0][i++] = features;
        }

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            try (OrtSession.Result result = session.run(Collections.singletonMap("input", inputTensor))) {
                float[][] output = (float[][]) result.get(0).getValue();
                return (double) output[0][0];
            }
        }
    }

    @Override
    public void close() throws Exception {
        if (session != null) session.close();
        if (env != null) env.close();
    }
}