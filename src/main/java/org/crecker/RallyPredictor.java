package org.crecker;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Collections;
import java.util.LinkedList;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;

import static org.crecker.Main_data_handler.INDICATOR_RANGE_MAP;
import static org.crecker.Main_data_handler.frameSize;

public class RallyPredictor implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;
    private final LinkedList<double[]> buffer;
    private final int windowSize;
    private final int numFeatures;
    private final ReentrantLock bufferLock = new ReentrantLock();

    public RallyPredictor(String modelPath, int windowSize, int numFeatures) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
        this.buffer = new LinkedList<>();
        this.windowSize = windowSize;
        this.numFeatures = numFeatures;
    }

    public static double predict(double[] features) {
        final int WINDOW_SIZE = frameSize; // Match your frameSize
        final int NUM_FEATURES = INDICATOR_RANGE_MAP.size();

        try (RallyPredictor predictor = new RallyPredictor("spike_predictor.onnx", WINDOW_SIZE, NUM_FEATURES)) {
            Double spikeProbability = predictor.updateAndPredict(features);

            if (spikeProbability != null && spikeProbability > 0.005) { // Change value after testing
                System.out.println("High spike probability: " + spikeProbability);
            }

            return Objects.requireNonNullElse(spikeProbability, 0.0);

        } catch (Exception e) {
            e.printStackTrace();
            return 0.0;
        }
    }

    /**
     * Adds new features to the buffer and predicts spike probability if the buffer is full.
     *
     * @param features The feature vector for the current time step.
     * @return The spike probability, or null if the buffer is not yet full.
     */
    public synchronized Double updateAndPredict(double[] features) throws OrtException {
        if (features.length != numFeatures) {
            throw new IllegalArgumentException("Feature vector length must be " + numFeatures);
        }

        bufferLock.lock();
        try {
            buffer.add(features);
            if (buffer.size() > windowSize) {
                buffer.removeFirst();
            }

            if (buffer.size() == windowSize) {
                return predictSpike();
            }
        } finally {
            bufferLock.unlock();
        }

        return null;
    }

    /**
     * Predicts the spike probability using the current buffer.
     *
     * @return The spike probability, or null if the buffer is not yet full.
     */
    private Double predictSpike() throws OrtException {
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