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

public class RallyPredictor implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;
    private final LinkedList<double[]> buffer;
    private final int numFeatures;
    private final ReentrantLock bufferLock = new ReentrantLock();

    public RallyPredictor(String modelPath, int numFeatures) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
        options.addCoreML();

        this.session = env.createSession(modelPath, options);

        this.buffer = new LinkedList<>();
        this.numFeatures = numFeatures;
    }

    public static double predict(double[] features) {
        final int NUM_FEATURES = INDICATOR_RANGE_MAP.size();

        String modelPath = "./rallyMLModel/spike_predictor.onnx";
        try (RallyPredictor predictor = new RallyPredictor(modelPath, NUM_FEATURES)) {
            Double spikeProbability = predictor.updateAndPredict(features);
            if (spikeProbability != null && spikeProbability > 0.3) { // Change value after testing
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
    public synchronized Double updateAndPredict(double[] features) {
        if (features.length != numFeatures) {
            throw new IllegalArgumentException("Feature vector length must be " + numFeatures);
        }

        bufferLock.lock();
        try {
            buffer.add(features);
            return predictSpike();

        } catch (Exception e) {
            e.printStackTrace();
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

        int windowSize = 20;  // Sequence length used in model training
        int numFeatures = 8;  // Number of features per time step

        float[][][] inputArray = new float[1][windowSize][numFeatures];

        int i = 0;
        for (double[] features : buffer) {
            for (int j = 0; j < features.length && j < numFeatures; j++) {
                inputArray[0][i][j] = (float) features[j];
            }
            i++;
        }

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            try (OrtSession.Result result = session.run(Collections.singletonMap("args_0", inputTensor))) {
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