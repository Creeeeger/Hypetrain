package org.crecker;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Collections;
import java.util.LinkedList;
import java.util.concurrent.locks.ReentrantLock;

public class RallyPredictor implements AutoCloseable {
    private static RallyPredictor instance;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final LinkedList<float[]> buffer;
    private final ReentrantLock bufferLock = new ReentrantLock();
    private final int length = weightRangeMap.INDICATOR_RANGE_MAP.size();

    // Private constructor to prevent instantiation
    private RallyPredictor(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.addCoreML();

        this.session = env.createSession(modelPath, options);

        this.buffer = new LinkedList<>();
    }

    // Singleton instance getter
    public static synchronized RallyPredictor getInstance(String modelPath) throws OrtException {
        if (instance == null) {
            instance = new RallyPredictor(modelPath);
        }
        return instance;
    }

    public static float predict(float[] features) {
        String modelPath = "./rallyMLModel/spike_predictor.onnx";
        try {
            RallyPredictor predictor = RallyPredictor.getInstance(modelPath);

            Float spikeProbability = predictor.updateAndPredict(features);
            if (spikeProbability != null) {
                if (pLTester.debug) {
                    System.out.println("High spike probability: " + spikeProbability);
                }
                return spikeProbability;
            } else {
                return 0;
            }

        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    /**
     * Adds new features to the buffer and predicts spike probability if the buffer is full.
     *
     * @param features The feature vector for the current time step.
     * @return The spike probability, or null if the buffer is not yet full.
     */
    public synchronized Float updateAndPredict(float[] features) {
        bufferLock.lock();
        try {
            buffer.clear();
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
    private Float predictSpike() throws OrtException {
        float[][][] inputArray = new float[1][length][length];

        for (int i = 0; i < Math.min(buffer.size(), length); i++) {
            float[] features = buffer.get(i);
            System.arraycopy(features, 0, inputArray[0][i], 0, Math.min(features.length, length));
        }

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            try (OrtSession.Result result = session.run(Collections.singletonMap("args_0", inputTensor))) {
                float[][] output = (float[][]) result.get(0).getValue();
                return output[0][0];
            }
        }
    }

    @Override
    public void close() throws Exception {
        if (session != null) session.close();
        if (env != null) env.close();
        instance = null; // Reset the singleton instance when closed
    }
}