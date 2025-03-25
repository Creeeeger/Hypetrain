package org.crecker;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.LinkedList;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RallyPredictor implements AutoCloseable {
    private static RallyPredictor instance;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final LinkedList<float[]> buffer;
    private final ReentrantLock bufferLock = new ReentrantLock();
    private final int length = mainDataHandler.INDICATOR_RANGE_MAP.size();
    private int dynamicBufferSize = 28; // Initial default value

    // Private constructor to prevent instantiation
    private RallyPredictor(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
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
        Path modelPath = Paths.get(System.getProperty("user.dir"), "rallyMLModel", "spike_predictor.onnx");
        try {
            return RallyPredictor.getInstance(modelPath.toString()).updateAndPredict(features);
        } catch (Exception e) {
            System.out.println("predict Function: " + e.getMessage());
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
            if (buffer.size() >= dynamicBufferSize) {
                buffer.remove(0); // Remove the oldest feature if the buffer is full
            }

            buffer.add(features); // Add the new feature to the buffer

            if (buffer.size() >= dynamicBufferSize) {
                return predictSpike(); // Only predict once the buffer is full
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            bufferLock.unlock();
        }
        return 0F;
    }

    /**
     * Predicts the spike probability using the current buffer.
     *
     * @return The spike probability, or null if the buffer is not yet full.
     */
    private Float predictSpike() throws OrtException {
        try {
            float[][][] inputArray = prepareInputArray();

            try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
                try (OrtSession.Result result = session.run(Collections.singletonMap("args_0", inputTensor))) {
                    float[][] output = (float[][]) result.get(0).getValue();
                    return output[0][0];
                }
            }
        } catch (OrtException e) {
            Integer expectedSize = parseExpectedSizeFromError(e.getMessage());
            if (expectedSize != null) {
                System.out.println("Adapting buffer size from " + dynamicBufferSize + " to " + expectedSize);
                dynamicBufferSize = expectedSize;
            } else {
                throw e;
            }
        }

        return null;
    }

    private float[][][] prepareInputArray() {
        float[][][] inputArray = new float[1][dynamicBufferSize][length];

        for (int i = 0; i < dynamicBufferSize; i++) {

            float[] features = buffer.get(buffer.size() - 1);

            System.arraycopy(
                    features, 0,
                    inputArray[0][i], 0,
                    Math.min(features.length, length)
            );
        }
        return inputArray;
    }

    private Integer parseExpectedSizeFromError(String errorMessage) {
        Pattern pattern = Pattern.compile("Expected: (\\d+)");
        Matcher matcher = pattern.matcher(errorMessage);

        if (matcher.find()) {
            return Integer.parseInt(matcher.group(1));
        }
        return null;
    }

    @Override
    public void close() throws Exception {
        if (session != null) session.close();
        if (env != null) env.close();
        instance = null; // Reset the singleton instance when closed
    }
}