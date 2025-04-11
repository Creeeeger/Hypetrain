package org.crecker;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RallyPredictor implements AutoCloseable {
    private static RallyPredictor instance;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final Map<String, LinkedList<float[]>> symbolBuffers = new ConcurrentHashMap<>();
    private final ReentrantLock sizeAdjustmentLock = new ReentrantLock();
    private int dynamicBufferSize = 28;
    private final int featureLength = mainDataHandler.INDICATOR_KEYS.size();

    // Private constructor to prevent instantiation
    private RallyPredictor(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        this.session = env.createSession(modelPath, options);
    }

    // Singleton instance getter
    public static synchronized RallyPredictor getInstance(String modelPath) throws OrtException {
        if (instance == null) {
            instance = new RallyPredictor(modelPath);
        }
        return instance;
    }

    public static float predict(float[] features, String symbol) {
        Path modelPath = Paths.get(System.getProperty("user.dir"), "rallyMLModel", "spike_predictor.onnx");
        try {
            Float prediction = RallyPredictor.getInstance(modelPath.toString()).updateAndPredict(symbol, features);
            return prediction != null ? prediction : 0F;
        } catch (Exception e) {
            System.out.println("predict Function: " + e.getMessage());
            return 0F;
        }
    }

    public Float updateAndPredict(String symbol, float[] features) {
        LinkedList<float[]> buffer = symbolBuffers.computeIfAbsent(symbol, k -> new LinkedList<>());

        synchronized (buffer) {
            if (buffer.size() >= dynamicBufferSize) {
                buffer.removeFirst(); // Keep buffer size in check
            }
            buffer.addLast(features); // Add new feature vector

            if (buffer.size() >= dynamicBufferSize) {
                try {
                    return predictSpike(buffer);
                } catch (OrtException e) {
                    handleModelError(e);
                }
            }
        }
        return 0F;
    }

    private Float predictSpike(LinkedList<float[]> buffer) throws OrtException {
        float[][][] inputArray = prepareInputArray(buffer);

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            try (OrtSession.Result result = session.run(Collections.singletonMap("args_0", inputTensor))) {
                float[][] output = (float[][]) result.get(0).getValue();
                return output[0][0];
            }
        }
    }

    /**
     * Prepares input by replicating the latest feature vector (like version 2 does).
     */
    private float[][][] prepareInputArray(LinkedList<float[]> buffer) {
        float[][][] inputArray = new float[1][dynamicBufferSize][featureLength];

        float[] latest = buffer.getLast();

        for (int i = 0; i < dynamicBufferSize; i++) {
            System.arraycopy(latest, 0, inputArray[0][i], 0, Math.min(latest.length, featureLength));
        }

        return inputArray;
    }

    private void handleModelError(OrtException e) {
        Integer expectedSize = parseExpectedSizeFromError(e.getMessage());
        if (expectedSize != null) {
            sizeAdjustmentLock.lock();
            try {
                if (expectedSize != dynamicBufferSize) {
                    System.out.println("Adjusting buffer size to " + expectedSize);
                    dynamicBufferSize = expectedSize;
                    trimAllBuffers();
                }
            } finally {
                sizeAdjustmentLock.unlock();
            }
        } else {
            e.printStackTrace();
        }
    }

    private void trimAllBuffers() {
        symbolBuffers.forEach((symbol, buffer) -> {
            synchronized (buffer) {
                while (buffer.size() > dynamicBufferSize) {
                    buffer.removeFirst();
                }
            }
        });
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
        instance = null; // Reset singleton on close
    }
}