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

/**
 * <h1>RallyPredictor</h1>
 * Singleton class for making predictions using a pre-trained ONNX machine learning model.
 * Handles buffer management per symbol, dynamic buffer sizing, model inference, and
 * error-driven auto-adjustment of input size for robustness.
 * <p>
 * Supports thread-safe updates and concurrent predictions for multiple symbols.
 */
public class RallyPredictor implements AutoCloseable {

    // --- Singleton instance and ONNX runtime management ---

    private static RallyPredictor instance; // The singleton instance

    private final OrtEnvironment env;       // ONNX Runtime environment (heavyweight, shareable)
    private final OrtSession session;       // Loaded ONNX model session

    // --- Per-symbol input feature history management ---

    // Each symbol (e.g., stock ticker) maps to a buffer holding its latest N feature vectors for prediction
    private final Map<String, LinkedList<float[]>> symbolBuffers = new ConcurrentHashMap<>();

    // Lock to make sure buffer size changes are atomic and race-free
    private final ReentrantLock sizeAdjustmentLock = new ReentrantLock();

    // --- Model/config parameters ---

    // How many time steps to keep for rolling predictions (dynamically adjusted to match model input)
    private int dynamicBufferSize = 28;

    // How many features per time step (matches the number of input features expected by the model)
    private final int featureLength = mainDataHandler.INDICATOR_KEYS.size();

    /**
     * Private constructor - only called via getInstance().
     * Loads the ONNX model at the given path and sets up the environment and session.
     *
     * @param modelPath Path to the ONNX file.
     * @throws OrtException If loading the model fails.
     */
    private RallyPredictor(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment(); // Singleton for ONNX environment
        OrtSession.SessionOptions options = new OrtSession.SessionOptions(); // Session options (can tweak performance here)
        this.session = env.createSession(modelPath, options); // Load the model
    }

    /**
     * Singleton getter for the RallyPredictor.
     * Loads the model only once; subsequent calls reuse the existing environment/session.
     *
     * @param modelPath Path to ONNX model file.
     * @return the RallyPredictor instance.
     * @throws OrtException If model loading fails.
     */
    public static synchronized RallyPredictor getInstance(String modelPath) throws OrtException {
        if (instance == null) {
            instance = new RallyPredictor(modelPath); // Construct singleton if needed
        }
        return instance;
    }

    /**
     * Static helper for simple predictions.
     * Accepts a feature vector and symbol, retrieves (or creates) the predictor singleton,
     * updates the per-symbol buffer, and returns the latest prediction.
     *
     * @param features Array of feature values for current time step.
     * @param symbol   Symbol identifier (e.g., stock ticker).
     * @return Predicted value (float) from the model, or 0F if not enough history or error.
     */
    public static float predict(float[] features, String symbol) {
        // Default model path (could be configurable)
        Path modelPath = Paths.get(System.getProperty("user.dir"), "rallyMLModel", "spike_predictor.onnx");
        try {
            Float prediction = RallyPredictor.getInstance(modelPath.toString()).updateAndPredict(symbol, features);
            return prediction != null ? prediction : 0F; // Fallback to 0F if model returns null
        } catch (Exception e) {
            System.out.println("predict Function: " + e.getMessage());
            return 0F;
        }
    }

    /**
     * Adds the provided features to the symbol's rolling buffer and performs prediction if enough history.
     * Thread-safe for each buffer.
     *
     * @param symbol   Symbol to track.
     * @param features Feature vector to add.
     * @return Prediction (Float) if model inference is performed, or 0F if not enough data or on error.
     */
    public Float updateAndPredict(String symbol, float[] features) {
        // Retrieve or create buffer for this symbol (thread-safe)
        LinkedList<float[]> buffer = symbolBuffers.computeIfAbsent(symbol, k -> new LinkedList<>());

        synchronized (buffer) {
            // Maintain rolling buffer size: discard the oldest if full
            if (buffer.size() >= dynamicBufferSize) {
                buffer.removeFirst();
            }
            buffer.addLast(features);

            // Only predict if we have enough history to match model's required input shape
            if (buffer.size() >= dynamicBufferSize) {
                try {
                    return predictSpike(buffer); // Run model inference
                } catch (OrtException e) {
                    handleModelError(e); // Try to recover from shape mismatch
                }
            }
        }
        return 0F; // Not enough history, or error occurred
    }

    /**
     * Runs the ONNX model inference using the latest buffer.
     * Builds the required input tensor shape, feeds to model, and extracts the output.
     *
     * @param buffer Rolling history of feature vectors (size == dynamicBufferSize).
     * @return Predicted value from the model output.
     * @throws OrtException If model inference fails.
     */
    private Float predictSpike(LinkedList<float[]> buffer) throws OrtException {
        // Prepare the model input: shape [1, time_steps, num_features]
        float[][][] inputArray = prepareInputArray(buffer);

        // Construct ONNX input tensor (auto-managed with try-with-resources)
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            // Run the model (input name is "args_0" as generated by most ONNX export tools)
            try (OrtSession.Result result = session.run(Collections.singletonMap("args_0", inputTensor))) {
                float[][] output = (float[][]) result.get(0).getValue(); // Extract prediction
                return output[0][0]; // Typically [batch, output_dim], so take the first value
            }
        }
    }

    /**
     * Prepares the input tensor for the ONNX model.
     * Fills the [1, dynamicBufferSize, featureLength] array with the latest feature vector for each step,
     * replicating the most recent input (matches model version 2's behaviour).
     *
     * @param buffer The rolling buffer of feature vectors.
     * @return Prepared input array for the model.
     */
    private float[][][] prepareInputArray(LinkedList<float[]> buffer) {
        float[][][] inputArray = new float[1][dynamicBufferSize][featureLength];

        // Use the latest feature vector for all time steps (model expects repeated latest state)
        float[] latest = buffer.getLast();

        for (int i = 0; i < dynamicBufferSize; i++) {
            // Defensive: if latest is shorter than featureLength, copy up to min length
            System.arraycopy(latest, 0, inputArray[0][i], 0, Math.min(latest.length, featureLength));
        }

        return inputArray;
    }

    /**
     * Attempts to auto-recover if model throws an exception due to mismatched input shape.
     * If the error message reveals the expected time_steps, resizes buffer and retries.
     *
     * @param e The OrtException caught during inference.
     */
    private void handleModelError(OrtException e) {
        // Try to extract the expected time_steps from error message (e.g., "Expected: 32")
        Integer expectedSize = parseExpectedSizeFromError(e.getMessage());
        if (expectedSize != null) {
            sizeAdjustmentLock.lock();
            try {
                // Only adjust if it's actually different
                if (expectedSize != dynamicBufferSize) {
                    System.out.println("Adjusting buffer size to " + expectedSize);
                    dynamicBufferSize = expectedSize;
                    trimAllBuffers(); // Trim all symbol buffers to the new required size
                }
            } finally {
                sizeAdjustmentLock.unlock();
            }
        } else {
            e.printStackTrace(); // Print stack trace if auto-recovery not possible
        }
    }

    /**
     * Trims all symbol buffers so that none exceed the current dynamicBufferSize.
     * Thread-safe for each buffer.
     */
    private void trimAllBuffers() {
        symbolBuffers.forEach((symbol, buffer) -> {
            synchronized (buffer) {
                while (buffer.size() > dynamicBufferSize) {
                    buffer.removeFirst(); // Remove oldest until buffer is at the right size
                }
            }
        });
    }

    /**
     * Extracts the expected input size (time_steps) from a model error message.
     * Assumes the format includes "Expected: N" where N is an integer.
     *
     * @param errorMessage The error message string.
     * @return Parsed integer, or null if not found.
     */
    private Integer parseExpectedSizeFromError(String errorMessage) {
        Pattern pattern = Pattern.compile("Expected: (\\d+)");
        Matcher matcher = pattern.matcher(errorMessage);

        if (matcher.find()) {
            return Integer.parseInt(matcher.group(1)); // Parse the captured group as int
        }
        return null;
    }

    /**
     * Cleanly releases model and ONNX runtime resources.
     * Must be called on shutdown or model reload to prevent resource leaks.
     */
    @Override
    public void close() throws Exception {
        if (session != null) session.close();
        if (env != null) env.close();
        instance = null; // Allow new instance creation if needed
    }
}