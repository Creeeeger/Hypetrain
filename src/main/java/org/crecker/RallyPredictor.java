package org.crecker;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;

import java.nio.file.Paths;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.crecker.mainDataHandler.frameSize;

/**
 * <h1>RallyPredictor</h1>
 * <p>
 * Manages one or more ONNX model sessions for real-time stock‐signal predictions.
 * Each unique ONNX file path yields exactly one RallyPredictor instance (per‐model singleton).
 * Instances handle:
 * <ul>
 *   <li>Loading ONNX models into OrtSession</li>
 *   <li>Rolling‐buffer maintenance per symbol for streaming predictions</li>
 *   <li>Auto‐adjusting the required history length if the model’s input shape changes</li>
 *   <li>Direct “batch” predictions from a fixed 30‐bar window</li>
 * </ul>
 * </p>
 *
 * <p>
 * Usage:
 * <pre>
 *   // For a streaming (rolling‐buffer) prediction:
 *   float score = RallyPredictor.predict(featuresArray, symbol);
 *
 *   // For a single‐window entry prediction from exactly `frameSize` bars:
 *   float entryScore = RallyPredictor.predictNotificationEntry(stockUnitList);
 * </pre>
 * The first method uses the “spike_predictor.onnx” model by default; the second uses “entryPrediction.onnx”.
 * Both methods share the same underlying registry so that each ONNX file is loaded only once.
 * </p>
 */
public class RallyPredictor implements AutoCloseable {

    // --- Registry of predictors and ONNX runtime management ---

    /**
     * Maps each ONNX model file path to its own RallyPredictor singleton.
     * This ensures that each ONNX graph is loaded only once (one OrtSession per file).
     */
    private static final Map<String, RallyPredictor> INSTANCES = new ConcurrentHashMap<>();

    /**
     * Shared ONNX Runtime environment.
     */
    private final OrtEnvironment env;

    /**
     * ONNX Runtime session for this particular model file.
     */
    private final OrtSession session;


    // --- Per-symbol input feature history management ---

    /**
     * For each stock symbol, holds a rolling window of the most recent feature vectors.
     * The window size may be adjusted if the model reports a different expected input length.
     */
    private final Map<String, LinkedList<float[]>> symbolBuffers = new ConcurrentHashMap<>();

    /**
     * Ensures that buffer‐size adjustments happen atomically across all symbols.
     */
    private final ReentrantLock sizeAdjustmentLock = new ReentrantLock();


    // --- Model/config parameters ---

    /**
     * Current required history length (number of time steps) for streaming predictions.
     * Default is 28, but may be updated if the ONNX model’s input shape changes.
     */
    private int dynamicBufferSize = 28;

    /**
     * Number of features per time step. Must match the model’s expected input “feature” dimension.
     * Pulled from mainDataHandler.INDICATOR_KEYS.size()
     */
    private final int featureLength = mainDataHandler.INDICATOR_KEYS.size();


    /**
     * Private constructor – only called by {@link #getInstance(String)}.
     * Initializes the OrtEnvironment and loads the ONNX model into a new OrtSession.
     *
     * @param modelPath Absolute path to the ONNX model file.
     * @throws OrtException if loading or parsing the ONNX graph fails.
     */
    private RallyPredictor(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment(); // Shared, thread-safe ONNX environment
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        // Options can be tweaked here (e.g., enable CPU or GPU flags)
        this.session = env.createSession(modelPath, options);
    }

    /**
     * Retrieves (or creates) a RallyPredictor bound to the specified ONNX file.
     * Internally, this is a per-path singleton: the first call loads the graph,
     * subsequent calls return the same OrtSession for that file.
     *
     * @param modelPath Absolute path to an ONNX model file (e.g. “spike_predictor.onnx”).
     * @return RallyPredictor instance whose {@link #session} corresponds to the given path.
     * @throws RuntimeException wrapping OrtException if model loading fails.
     */
    public static RallyPredictor getInstance(String modelPath) {
        return INSTANCES.computeIfAbsent(modelPath, path -> {
            try {
                return new RallyPredictor(path);
            } catch (OrtException e) {
                throw new RuntimeException("Failed to load ONNX model at " + path, e);
            }
        });
    }

    /**
     * Static helper for simple rolling‐buffer predictions using the default spike model.
     * Internally calls {@link #updateAndPredict(String, float[])} on the singleton
     * for “spike_predictor.onnx”.
     *
     * @param features Current feature vector (length == featureLength).
     * @param symbol   Stock symbol (ticker) used to select the rolling buffer.
     * @return Model’s float prediction, or 0F if insufficient history or on error.
     */
    public static float predict(float[] features, String symbol) {
        // Use the “spike_predictor.onnx” model by default
        String spikeModelPath = Paths.get(System.getProperty("user.dir"), "rallyMLModel", "spike_predictor.onnx").toString();
        try {
            RallyPredictor predictor = RallyPredictor.getInstance(spikeModelPath);
            return predictor.updateAndPredict(symbol, features);
        } catch (Exception e) {
            System.out.println("predict Function: " + e.getMessage());
            return 0F;
        }
    }

    /**
     * Appends the given feature vector to the symbol’s rolling buffer, then attempts
     * to run streaming prediction if the buffer has at least {@link #dynamicBufferSize} entries.
     *
     * @param symbol   Stock symbol used as the key for its rolling buffer.
     * @param features Feature vector for the current time step (length == featureLength).
     * @return Model’s float prediction if buffer size ≥ dynamicBufferSize; otherwise 0F.
     * @throws OrtException if ONNX inference fails (e.g. shape mismatch).
     */
    public Float updateAndPredict(String symbol, float[] features) {
        // Get or create the buffer for this symbol
        LinkedList<float[]> buffer = symbolBuffers.computeIfAbsent(symbol, k -> new LinkedList<>());

        synchronized (buffer) {
            // If buffer is already full, drop the oldest entry
            if (buffer.size() >= dynamicBufferSize) {
                buffer.removeFirst();
            }
            buffer.addLast(features);

            // Only run inference once we have at least dynamicBufferSize vectors
            if (buffer.size() >= dynamicBufferSize) {
                try {
                    return predictSpike(buffer); // Run model inference
                } catch (OrtException e) {
                    handleModelError(e); // Try to recover from shape mismatch
                }
            }
        }
        return 0F;
    }

    /**
     * Runs ONNX model inference on a “rolling” buffer of feature vectors.
     * Constructs a 3D tensor of shape [1, time_steps, featureLength], fills it
     * by repeating the latest vector if needed, and invokes the session.
     *
     * @param buffer LinkedList of float[] with size == dynamicBufferSize.
     * @return The model’s float output (batch[0][0]), or throws OrtException if something goes wrong.
     * @throws OrtException if ONNX Runtime fails (e.g. data type or shape mismatch).
     */
    private Float predictSpike(LinkedList<float[]> buffer) throws OrtException {
        // Prepare the model input: shape [1, time_steps, num_features]
        float[][][] inputArray = prepareInputArray(buffer);

        // Construct ONNX input tensor (auto-managed with try-with-resources)
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            // Note:	ONNX input key must match the model’s actual input name
            OrtSession.Result result = session.run(Collections.singletonMap("args_0", inputTensor));
            float[][] output = (float[][]) result.get(0).getValue(); // Extract prediction
            return output[0][0]; // Typically [batch, output_dim], so take the first value
        }
    }

    /**
     * Builds a [1 × dynamicBufferSize × featureLength] array by taking the last
     * buffer entry (float[]) and copying it to every time step. This replicates
     * the “latest state” for each required time step.
     *
     * @param buffer LinkedList of float[] with size == dynamicBufferSize.
     * @return 3D float array ready for OnnxTensor.createTensor().
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
     * If ONNX inference fails due to an unexpected input shape (e.g., “Expected: N”),
     * this method parses the required N from the error message, updates dynamicBufferSize,
     * and prunes any existing rolling buffers to that new length.
     *
     * @param e OrtException that includes “Expected: <integer>” in its message.
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
     * Removes oldest entries from each symbol’s buffer so no buffer exceeds
     * the new {@link #dynamicBufferSize}.
     * This ensures that future calls to predictSpike(...) match the model’s input length.
     */
    private void trimAllBuffers() {
        symbolBuffers.forEach((symbol, buffer) -> {
            while (buffer.size() > dynamicBufferSize) {
                buffer.removeFirst();// Remove oldest until buffer is at the right size
            }
        });
    }

    /**
     * Parses “Expected: N” from an ONNX Runtime exception message. For example,
     * if e.getMessage() contains “Expected: 32”, this returns 32.
     *
     * @param errorMessage The raw exception message from OrtException.
     * @return Parsed integer, or null if the pattern isn’t found.
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
     * Immediately predicts from a fixed, non‐streaming window of exactly {@code frameSize} bars.
     * Builds a [1 × frameSize × 6] tensor from the List of StockUnit, calls ONNX, and returns
     * the scalar output. This is typically used for “entry” predictions after a notification.
     *
     * @param stockUnits List of StockUnit length ≥ frameSize. Each StockUnit must provide:
     *                   getOpen(), getHigh(), getLow(), getClose(), getVolume(), getPercentageChange().
     * @return ONNX model’s float output (batch[0][0]), or 0F if insufficient data or on error.
     */
    public float predictFromWindow(List<StockUnit> stockUnits) {
        // If the list is null or shorter than the required frameSize, we cannot run inference.
        // Return 0F as a safe fallback.
        if (stockUnits == null || stockUnits.size() < frameSize) {
            return 0F;
        }

        // Create a 3D float array with dimensions [batch=1][time_steps=frameSize][features=6].
        // This will hold the converted StockUnit values for ONNX.
        float[][][] inputArray = new float[1][frameSize][6];

        // Iterate over each of the first frameSize StockUnit entries.
        for (int i = 0; i < frameSize; i++) {
            StockUnit u = stockUnits.get(i);

            // Map each StockUnit field to the corresponding index in the features dimension:
            //   [0] = open price
            inputArray[0][i][0] = (float) u.getOpen();
            //   [1] = high price
            inputArray[0][i][1] = (float) u.getHigh();
            //   [2] = low price
            inputArray[0][i][2] = (float) u.getLow();
            //   [3] = close price
            inputArray[0][i][3] = (float) u.getClose();
            //   [4] = volume
            inputArray[0][i][4] = (float) u.getVolume();
            //   [5] = percentageChange
            inputArray[0][i][5] = (float) u.getPercentageChange();
        }

        // Try-with-resources: create an OnnxTensor from the inputArray and automatically close it.
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, inputArray)) {
            // Run inference on the model, using "input" as the ONNX graph’s input name.
            // The result is a list of OrtValues; we expect the first to contain our float output.
            OrtSession.Result result = session.run(Collections.singletonMap("input", tensor));

            // Extract the float[][] from the first output OrtValue. This is typically shape [1][1].
            float[][] out = (float[][]) result.get(0).getValue();

            // Return the single scalar at out[0][0], which is the model’s predicted value.
            return out[0][0];
        } catch (OrtException e) {
            // If ONNX Runtime throws an exception (e.g., tensor creation or session run failure),
            // print the stack trace for debugging, then return 0F as a fallback.
            e.printStackTrace();
            return 0F;
        }
    }

    /**
     * Static helper for “entry” predictions using entryPrediction.onnx.
     * Calls {@link #predictFromWindow(List)} on the singleton for that file.
     *
     * @param stockUnits Exactly frameSize StockUnit objects (the 30-bar lookback).
     * @return ONNX entry model’s float output, or 0F on error.
     */
    public static float predictNotificationEntry(List<StockUnit> stockUnits) {
        try {
            // Build the absolute path to the ONNX model file “entryPrediction.onnx” under rallyMLModel
            String entryPredictionPath = Paths
                    .get(System.getProperty("user.dir"), "rallyMLModel", "entryPrediction.onnx")
                    .toString();

            // Retrieve (or load) the RallyPredictor instance tied to that ONNX path
            RallyPredictor predictor = RallyPredictor.getInstance(entryPredictionPath);

            // Call the predictor’s predictFromWindow method, passing in the StockUnit window
            return predictor.predictFromWindow(stockUnits);
        } catch (Exception ex) {
            // If anything goes wrong (e.g., model load or inference fails), print stack trace
            ex.printStackTrace();
            // Return 0F as a safe fallback probability
            return 0F;
        }
    }

    /**
     * Releases the OrtSession and OrtEnvironment for this RallyPredictor instance.
     * Also removes it from the registry so it can be garbage‐collected.
     * Must be called when shutting down or reloading a model to free native resources.
     */
    @Override
    public void close() throws Exception {
        if (session != null) session.close();
        if (env != null) env.close();
        INSTANCES.values().remove(this);
    }
}