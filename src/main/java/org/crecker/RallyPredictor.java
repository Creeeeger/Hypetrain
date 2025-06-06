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
 *   // For a single‐window entry prediction from exactly `size` bars:
 *   float entryScore = RallyPredictor.predictNotificationEntry(stockUnitList);
 * </pre>
 * The first method uses the “spike_predictor.onnx” model by default; the second uses “entryPrediction.onnx”.
 * Both methods share the same underlying registry so that each ONNX file is loaded only once.
 * </p>
 */
public class RallyPredictor implements AutoCloseable {
    /**
     * Number of time‐steps the ONNX model expects as input.
     * This should match the window length (frameSize) you trained the model with.
     */
    private int size = 20;

    /**
     * Total number of features per time‐step. Must match the model’s input “feature” dimension.
     * In our case, we have exactly six features:
     * 0 → open price
     * 1 → high price
     * 2 → low price
     * 3 → close price
     * 4 → volume
     * 5 → percentageChange
     */
    private static final int N_FEATURES = 6;

    /**
     * Mean values computed by the Python StandardScaler on the training data.
     * The ordering here must exactly match FEATURE_COLS = [open, high, low, close, volume, percentageChange].
     * Each entry was obtained from scaler.mean_ in Python:
     * FEATURE_MEANS[0] = mean of “open”
     * FEATURE_MEANS[1] = mean of “high”
     * FEATURE_MEANS[2] = mean of “low”
     * FEATURE_MEANS[3] = mean of “close”
     * FEATURE_MEANS[4] = mean of “volume”
     * FEATURE_MEANS[5] = mean of “percentageChange”
     * <p>
     * These floats are used to center each raw feature before dividing by its scale.
     */
    private static final float[] FEATURE_MEANS = new float[]{
            4.18877424e+01f,
            4.20415464e+01f,
            4.17932113e+01f,
            4.19515795e+01f,
            2.42354397e+05f,
            2.10162414e-01f,
    };

    /**
     * Scale (standard deviation) values computed by the Python StandardScaler on the training data.
     * The ordering here must match FEATURE_MEANS above. Each entry was obtained from scaler.scale_ in Python:
     * FEATURE_SCALES[0] = std of “open”
     * FEATURE_SCALES[1] = std of “high”
     * FEATURE_SCALES[2] = std of “low”
     * FEATURE_SCALES[3] = std of “close”
     * FEATURE_SCALES[4] = std of “volume”
     * FEATURE_SCALES[5] = std of “percentageChange”
     * <p>
     * These floats are used to divide (after subtracting the mean) so each feature has unit variance.
     */
    private static final float[] FEATURE_SCALES = new float[]{
            3.81914738e+01f,
            3.82674599e+01f,
            3.81453129e+01f,
            3.82160052e+01f,
            4.71319662e+05f,
            7.79173064e-01f,
    };

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
     * Immediately predicts from a fixed, non‐streaming window of exactly {@code size} bars.
     * Builds a [1 × size × 6] tensor from the List of StockUnit, normalizes each feature,
     * calls ONNX, and returns the scalar output. This is typically used for “entry” predictions
     * after a notification has fired.
     *
     * @param stockUnits List of StockUnit length ≥ size. Each StockUnit must provide:
     *                   getOpen(), getHigh(), getLow(), getClose(), getVolume(), getPercentageChange().
     * @return ONNX model’s float output (batch[0][0]), or 0F if insufficient data or on error.
     */
    public float predictFromWindow(List<StockUnit> stockUnits) {
        // If the list is null or shorter than the required number of time‐steps, we cannot form a valid input.
        if (stockUnits == null || stockUnits.size() < size) {
            // Return 0F as a safe fallback probability when data is insufficient.
            return 0F;
        }

        // Create a 3D float array with dimensions [batch=1][time_steps=size][features=N_FEATURES].
        // This array will hold the normalized feature vectors for all 'size' timesteps.
        float[][][] inputArray = new float[1][size][N_FEATURES];

        // Iterate over each of the first 'size' StockUnit entries to fill and normalize.
        for (int i = 0; i < size; i++) {
            StockUnit u = stockUnits.get(i);

            // 1) Extract raw feature values from StockUnit:
            float rawOpen = (float) u.getOpen();             // open price
            float rawHigh = (float) u.getHigh();             // high price
            float rawLow = (float) u.getLow();              // low price
            float rawClose = (float) u.getClose();            // close price
            float rawVol = (float) u.getVolume();           // traded volume
            float rawPct = (float) u.getPercentageChange(); // percentage change

            // 2) Normalize each feature: (raw value – precomputed mean) / precomputed scale
            //    The order of indices must match FEATURE_MEANS and FEATURE_SCALES arrays:
            //      index 0 → 'open', index 1 → 'high', index 2 → 'low',
            //      index 3 → 'close', index 4 → 'volume', index 5 → 'percentageChange'.
            inputArray[0][i][0] = (rawOpen - FEATURE_MEANS[0]) / FEATURE_SCALES[0];
            inputArray[0][i][1] = (rawHigh - FEATURE_MEANS[1]) / FEATURE_SCALES[1];
            inputArray[0][i][2] = (rawLow - FEATURE_MEANS[2]) / FEATURE_SCALES[2];
            inputArray[0][i][3] = (rawClose - FEATURE_MEANS[3]) / FEATURE_SCALES[3];
            inputArray[0][i][4] = (rawVol - FEATURE_MEANS[4]) / FEATURE_SCALES[4];
            inputArray[0][i][5] = (rawPct - FEATURE_MEANS[5]) / FEATURE_SCALES[5];
            // At this point, inputArray[0][i] is a length-6 vector of z-scored features for timestep i.
        }

        // Run ONNX inference inside a try-with-resources to ensure the OnnxTensor closes automatically.
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, inputArray)) {
            // session.run() takes a map of input names to OnnxTensor.
            // We assume the ONNX graph’s input node is named "input".
            OrtSession.Result result = session.run(Collections.singletonMap("input", tensor));

            // The ONNX model’s first output is expected to be a float[][] of shape [1][1].
            float[][] out = (float[][]) result.get(1).getValue();

            // Return the single scalar at out[0][0], which is the model’s predicted float.
            return out[0][0];
        } catch (OrtException e) {
            // If ONNX Runtime throws an exception (e.g., incorrect tensor shape),
            // parse the error message to see if the model expects a different 'size'.
            Integer expectedSize = parseExpectedSizeFromError(e.getMessage());
            if (expectedSize != null) {
                // Update our 'size' so future calls use the new required window length.
                size = expectedSize;
            }
            // Print stack trace for debugging, and return 0F as a safe fallback.
            e.printStackTrace();
            return 0F;
        }
    }

    /**
     * Static helper for “entry” predictions using entryPrediction.onnx.
     * Calls {@link #predictFromWindow(List)} on the singleton for that file.
     *
     * @param stockUnits Exactly size StockUnit objects (the 30-bar lookback).
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