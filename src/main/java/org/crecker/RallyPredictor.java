package org.crecker;

import ai.onnxruntime.*;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;

import java.nio.file.Paths;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

import static org.crecker.NotificationLabelingUI.normalizeWindowMinMax;

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
     */
    private static int size;

    /**
     * Total number of features per time‐step. Must match the model’s input “feature” dimension.
     */
    private static int N_FEATURES;

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
     * For each stock symbol, maintains a rolling window of the most recent feature vectors
     * used for uptrend detection. The window size is driven by {@code dynamicUptrendBufferSize}
     * and will automatically adjust if the uptrend model’s expected input length changes.
     * <p>
     * Key:   stock symbol
     * Value: linked list of float arrays, each array representing a feature vector at one time step
     */
    private final Map<String, LinkedList<float[]>> uptrendBuffer = new ConcurrentHashMap<>();

    /**
     * Ensures that buffer‐size adjustments happen atomically across all symbols.
     */
    private final ReentrantLock sizeAdjustmentLock = new ReentrantLock();

    /**
     * Current required history length (number of time steps) for streaming predictions.
     */
    private static int dynamicBufferSize;

    /**
     * Current required history length (number of time steps) for streaming uptrend predictions.
     */
    private static int dynamicUptrendBufferSize;

    /**
     * Number of features per time step for the uptrend model.
     */
    private static int featureLengthUptrend;

    /**
     * Number of features per time step. Must match the model’s expected input “feature” dimension.
     */
    private static int featureLength;

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
            e.printStackTrace();
            return 0F;
        }
    }

    /**
     * Appends the given feature vector into a per-symbol rolling window buffer,
     * and when the buffer has reached the configured {@link #dynamicBufferSize},
     * performs a prediction using the full window snapshot.
     * <p>
     * Thread-safety:
     * The per-symbol buffer is protected by synchronizing on the buffer instance
     * only for the minimal duration needed to update its contents and capture a snapshot.
     * Inference is performed outside the synchronized block to avoid blocking other threads.
     * </p>
     *
     * @param symbol   The stock symbol whose buffer this update belongs to.
     *                 Buffers are created on demand for new symbols.
     * @param features A feature vector representing the latest observation.
     *                 Must have length == {@link #featureLength}.
     * @return A non-null {@link Float} containing the model’s prediction if the rolling
     * buffer has reached size {@link #dynamicBufferSize}; otherwise returns 0F.
     * If an {@link OrtException} occurs during inference, the stack trace is
     * printed and 0F is returned.
     */
    public Float updateAndPredict(String symbol, float[] features) {
        // Retrieve or create the per-symbol buffer.
        LinkedList<float[]> buffer = symbolBuffers.computeIfAbsent(symbol, k -> new LinkedList<>());
        LinkedList<float[]> windowSnapshot = null;

        // 1) Update the buffer under lock:
        synchronized (buffer) {
            // Add the newest feature vector to the end.
            buffer.addLast(features);

            // Remove the oldest if we exceed the target window size.
            if (buffer.size() > dynamicBufferSize) {
                buffer.removeFirst();
            }

            // If we've accumulated enough data, capture a snapshot for inference.
            if (buffer.size() == dynamicBufferSize) {
                windowSnapshot = new LinkedList<>(buffer);
            }
        }

        // 2) Perform inference outside the lock to avoid blocking other threads.
        if (windowSnapshot != null) {
            try {
                return predictSpike(windowSnapshot);
            } catch (OrtException e) {
                // Log the exception and fall through to return 0F.
                e.printStackTrace();
            }
        }

        // 3) Not enough data or an error occurred: indicate “no prediction yet.”
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
        } catch (OrtException e) {
            // Print stack trace for debugging, and return 0F as a safe fallback.
            e.printStackTrace();
            return 0F;
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
     * Immediately predicts from a fixed, non‐streaming window of exactly {@link #size} bars.
     * <p>
     * Constructs a tensor of shape [1 × {@code size} × {@link #N_FEATURES}] by extracting and
     * normalizing the required fields from each {@link StockUnit} in the provided list,
     * then invokes the ONNX model and returns its scalar output.
     * This is typically used for “entry” predictions once sufficient data has been collected.
     * </p>
     *
     * @param stockUnits A non-null List of {@link StockUnit} instances with length ≥ {@link #size}.
     *                   Each {@code StockUnit} must supply values for:
     *                   {@link StockUnit#getOpen()}, {@link StockUnit#getHigh()},
     *                   {@link StockUnit#getLow()}, {@link StockUnit#getClose()},
     *                   {@link StockUnit#getVolume()}, and {@link StockUnit#getPercentageChange()}.
     * @return The model’s predicted score (the scalar at output[0][0]) if the input list
     * has at least {@link #size} elements and inference succeeds; otherwise returns 0F.
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

        // Iterate over each of the first 'size' StockUnit entries to fill
        for (int i = 0; i < size; i++) {
            StockUnit u = stockUnits.get(i);

            // Directly use raw values
            inputArray[0][i][0] = (float) u.getOpen();
            inputArray[0][i][1] = (float) u.getHigh();
            inputArray[0][i][2] = (float) u.getLow();
            inputArray[0][i][3] = (float) u.getClose();
            inputArray[0][i][4] = (float) u.getVolume();
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
            // Print stack trace for debugging, and return 0F as a safe fallback.
            e.printStackTrace();
            return 0F;
        }
    }

    /**
     * Static helper for “entry” predictions using entryPrediction.onnx.
     * Calls {@link #predictFromWindow(List)} on the singleton for that file.
     *
     * @param stockUnits Exactly size StockUnit objects (the 20-bar lookback).
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

            // slice to the correct size for inference
            int fromIdx = Math.max(0, stockUnits.size() - 20);
            List<StockUnit> slice = normalizeWindowMinMax(stockUnits.subList(fromIdx, stockUnits.size()));

            // Call the predictor’s predictFromWindow method, passing in the StockUnit window
            return predictor.predictFromWindow(slice);
        } catch (Exception ex) {
            // If anything goes wrong (e.g., model load or inference fails), print stack trace
            ex.printStackTrace();
            // Return 0F as a safe fallback probability
            return 0F;
        }
    }

    /**
     * Performs a streaming uptrend prediction for the given symbol using a preloaded ONNX model.
     * Loads the ONNX model from disk (under rallyMLModel/uptrendPredictor.onnx or tinyUptrend.onnx) the first time,
     * then uses a singleton RallyPredictor instance to update the symbol’s feature buffer
     * and return the latest uptrend prediction.
     *
     * @param features an array of feature values for the current time step
     * @param symbol   the stock symbol associated with these features
     * @return the model’s uptrend score, or 0.0f if an error occurs
     */
    public static float predictUptrend(float[] features, String symbol) {
        // Build the absolute path to the uptrend ONNX model file
        String uptrendModelPath = Paths
                .get(System.getProperty("user.dir"), "rallyMLModel", "tinyUptrend.onnx")
                .toString();
        try {
            // Get (or create) the singleton predictor for this model
            RallyPredictor predictor = RallyPredictor.getInstance(uptrendModelPath);

            // Delegate to the instance method to update buffer & predict
            return predictor.updateAndPredictUptrend(symbol, features);
        } catch (Exception e) {
            // Log any errors and fall back to 0.0f
            System.out.println("predictUptrend(streaming) error: " + e.getMessage());
            return 0F;
        }
    }

    /**
     * Updates the in‐memory rolling buffer of feature vectors for the given symbol
     * and, once the buffer reaches the required length, invokes the model inference.
     *
     * @param symbol   the stock symbol whose buffer is being updated
     * @param features the new feature vector to append
     * @return the uptrend prediction once enough history is collected, otherwise 0.0f
     * @throws OrtException if the ONNX runtime encounters an error
     */
    public Float updateAndPredictUptrend(String symbol, float[] features) throws OrtException {
        // Get or create the symbol’s feature buffer
        LinkedList<float[]> buffer = uptrendBuffer.computeIfAbsent(symbol, k -> new LinkedList<>());

        synchronized (buffer) {
            // Append the latest feature vector to the buffer
            buffer.addLast(features);
            // If buffer exceeds the model’s expected window, drop the oldest step
            if (buffer.size() > dynamicUptrendBufferSize) {
                buffer.removeFirst();
            }
            // If we have exactly the needed number of steps, perform inference
            if (buffer.size() == dynamicUptrendBufferSize) {
                return predictUptrendFromBuffer(buffer);
            }
        }
        // Not enough data yet: return default
        return 0F;
    }

    /**
     * Constructs the ONNX model input tensor from the buffered feature vectors
     * and executes the model to obtain a single uptrend score.
     *
     * @param buffer the rolling window of feature vectors (size == dynamicUptrendBufferSize)
     * @return the first element of the model’s output tensor
     * @throws OrtException if tensor creation or model execution fails
     */
    private Float predictUptrendFromBuffer(LinkedList<float[]> buffer) throws OrtException {
        // Prepare a [1 x time_steps x features] array for ONNX
        float[][][] inputArray = new float[1][dynamicUptrendBufferSize][featureLengthUptrend];

        // Copy each time step’s features into the input array
        for (int t = 0; t < dynamicUptrendBufferSize; t++) {
            float[] stepFeatures = buffer.get(t);
            // Copy up to featureLengthUptrend elements (or fewer if feature vector is shorter)
            System.arraycopy(
                    stepFeatures,
                    0,
                    inputArray[0][t],
                    0,
                    Math.min(stepFeatures.length, featureLengthUptrend)
            );
        }

        // Create the ONNX tensor and run the session
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            OrtSession.Result result = session.run(Collections.singletonMap("input", inputTensor));
            // Extract and return the scalar prediction from the output tensor
            float[][] output = (float[][]) result.get(0).getValue();
            return output[0][0];
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

    /**
     * Inspect each ONNX model under rallyMLModel/ and set the static fields:
     * <ul>
     *   <li><code>dynamicUptrendBufferSize</code> & <code>featureLengthUptrend</code>
     *       from <code>uptrendPredictor.onnx</code></li>
     *   <li><code>size</code> & <code>N_FEATURES</code>
     *       from <code>entryPrediction.onnx</code></li>
     *   <li><code>dynamicBufferSize</code> & <code>featureLength</code>
     *       from <code>spike_predictor.onnx</code></li>
     * </ul>
     *
     * @throws OrtException if any ONNX model fails to load or introspection fails.
     */
    public static void setParameters() throws OrtException {
        // Base directory where all ONNX models live
        String base = Paths.get(System.getProperty("user.dir"), "rallyMLModel").toString();

        // 1) uptrendPredictor.onnx or tinyUptrend.onnx → dynamicUptrendBufferSize & featureLengthUptrend
        int[] up = inspect(Paths.get(base, "tinyUptrend.onnx").toString());
        dynamicUptrendBufferSize = up[0];
        featureLengthUptrend = up[1];

        // 2) entryPrediction.onnx → size & N_FEATURES
        int[] entry = inspect(Paths.get(base, "entryPrediction.onnx").toString());
        size = entry[0];
        N_FEATURES = entry[1];

        // 3) spike_predictor.onnx → dynamicBufferSize & featureLength
        int[] spike = inspect(Paths.get(base, "spike_predictor.onnx").toString());
        dynamicBufferSize = spike[0];
        featureLength = spike[1];
    }

    /**
     * Loads an ONNX model at the given path, inspects its first
     * input tensor, and returns the expected window size (time_steps) and feature count.
     * <p>
     * <code>[batch, time_steps, num_features]</code>. We ignore the batch dimension
     * (must be 1 at runtime) and extract the remaining two axes.
     * </p>
     *
     * @param modelPath Absolute path to the ONNX file to inspect.
     * @return A two-element <code>int[]</code>: <code>[windowSize, nFeatures]</code>.
     * @throws OrtException          if creating the ONNX session or querying the metadata fails.
     * @throws IllegalStateException if the graph’s first input is not a Tensor.
     */
    public static int[] inspect(String modelPath) throws OrtException {
        // Acquire the shared ONNX Runtime environment
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

        // Open a session for introspection
        try (OrtSession session = env.createSession(modelPath, opts)) {
            // We only expect one input; grab its name
            String inputName = session.getInputNames().iterator().next();

            // Retrieve its NodeInfo to inspect shape
            NodeInfo info = session.getInputInfo().get(inputName);
            if (!(info.getInfo() instanceof TensorInfo)) {
                throw new IllegalStateException("ONNX model at [" + modelPath + "] "
                        + "does not expose a Tensor input");
            }

            // Extract the declared shape: [batch, time_steps, num_features]
            long[] shape = ((TensorInfo) info.getInfo()).getShape();

            // Cast down to int; if shape dims > Integer.MAX_VALUE you’ve bigger problems
            int windowSize = (int) shape[1];
            int nFeatures = (int) shape[2];

            return new int[]{windowSize, nFeatures};
        }
    }
}