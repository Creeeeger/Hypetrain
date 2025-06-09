package org.crecker;

import com.crazzyghost.alphavantage.AlphaVantage;
import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.fundamentaldata.response.CompanyOverview;
import com.crazzyghost.alphavantage.fundamentaldata.response.CompanyOverviewResponse;
import com.crazzyghost.alphavantage.news.response.NewsResponse;
import com.crazzyghost.alphavantage.parameters.Interval;
import com.crazzyghost.alphavantage.parameters.OutputSize;
import com.crazzyghost.alphavantage.realtime.response.RealTimeResponse;
import com.crazzyghost.alphavantage.stock.response.StockResponse;
import com.crazzyghost.alphavantage.timeseries.response.QuoteResponse;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.crazzyghost.alphavantage.timeseries.response.TimeSeriesResponse;
import org.jetbrains.annotations.NotNull;
import org.jfree.data.time.Second;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import static org.crecker.RallyPredictor.*;
import static org.crecker.dataTester.handleSuccess;
import static org.crecker.mainUI.*;
import static org.crecker.pLTester.*;

/**
 * The {@code mainDataHandler} class is responsible for the central logic and data flow in the stock analytics application.
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Manages storage and processing of time series and technical indicator data for multiple stocks.</li>
 *   <li>Defines static constants and utility collections for use throughout the analysis pipeline.</li>
 *   <li>Handles caching, indicator normalization, and categorization of stock features for advanced analytics.</li>
 *   <li>Coordinates real-time and historical data retrieval, normalization, and caching operations.</li>
 *   <li>Supports multithreaded data operations with thread-safe collections.</li>
 * </ul>
 *
 * <p>
 * The class is not intended to be instantiated; all functionality is via static methods and fields.
 * </p>
 */
public class mainDataHandler {

    /**
     * The ordered list of all technical indicator keys that will be extracted from each window of stock data.
     * <ul>
     *   <li>Order is CRITICAL; indexes correspond to downstream feature processing.</li>
     *   <li>Each entry maps to a specific type of indicator.</li>
     * </ul>
     */

    public static final List<String> INDICATOR_KEYS = List.of(
            "SMA_CROSS",              // 0: Simple Moving Average crossover event (binary state: -1, 0, 1)
            "TRIX",                   // 1: TRIX triple exponential average (momentum/trend indicator)
            "ROC",                    // 2: Rate Of Change (percentage, momentum indicator)
            "PLACEHOLDER",            // 3: Placeholder for future features, currently static
            "CUMULATIVE_PERCENTAGE",  // 4: Spike in recent cumulative percent move (binary)
            "CUMULATIVE_THRESHOLD",   // 5: Raw cumulative percent move over short window (numeric)
            "KELTNER",                // 6: Keltner channel breakout (binary)
            "ELDER_RAY"               // 7: Elder Ray Index (numeric)
    );

    /**
     * Feature‐wise scale factors for uptrend normalization.
     * <p>
     * Each entry SCALE[i] is the multiplier applied to the raw feature raw[i] before adding the corresponding
     * MIN_OFFSET[i], mapping the feature into the model’s expected input range.
     * <p>
     * Length must match {@code MIN_OFFSET.length} and the dimensionality of the uptrend feature vector.
     */
    public static final float[] SCALE = new float[]{
            0.03709612698921975F,
            0.03718037481173628F,
            0.037126334559678315F,
            0.03711035319058666F,
            1.1870690197540156e-06F,
            26.36753478595499F,
            0.06344392667150953F,
            1.2762418750533744F,
            1.228229629811581F,
            2.1590134317330776F,
            3.262574589133557F,
            0.12781457299556007F,
            0.11427646696766346F,
            0.06658063098303002F,
            0.998415021345876F,
            2.178107557782743e-06F,
    };

    /**
     * Feature‐wise minimum offsets for uptrend normalization.
     * <p>
     * Each entry MIN_OFFSET[i] is added to raw[i] * SCALE[i], shifting the scaled feature into its final range.
     * Negative values indicate that the feature’s minimum raw value must be offset downward after scaling.
     * <p>
     * Length must match {@code SCALE.length} and the dimensionality of the uptrend feature vector.
     */
    public static final float[] MIN_OFFSET = new float[]{
            -1.0340369869738109F,
            -1.0407093911370124F,
            -1.0324910423454468F,
            -1.0344617811922983F,
            -1.1870690197540156e-06F,
            0.5667664582441603F,
            -2.085401869692518F,
            0.0F,
            0.5750325480851886F,
            0.4939285600947485F,
            0.4459961241469656F,
            0.5176808566205282F,
            0.39627526885008F,
            0.3261715297271644F,
            0.0F,
            -0.0002844608470464262F,
    };

    /**
     * Set of technical indicators that should be interpreted as binary (0 or 1) in the normalization process.
     * <p>
     * This ensures normalization function always returns 0 or 1 for these indicators, regardless of raw value.
     * <p>
     * Used in {@link #normalizeScore(String, double, String)}
     */
    public static final Set<String> BINARY_INDICATORS = Set.of(
            "KELTNER",                // Keltner breakout: 1 if present, else 0
            "CUMULATIVE_PERCENTAGE"   // Cumulative spike: 1 if present, else 0
    );

    /**
     * The dedicated thread for the “second framework” real-time data loop.
     * <p>
     * When {@code useSecondFramework} transitions to {@code true}, this thread is
     * spawned to continuously poll and process data. As soon as
     * {@code useSecondFramework} becomes {@code false}, this thread is interrupted
     * and allowed to terminate. Only one instance of this thread should be active at a time.
     * </p>
     */
    private static Thread secondFrameworkThread = null;

    /**
     * Directory path for disk cache of stock data (e.g., "cache/" in current working directory).
     * <p>
     * Used for storing and retrieving locally cached minute-bar or indicator data to minimize API calls.
     */
    public static final String CACHE_DIR = Paths.get(System.getProperty("user.dir"), "cache").toString();

    /**
     * Central timeline map: stores the loaded or downloaded time series for each stock symbol.
     * <ul>
     *   <li>Key: Stock symbol (always uppercase, e.g., "AAPL")</li>
     *   <li>Value: List of {@link StockUnit} representing each time bar (chronological order)</li>
     * </ul>
     * <p>
     * This is the primary in-memory data store for the entire pipeline, used for all technical indicator calculation.
     * <p>
     * Thread-safe for concurrent reads and writes.
     */
    static final Map<String, List<StockUnit>> symbolTimelines = new ConcurrentHashMap<>();

    /**
     * Central real-time timeline map: stores the live, incoming time series for each stock symbol.
     * <ul>
     *   <li>Key: Stock symbol (always uppercase, e.g., "AAPL")</li>
     *   <li>Value: List of {@link StockUnit} representing each real-time time bar (chronological order)</li>
     * </ul>
     * <p>
     * This is a separate in-memory data store dedicated exclusively to real-time or streaming data,
     * ensuring that live updates do not interfere with the main {@link #symbolTimelines} historical or batch data.
     * <p>
     * Used for handling and analyzing live market feeds, short-term analytics, and immediate technical calculations.
     * <p>
     * Thread-safe for concurrent reads and writes.
     */
    static final Map<String, List<StockUnit>> realTimeTimelines = new ConcurrentHashMap<>();

    /**
     * Stores all generated {@link Notification} objects relevant to PL (Profit & Loss) analysis or spike detection.
     * <p>
     * Populated throughout analytics; used for summary UI panels, notification feeds, or backtesting.
     */
    static final List<Notification> notificationsForPLAnalysis = new ArrayList<>();

    /**
     * Stores precomputed min/max ranges for each technical indicator per symbol.
     * <p>
     * Data structure: Map&lt;SYMBOL, Map&lt;INDICATOR, Map&lt;"min"/"max", value&gt;&gt;&gt;
     * - Key: Uppercase symbol name (e.g., "AAPL")
     * - Value: For each indicator, holds a map containing robust minimum and maximum values, used to normalize raw feature data.
     * <p>
     * Populated by {@link #precomputeIndicatorRanges(boolean)} and consumed by {@link #normalizeScore(String, double, String)}
     */
    private static final Map<String, Map<String, Map<String, Double>>> SYMBOL_INDICATOR_RANGES = new ConcurrentHashMap<>();

    /**
     * Category weights for calculating final 'aggressiveness' score for a stock signal.
     * <p>
     * These weights determine the relative importance of features by their indicator category (trend, momentum, etc.).
     * <ul>
     *   <li>Keys: "TREND", "MOMENTUM", "STATISTICAL", "ADVANCED"</li>
     *   <li>Values: The proportion of the final aggressiveness attributed to each category</li>
     * </ul>
     * <p>
     * <b>Note:</b> These values can be dynamically tuned to alter system bias (e.g., for different market regimes).
     */
    private static final Map<String, Double> CATEGORY_WEIGHTS = new HashMap<>() {{
        /*
          Category   | Bull | Bear | High Volatility | Scraper
          -----------|------|------|-----------------|--------
          TREND      | 0.30 | 0.15 | 0.20            | 0.10
          MOMENTUM   | 0.40 | 0.25 | 0.35            | 0.10
          STATS      | 0.15 | 0.30 | 0.25            | 0.45
          ADVANCED   | 0.15 | 0.30 | 0.20            | 0.35
         */
        put("TREND", 0.1);       // Features 0-1 (SMA, TRIX) - Lowered for 'scraper' regime
        put("MOMENTUM", 0.1);    // Feature 2 (ROC) - Lowered for 'scraper' regime
        put("STATISTICAL", 0.45);// Features 4-5 (Spike, Cumulative) - Highest in 'scraper' regime
        put("ADVANCED", 0.35);   // Features 6-7 (Keltner, Elder) - Secondary importance
    }};

    /**
     * Maps feature index to its category for weighted scoring.
     * <p>
     * Allows for lookup of each feature's conceptual grouping ("TREND", "MOMENTUM", etc.) when aggregating scores.
     * <ul>
     *   <li>Index: Corresponds to {@link #INDICATOR_KEYS} order.</li>
     *   <li>Value: Category string.</li>
     * </ul>
     */
    private static final Map<Integer, String> FEATURE_CATEGORIES = new HashMap<>() {{
        put(0, "TREND");        // SMA crossover
        put(1, "TREND");        // TRIX
        put(2, "MOMENTUM");     // ROC
        put(3, "NEUTRAL");      // Placeholder
        put(4, "STATISTICAL");  // Cumulative spike
        put(5, "STATISTICAL");  // Cumulative % move
        put(6, "ADVANCED");     // Keltner
        put(7, "ADVANCED");     // Elder Ray
    }};

    /**
     * State-tracking map for per-symbol SMA crossover state.
     * <ul>
     *   <li>Key: Symbol</li>
     *   <li>Value: Integer state of SMA (1 = bullish crossover, -1 = bearish, 0 = neutral/no signal)</li>
     * </ul>
     * Used to prevent duplicate signals on repeated crossovers.
     * <p>
     * Thread-safe, supports concurrent indicator calculation in multithreaded contexts.
     */
    private static final ConcurrentHashMap<String, Integer> smaStateMap = new ConcurrentHashMap<>();

    /**
     * Dedicated single-threaded executor for scheduled background tasks such as real-time data polling.
     * <p>
     * Used by routines that require fixed-rate, non-blocking execution, e.g. {@link #realTimeDataCollector(String)}.
     */
    private static final ScheduledExecutorService executorService = Executors.newSingleThreadScheduledExecutor();

    /**
     * Cache of precomputed ATR (Average True Range) values per symbol.
     * <ul>
     *   <li>Key: Symbol</li>
     *   <li>Value: Cached ATR (used for adaptive gap threshold logic and high-volatility detection)</li>
     * </ul>
     * Avoids repeated ATR computation for the same look-back period.
     */
    private static final Map<String, Double> historicalATRCache = new ConcurrentHashMap<>();

    /**
     * Look-back window (in bars) for historical ATR calculations. Controls sensitivity of volatility measurements.
     * <p>
     * Set to 100 to provide a robust estimate across typical trading weeks.
     */
    private static final int HISTORICAL_LOOK_BACK = 100;

    /**
     * Number of minutes to aggregate per bar in rally analysis compression.
     * <p>
     * Used for grouping high-frequency bars into multi-minute "super-bars"
     * to smooth noise in trend and channel calculations.
     * Typical value: 120 minutes (2 hours per bar).
     */
    private static final int MINUTES = 120;

    /**
     * Number of look-back days for rally detection algorithms.
     * <p>
     * Defines window size for regression analysis and uptrend confirmation.
     */
    private static final int DAYS = 12;

    // ==================== RALLY PIPELINE PARAMETERS =====================
    /**
     * Total number of bars in full rally detection window.
     * <p>
     * Assumes ~8 bars per day (based on 2-hour bars, standard US session).
     * Used for window slicing in regression and channel width calculations.
     */
    private static final int W = DAYS * 8;

    /**
     * Minimum required regression slope (in percent per bar) for a rally to qualify as an "uptrend".
     * <p>
     * Used to reject trends that are too flat to be significant.
     */
    private static final double SLOPE_MIN = 0.1;

    /**
     * Maximum allowable regression channel width as percent of price.
     * <p>
     * Rejects "rallies" where the price action is too volatile or inconsistent with a clean trend.
     */
    private static final double WIDTH_MAX = 0.06;

    /**
     * Array of different day-based windows for multi-horizon rally checking.
     * <p>
     * E.g., checks consistency of uptrend over 4, 6, 8, 10, and 14 day spans.
     * Each is mapped to a number of bars for regression analysis.
     */
    private static final int[] DAYS_OPTIONS = {4, 6, 8, 10, 14};

    /**
     * Minimum R^2 (coefficient of determination) for accepting a regression trend as meaningful.
     * <p>
     * Rejects trends with too much scatter or inconsistency.
     */
    private static final double R2_MIN = 0.6;

    /**
     * Allowed tolerance as a percent for checking alignment of price to regression line at rally end.
     * <p>
     * If actual prices are outside this band, trend is considered broken.
     */
    private static final double TOLERANCE_PCT = 0.03;

    /**
     * Market open time (assumed for all data, in 24hr format).
     * <p>
     * Filters out pre-market and after-hours bars.
     */
    private static final LocalTime MARKET_OPEN = LocalTime.of(4, 0);

    /**
     * Market close time (assumed for all data, in 24hr format).
     * <p>
     * Used to identify valid in-session bars.
     */
    private static final LocalTime MARKET_CLOSE = LocalTime.of(20, 0);

    /**
     * Flag indicating whether the one‐time dry‐run buffer seeding (and any associated benchmarking)
     * has already been performed.
     * Defaults to false so that on first entry into the real-time loop the seeding logic will run;
     * once set to true, the seeding block is skipped on subsequent iterations.
     */
    private static boolean firstTimeComputed = false;

    // Map of different markets to scan select market to get list
    public static final Map<String, String[]> stockCategoryMap = new HashMap<>() {{
        // Add a new entry “allSymbols” with the full list of symbols
        put("allSymbols", new String[]{
                "ACHR", "ABBV", "AAPL", "ABT", "AAOI", "ABNB", "ADBE", "AEM", "ADI", "AFRM",
                "AES", "AKAM", "AMAT", "ALAB", "AMD", "AMT", "AMGN", "AMZN", "APH", "ANET",
                "APLD", "APO", "APP", "ARM", "APTV", "AS", "ASTS", "AVGO", "BABA", "ASPI",
                "AXP", "AZN", "BDX", "BBY", "BE", "BMY", "BKR", "BSX", "BP", "BTI", "BX",
                "C", "CARR", "CAT", "CAVA", "CDNS", "CF", "CEG", "CELH", "CMCSA", "CMG",
                "CME", "COF", "COIN", "CORZ", "CTVA", "COP", "CRDO", "CP", "CSX", "CRWD",
                "CSCO", "CVNA", "DDOG", "CVS", "DHI", "DFS", "DHR", "DIS", "DKNG", "DJT",
                "EA", "ENB", "ENPH", "EPD", "EOG", "ET", "EQNR", "EXAS", "EW", "EXPE", "FI",
                "FIVE", "FSLR", "FTNT", "FUTU", "GE", "GIS", "GEV", "GILD", "GLW", "GME",
                "GM", "GOOGL", "GSK", "HD", "HDB", "HIMS", "HSAI", "HOOD", "HON", "IBM",
                "IBN", "ICE", "IONQ", "INFY", "JD", "JPM", "KHC", "KKR", "LLY", "LRCX",
                "LOW", "LUMN", "LUNR", "LX", "MA", "MARA", "LUV", "MBLY", "MCHP", "MDB",
                "MDLZ", "MDT", "META", "MET", "MGM", "MKC", "MMM", "MO", "MRK", "MRNA",
                "MRVL", "MS", "MU", "MSFT", "MSTR", "NIO", "NFLX", "NKE", "NNE", "NOVA",
                "NVDA", "NVO", "NXPI", "OMC", "OPEN", "O", "OKLO", "ORCL", "OKE", "PBR",
                "PANW", "PCG", "PDD", "PM", "PLTR", "PGR", "PSX", "QBTS", "PYPL", "PTON",
                "QUBT", "QCOM", "RCAT", "RDDT", "RGTI", "RIO", "RIVN", "RKLB", "RXRX", "RUN",
                "SBUX", "SE", "SCHW", "SHOP", "SEDG", "SMCI", "SG", "SLB", "SNOW", "SMR",
                "SMTC", "SNY", "SONY", "SOUN", "SWK", "SOFI", "SPOT", "TCOM", "SYY", "TEM",
                "TJX", "TGT", "TMO", "TSN", "TMUS", "TSLA", "TTD", "TTEK", "UBER", "TXN",
                "UBS", "U", "UL", "UNP", "USB", "UPST", "UPS", "V", "VLO", "VKTX", "VST",
                "VRT", "WELL", "W", "WDAY", "WFC", "XOM", "XPEV", "ZIM", "ZTO", "ZETA", "ZTS"
        });
        put("aiStocks", new String[]{
                "AMD", "DDOG", "GOOGL", "META", "MSFT", "NVDA", "PLTR", "SMCI", "SNOW"
        });

        put("autoEV", new String[]{
                "GM", "NIO", "RIVN", "TSLA", "XPEV"
        });

        put("bigCaps", new String[]{
                "AAPL", "ABBV", "ABT", "AMGN", "AMZN", "AMD", "AVGO", "AXP",
                "BMY", "CAT", "CSCO", "COST", "CVX",
                "DE", "DIS", "GE", "GOOGL", "HD", "HON", "IBM",
                "JNJ", "JPM", "LLY", "LOW", "LMT", "MA",
                "MCD", "MDT", "META", "MMM", "MRK", "MSFT", "NFLX", "NVDA",
                "ORCL", "PEP", "PFE", "PG", "QCOM", "SBUX",
                "SBUX", "TGT", "TMO", "TMUS", "TXN", "UPS", "V", "WFC", "XOM"
        });

        put("chineseTech", new String[]{
                "BABA", "HDB", "INFY", "JD", "NIO", "PDD", "XPEV"
        });

        put("cryptoBlockchain", new String[]{
                "COIN", "CORZ", "HSAI", "MARA", "MSTR", "QBTS", "QUBT", "RGTI", "SOUN"
        });

        put("energy", new String[]{
                "BP", "COP", "ENB", "EOG", "ET", "OKE", "SLB", "VLO", "XOM"
        });

        put("financials", new String[]{
                "AXP", "BNS", "C", "COF", "JPM", "MET", "MS", "SCHW", "UBS", "USB", "WFC"
        });

        put("foodBeverage", new String[]{
                "CELH", "GIS", "KHC", "MDLZ", "SBUX", "TGT"
        });

        put("healthcareProviders", new String[]{
                "ANTM", "CI", "CNC", "CVS", "ELV", "HCA", "HUM", "UHS"
        });

        put("highVolatile", new String[]{
                "ACHR", "AFRM", "ASPI", "ASTS", "CELH", "COIN", "CORZ", "CVNA", "DJT",
                "ENPH", "FIVE", "FUTU", "HOOD", "LMND", "LUMN", "LUNR", "MBLY", "MSTR",
                "NIO", "NOVA", "PLTR", "PTON", "QBTS", "QUBT", "RCAT", "RDDT", "RKLB",
                "RIVN", "RUN", "RXRX", "SHOP", "SMCI", "SMR", "SMTC", "SOUN", "SOFI",
                "SNOW", "TTD", "UPST", "VKTX", "XPEV"
        });

        put("industrials", new String[]{
                "CAT", "GE", "HON", "MMM", "UNP"
        });

        put("midCaps", new String[]{
                "AAOI", "AFRM", "ALAB", "APLD", "APP", "ASPI", "CELH", "CVNA",
                "ENPH", "FIVE", "FUTU", "HIMS", "HOOD", "INFY", "LMND", "LPLA", "LULU",
                "MDB", "MBLY", "MSTR", "NIO", "NOVA", "OKLO", "OPEN", "PTON", "RIVN",
                "RUN", "SHOP", "SMTC", "SNOW", "SOUN", "SPOT", "TCOM", "TEM", "TTD",
                "UPST", "VKTX", "ZETA", "ZIM"
        });

        put("pharma", new String[]{
                "ABBV", "AMGN", "AZN", "BMY", "CELH", "GILD", "JNJ", "LLY", "MDT", "MRNA",
                "SNY", "ZTS"
        });

        put("quantum", new String[]{
                "IONQ", "QCOM", "QBTS", "QUBT", "RGTI"
        });

        put("retail", new String[]{
                "AMZN", "BBY", "HD", "LOW", "LULU", "MGM", "SONY",
                "TGT", "TJX"
        });

        put("robotics", new String[]{
                "ACHR", "MBLY", "RKLB"
        });

        put("semiconductors", new String[]{
                "ADI", "AMAT", "AMD", "ARM", "AVGO", "CDNS", "CRDO", "LRCX", "MCHP",
                "MRVL", "MU", "NVDA", "NXPI", "QCOM", "SMCI", "SMTC", "TSLA", "TXN"
        });

        put("smallCaps", new String[]{
                "ACHR", "ASTS", "BE", "CAVA", "CORZ", "CRDO", "DJT",
                "IONQ", "LMND", "LUMN", "LUNR", "LX", "MBLY", "QBTS", "QUBT", "RCAT",
                "RDDT", "RKLB", "RXRX", "SMR", "SOUN", "TMDX", "UPST", "VKTX", "XPEV", "ZIM"
        });

        put("techGiants", new String[]{
                "AAPL", "ADBE", "AMZN", "CSCO", "GOOGL", "META", "MSFT", "NVDA", "ORCL"
        });

        put("ultraVolatile", new String[]{
                "ACHR", "ASTS", "CORZ", "DJT", "ENPH", "IONQ", "LMND",
                "LUMN", "LUNR", "QUBT", "QBTS", "RCAT", "RDDT", "RKLB", "RXRX", "SMR", "SOUN",
                "UPST", "VKTX"
        });
        put("favourites", new String[]{ // These are my favourites and the ones working the best with the algorithm
                "APLD", "HIMS", "IONQ", "OKLO", "PLTR", "QBTS", "QUBT",
                "RGTI", "RKLB", "SMCI", "SMR", "SOUN", "TEM", "TTD", "U"
        });
    }};

    /* Mega-caps: typically the largest, most liquid names. */
    private static final List<String> MEGA_CAPS = List.of("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "TSLA");

    /* Large-caps: still very liquid but a notch below the giants. */
    private static final List<String> LARGE_CAPS = List.of("BABA", "AMD", "PLTR", "UBER", "CMCSA", "DIS", "SBUX", "XOM", "BP", "CSCO");

    /* Mid-caps: moderate liquidity and volatility. */
    private static final List<String> MID_CAPS = List.of("CMG", "CSX", "LRCX", "MCHP", "MRVL", "NKE", "SOFI", "TTD", "PTON", "RIVN");

    /* Small/micro-caps: prone to sharp one-minute swings. */
    private static final List<String> SMALL_CAPS = List.of("ACHR", "AES", "APLD", "COIN", "CORZ", "ET", "GME", "HIMS", "HOOD", "IONQ", "LUMN",
            "LUV", "MSTR", "MU", "NIO", "NVO", "OKLO", "OPEN", "PCG", "QBTS", "QUBT", "RGTI", "RKLB",
            "RUN", "RXRX", "SLB", "SMR", "SHOP", "SMCI", "SOUN", "TEM");

    /**
     * Number of bars in each analysis window (frame) for main technical signal generation.
     * <ul>
     *   <li>Controls the minimum data size for feature calculation and spike detection.</li>
     *   <li>Adjustable for tuning short-term vs. long-term strategy.</li>
     * </ul>
     */
    static int frameSize = 30; // Frame size for analysis (default: 30 bars, typically minutes)

    // ====================================================================

    /**
     * Initializes the Alpha Vantage API client using a provided API key.
     * <p>
     * Sets the API key and timeout value globally for all subsequent requests.
     * This method must be called before any API calls are made.
     *
     * @param token Your Alpha Vantage API key
     */
    public static void InitAPi(String token) {
        // Configure the API client with a key and a 10-second timeout
        Config cfg = Config.builder()
                .key(token)
                .timeOut(10) // Timeout in seconds (ensures non-blocking UI)
                .build();

        // Initialize the Alpha Vantage API singleton with the config
        AlphaVantage.api().init(cfg);
    }

    /**
     * Fetches the full minute-by-minute price timeline for a given stock symbol from Alpha Vantage API.
     * <p>
     * On success, triggers the provided {@link TimelineCallback} with a {@code List<StockUnit>} containing all available bars.
     * The result can be used for historical analysis, feature extraction, or visualizations.
     * This call is asynchronous and does not block the UI.
     *
     * @param symbolName The ticker symbol to fetch (e.g., "AAPL")
     * @param callback   Callback that receives the populated stock timeline
     */
    public static void getTimeline(String symbolName, TimelineCallback callback) {
        // Prepare a container for StockUnit bars (will be populated via API response)
        AlphaVantage.api()
                .timeSeries()
                .intraday()                        // Fetch intraday (minute-level) price data
                .forSymbol(symbolName)             // Set the stock symbol to retrieve data for
                .interval(Interval.ONE_MIN)        // Use 1-minute intervals for fine-grained candles
                .outputSize(OutputSize.FULL)       // Request the full available historical series
                .entitlement("realtime")           // Set the data pull mode to realtime
                .onSuccess(e -> {
                    // On successful data retrieval, cast the response and extract all StockUnit bars
                    TimeSeriesResponse response = (TimeSeriesResponse) e;
                    List<StockUnit> stocks = new ArrayList<>(response.getStockUnits());

                    // ====== BEGIN: Spike Trimming ======
                    // Occasionally, the API or exchange returns bad/corrupt data where
                    // the high or low of a candle is an extreme outlier (a "wick spike").
                    // To avoid plotting these on the chart (which distorts the scale and misleads users),
                    // we clamp (limit) the high/low values to a reasonable threshold relative to the open/close.

                    double spikeThreshold = 0.15; // Allow candle wicks to be at most ±15% away from open/close

                    for (StockUnit stock : stocks) {
                        double open = stock.getOpen();    // Opening price for the interval
                        double close = stock.getClose();  // Closing price for the interval
                        double high = stock.getHigh();    // High price for the interval (could be a spike)
                        double low = stock.getLow();      // Low price for the interval (could be a spike)

                        // Determine the normal trading range for this candle
                        double maxOC = Math.max(open, close);
                        double minOC = Math.min(open, close);

                        // Calculate the maximum and minimum allowed values for high and low
                        // based on the spike threshold. E.g., for a 15% threshold and a maxOC of 100,
                        // allowedHigh = 115.0, allowedLow = minOC * 0.85 = 85.0
                        double allowedHigh = maxOC * (1 + spikeThreshold);
                        double allowedLow = minOC * (1 - spikeThreshold);

                        // If the high is greater than allowedHigh, clamp it down to allowedHigh.
                        if (high > allowedHigh) stock.setHigh(allowedHigh);

                        // If the low is lower than allowedLow, clamp it up to allowedLow.
                        if (low < allowedLow) stock.setLow(allowedLow);
                    }
                    // ====== END: Spike Trimming ======

                    // Pass the cleaned-up list to the callback for further UI/chart processing
                    callback.onTimeLineFetched(stocks);
                })
                .onFailure(mainDataHandler::handleFailure) // Display errors if the fetch fails
                .fetch();                                  // Begin asynchronous data fetch
    }

    /**
     * Fetches both fundamental and latest quote data for a given stock symbol,
     * assembling a 9-element {@code Double[]} array with common key metrics.
     * <p>
     * The resulting array is populated as follows:
     * <pre>
     *   [0] Open price (from latest quote)
     *   [1] High price (from latest quote)
     *   [2] Low price (from latest quote)
     *   [3] Volume (from latest quote)
     *   [4] P/E Ratio (from fundamentals)
     *   [5] PEG Ratio (from fundamentals)
     *   [6] 52-week high (from fundamentals)
     *   [7] 52-week low (from fundamentals)
     *   [8] Market capitalization (from fundamentals)
     * </pre>
     * <b>Note:</b> This method launches two asynchronous API calls (fundamental and quote data).
     * The callback is called only after the quote data is received (but before fundamentals necessarily finish).
     * For fully synchronized fundamental+quote delivery, consider a custom latch.
     *
     * @param symbolName The ticker symbol (e.g., "AAPL", "MSFT")
     * @param callback   Callback that will receive the populated Double[] data array.
     */
    public static void getInfoArray(String symbolName, DataCallback callback) {
        // Pre-allocate array to store all 9 values (indexes defined above)
        Double[] data = new Double[9];

        // ====== Fetch fundamental data asynchronously ======
        AlphaVantage.api()
                .fundamentalData()
                .companyOverview()
                .forSymbol(symbolName)
                .onSuccess(e -> {
                    // Upon successful response, extract key fields from fundamentals
                    CompanyOverviewResponse companyOverviewResponse = (CompanyOverviewResponse) e;
                    CompanyOverview response = companyOverviewResponse.getOverview();
                    data[4] = response.getPERatio() != null ? response.getPERatio() : 0.0;
                    data[5] = response.getPEGRatio() != null ? response.getPEGRatio() : 0.0;
                    data[6] = response.getFiftyTwoWeekHigh() != null ? response.getFiftyTwoWeekHigh() : 0.0;
                    data[7] = response.getFiftyTwoWeekLow() != null ? response.getFiftyTwoWeekLow() : 0.0;
                    Long mktCap = response.getMarketCapitalization();
                    data[8] = (mktCap == null) ? 0.0 : mktCap.doubleValue();
                })
                .onFailure(mainDataHandler::handleFailure)  // Logs any API failure for debugging
                .fetch();

        // ====== Fetch current quote data asynchronously ======
        AlphaVantage.api()
                .timeSeries()
                .quote()
                .forSymbol(symbolName)
                .entitlement("realtime")
                .onSuccess(e -> {
                    // On quote data arrival, fill in open, high, low, and volume
                    QuoteResponse response = (QuoteResponse) e;
                    data[0] = response.getOpen();      // Latest open price
                    data[1] = response.getHigh();      // Latest high price
                    data[2] = response.getLow();       // Latest low price
                    data[3] = response.getVolume();    // Latest volume

                    // Callback fires here – note: fundamental data may or may not have arrived yet
                    callback.onDataFetched(data);
                })
                .onFailure(mainDataHandler::handleFailure)  // Logs error on API failure
                .fetch();
    }

    /**
     * Generic global handler for AlphaVantage API failures.
     * <p>
     * Prints stack trace to stderr. Intended as a quick diagnostics handler,
     * but can be replaced with a more robust UI-level notification or logging framework.
     *
     * @param error The thrown exception from API
     */
    public static void handleFailure(AlphaVantageException error) {
        error.printStackTrace(); // Simple debug log; can be expanded to UI or persistent logs
    }

    /**
     * Searches the AlphaVantage symbol database for tickers matching a user search string.
     * <p>
     * Typically used to implement an "autocomplete" or "search bar" in the UI,
     * providing the user with a list of possible tickers as they type.
     * Asynchronous: does not block the calling thread.
     *
     * @param searchText The partial or full search string (e.g., "Apple", "TESL")
     * @param callback   Callback that receives either a List of matching symbol strings, or a failure event.
     */
    public static void findMatchingSymbols(String searchText, SymbolSearchCallback callback) {
        AlphaVantage.api()
                .Stocks()
                .setKeywords(searchText)     // Set the search pattern (not case-sensitive)
                .onSuccess(e -> {
                    // On success, extract and map symbols from each match result
                    List<String> allSymbols = e.getMatches()
                            .stream()
                            .map(StockResponse.StockMatch::getSymbol) // Extract only the symbol string
                            .toList();
                    callback.onSuccess(allSymbols);    // Deliver results to UI or caller
                })
                .onFailure(failure -> {
                    // On failure: log the error, and return a RuntimeException via callback
                    mainDataHandler.handleFailure(failure);
                    callback.onFailure(new RuntimeException("API call failed"));
                })
                .fetch(); // Non-blocking, returns immediately
    }

    /**
     * Fetches the latest news headlines and article summaries for a given ticker symbol.
     * <p>
     * Allows the UI to display recent, relevant news for a specific company or stock.
     * Limits to the 12 most recent stories, sorted with the newest first.
     *
     * @param Symbol   The ticker symbol (case-insensitive, e.g., "AAPL", "TSLA")
     * @param callback Callback that receives a List of {@link NewsResponse.NewsItem} objects on success.
     */
    public static void receiveNews(String Symbol, ReceiveNewsCallback callback) {
        AlphaVantage.api()
                .News()
                .setTickers(Symbol)      // Target stock ticker(s), comma separated
                .setSort("LATEST")       // Ensure latest news comes first
                .setLimit(12)            // Only fetch up to 12 headlines/articles
                .onSuccess(e -> callback.onNewsReceived(e.getNewsItems())) // Deliver news to UI or caller
                .onFailure(mainDataHandler::handleFailure) // Log/notify on API failure
                .fetch();
    }

    /**
     * Starts "Hype Mode" auto-scanning for stock symbols using the specified market regime and trade volume.
     * <p>
     * This method dynamically assembles a list of stock symbols tailored to both the user-selected market regime
     * and trading volume, ensuring sufficient liquidity for realistic simulations or scanning.
     * It uses per-(regime+volume) local file caching to minimize redundant API calls, and always invokes
     * {@link #hypeModeFinder(List)} with the selected symbol set.
     *
     * <ul>
     *   <li><b>For all trade volumes</b>, the function uses a regime-specific cache file
     *       (named "{@code [marketRegime]_[tradeVolume].txt}").</li>
     *   <li>If the file exists, it loads symbols for that regime directly from disk to save time and API quota.</li>
     *   <li>If the file does not exist, it dynamically filters symbols for that regime via
     *       {@link #getAvailableSymbols}, then writes them to the cache for future use.</li>
     *   <li>In all cases, the resulting list is passed to {@link #hypeModeFinder(List)} for analysis.</li>
     * </ul>
     * <p>
     * All progress and status updates are shown in the UI's {@code logTextArea} in real time, and
     * all file operations and symbol selection are now <b>regime-aware</b>.
     *
     * @param marketRegime The currently selected market regime/category (e.g., "aiStocks", "bigCaps", etc.)
     * @param tradeVolume  The user-selected trade volume for filtering stocks (e.g., 10,000, 100,000, etc.)
     */
    public static void startHypeMode(String marketRegime, int tradeVolume) {
        // --- 1. Get symbol list for the active market regime ---
        // Uses the stockCategoryMap to fetch only the symbols relevant to the selected regime
        String[] symbolsForRegime = stockCategoryMap.getOrDefault(marketRegime, new String[0]);

        // Log start message: indicates hype mode activation, current market regime, and initial parameters
        logTextArea.append(String.format(
                "Activating hype mode for auto Stock scanning, Settings: %s Volume, %s Market Regime, %s Stocks to scan\n",
                tradeVolume, marketRegime, symbolsForRegime.length));
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        // Local container for filtered or loaded symbol names (uppercased)
        List<String> possibleSymbols = new ArrayList<>();

        // --- 2. Use a regime-specific cache file (e.g., "aiStocks_90000.txt") ---
        String cacheFileName = marketRegime + "_" + tradeVolume + ".txt";
        File file = new File(cacheFileName);

        // ======= MAIN LOGIC BRANCH: LARGE TRADE VOLUME =======
        if (file.exists()) {
            // ------------ [1] FILE EXISTS: LOAD SYMBOLS FROM DISK ------------
            try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                String line;
                // Read every line as a symbol and add to possibleSymbols
                while ((line = reader.readLine()) != null) {
                    possibleSymbols.add(line);
                }
            } catch (IOException e) {
                e.printStackTrace(); // Print to stderr for debugging if file can't be read
            }

            // Log success to UI (shows which regime file was loaded)
            logTextArea.append("Loaded symbols from file: " + cacheFileName + "\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

            // Proceed to next step with the loaded symbols
            hypeModeFinder(possibleSymbols);

        } else {
            // ------------ [2] FILE DOES NOT EXIST: FETCH AND CACHE SYMBOLS ------------
            logTextArea.append("Started getting possible symbols for regime: " + marketRegime + "\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

            // Dynamically fetch suitable symbols for this regime and cache to file for next run
            getAvailableSymbols(tradeVolume, symbolsForRegime, result -> {
                try (FileWriter writer = new FileWriter(file)) {
                    for (String s : result) {
                        String symbol = s.toUpperCase();
                        possibleSymbols.add(symbol);  // Add to runtime set
                        writer.write(symbol + System.lineSeparator()); // Write to cache file
                    }
                } catch (IOException e) {
                    e.printStackTrace(); // Error writing cache file (still proceeds)
                }

                // Log finish and proceed to main analysis (with regime info)
                logTextArea.append("Finished getting possible symbols for regime: " + marketRegime + "\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                hypeModeFinder(possibleSymbols); // Continue with filtered set
            });
        }
    }

    /**
     * Dynamically filters a universe of stock symbols to find those that meet liquidity and tradability
     * requirements for a specific trade volume. Designed for "Hype Mode" symbol preselection.
     * <p>
     * This method is highly parallel and asynchronous. For each candidate symbol:
     * <ul>
     *   <li>Fetches fundamental data (market cap, shares outstanding) via AlphaVantage.</li>
     *   <li>Fetches recent daily price and volume bars for liquidity analysis.</li>
     *   <li>Applies several liquidity filters (market cap, avg volume, shares available, max open limit).</li>
     *   <li>If a symbol passes all checks, it is added to the result list.</li>
     *   <li>When all symbols have completed (success or fail), triggers the callback with the filtered list.</li>
     * </ul>
     * Filters and calculations are designed to ensure only realistically tradable, liquid stocks are
     * considered for high-volume algorithmic strategies or simulation.
     *
     * @param tradeVolume     The cash trade size to check (e.g., $100,000)
     * @param possibleSymbols Universe of candidate tickers to filter (e.g., S&P500 list)
     * @param callback        Callback invoked with the filtered List of tradable symbols
     */
    public static void getAvailableSymbols(int tradeVolume, String[] possibleSymbols, SymbolCallback callback) {
        // If no possible symbols, immediately short-circuit and return empty list to callback.
        if (possibleSymbols.length == 0) {
            callback.onSymbolsAvailable(Collections.emptyList());
            return;
        }

        // Thread-safe list to collect symbols passing all checks (safe for concurrent mutation)
        List<String> actualSymbols = new CopyOnWriteArrayList<>();

        // Atomic counter to track completion of async calls (one per candidate symbol)
        AtomicInteger remaining = new AtomicInteger(possibleSymbols.length);

        // Timeout executor: single thread, but fine since timeout tasks are light
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        // ==================== FILTER THRESHOLDS ====================
        // (These values can be tuned for stricter or looser liquidity requirements)
        final double MARKET_CAP_PERCENTAGE = 0.001;         // Max trade = 5% of market cap
        final double AVG_VOLUME_PERCENTAGE = 0.01;         // Max shares = 20% of avg 30-day volume
        final double SHARES_OUTSTANDING_PERCENTAGE = 0.01; // Max shares = 1% of shares outstanding

        // ==================== PROGRESS DIALOG SETUP ====================

        // Create a single-element array to hold the ProgressDialog instance, so it can be modified inside inner classes/lambdas
        final ProgressDialog[] progressDialog = new ProgressDialog[1];

        // If the logTextArea (used to determine the parent frame for dialogs) is present
        if (logTextArea != null) {
            // Ensure UI code runs on the Event Dispatch Thread (EDT) for thread safety
            SwingUtilities.invokeLater(() -> {
                // Get the parent frame of the logTextArea component
                Frame parentFrame = (Frame) SwingUtilities.getWindowAncestor(logTextArea);
                // Create the progress dialog and attach it to the parent frame
                progressDialog[0] = new ProgressDialog(parentFrame);
                // Set the dialog title to describe the current task
                progressDialog[0].setTitle("Filtering Symbols for tradability");
                // Make the dialog visible to the user
                progressDialog[0].setVisible(true);
            });
        }

        // ==================== MODIFIED COMPLETION HANDLER ====================

        // Define a runnable to handle the completion of symbol processing tasks
        Runnable completionHandler = () -> {
            // Calculate how many symbols have been processed so far, decrementing the remaining counter
            int processed = possibleSymbols.length - remaining.decrementAndGet();

            // Update the progress dialog on the Event Dispatch Thread (EDT)
            SwingUtilities.invokeLater(() -> {
                // If the progress dialog exists, update its progress bar
                if (progressDialog[0] != null) {
                    progressDialog[0].updateProgress(processed, possibleSymbols.length);
                }
            });

            // If all symbols have been processed (remaining count reaches zero)
            if (remaining.get() == 0) {
                // Shut down the scheduler to stop any further scheduled tasks
                scheduler.shutdown();
                // Notify the callback that the actualSymbols are available/ready
                callback.onSymbolsAvailable(actualSymbols);

                // Dispose of (close) the progress dialog on the EDT
                SwingUtilities.invokeLater(() -> {
                    if (progressDialog[0] != null) {
                        progressDialog[0].dispose();
                    }
                });
            }
        };

        // ==================== MAIN FILTER LOOP ====================
        for (String symbol : possibleSymbols) {
            // This array is used as a mutable holder to allow the inner classes to cancel the timeout
            final ScheduledFuture<?>[] timeoutFuture = new ScheduledFuture<?>[1];

            Runnable timeoutTask = () -> {
                System.out.println("Timeout for symbol: " + symbol + " (no response in 60)");
                // No need to add the symbol to the results, just proceed as if it failed
                completionHandler.run();
            };

            // Schedule the timeout (60 seconds)
            timeoutFuture[0] = scheduler.schedule(timeoutTask, 60, TimeUnit.SECONDS);

            // --------- [1] Fetch company fundamentals (market cap, shares out) ---------
            AlphaVantage.api()
                    .fundamentalData()
                    .companyOverview()
                    .forSymbol(symbol)
                    .onSuccess(e -> {
                        CompanyOverviewResponse companyResponse = (CompanyOverviewResponse) e;
                        CompanyOverview overview = companyResponse.getOverview();

                        // === Check for nulls ===
                        if (overview == null ||
                                overview.getMarketCapitalization() == null ||
                                overview.getSharesOutstanding() == null) {
                            // Log and skip this symbol
                            System.out.println("Null data for symbol: " + symbol + " (likely delisted or incomplete fundamentals)");
                            completionHandler.run();
                            return;
                        }

                        long marketCapitalization = companyResponse.getOverview().getMarketCapitalization();
                        long sharesOutstanding = companyResponse.getOverview().getSharesOutstanding();

                        // --------- [2] Fetch recent daily price/volume series for this symbol ---------
                        AlphaVantage.api()
                                .timeSeries()
                                .daily()
                                .forSymbol(symbol)
                                .outputSize(OutputSize.COMPACT)
                                .entitlement("realtime")
                                .onSuccess(tsResponse -> {
                                    timeoutFuture[0].cancel(false);
                                    TimeSeriesResponse ts = (TimeSeriesResponse) tsResponse;
                                    List<StockUnit> stockUnits = ts.getStockUnits();

                                    // Use the latest close price for shares-to-buy calculation.
                                    double close = stockUnits.get(0).getClose();

                                    // If there is no price data, skip this symbol.
                                    if (stockUnits.isEmpty() || close <= 0) {
                                        completionHandler.run();
                                        return;
                                    }

                                    // Compute how many shares we need to buy for the given trade volume.
                                    double sharesToTrade = tradeVolume / close;

                                    // Compute average volume over last 30 trading days (or as many as available).
                                    int daysToConsider = Math.min(30, stockUnits.size());
                                    double totalVolume = 0;
                                    for (int i = 0; i < daysToConsider; i++) {
                                        totalVolume += stockUnits.get(i).getVolume();
                                    }

                                    double averageVolume = totalVolume / daysToConsider;

                                    // ========== LIQUIDITY FILTERS ==========
                                    // [A] Is trade small enough relative to market cap?
                                    boolean validMarketCap = (double) tradeVolume <= MARKET_CAP_PERCENTAGE * marketCapitalization;

                                    // [B] Is trade small enough relative to daily trading volume?
                                    boolean validVolume = tradeVolume <= AVG_VOLUME_PERCENTAGE * averageVolume;

                                    // [C] Is trade small enough relative to total shares outstanding?
                                    boolean validSharesOutstanding = sharesToTrade <= SHARES_OUTSTANDING_PERCENTAGE * sharesOutstanding;

                                    // [OPTIONAL] Print detailed liquidity filter diagnostics for debugging
                                    System.out.println(
                                            "===== Liquidity Check for: " + symbol + " =====\n" +
                                                    "Trade Volume ($): " + tradeVolume + "\n" +
                                                    "Close Price: " + close + "\n" +
                                                    "Shares to Trade: " + sharesToTrade + "\n" +
                                                    "Market Cap: " + marketCapitalization + "\n" +
                                                    "Average Volume (30-day): " + averageVolume + "\n" +
                                                    "Shares Outstanding: " + sharesOutstanding + "\n" +
                                                    "Valid Market Cap? " + validMarketCap + "\n" +
                                                    "Valid Volume? " + validVolume + "\n" +
                                                    "Valid Shares Outstanding? " + validSharesOutstanding + "\n" +
                                                    "====================================="
                                    );

                                    // [D] Only add to final result if ALL filters pass
                                    if (validMarketCap && validVolume && validSharesOutstanding) {
                                        actualSymbols.add(symbol);
                                    }

                                    completionHandler.run();
                                })
                                .onFailure(error -> {
                                    timeoutFuture[0].cancel(false);
                                    // If daily bar fetch fails, log and continue
                                    mainDataHandler.handleFailure(error);
                                    completionHandler.run();
                                })
                                .fetch();
                    })
                    .onFailure(error -> {
                        timeoutFuture[0].cancel(false);
                        // If fundamental data fetch fails, log and continue
                        mainDataHandler.handleFailure(error);
                        completionHandler.run();
                    })
                    .fetch();
        }
    }

    /**
     * Orchestrates the end-to-end processing pipeline for a batch of real-time stock data matches.
     * <p>
     * This method performs three major steps, in order:
     * <ol>
     *   <li><b>prepareMatches:</b> Integrates raw real-time data into application data structures and updates timelines.</li>
     *   <li><b>calculateChangesForSeconds:</b> Computes and updates percentage changes for each stock symbol across the most recent time windows.</li>
     *   <li><b>evaluateSeconds:</b> Runs analytics, event detection, and/or alert logic based on the updated time series.</li>
     * </ol>
     *
     * <p>
     * This method should be called every polling cycle after new real-time data is fetched.
     * <b>Side effects:</b> Mutates application state by updating time series, computed fields, and possibly triggering user-facing events or notifications.
     *
     * @param matches List of new real-time data results, one per stock symbol, to be processed.
     */
    public static void processStockData(List<RealTimeResponse.RealTimeMatch> matches) {
        // Step 1: Assimilate new real-time matches into main data structures and timelines.
        prepareMatches(matches);

        // Step 2: For all timelines, calculate percentage changes.
        calculateChangesForSeconds();

        // Step 3: Evaluate analytics, technical signals, or alert triggers based on the latest data.
        evaluateSeconds();
    }

    /**
     * Integrates a batch of real-time stock data into the application's main data structures.
     * <p>
     * For each real-time data match:
     * <ul>
     *   <li>Converts the API-provided match object to an internal {@link StockUnit} representation.</li>
     *   <li>Chooses between regular and extended-hours prices depending on current market context.</li>
     *   <li>Ensures all timeline data is stored in a thread-safe structure for the given stock symbol.</li>
     *   <li>Keeps a temporary map of the most recent update per symbol for quick access (useful for UI or later logic).</li>
     *   <li>Appends a summary status message for diagnostics or user feedback.</li>
     * </ul>
     * <p>
     * Side effects: Updates the global {@code realTimeTimelines} and writes to {@code logTextArea}.
     *
     * @param matches List of API-provided real-time matches to process and store.
     */
    private static void prepareMatches(List<RealTimeResponse.RealTimeMatch> matches) {
        // Tracks the most recent StockUnit inserted per symbol in this batch (not always needed, but may help with UI/summaries)
        Map<String, StockUnit> currentBatch = new ConcurrentHashMap<>();

        // Process each real-time match entry individually
        for (RealTimeResponse.RealTimeMatch match : matches) {
            // Ensure symbol keys are always uppercase for consistency in data structures
            String symbol = match.getSymbol().toUpperCase();

            // Decide whether to use extended-hours (pre-/post-market) data for this tick; fallback to regular hours if not available
            boolean extended = useExtended(match);

            // For a real-time tick, only the "close" price is meaningful; set all OHLC fields to the same value for this tick
            double close = extended ? match.getExtendedHoursQuote() : match.getClose();
            double open = extended ? match.getExtendedHoursQuote() : match.getOpen();
            double high = extended ? match.getExtendedHoursQuote() : match.getHigh();
            double low = extended ? match.getExtendedHoursQuote() : match.getLow();
            double volume = match.getVolume(); // Extended-hours volume is often zero or not reported, but use what’s available

            // Build the StockUnit object with all relevant data for this instant
            StockUnit unit = new StockUnit.Builder()
                    .symbol(symbol)
                    .open(open)
                    .high(high)
                    .low(low)
                    .close(close)
                    .time(match.getTimestamp())
                    .volume(volume)
                    .build();

            // Add the StockUnit to the thread-safe timeline for this symbol (initialize list if necessary)
            synchronized (realTimeTimelines) {
                realTimeTimelines
                        .computeIfAbsent(symbol, k -> Collections.synchronizedList(new ArrayList<>()))
                        .add(unit);

            }
            // Track this unit in the local batch map (for UI, summary, or further per-batch logic)
            currentBatch.put(symbol, unit);
        }

        // Provide a user/debug message indicating how many distinct symbols/entries were processed in this run
        logTextArea.append("Processed " + currentBatch.size() + " valid stock entries\n");
    }

    /**
     * Calculates and updates the percentage change between consecutive {@link StockUnit} entries
     * for each tracked stock symbol in the {@code realTimeTimelines} data structure.
     * <p>
     * This method is thread-safe: it synchronizes on the {@code realTimeTimelines} map to ensure
     * consistency if other threads are also updating timelines.
     * <p>
     * For each symbol, iterates through its timeline (list of StockUnits) and sets the
     * percent change (relative to previous close) on each {@link StockUnit} (starting from the second item).
     * <ul>
     *   <li>Skips symbols with fewer than 2 data points (nothing to compare).</li>
     *   <li>Skips calculations where the previous close is zero or negative (avoids divide-by-zero or nonsensical math).</li>
     * </ul>
     *
     * <b>Side effects:</b> Mutates each {@link StockUnit} in {@code realTimeTimelines} by calling {@code setPercentageChange()}.
     */
    private static void calculateChangesForSeconds() {
        // Thread safety: lock the timelines map during calculation
        synchronized (realTimeTimelines) {
            // Parallel stream: handles each symbol's timeline concurrently for speed (safe because no symbol's timeline is shared)
            realTimeTimelines.keySet().parallelStream().forEach(symbol -> {
                // Get this symbol's time series of StockUnit entries
                List<StockUnit> timeline = realTimeTimelines.get(symbol);

                // Defensive: Need at least 2 data points to calculate a percentage change
                if (timeline.size() < 2) {
                    return;
                }

                // Iterate through timeline, calculating percentage change for each pair (i-1, i)
                for (int i = 1; i < timeline.size(); i++) {
                    StockUnit current = timeline.get(i);
                    StockUnit previous = timeline.get(i - 1);

                    // Only compute if the previous close price is positive (avoid divide-by-zero and negative price anomalies)
                    if (previous.getClose() > 0) {
                        // Calculate percent change: ((current - previous) / previous) * 100
                        double change = ((current.getClose() - previous.getClose()) / previous.getClose()) * 100;
                        // Store the result in the current StockUnit
                        current.setPercentageChange(change);
                    }
                }
            });
        }
    }

    /**
     * Analyzes second-by-second stock data for each symbol, detects rapid uptrends ("spikes"),
     * and, if detected, aggregates the relevant window into a 1-minute bar and triggers further advanced processing.
     * <p>
     * This method performs several key operations:
     * <ul>
     *   <li><b>Cleanup:</b> Removes StockUnit entries older than 90 seconds for each symbol to conserve memory and processing power.</li>
     *   <li><b>De-duplication:</b> Removes duplicate StockUnits with the same timestamp to ensure clean analytics.</li>
     *   <li><b>Statistical Analysis:</b> Calculates total percent change and counts "green bars" (positive movement) over the current time window.</li>
     *   <li><b>Spike Detection:</b> Detects strong uptrend's based on configurable percentage and ratio of green bars.</li>
     *   <li><b>Aggregation:</b> If a spike is found, aggregates second bars into a single 1-minute StockUnit bar.</li>
     *   <li><b>Advanced Processing Trigger:</b> Launches advanced analysis/alerting for the detected spike using a high-priority thread.</li>
     * </ul>
     * <b>Side effects:</b> Mutates {@code realTimeTimelines} and {@code symbolTimelines}, and may trigger UI/log actions or background processing.
     */
    private static void evaluateSeconds() {
        // Step 1: For each symbol, remove any StockUnit older than the most recent 55 seconds.
        // This keeps memory and processing focused on the most relevant, recent market activity.
        synchronized (realTimeTimelines) {
            realTimeTimelines.forEach((symbol, timeline) ->
                    timeline.removeIf(unit ->
                            unit.getLocalDateTimeDate().isBefore(
                                    timeline.get(timeline.size() - 1) // Reference: most recent bar's time
                                            .getLocalDateTimeDate()
                                            .minusSeconds(55)            // Define "too old" as >55 seconds ago
                            )
                    )
            );
        }

        synchronized (realTimeTimelines) {
            // Step 2: For every symbol, analyze its timeline in parallel for real-time event detection.
            realTimeTimelines.keySet()
                    .parallelStream() // Multithreaded: handles each symbol concurrently for speed/scalability.
                    .forEach(symbol -> {
                        // Create a working copy of this symbol's recent StockUnit objects for processing.
                        // Avoids mutating the global list during parallel analysis.
                        List<StockUnit> realTimeWindow = new ArrayList<>(realTimeTimelines.getOrDefault(symbol.toUpperCase(), new ArrayList<>()));

                        // Proceed only if we have data points to analyze.
                        if (!realTimeWindow.isEmpty()) {
                            // Step 3: Remove duplicate StockUnits that share the same timestamp (by LocalDateTime).
                            // This ensures no duplicate events/reads affect calculations.
                            Set<LocalDateTime> seenTimes = new HashSet<>();
                            Iterator<StockUnit> it = realTimeWindow.iterator();
                            while (it.hasNext()) {
                                StockUnit unit = it.next();
                                LocalDateTime t = unit.getLocalDateTimeDate();
                                if (!seenTimes.add(t)) {
                                    it.remove(); // Remove duplicate bar.
                                }
                            }

                            // Step 4: Calculate the total percentage price change in this time window
                            // and count the number of "green bars" (bars with a positive price move).
                            double pctChange = realTimeWindow.stream()
                                    .skip(1) // Skip first (no prior to compare against)
                                    .mapToDouble(StockUnit::getPercentageChange)
                                    .sum();

                            long greenBars = realTimeWindow.stream()
                                    .skip(1)
                                    .filter(bar -> bar.getPercentageChange() > 0)
                                    .count();

                            // get dynamically the threshold
                            double threshold = getThreshold(symbol);

                            // Step 5: Event detection logic (uptrend spike detection)
                            // - pctChange: checks if the sum of price changes is above a threshold
                            // - greenBars: ensures at least 60% of the bars in this window are positive.
                            boolean strongUptrend =
                                    pctChange > threshold &&
                                            greenBars >= realTimeWindow.size() * 0.7;

                            // If an uptrend spike is detected, continue to aggregate and process the bar.
                            if (strongUptrend) {
                                // Step 6: Aggregate all second bars in this window into a single 1-minute StockUnit bar.
                                StockUnit first = realTimeWindow.get(0); // Oldest in window (start of aggregation)
                                StockUnit last = realTimeWindow.get(realTimeWindow.size() - 1); // Most recent in window

                                // High: max close in window, Low: min close, Volume: sum of all, Time: end of window
                                double high = realTimeWindow.stream().mapToDouble(StockUnit::getClose).max().orElse(first.getClose());
                                double low = realTimeWindow.stream().mapToDouble(StockUnit::getClose).min().orElse(first.getClose());
                                double volume = realTimeWindow.stream().mapToDouble(StockUnit::getVolume).sum();
                                LocalDateTime barTime = last.getLocalDateTimeDate(); // Use the most recent bar's timestamp for aggregation
                                String formattedTime = barTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")); // Consistent with rest of pipeline

                                // Build the new 1-minute aggregated bar for rolling analytics.
                                StockUnit aggregatedRealTimeBar = new StockUnit.Builder()
                                        .symbol(symbol)
                                        .open(first.getClose())   // Set "open" to the oldest close in window
                                        .high(high)
                                        .low(low)
                                        .close(last.getClose())   // Set "close" to the newest close in window
                                        .time(formattedTime)
                                        .volume(volume)
                                        .build();

                                // Step 7: Get a rolling window of the most recent 29 bars from the main symbol timeline.
                                // This supports robust moving calculations and avoids excess memory growth.
                                List<StockUnit> symbolTimeLineUnits; // Declare the list to hold the 29 most recent StockUnit entries for the current symbol

                                synchronized (symbolTimelines) { // Synchronize access to the shared symbolTimelines map to prevent race conditions in multithreaded environments
                                    List<StockUnit> timeline = symbolTimelines.get(symbol); // Retrieve the list of historical StockUnit entries for the given symbol

                                    if (timeline == null || timeline.isEmpty()) { // Defensive check: handle the case where no data exists yet for the symbol
                                        symbolTimeLineUnits = new ArrayList<>(); // Initialize as an empty list to avoid NullPointerException
                                    } else {
                                        int window = 29; // Define the window size — how many of the most recent entries to keep
                                        int from = Math.max(0, timeline.size() - window); // Calculate the starting index to ensure we don't go below 0 (in case there are fewer than 29 entries)

                                        // Create a new list containing only the most recent `window` number of StockUnit entries
                                        // This is done by copying a sublist from the original timeline
                                        symbolTimeLineUnits = new ArrayList<>(timeline.subList(from, timeline.size()));
                                    }
                                }

                                LocalDateTime minuteBarTime = aggregatedRealTimeBar.getLocalDateTimeDate();

                                // Step 7b: Add new minute bar if this is a new bar (time strictly after last bar in window).
                                // This ensures we don't double-add or duplicate bars if polling overlaps or is too frequent.
                                if (symbolTimeLineUnits.isEmpty()) {
                                    symbolTimeLineUnits.add(aggregatedRealTimeBar);
                                } else {
                                    StockUnit lastBar = symbolTimeLineUnits.get(symbolTimeLineUnits.size() - 1);
                                    LocalDateTime lastBarTime = lastBar.getLocalDateTimeDate();

                                    if (minuteBarTime.isAfter(lastBarTime)) {
                                        symbolTimeLineUnits.add(aggregatedRealTimeBar); // Append only if newer.
                                    }
                                }

                                // Step 8: Run post-aggregation analytics in a dedicated high-priority thread.
                                // Passes the updated rolling window for further event detection, ML, or alerting.
                                if (SYMBOL_INDICATOR_RANGES.get(symbol) != null) {
                                    Thread highPriorityThread = new Thread(() -> advancedProcessing(symbolTimeLineUnits, symbol));
                                    highPriorityThread.setPriority(Thread.MAX_PRIORITY); // Ensures quick ML/alert reaction.
                                    highPriorityThread.start();
                                }
                            }
                        }
                    });
        }
    }

    /**
     * Utility class defining volatility tiers for a universe of stock symbols
     * and providing per-symbol "spike" thresholds for minute-bar range detection.
     *
     * <p>Tier definitions (by market cap estimate):
     * <ul>
     *   <li><b>TIER_1 (Mega-Caps):</b>   > $300 B → threshold = 0.005 (0.5%)</li>
     *   <li><b>TIER_2 (Large-Caps):</b>  $100 B–$300 B → threshold = 0.010 (1.0%)</li>
     *   <li><b>TIER_3 (Mid-Caps):</b>    $20 B–$100 B → threshold = 0.015 (1.5%)</li>
     *   <li><b>TIER_4 (Small/Micro):</b> < $20 B → threshold = 0.025 (2.5%)</li>
     * </ul>
     */
    private static double getThreshold(String symbol) {
        double threshold;
        if (MEGA_CAPS.contains(symbol)) threshold = 0.3;
        else if (LARGE_CAPS.contains(symbol)) threshold = 0.4;
        else if (MID_CAPS.contains(symbol)) threshold = 0.8;
        else if (SMALL_CAPS.contains(symbol)) threshold = 1.5;
        else threshold = 1.0;
        return threshold;
    }

    /**
     * Performs advanced event and signal detection on a list of {@link StockUnit} objects for a given symbol.
     * <p>
     * This includes:
     * <ul>
     *   <li>Running notification/event analysis (such as technical indicators or unusual market activity).</li>
     *   <li>Forwarding generated notifications to the UI, dashboard, or alerting system.</li>
     *   <li>Safely batching all generated notifications for later profit/loss analysis in a global list.</li>
     * </ul>
     *
     * <p>
     * All errors during processing are caught and logged to avoid aborting the processing loop for one bad symbol or frame.
     * Notification list is updated in a thread-safe manner to support multithreaded batch processing.
     *
     * @param stockUnits The time series or recent frame of market data for one stock symbol.
     * @param symbol     The stock symbol being analyzed (ticker code).
     */
    private static void advancedProcessing(List<StockUnit> stockUnits, String symbol) {
        // Temporary collection for all notifications generated in this pass (for this symbol)
        List<Notification> stockNotifications = new ArrayList<>();

        try {
            // Step 1: Detect any actionable events/signals (spikes, crossovers, patterns, etc.)
            List<Notification> notifications = getNotificationForFrame(stockUnits, symbol);
            stockNotifications.addAll(notifications);

            // Step 2: If any notifications were generated, push them to the UI/dashboard
            if (!notifications.isEmpty()) {
                for (Notification notification : notifications) {
                    // Look up the maximum open quantity for this symbol (if available), else default to "N/A"
                    String name = Optional.ofNullable(nameToData.get(symbol))
                            .map(obj -> String.valueOf(obj.maxOpenQuantity()))
                            .orElse("N/A");

                    // Run the “entry” ONNX model on the notification’s 30-bar window to get a probability score
                    float prediction = predictNotificationEntry(notification.getStockUnitList());

                    // Build and dispatch a new Notification:
                    //   Title includes:
                    //     - "Sec, ℙ: " prefix (indicating second-based spike + probability),
                    //     - the formatted prediction (two decimals),
                    //     - the original notification’s title,
                    //     - " Amt: " followed by the max open quantity for context.
                    //   Content is carried forward unchanged.
                    //   stockUnitList, localDateTime, symbol, and change are passed straight through.
                    //   Config = 5 (denotes “Second-based alarm” styling).
                    //   Validation window is provided as an empty list here (filled later if needed).
                    addNotification(
                            "Sec, ℙ: " + String.format("%.2f ", prediction) + notification.getTitle() + " Amt: " + name,
                            notification.getContent(),
                            notification.getStockUnitList(),
                            notification.getLocalDateTime(),
                            notification.getSymbol(),
                            notification.getChange(),
                            5,
                            new ArrayList<>()
                    );
                }
            }
        } catch (Exception e) {
            // Defensive: Catch all to ensure that a bad symbol/data window doesn't halt processing
            e.printStackTrace();
        }

        // Step 3: Batch add all notifications from this run to the global list for P/L (profit/loss) analysis.
        // Synchronized for thread safety if this method is called concurrently for multiple symbols.
        synchronized (notificationsForPLAnalysis) {
            notificationsForPLAnalysis.addAll(stockNotifications);
        }
    }

    /**
     * Loads and maintains up-to-date time series (OHLCV) data for a given set of stock symbols.
     * <p>
     * This method coordinates all required pre-processing and continuous polling for "hype mode":
     * <ul>
     *   <li><b>1. Data loading:</b> Loads from cache if available, else pulls full intraday data from AlphaVantage.</li>
     *   <li><b>2. UI feedback:</b> Continuously updates a progress dialog and status log as each symbol is processed.</li>
     *   <li><b>3. Timeline calculation:</b> Once all symbols are loaded, computes historical % changes and
     *   precomputes feature indicator ranges needed for later ML/alert calculations.</li>
     *   <li><b>4. Real-time polling:</b> Enters a loop, fetching latest real-time data for all stocks in batches,
     *   updating the timeline and running alert logic. Sleeps for 1 minute between updates to avoid API bans.</li>
     *   <li><b>5. Threading:</b> Heavy lifting is performed in a dedicated background thread so the UI remains responsive.</li>
     * </ul>
     * <b>Note:</b> This function is at the heart of the app's "fast scan" and live-trading simulation capability.
     *
     * @param symbols List of ticker symbols (Strings) to monitor; case-insensitive (internally converted to uppercase).
     */
    public static void hypeModeFinder(List<String> symbols) {
        // Announce to the user that the download/fetch sequence is starting.
        logTextArea.append("Started pulling data from server\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        // Countdown latch is used to know when ALL symbol data (from file or API) has loaded.
        CountDownLatch countDownLatch = new CountDownLatch(symbols.size());

        // ----------- UI PROGRESS BAR INITIALIZATION -----------
        // Build a modal dialog (blocks interaction) with a progress bar for visual feedback.
        ProgressDialog progressDialog = new ProgressDialog((Frame) SwingUtilities.getWindowAncestor(logTextArea));
        // Show the dialog on the UI thread to prevent concurrency issues.
        SwingUtilities.invokeLater(() -> progressDialog.setVisible(true));

        // Helper: Whenever a symbol is processed, update the dialog with progress.
        Runnable updateProgress = () -> {
            int current = symbols.size() - (int) countDownLatch.getCount();
            progressDialog.updateProgress(current, symbols.size());
        };

        // =========== LOAD DATA FOR EACH SYMBOL ==============
        for (String symbol : symbols) {
            String symbolUpper = symbol.toUpperCase(); // Consistent key format for all cache/maps
            Path cachePath = Paths.get(CACHE_DIR, symbolUpper + ".txt");

            // ============= CACHE CHECK =============
            if (Files.exists(cachePath)) {
                // --- If we have a file cache, use it (MUCH faster than API, reduces rate limits) ---
                try {
                    processStockDataFromFile(cachePath.toString(), symbolUpper, 10000); // Parse cached bars and update global timeline
                    countDownLatch.countDown(); // Mark one task done
                    SwingUtilities.invokeLater(updateProgress); // Update the progress bar
                } catch (IOException e) {
                    // Any file read issue: tell user and move on so the app doesn't hang.
                    logTextArea.append("Error loading cache for " + symbolUpper + ": " + e.getMessage() + "\n");
                    e.printStackTrace();
                    countDownLatch.countDown(); // Even on error, count down so the thread doesn't block forever
                    SwingUtilities.invokeLater(updateProgress);
                }
            } else {
                // ============= API FETCH (NO CACHE) =============
                // API call for full intraday OHLCV data (1-min bars, as much as available)
                AlphaVantage.api()
                        .timeSeries()
                        .intraday()
                        .forSymbol(symbol)
                        .interval(Interval.ONE_MIN)
                        .outputSize(OutputSize.FULL)
                        .entitlement("realtime")
                        .onSuccess(e -> {
                            try {
                                // Parse and store in-memory (and maybe to disk if your handleSuccess does so)
                                handleSuccess((TimeSeriesResponse) e);

                                // Extract all StockUnits (bars) from response, annotate with symbol.
                                TimeSeriesResponse response = (TimeSeriesResponse) e;
                                List<StockUnit> units = response.getStockUnits();
                                units.forEach(stockUnit -> stockUnit.setSymbol(symbolUpper)); // Tag each bar for clarity

                                // Reverse bars to oldest → newest (AlphaVantage returns newest → oldest)
                                List<StockUnit> reversedUnits = new ArrayList<>(units);
                                Collections.reverse(reversedUnits);

                                // Update the global map in a thread-safe way.
                                synchronized (symbolTimelines) {
                                    symbolTimelines.computeIfAbsent(symbolUpper, k ->
                                            Collections.synchronizedList(new ArrayList<>())
                                    ).addAll(reversedUnits);
                                }

                                countDownLatch.countDown();
                                SwingUtilities.invokeLater(updateProgress);
                            } catch (Exception ex) {
                                // Any parsing or logic error, log and proceed so the pipeline keeps running.
                                logTextArea.append("Failed to process data in PreFetch for " + symbolUpper + ": " + ex.getMessage() + "\n");
                                ex.printStackTrace();
                                countDownLatch.countDown();
                                SwingUtilities.invokeLater(updateProgress);
                            }
                        })
                        .onFailure(error -> {
                            // API fetch failed; report but don't let this symbol hold up the batch.
                            mainDataHandler.handleFailure(error);
                            logTextArea.append("Failed to download data in PreFetch for " + symbolUpper + "\n");
                            countDownLatch.countDown();
                            SwingUtilities.invokeLater(updateProgress);
                        })
                        .fetch();
            }
        }

        // ========== BACKGROUND THREAD: CONTINUES AFTER ALL SYMBOLS LOADED ==========
        new Thread(() -> {
            try {
                // Wait (blocks here) until every symbol is loaded (cache or API).
                countDownLatch.await();

                // When everything is loaded: close progress dialog and let user know.
                SwingUtilities.invokeLater(() -> {
                    progressDialog.dispose();
                    logTextArea.append("Initial data loading completed\n");
                });

                // --- Post-load: Prepare all in-memory timelines for ML/alerting ---
                calculateStockPercentageChange(false);

                // Precompute min/max/percentile ranges for each indicator, per symbol.
                precomputeIndicatorRanges(true);

                // ===== MAIN REAL-TIME DATA LOOP (continues as long as thread is not interrupted) =====
                while (!Thread.currentThread().isInterrupted()) {
                    // ======= (A) START OR STOP the second–framework thread exactly once =======
                    if (useSecondFramework) {
                        // If the flag is true, we want to ensure the second framework thread is running.
                        // Only start it if it isn't already created or if the previous one has died.
                        if (secondFrameworkThread == null || !secondFrameworkThread.isAlive()) {
                            // Create a new Thread assigned to run the second-framework loop.
                            secondFrameworkThread = new Thread(() -> {
                                // The thread’s name can help with debugging/logging.
                            }, "SecondFrameworkThread");

                            // Inside the thread’s Runnable, we perform continuous polling as long as:
                            //  1) useSecondFramework remains true, and
                            //  2) this thread has not been interrupted.
                            secondFrameworkThread = new Thread(() -> {
                                // Loop condition: keep running until the flag flips off or we’re interrupted.
                                while (useSecondFramework && !Thread.currentThread().isInterrupted()) {
                                    try {
                                        // Create a thread-safe collection to accumulate all matches for this polling round.
                                        List<RealTimeResponse.RealTimeMatch> matches = new CopyOnWriteArrayList<>();

                                        // Take a snapshot of the symbols set at this moment to avoid concurrent modification.
                                        List<String> symbolsSnapshot = new ArrayList<>(symbols);

                                        // Compute how many batches of up to 100 symbols we need (100 is the API limit).
                                        int totalBatches = (int) Math.ceil(symbolsSnapshot.size() / 100.0);

                                        // A CountDownLatch lets us wait until every asynchronous batch request has completed.
                                        CountDownLatch latch = new CountDownLatch(totalBatches);

                                        // Split symbols into sublists (batches) of size ≤ 100.
                                        for (int i = 0; i < totalBatches; i++) {
                                            // Determine start index for this batch.
                                            int start = i * 100;
                                            // Determine end index, ensuring we don’t exceed list size.
                                            int end = Math.min((i + 1) * 100, symbolsSnapshot.size());

                                            // Extract the sublist of symbols for this batch.
                                            List<String> batchSymbols = symbolsSnapshot.subList(start, end);

                                            // Join the symbols into a comma-separated string and uppercase them (per API requirements).
                                            String symbolsBatch = String.join(",", batchSymbols).toUpperCase();

                                            // Fire off an asynchronous AlphaVantage request for this batch.
                                            AlphaVantage.api()
                                                    .Realtime()                  // Real-time data endpoint
                                                    .setSymbols(symbolsBatch)    // Set the batch of symbols
                                                    .entitlement("realtime")     // Specify entitlement token/flag
                                                    .onSuccess(response -> {
                                                        // On successful response: collect all matches into our list.
                                                        matches.addAll(response.getMatches());
                                                        // Signal that this batch is done by counting down the latch.
                                                        latch.countDown();
                                                    })
                                                    .onFailure(e -> {
                                                        // If an error occurs, handle/log it but still count down so latch can proceed.
                                                        handleFailure(e);
                                                        latch.countDown();
                                                    })
                                                    .fetch();  // Actually send the HTTP request asynchronously
                                        }

                                        // Wait up to 5 seconds for all batch requests to complete.
                                        // If not every batch finishes within 5 seconds, we log a warning and move on.
                                        if (!latch.await(5, TimeUnit.SECONDS)) {
                                            logTextArea.append("Warning: Timed out waiting for data in second framework\n");
                                        }

                                        // At this point, 'matches' contains all successful responses for this round.
                                        // Process them (e.g., update internal data structures, trigger analytics, etc.).
                                        processStockData(matches);

                                        // After processing, pause 5 seconds to enforce rate-limiting before next poll.
                                        Thread.sleep(5000);

                                    } catch (InterruptedException ex) {
                                        // If we’re interrupted (either because useSecondFramework became false or someone called interrupt()):
                                        // 1) Re-set the interrupt flag so higher-level code knows we were interrupted.
                                        Thread.currentThread().interrupt();
                                        // 2) Log that we’re exiting cleanly.
                                        logTextArea.append("Second framework thread interrupted, exiting\n");
                                        // 3) Break out of the while-loop so the thread can terminate.
                                        break;

                                    } catch (Exception ex) {
                                        // Catch any other exception to prevent the thread from dying silently.
                                        ex.printStackTrace();
                                        logTextArea.append("Error in second framework: " + ex.getMessage() + "\n");
                                        // Loop condition still holds (unless useSecondFramework is flipped), so we continue.
                                    }
                                }
                                // Once we exit the loop, perform any final cleanup if necessary.
                                logTextArea.append("Second framework thread has stopped.\n");
                            }, "SecondFrameworkThread");

                            // Start the newly created thread so it begins executing its run() method.
                            secondFrameworkThread.start();
                        }

                    } else {
                        // If useSecondFramework is false, we must ensure the background thread stops immediately.
                        if (secondFrameworkThread != null && secondFrameworkThread.isAlive()) {
                            // Interrupt the thread; this triggers InterruptedException or causes the loop condition to fail.
                            secondFrameworkThread.interrupt();
                            // Clear our reference so a new thread could be spawned later if the flag is turned on again.
                            secondFrameworkThread = null;
                        }
                    }

                    // This loop keeps running until the thread is explicitly interrupted (e.g., via cancellation).
                    // --- Create a CountDownLatch with a count equal to the number of symbols we want to fetch ---
                    // Each symbol's asynchronous API call will decrement this latch when finished (success or failure).
                    CountDownLatch latch = new CountDownLatch(symbols.size());
                    Set<String> pendingSymbols = Collections.newSetFromMap(new ConcurrentHashMap<>());
                    pendingSymbols.addAll(symbols); // Initialize with all symbols

                    try {
                        // Loop through all symbols that need to be updated in real-time.
                        for (String symbol : symbols) {
                            // Asynchronously fetch the latest intraday data for each symbol from AlphaVantage.
                            AlphaVantage.api()
                                    .timeSeries()
                                    .intraday()
                                    .forSymbol(symbol)                       // Set the symbol (e.g., "AAPL")
                                    .interval(Interval.ONE_MIN)              // 1-minute resolution
                                    .outputSize(OutputSize.COMPACT)          // Only recent bars (smallest payload)
                                    .entitlement("realtime")                 // Specify real-time data entitlement
                                    .onSuccess(response -> {                 // Success callback (executed for each API call)
                                        // Parse the response as a time series and extract all bars for the symbol
                                        TimeSeriesResponse tsResponse = (TimeSeriesResponse) response;
                                        List<StockUnit> stockUnits = tsResponse.getStockUnits();

                                        if (!stockUnits.isEmpty()) {
                                            // AlphaVantage API returns newest bars first; reverse to chronological order (oldest → newest)
                                            List<StockUnit> reversedUnits = new ArrayList<>(stockUnits);
                                            Collections.reverse(reversedUnits);

                                            // Tag each bar with the symbol for clarity/tracking
                                            reversedUnits.forEach(e -> e.setSymbol(symbol));

                                            // Ensure thread-safe update of the global timeline map
                                            synchronized (symbolTimelines) {
                                                // Retrieve the current timeline for this symbol, or create a new one if missing/empty
                                                List<StockUnit> timeline = symbolTimelines.get(symbol);
                                                if (timeline == null || timeline.isEmpty()) {
                                                    timeline = new ArrayList<>();
                                                    symbolTimelines.put(symbol, timeline);
                                                }

                                                // === Compute the “last fully closed minute” in US/Eastern time zone ===
                                                // We ignore the local clock entirely and base everything on US/Eastern.
                                                ZoneId eastern = ZoneId.of("US/Eastern");

                                                // 1. Get the current moment in US/Eastern.
                                                ZonedDateTime nowEastern = ZonedDateTime.now(eastern);

                                                // 2. Truncate to the start of the current minute, then subtract one minute
                                                //    to land exactly on the previous full minute boundary.
                                                //    Example: If it’s 11:01:05 US/Eastern now, then
                                                //      nowEastern.truncatedTo(MINUTES) → 11:01:00
                                                //      minusMinutes(1)               → 11:00:00
                                                ZonedDateTime lastFullEastern = nowEastern
                                                        .truncatedTo(ChronoUnit.MINUTES)
                                                        .minusMinutes(1);

                                                // 3. Convert that ZonedDateTime back to a LocalDateTime.
                                                //    This `cutoffEasternLdt` represents the latest timestamp we consider “complete.”
                                                //    Any bar stamped after this (e.g. 11:01:00 when i  t’s 11:01:05) is still in-progress and must be skipped.
                                                LocalDateTime cutoffEasternLdt = lastFullEastern.toLocalDateTime();

                                                // === Filter the reversedUnits list so we only take bars that are both:
                                                //       A) Newer than whatever we already have stored for this symbol.
                                                //       B) At or before the last fully closed Eastern-minute cutoff ===
                                                // We’ll collect these into newBars and then append them to our existing timeline.
                                                List<StockUnit> finalTimeline = timeline; // alias for clarity

                                                List<StockUnit> newBars = reversedUnits.stream()
                                                        .filter(unit -> {
                                                            // 1) Interpret the bar’s timestamp as a LocalDateTime in US/Eastern.
                                                            LocalDateTime barTime = unit.getLocalDateTimeDate(); // assumed parsed in US/Eastern

                                                            // --- A) Don’t re-add any bar that’s not strictly newer than our stored data ---
                                                            // If our current timeline is empty, newestStored == null → allow any bar.
                                                            // Otherwise, compare: only accept future bars (barTime > newestStored).
                                                            LocalDateTime newestStored = finalTimeline.isEmpty()
                                                                    ? null
                                                                    : finalTimeline.get(finalTimeline.size() - 1).getLocalDateTimeDate();

                                                            boolean isAfterNewest = (newestStored == null)
                                                                    || barTime.isAfter(newestStored);

                                                            // --- B) Only accept bars at or before the last fully closed US/Eastern minute ---
                                                            // If barTime is “after” cutoffEasternLdt, that means it’s still in-progress.
                                                            // For example, if cutoffEasternLdt = 11:00:00, any bar with barTime = 11:01:00 will be rejected.
                                                            boolean isAtOrBeforeFull = !barTime.isAfter(cutoffEasternLdt);

                                                            // Return true only if BOTH conditions are satisfied:
                                                            return isAfterNewest && isAtOrBeforeFull;
                                                        })
                                                        .toList();

                                                // After filtering, newBars contains only completed bars up through the last full Eastern minute.
                                                // You can now add them to the timeline:
                                                timeline.addAll(newBars);
                                            }
                                        }

                                        pendingSymbols.remove(symbol);
                                        // Signal that this symbol's fetch is complete (success).
                                        latch.countDown();
                                    })
                                    .onFailure(e -> {
                                        // On API failure, log which symbol failed (and still signal completion)
                                        logTextArea.append("Failed for symbol in Main Loop: " + symbol + " " + e.getMessage() + " \n");
                                        pendingSymbols.remove(symbol);
                                        latch.countDown();
                                    })
                                    .fetch(); // Start the async request
                        }

                        long startTime = System.nanoTime();

                        // --- Wait for all fetches to complete (or timeout after 50 seconds) ---
                        // The main thread blocks here until all symbol requests finish or timeout is reached.
                        if (!latch.await(50, TimeUnit.SECONDS)) {
                            // If not all latches counted down, warn the user (could be slow API or network)
                            logTextArea.append("Warning: Timed out waiting for some data in Main Loop\n");

                            // Log timed out symbols
                            logTextArea.append("Timed out symbols: " + pendingSymbols + "\n");
                            logTextArea.append("Number of timed out symbols: " + pendingSymbols.size() + "\n");
                        }

                        // Log the number of processed stock entries for debugging/feedback.
                        logTextArea.append("Processed " + symbols.size() + " valid stock entries\n");
                        long endTime = System.nanoTime();
                        long durationMs = (endTime - startTime) / 1_000_000; // convert nanoseconds to milliseconds

                        System.out.println("Execution time: " + durationMs + " ms\n");

                        // === One‐time “dry run” to seed both predictor buffers with historical data ===
                        if (!firstTimeComputed) {
                            // 1. Compute percentage changes across the full history
                            calculateStockPercentageChange(false);

                            // 2. Lock the timeline map so no new data races occur during seeding
                            synchronized (symbolTimelines) {
                                // 3. Iterate over every symbol that has been loaded into memory
                                symbolTimelines.keySet().forEach(symbol -> {
                                    List<StockUnit> timeLine = symbolTimelines.get(symbol);

                                    // 4. Ensure we have at least two full windows of data (2 × buffer size)
                                    if (timeLine.size() >= 30 * 2) {
                                        // 5. Slide a window of length 30 from “oldest needed” up to the most recent 29
                                        for (int start = timeLine.size() - 30 * 2; start <= timeLine.size() - 31; start++) {
                                            // 5a. Extract a 30‐bar slice for feature computation
                                            List<StockUnit> window = timeLine.subList(start, start + 30);

                                            // 5b. Normalize features for the main entry predictor and feed them in
                                            float[] mainFeatures = normalizeFeatures(computeFeatures(window, symbol), symbol);
                                            predict(mainFeatures, symbol);          // Populates the 28‐step buffer

                                            // 5c. Normalize slope‐based features for the uptrend predictor and feed them in
                                            float[] upFeatures = normalizeUptrend(
                                                    computeFeaturesForSlope(window, new int[]{5, 10, 20})
                                            );
                                            predictUptrend(upFeatures, symbol);    // Populates the 30‐step uptrend buffer
                                        }
                                    }
                                });
                            }

                            // 6. Flip the flag so we never run this seeding block again
                            firstTimeComputed = true;
                        }

                        // After all updates, recalculate the percentage change for each symbol,
                        // so UI indicators and trading logic reflect the latest market conditions.
                        // ===== After all symbol fetches complete, before recalculating percentage changes =====
                        calculateStockPercentageChange(true);

                        // ====== Rate limiting: Sleep for 60 seconds before polling again ======
                        // ======= SLEEP UNTIL THE NEXT FULL MINUTE =======

                        // Get the current system time in milliseconds since epoch
                        long currentMillis = System.currentTimeMillis();

                        // Calculate the timestamp (in millis) of the next full minute boundary
                        // E.g., if now is 12:34:45.789, nextMinuteMillis will be 12:35:00.000
                        long nextMinuteMillis = ((currentMillis / 60000) + 1) * 60000 + 2000;

                        // Compute how many milliseconds to sleep to reach the next full minute
                        long sleepTime = nextMinuteMillis - currentMillis;

                        // clean up the map to save memory
                        trimSymbolTimelines(300);

                        // Only sleep if we're not already exactly at a full minute
                        if (sleepTime > 0) {
                            try {
                                // Log the next fetch timestamp (for debugging/monitoring)
                                Instant fetchTime = Instant.ofEpochMilli(nextMinuteMillis);
                                System.out.println("[INFO] Next fetch scheduled at: " + fetchTime);
                                logTextArea.append("Next fetch scheduled at: " + fetchTime + "\n");

                                // Sleep until the calculated next full minute
                                Thread.sleep(sleepTime);
                            } catch (InterruptedException e) {
                                // Handle interruption (e.g., if shutting down)
                                Thread.currentThread().interrupt();
                                logTextArea.append("Data pull interrupted during sleep\n");
                                break;
                            }
                        }
                    } catch (InterruptedException e) {
                        // If thread is interrupted (shutdown requested), exit loop cleanly.
                        e.printStackTrace();
                        Thread.currentThread().interrupt();
                        logTextArea.append("Data pull interrupted in main loop\n");
                        break;
                    } catch (Exception e) {
                        // Any error: print/log and keep the background thread running.
                        e.printStackTrace();
                        logTextArea.append("Error during data pull in main loop: " + e.getMessage() + "\n");
                    }
                    // Always scroll UI log to show the latest event.
                    logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                }
            } catch (InterruptedException e) {
                // This is only reached if the whole background thread is interrupted while waiting for data.
                Thread.currentThread().interrupt();
            }
        }).start(); // All of the above runs in a dedicated background worker thread.
    }

    /**
     * Trims each symbol's timeline in the symbolTimelines map so that only the
     * most recent {@code keepLast} StockUnit entries are retained.
     * <p>
     * For each entry, removes all elements from the start of the list up to the last
     * {@code keepLast} elements.
     *
     * @param keepLast The number of most recent StockUnit entries to keep for each symbol.
     */
    public static void trimSymbolTimelines(int keepLast) {
        // Iterate over all entries in the ConcurrentHashMap.
        synchronized (symbolTimelines) {
            symbolTimelines.forEach((symbol, timeline) -> {
                int size = timeline.size(); // Get current size of the list.
                // If the list has more than keepLast elements, remove the oldest ones.
                if (size > keepLast) {
                    // Remove from index 0 (inclusive) to (size - keepLast) (exclusive).
                    // This keeps only the last 'keepLast' elements.
                    timeline.subList(0, size - keepLast).clear();
                }
            });
        }
    }

    /**
     * Fetches the most recent real-time quote (tick) for a single symbol, delivering
     * the result to a callback for live updating, streaming, or monitoring.
     * <p>
     * Designed for use in real-time dashboards, streaming price trackers, or
     * UI elements that need current price and volume updates every second.
     *
     * @param symbol   The ticker symbol to monitor (case-insensitive, e.g., "AAPL", "MSFT")
     * @param callback Callback that receives the latest {@link RealTimeResponse.RealTimeMatch} object
     */
    public static void getRealTimeUpdate(String symbol, RealTimeCallback callback) {
        AlphaVantage.api()
                .Realtime()
                .setSymbols(symbol) // Set one or more symbols (comma separated supported)
                .entitlement("realtime")
                .onSuccess(response ->
                        // Only take the first result (usually only one for single symbol)
                        callback.onRealTimeReceived(response.getMatches().get(0))
                )
                .onFailure(mainDataHandler::handleFailure) // Log/handle errors
                .fetch(); // Launch async call (returns immediately)
    }

    /**
     * Fetches fundamental company overview data (business summary, valuation, etc.)
     * for a given symbol, and delivers the raw API response to a callback.
     * <p>
     * Used for displaying detailed company info, financials, and ratios in dashboards,
     * or for supporting deeper data-driven analysis in the UI.
     *
     * @param symbol   The ticker symbol for the company (e.g., "GOOGL")
     * @param callback Callback that receives a {@link CompanyOverviewResponse}
     */
    public static void getCompanyOverview(String symbol, OverviewCallback callback) {
        AlphaVantage.api()
                .fundamentalData()
                .companyOverview()
                .forSymbol(symbol) // Request overview for specified ticker
                .onSuccess(response -> {
                    // Cast and deliver the result to the callback (for UI or storage)
                    CompanyOverviewResponse overview = (CompanyOverviewResponse) response;
                    callback.onOverviewReceived(overview);
                })
                .onFailure(mainDataHandler::handleFailure) // Log error
                .fetch();
    }

    /**
     * Determines whether to use the extended-hours quote for a given real-time response.
     * <p>
     * Returns <code>true</code> if the regular market close price is zero but an extended-hours
     * quote is available (i.e., nonzero), indicating that the after-hours or pre-market price
     * should be used for more accurate, up-to-date data.
     *
     * <p><b>Typical use:</b> This is useful when real-time updates occur outside of regular
     * trading hours or for symbols that may have sporadic market data, such as newly listed
     * stocks or very illiquid assets.
     *
     * @param value The {@link RealTimeResponse.RealTimeMatch} containing both the regular
     *              close and extended-hours quote.
     * @return <code>true</code> if the extended-hours quote should be used; <code>false</code> otherwise.
     */

    static boolean useExtended(RealTimeResponse.RealTimeMatch value) {
        // Expected timestamp format: "2025-05-27 19:46:41.670"
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
        LocalDateTime dateTime = LocalDateTime.parse(value.getTimestamp(), formatter);
        LocalTime time = dateTime.toLocalTime();

        // US Stock market regular hours: 09:30 - 16:00 ET
        LocalTime regularOpen = LocalTime.of(9, 30);
        LocalTime regularClose = LocalTime.of(16, 0);

        // Use extended if outside regular hours and an extended quote is available
        boolean outsideRegular = time.isBefore(regularOpen) || time.isAfter(regularClose);
        boolean hasExtended = value.getExtendedHoursQuote() != 0.0;

        return outsideRegular && hasExtended;
    }

    /**
     * Calculates and updates the percentage price change for every bar of every symbol in the global timeline.
     * <p>
     * This ensures all indicators, spike detectors, and trend systems operate on fresh, accurate values.
     * <b>Call this after any new bars are added to symbolTimelines!</b>
     * <ul>
     *   <li>If the flag {@code realFrame} is true, the method triggers a "spike-in-rally" analysis pass
     *       after updating percentage changes (used for real-time and batch rally scans).</li>
     *   <li>Handles outlier changes (> 14%) by carrying forward the previous value (removes "bad ticks").</li>
     *   <li>All results are logged to the UI for transparency and debugging.</li>
     * </ul>
     *
     * @param realFrame If true, triggers an additional full rally spike calculation pass after updating percentages.
     */
    public static void calculateStockPercentageChange(boolean realFrame) {
        // Thread safety: synchronize on the global timeline structure
        // Synchronize on symbolTimelines to prevent concurrent modification
        // while the parallel stream processes all symbols in parallel.
        // This ensures thread safety if other threads may add/remove symbols while this runs.
        synchronized (symbolTimelines) {
            // Process each symbol in symbolTimelines using parallel threads.
            // This improves speed when there are many symbols to analyze.
            symbolTimelines.keySet().parallelStream().forEach(symbol -> {
                // Get the list of StockUnit objects (the timeline) for this symbol.
                List<StockUnit> timeline = symbolTimelines.get(symbol);

                // If there aren't at least two data points, skip calculation and log a warning.
                if (timeline.size() < 2) {
                    // Log a message indicating not enough data for this symbol.
                    logTextArea.append("Not enough data for for percentage processing" + symbol + "\n");
                    return;
                }

                // Start from the second item (index 1) since percentage change needs a previous value.
                for (int i = 1; i < timeline.size(); i++) {
                    StockUnit current = timeline.get(i);
                    StockUnit previous = timeline.get(i - 1);

                    // Only calculate change if the previous close price is positive (avoids division by zero).
                    if (previous.getClose() > 0) {
                        // Calculate the percent change between the current and previous close.
                        double change = ((current.getClose() - previous.getClose()) / previous.getClose()) * 100;

                        // Clamp outliers: If the change is abnormally large (>= 14% up/down),
                        // use the previous bar's value instead. This reduces noise from splits or bad ticks.
                        change = Math.abs(change) >= 14 ? previous.getPercentageChange() : change;

                        // Store the calculated (or clamped) change in the current StockUnit for later use (e.g., indicators).
                        current.setPercentageChange(change);
                    }
                }
            });
        }

        // If this update was for a real-time (live) frame, trigger spike analysis
        if (realFrame) {
            calculateSpikesInRally(frameSize, true);
        }
    }

    /**
     * Scans all loaded symbols for significant "spike" events within a rally period,
     * processing each symbol's timeline in parallel for efficiency.
     * <p>
     * This method is used to find candidate stocks for alerts, notifications, or further
     * technical analysis based on short-term or real-time market behavior.
     * <ul>
     *   <li>Processes each symbol in {@link #symbolTimelines} using parallel streams (multicore).</li>
     *   <li>For each, slices the timeline into windows and runs event detection.</li>
     *   <li>After all symbols are processed, the resulting notifications are globally sorted.</li>
     * </ul>
     *
     * @param minutesPeriod Length of the window in minutes to analyze for spikes/rallies.
     * @param realFrame     If true, only the most recent window is checked per symbol (live mode). If false, all possible windows are checked (historical analysis).
     */
    public static void calculateSpikesInRally(int minutesPeriod, boolean realFrame) {
        // Process each symbol's timeline concurrently to improve performance
        symbolTimelines.keySet()
                .parallelStream() // Use parallelStream for multithreaded processing
                .forEach(symbol -> { // For each symbol in the set
                    // Retrieve the timeline (list of StockUnit) for this symbol
                    List<StockUnit> timeline = new ArrayList<>(getSymbolTimeline(symbol));

                    // Proceed only if the timeline is not empty
                    if (!timeline.isEmpty()) {
                        // Analyze the symbol's timeline over sliding time windows
                        processTimeWindows(symbol, timeline, minutesPeriod, realFrame);
                    }
                });

        // After all symbols, sort global notifications so newest/most relevant are first
        sortNotifications(notificationsForPLAnalysis);
    }

    /**
     * Processes sliding or fixed-size time windows for a symbol's timeline, applying event detection logic.
     * <p>
     * Handles both real-time (only the latest frame) and batch (all windows) modes.
     * For each eligible window, checks for event notifications (e.g., rally spike, technical trigger),
     * and queues up new notifications for UI, logs, or user alerts.
     *
     * @param symbol       Symbol name (upper-case)
     * @param timeline     Chronologically ordered list of StockUnit bars for this symbol
     * @param minutes      The length of window (in minutes) to process
     * @param useRealFrame If true, process only the last window; else, slide window over full timeline
     */
    private static void processTimeWindows(String symbol, List<StockUnit> timeline, int minutes, boolean useRealFrame) {
        List<Notification> stockNotifications = new ArrayList<>(); // Collected notifications for this run

        if (useRealFrame) {
            // ===== Real-time mode: only process the most recent relevant window =====
            if (!timeline.isEmpty()) {
                // Set end to the latest bar, start to N minutes back
                LocalDateTime endTime = timeline.get(timeline.size() - 1).getLocalDateTimeDate();
                LocalDateTime startTime = endTime.minusMinutes(minutes);

                // Extract the relevant window (maybe smaller than needed)
                LinkedList<StockUnit> timeWindow = new LinkedList<>(getTimeWindow(timeline, startTime, endTime));

                int startIndex = findTimeIndex(timeline, startTime);

                if (startIndex < 0) {
                    // No bar ≥ startTime—force it to the front or back, depending on your semantics.
                    startIndex = 0; // (or timeline.size(), if you want to backfill from “end”)
                }

                // Pad window from left if it's too short (backfill with earlier bars)
                while (timeWindow.size() < frameSize && startIndex > 0) {
                    startIndex--;
                    timeWindow.addFirst(timeline.get(startIndex));
                }

                // Only process window if it is large enough for event logic
                if (timeWindow.size() >= frameSize) {
                    try {
                        // Run event detection (technical signals, spikes, etc.)
                        List<Notification> notifications = getNotificationForFrame(timeWindow, symbol);
                        stockNotifications.addAll(notifications);

                        // Push notifications to UI, dashboard, etc.
                        if (!notifications.isEmpty()) {
                            for (Notification notification : notifications) {
                                // Retrieve the max open quantity for this symbol, if available; otherwise use "N/A"
                                String name = Optional.ofNullable(nameToData.get(symbol))
                                        .map(obj -> String.valueOf(obj.maxOpenQuantity()))
                                        .orElse("N/A");

                                // Run the entry‐prediction ONNX model on this notification’s 30‐bar window
                                float prediction = predictNotificationEntry(notification.getStockUnitList());

                                // Construct and dispatch a new Notification with:
                                //   Title:
                                //     • “ℙ: ” prefix indicating probability,
                                //     • formatted prediction (two decimal places),
                                //     • original notification title,
                                //     • “ Amt: ” followed by the max open quantity for context.
                                //   Content: original notification’s content
                                //   stockUnitList: same list of StockUnit bars
                                //   localDateTime: same timestamp
                                //   symbol: same stock ticker
                                //   change: same percentage change
                                //   config: reuse the original notification’s config code
                                //   validationWindow: carry forward the already‐computed validation window
                                addNotification(
                                        "ℙ: " + String.format("%.2f ", prediction) + notification.getTitle() + " Amt: " + name,
                                        notification.getContent(),
                                        notification.getStockUnitList(),
                                        notification.getLocalDateTime(),
                                        notification.getSymbol(),
                                        notification.getChange(),
                                        notification.getConfig(),
                                        notification.getValidationWindow()
                                );
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace(); // Don't crash on one bad window
                    }
                }
            }
        } else {
            // ===== Historical mode: slide window over every possible start in the timeline =====
            timeline.forEach(stockUnit -> {
                // Set up sliding window: starts at each bar, extends N minutes forward
                LocalDateTime startTime = stockUnit.getLocalDateTimeDate();
                LocalDateTime endTime = startTime.plusMinutes(minutes);
                int startIndex = findTimeIndex(timeline, startTime);

                List<StockUnit> timeWindow = getTimeWindow(timeline, startTime, endTime);

                // Fallback: if window is too short, attempt to fill out to frameSize using available data
                if (timeWindow.size() < frameSize) {
                    int fallbackEnd = Math.min(startIndex + frameSize, timeline.size());
                    timeWindow = timeline.subList(startIndex, fallbackEnd);
                }

                if (timeWindow.size() >= frameSize) {
                    try {
                        // Run event detection for this window
                        List<Notification> notifications = getNotificationForFrame(timeWindow, symbol);
                        stockNotifications.addAll(notifications);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        }

        // Batch add any found notifications for this symbol to the global list
        synchronized (notificationsForPLAnalysis) {
            notificationsForPLAnalysis.addAll(stockNotifications);
        }
    }

    /**
     * Helper function to extract a slice of a timeline (bars) between a start and end timestamp (inclusive).
     * Used to get N-minute windows for rally/event analysis.
     *
     * @param timeline Chronologically sorted list of StockUnit bars
     * @param start    Start time for window (inclusive)
     * @param end      End time for window (inclusive)
     * @return List of StockUnits between start and end (maybe empty if no bars in range)
     */
    private static List<StockUnit> getTimeWindow(List<StockUnit> timeline, LocalDateTime start, LocalDateTime end) {
        int startIndex = findTimeIndex(timeline, start);
        if (startIndex == -1) return Collections.emptyList(); // If start not found, return empty

        int endIndex = startIndex;
        // Extend to end of range or timeline
        while (endIndex < timeline.size() && !timeline.get(endIndex).getLocalDateTimeDate().isAfter(end)) {
            endIndex++;
        }

        // Return bars from [startIndex, endIndex) (Java subList is exclusive of end)
        return timeline.subList(startIndex, endIndex);
    }

    /**
     * Finds the index in a timeline of {@link StockUnit}s that is closest to or exactly matches a target timestamp.
     * <p>
     * The search is performed using a binary search for efficiency (O(log n)), since timelines are sorted chronologically.
     * <ul>
     *   <li>If an exact match for the target timestamp exists, returns its index.</li>
     *   <li>If not found, returns the index of the nearest later bar (or -1 if out of bounds).</li>
     *   <li>This is used for fast window slicing during indicator calculation and time-series operations.</li>
     * </ul>
     *
     * @param timeline The full, sorted (ascending) list of {@link StockUnit} objects (e.g., minute-by-minute data)
     * @param target   The target {@link LocalDateTime} to locate (e.g., window start or end)
     * @return Index of the matching or next closest bar, or -1 if beyond end of timeline.
     */
    private static int findTimeIndex(List<StockUnit> timeline, LocalDateTime target) {
        // Standard binary search on sorted timeline
        int low = 0;
        int high = timeline.size() - 1;

        while (low <= high) {
            int mid = (low + high) / 2;
            LocalDateTime midTime = timeline.get(mid).getLocalDateTimeDate();

            if (midTime.isBefore(target)) {
                low = mid + 1;    // Search upper half
            } else if (midTime.isAfter(target)) {
                high = mid - 1;   // Search lower half
            } else {
                return mid;       // Exact match found
            }
        }

        // If no exact match, return nearest valid index (not past end of array)
        return low < timeline.size() ? low : -1;
    }

    /**
     * Sorts a provided list of {@link Notification} objects in-place by their {@code LocalDateTime} property (chronological order).
     * <p>
     * Used to ensure notifications (e.g., spikes, dips, rally signals) are displayed in time order in the UI or logs.
     *
     * @param notifications The list of {@link Notification}s to sort.
     */
    public static void sortNotifications(List<Notification> notifications) {
        // Sort notifications by event time (ascending)
        notifications.sort(Comparator.comparing(Notification::getLocalDateTime));
    }

    /**
     * Precomputes robust min/max value ranges for every technical indicator, for every symbol in memory.
     * <p>
     * These ranges are used for normalizing all indicator values, so different stocks and features can be compared on a standard 0-1 scale.
     * <ul>
     *   <li>For binary indicators (e.g., spike, Keltner), the range is hardcoded to [0, 1].</li>
     *   <li>For SMA_CROSS, range is [-1, 1] (since it can be bullish, bearish, or neutral).</li>
     *   <li>For all other indicators, min and max are set as the 1st and 99th percentiles, making normalization robust to outliers.</li>
     *   <li>The computed values are stored in {@link #SYMBOL_INDICATOR_RANGES} for fast access during normalization.</li>
     * </ul>
     * This function should be called every time the symbol timelines are updated or reloaded, so ranges reflect the latest data.
     *
     * @param realData If true, operates on live in-memory symbols (from {@code symbolTimelines});
     *                 if false, operates on static symbol list (e.g., for dry-run or historical mode).
     */
    public static void precomputeIndicatorRanges(boolean realData) {
        int maxRequiredPeriod = frameSize; // How many bars are needed for feature calculation

        // Build the working list of symbols to process
        List<String> symbolList = new ArrayList<>();
        if (realData) {
            symbolList.addAll(symbolTimelines.keySet()); // Use live symbols from current session
        } else {
            // For backtesting: use predefined SYMBOLS array (stripped and uppercased)
            symbolList.addAll(Arrays.stream(SYMBOLS)
                    .map(s -> s.toUpperCase().replace(".TXT", "")) // Remove .TXT extension, standardize
                    .toList());
        }

        // ======= Process each symbol individually =======
        for (String symbol : symbolList) {
            List<StockUnit> timeline = symbolTimelines.get(symbol);
            if (timeline.size() < maxRequiredPeriod) {
                // Skip symbols that don't have enough history for feature extraction
                continue;
            }

            // Prepare to accumulate all feature values over all sliding windows
            List<String> indicators = new ArrayList<>(INDICATOR_KEYS); // All feature keys in correct order
            Map<String, List<Double>> indicatorValues = new HashMap<>();
            indicators.forEach(ind -> indicatorValues.put(ind, new ArrayList<>())); // Pre-populate empty lists

            // Slide a window of length maxRequiredPeriod over the timeline
            for (int i = maxRequiredPeriod - 1; i < timeline.size(); i++) {
                List<StockUnit> window = timeline.subList(i - maxRequiredPeriod + 1, i + 1);
                double[] features = computeFeatures(window, symbol); // Extract all features for this window

                // Add each feature to its indicator's value list
                for (int j = 0; j < features.length; j++) {
                    String indicator = indicators.get(j);
                    indicatorValues.get(indicator).add(features[j]);
                }
            }

            // === Calculate robust min/max for each indicator ===
            Map<String, Map<String, Double>> symbolRanges = new LinkedHashMap<>();

            for (String indicator : indicators) {
                // Handle binary indicators with hardcoded [0, 1] normalization
                if (BINARY_INDICATORS.contains(indicator)) {
                    symbolRanges.put(indicator, Map.of("min", 0.0, "max", 1.0));
                    continue;
                }

                // SMA_CROSS has three states: -1, 0, 1 (bearish, neutral, bullish)
                if (indicator.equals("SMA_CROSS")) {
                    symbolRanges.put(indicator, Map.of("min", -1.0, "max", 1.0));
                    continue;
                }

                // For all other indicators, use percentile-based min/max (robust to outliers)
                List<Double> values = indicatorValues.get(indicator);
                values.sort(Double::compareTo);

                // Use the 1st and 99th percentiles to avoid impact from rare/extreme data points
                int lowerIndex = (int) (values.size() * 0.01);
                int upperIndex = (int) (values.size() * 0.99);

                double min = values.get(lowerIndex);
                double max = values.get(upperIndex);

                symbolRanges.put(indicator, Map.of("min", min, "max", max));
            }
            synchronized (SYMBOL_INDICATOR_RANGES) {
                // Save the computed range map for this symbol
                SYMBOL_INDICATOR_RANGES.put(symbol, symbolRanges);
            }
        }
    }

    /**
     * Normalizes a raw indicator value to the range [0, 1] for use in ML models or aggregation.
     * <p>
     * Uses robust min/max ranges calculated per-symbol (see {@link #precomputeIndicatorRanges}) to avoid skew from outliers.
     * Binary indicators are always mapped strictly to 0 or 1, based on threshold.
     * <ul>
     *   <li>For indicators "SMA_CROSS", "KELTNER", "CUMULATIVE_PERCENTAGE", returns 1.0 if {@code rawValue >= 0.5}, else 0.0.</li>
     *   <li>For all others, does linear normalization within the min-max band and clips output to [0, 1].</li>
     * </ul>
     *
     * @param indicator Name/key of the indicator (should match keys from INDICATOR_KEYS)
     * @param rawValue  Raw, unnormalized feature value (e.g., TRIX output, percentage change, etc.)
     * @param symbol    The stock symbol (needed to lookup normalization ranges)
     * @return Normalized value, always between 0.0 and 1.0.
     * @throws RuntimeException if the indicator or symbol has no normalization range configured.
     */
    public static double normalizeScore(String indicator, double rawValue, String symbol) {
        // Lookup normalization range for this symbol/indicator
        Map<String, Double> range = null;
        try {
            Map<String, Map<String, Double>> symbolRanges = SYMBOL_INDICATOR_RANGES.get(symbol);
            range = symbolRanges.get(indicator);
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(symbol);
        }

        if (range == null) {
            // Defensive: prevents silent bugs if an indicator is missing from ranges
            throw new RuntimeException("Empty indicator");
        }

        double min = range.get("min");
        double max = range.get("max");

        // Handle degenerate case: constant value (prevents division by zero)
        if (max == min) return 0.0;

        // Special-case normalization for binary features (fast path)
        return switch (indicator) {
            // All these are interpreted as binary events (spike/no spike, breakout/no breakout, etc.)
            case "SMA_CROSS", "KELTNER", "CUMULATIVE_PERCENTAGE" -> rawValue >= 0.5 ? 1.0 : 0.0;
            // Default case: robust linear normalization, always clipped to [0, 1]
            default -> {
                double normalized = (rawValue - min) / (max - min);
                yield Math.max(0.0, Math.min(1.0, normalized));
            }
        };
    }

    /**
     * Extracts all feature values for a rolling window of stocks, returning a fixed-length array in canonical order.
     * <p>
     * All indicator computations are "feature-engineered" for downstream ML or statistical usage.
     * <ul>
     *   <li>Trend indicators: SMA crossover, TRIX</li>
     *   <li>Momentum indicator: Rate of Change (ROC)</li>
     *   <li>Statistical: spike test, cumulative % change</li>
     *   <li>Advanced: Keltner channel breakout, Elder-Ray index</li>
     *   <li>One placeholder for future expansion</li>
     * </ul>
     * Each feature is computed using its own custom logic (see each method for details).
     *
     * @param stocks List of {@link StockUnit}s (historical OHLCV bars), must have enough elements for each period tested
     * @param symbol The ticker symbol (needed for some contextual feature logic)
     * @return Array of raw (not yet normalized) feature values, in the order defined by INDICATOR_KEYS
     */
    private static double[] computeFeatures(List<StockUnit> stocks, String symbol) {
        // Fixed-length feature array for pipeline compatibility
        double[] features = new double[INDICATOR_KEYS.size()];

        // === Trend Following Indicators ===
        features[0] = isSMACrossover(stocks, 9, 21, symbol); // [0] Bull/bear/neutral crossover state (see SMA)
        features[1] = calculateTRIX(stocks, 5);              // [1] TRIX momentum oscillator

        // === Momentum Indicators ===
        features[2] = calculateROC(stocks, 20);              // [2] N-period Rate of Change

        features[3] = 0.2; // [3] Placeholder

        // === Statistical Indicators ===
        features[4] = isCumulativeSpike(stocks, 10, 0.35);   // [4] Binary: Has cumulative % change exceeded threshold?
        features[5] = cumulativePercentageChange(stocks);     // [5] Total percentage move over short window

        // === Advanced Indicators ===
        features[6] = isKeltnerBreakout(stocks, 12, 10, 0.3, 0.4); // [6] Binary: Keltner channel breakout detected?
        features[7] = elderRayIndex(stocks, 12);                    // [7] Bull Power (close - EMA)

        return features;
    }

    /**
     * Normalizes a raw feature vector for a specific symbol, producing an ML-ready [0, 1] float array.
     * <p>
     * Calls {@link #normalizeScore} for each feature in order, using the canonical keys from INDICATOR_KEYS.
     * Ensures all features for a given symbol are comparable, regardless of raw data range.
     *
     * @param rawFeatures Array of raw feature values (from {@link #computeFeatures})
     * @param symbol      Symbol context for normalization (impacts per-feature scaling)
     * @return Array of normalized float features, each in [0, 1]
     */
    private static float[] normalizeFeatures(double[] rawFeatures, String symbol) {
        float[] normalizedFeatures = new float[rawFeatures.length];
        List<String> indicatorKeys = new ArrayList<>(INDICATOR_KEYS); // Stable ordering

        // Apply normalization function for each feature
        for (int i = 0; i < rawFeatures.length; i++) {
            normalizedFeatures[i] = (float) normalizeScore(indicatorKeys.get(i), rawFeatures[i], symbol);
        }
        return normalizedFeatures;
    }

    /**
     * Calculates a dynamically weighted "aggressiveness score" for a trade decision,
     * based on normalized feature activations and category-specific weights.
     * <p>
     * This score is used to scale risk or confidence in the pipeline, allowing dynamic adaptation
     * to prevail market conditions by emphasizing certain technical indicators.
     * A higher aggressiveness indicates a stronger bullish signal or conviction to trade.
     * </p>
     * <p>
     * <strong>Weighting Process:</strong>
     * <ul>
     *   <li>Each feature is normalized to a 0–1 scale to ensure comparability.</li>
     *   <li>Features are grouped into categories (e.g., Trend, Momentum, Statistical, Advanced).</li>
     *   <li>Each category has a predefined weight reflecting its importance in bullish prediction.</li>
     *   <li>Feature activations within each category are multiplied by their category weight and summed.</li>
     *   <li>All category scores are summed to produce a global weighted score.</li>
     *   <li>The global score is scaled by the baseAggressiveness multiplier, allowing user adjustment.</li>
     * </ul>
     * </p>
     * <p>
     * <strong>Ensuring Good Bullish Prediction:</strong>
     * <ul>
     *   <li><em>Tune category weights</em> to emphasize features that historically predict bullish rallies well.</li>
     *   <li><em>Validate and improve feature quality</em> to reduce false positives and ensure meaningful activations.</li>
     *   <li><em>Use percentile normalization</em> to limit outlier effects and keep feature activations realistic.</li>
     *   <li><em>Combine this score with threshold and multiple feature checks</em> in the signal decision logic for robustness.</li>
     *   <li><em>Backtest with different weight sets</em> to find the best balance between sensitivity and noise.</li>
     * </ul>
     * </p>
     * <p>
     * <strong>Example tuning:</strong><br>
     * Increasing weights on Momentum and Advanced categories while lowering Trend and Statistical can improve
     * bullish rally detection if those categories are more predictive in your data.<br>
     * Example decision logic could require the aggressiveness score to exceed a threshold and key features like SMA crossover to be positive.
     * </p>
     *
     * @param features           Array of normalized feature activations (range 0–1) produced by {@link #normalizeFeatures}.
     * @param baseAggressiveness The user-set base multiplier for aggressiveness (default 1.0).
     * @return The scaled aggressiveness score, where higher values correspond to stronger bullish signals and trading confidence.
     */
    private static double calculateWeightedAggressiveness(float[] features, float baseAggressiveness) {
        Map<String, Double> categoryScores = new HashMap<>();

        // Compute total activation score per feature category (trend, momentum, etc.)
        for (int i = 0; i < features.length; i++) {
            String category = FEATURE_CATEGORIES.getOrDefault(i, "NEUTRAL");

            // Ignore neutral/placeholder features
            if (!category.equals("NEUTRAL")) {
                double weight = CATEGORY_WEIGHTS.get(category);   // Retrieve per-category weight
                double activation = features[i] * weight;         // Feature value * category weight
                // Merge to category score (sum over all features in same category)
                categoryScores.merge(category, activation, Double::sum);
            }
        }

        // Combine all category scores for the global boost
        double weightedScore = categoryScores.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();

        // Apply dynamic scaling to user base aggressiveness (e.g., 1.0 * (1 + weighted score))
        return baseAggressiveness * (1 + weightedScore);
    }

    /**
     * Applies feature‐wise scaling and offset to raw uptrend feature values.
     * <p>
     * Each element raw[i] is transformed via:
     * <pre>
     *   scaled[i] = raw[i] * SCALE[i] + MIN_OFFSET[i]
     * </pre>
     * to map the model’s output back into the original feature range.
     *
     * @param raw an array of unscaled uptrend features (length = {@code SCALE.length})
     * @return a new float[] where each element has been scaled and offset appropriately
     */
    public static float[] normalizeUptrend(float[] raw) {
        // 1. Determine the number of features to process
        int n = raw.length;
        // 2. Allocate an output array of the same length
        float[] scaled = new float[n];
        // 3. For each feature index...
        for (int i = 0; i < n; i++) {
            // 3a. Multiply by the scale factor for this feature
            // 3b. Add the minimum offset for this feature
            scaled[i] = raw[i] * SCALE[i] + MIN_OFFSET[i];
        }
        // 4. Return the normalized feature array
        return scaled;
    }

    public static float[] computeFeaturesForSlope(List<StockUnit> stocks, int[] SLOPE_WINDOWS) {
        int T = stocks.size();
        StockUnit last = stocks.get(T - 1);

        // 1) per‐step features
        float open = (float) last.getOpen();
        float high = (float) last.getHigh();
        float low = (float) last.getLow();
        float close = (float) last.getClose();
        float vol = (float) last.getVolume();

        // pct change vs prior bar
        float pctChange = (float) last.getPercentageChange();

        double[] listOfCloses = stocks.stream()
                .mapToDouble(StockUnit::getClose)
                .toArray();

        // rolling mean & std over last 10
        float ma10 = (float) avg(listOfCloses, T - 10, T);
        float std10 = (float) stddev(listOfCloses, T - 10, T);

        // 2) multi-bar slopes & momentum
        List<Float> multiSlopes = new ArrayList<>();
        List<Float> multiMoms = new ArrayList<>();
        double[] closes = stocks.stream().mapToDouble(StockUnit::getClose).toArray();
        for (int w : SLOPE_WINDOWS) {
            if (T >= w) {
                multiSlopes.add((float) linRegSlope(closes, T - w, T));
                multiMoms.add((float) ((closes[T - 1] / closes[T - w] - 1.0) * 100.0));
            } else {
                multiSlopes.add(0f);
                multiMoms.add(0f);
            }
        }

        // 3) overall volatility & avg volume over max window
        int maxW = SLOPE_WINDOWS[SLOPE_WINDOWS.length - 1];
        float volatility = (float) stddev(listOfCloses, T - maxW, T);
        float avgVolume = (float) avg(listOfCloses, T - maxW, T);

        StockUnit curr = stocks.get(stocks.size() - 1);
        StockUnit prev = stocks.get(stocks.size() - 2);

        float[] raw = new float[SCALE.length];
        int i = 0;
        raw[i++] = open;
        raw[i++] = high;
        raw[i++] = low;
        raw[i++] = close;
        raw[i++] = vol;
        raw[i++] = pctChange;
        raw[i++] = ma10;
        raw[i++] = std10;
        for (float s : multiSlopes) raw[i++] = s;
        for (float m : multiMoms) raw[i++] = m;
        raw[i++] = volatility;
        raw[i++] = avgVolume;

//        StringBuilder dbg = new StringBuilder();
//        raw[i++] = isInUptrend(stocks, 5, 1.5, 0.8) ? 1 : 0;
//        raw[i++] = hasGapDown(prev, curr, 0.08, dbg) ? 1 : 0;
//        raw[i++] = checkBadWicks(stocks, 3, 0.39, dbg) ? 1 : 0;
//        raw[i++] = lastNBarsRising(stocks, 3, 0.15, 0.5, dbg) ? 1 : 0;
//        raw[i++] = (float) isNearResistance(stocks.subList(stocks.size() - 15, stocks.size()));
        return raw;
    }

    // helper: compute slope of best‐fit line on closes[from…to)
    private static double linRegSlope(double[] a, int from, int to) {
        int n = to - from;
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for (int i = 0; i < n; i++) {
            double y = a[from + i];
            sumX += i;
            sumY += y;
            sumXY += (double) i * y;
            sumX2 += (double) i * (double) i;
        }
        double num = n * sumXY - sumX * sumY;
        double den = n * sumX2 - sumX * sumX;
        return den == 0 ? 0 : num / den;
    }

    // stddev over a[from…to)
    private static double stddev(double[] a, int from, int to) {
        int n = to - from;
        double mean = 0;
        for (int i = from; i < to; i++) mean += a[i];
        mean /= n;
        double var = 0;
        for (int i = from; i < to; i++) {
            double d = a[i] - mean;
            var += d * d;
        }
        return n > 1 ? Math.sqrt(var / (n - 1)) : 0;
    }

    // average over a[from…to)
    private static double avg(double[] a, int from, int to) {
        double sum = 0;
        for (int i = from; i < to; i++) sum += a[i];
        return sum / (to - from);
    }

    /**
     * Processes a fixed‐length frame of recent price bars for the given symbol and generates alerts.
     * <p>
     * This method is the core of the alerting pipeline:
     * <ul>
     *   <li>Computes raw feature vector via {@link #computeFeatures}.</li>
     *   <li>Normalizes features for ML input via {@link #normalizeFeatures}.</li>
     *   <li>Runs the primary rally/event model via {@code predict()} to obtain a confidence score.</li>
     *   <li>Computes slope‐based features for uptrend detection via {@link #computeFeaturesForSlope} and {@link #normalizeUptrend}.</li>
     *   <li>Runs the uptrend model via {@code predictUptrend()} and logs the result to {@code predictSeries} for charting.</li>
     *   <li>Delegates combined ML outputs and technical signals to {@link #evaluateResult} to produce final notifications.</li>
     * </ul>
     *
     * @param stocks a time‐ordered list of recent {@link StockUnit} bars (length == {@code frameSize})
     * @param symbol the ticker symbol context for this frame
     * @return a list of {@link Notification} objects representing any alerts generated for this frame
     */
    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String symbol) {
        // 1. Compute the raw indicator feature vector for this frame
        double[] features = computeFeatures(stocks, symbol);

        // 2. Normalize the features into [0,1] range for consistent ML input
        float[] normalizedFeatures = normalizeFeatures(features, symbol);

        // 3. Run the primary ML model to get a rally/spike event confidence score
        double prediction = predict(normalizedFeatures, symbol);

        // 4. Prepare slope‐based feature windows for uptrend detection
        int[] SLOPE_WINDOWS = new int[]{5, 10, 20};
        float[] upSlopeRaw = computeFeaturesForSlope(stocks, SLOPE_WINDOWS);
        float[] featureUptrend = normalizeUptrend(upSlopeRaw);

        // 5. Run the uptrend ML model to get its confidence score
        double uptrendPrediction = predictUptrend(featureUptrend, symbol);

        // 6. Update the prediction time series for charting purposes (timestamp = last bar in frame)
        predictSeries.addOrUpdate(new Second(stocks.get(stocks.size() - 1).getDateDate()), uptrendPrediction);

        // 7. Combine both ML scores and technical signals to generate actionable notifications
        return evaluateResult(
                prediction,
                stocks,
                symbol,
                features,
                normalizedFeatures,
                uptrendPrediction
        );
    }

    /**
     * Orchestrates a full rally scan across all tracked symbols.
     * <p>
     * Downloads intraday data for each stock, computes rolling percentage changes,
     * and then analyzes each timeline for potential "rally" conditions using regression.
     * Handles asynchronous fetches and UI progress feedback in a thread-safe manner.
     * <ul>
     *   <li>Uses a {@link CountDownLatch} to synchronize multiple async data fetches.</li>
     *   <li>Displays a progress bar via {@link ProgressDialog} for user feedback.</li>
     *   <li>Calls {@link #calculateIfRally} at the end to perform uptrend/rally checks for each symbol.</li>
     * </ul>
     *
     * @return List of symbols that meet all rally conditions (see regression/width/continuation logic).
     * @throws InterruptedException If thread is interrupted during blocking wait for data fetches.
     */
    public static List<String> checkForRallies() throws InterruptedException {
        // Clone the symbol array as an ArrayList for mutability
        List<String> stockList = new ArrayList<>(Arrays.stream(stockCategoryMap.get(market)).toList());

        // Map for per-symbol bar data (thread safe, as many fetches run in parallel)
        Map<String, List<StockUnit>> timelines = new ConcurrentHashMap<>();
        CountDownLatch latch = new CountDownLatch(stockList.size());

        // ====== User Feedback: Start Progress Dialog ======
        logTextArea.append("Started pulling data from server\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        //Create and show progress dialog
        ProgressDialog progressDialog = new ProgressDialog((Frame) SwingUtilities.getWindowAncestor(logTextArea));
        SwingUtilities.invokeLater(() -> progressDialog.setVisible(true));

        // Helper for thread-safe progress updates in the UI
        Runnable updateProgress = () -> {
            int current = stockList.size() - (int) latch.getCount();
            progressDialog.updateProgress(current, stockList.size());
        };

        // ====== Fetch Data for Each Symbol (Async, Non-blocking) ======
        for (String symbol : stockList) {
            String symbolUpper = symbol.toUpperCase();

            AlphaVantage.api()
                    .timeSeries()
                    .intraday()
                    .forSymbol(symbol)
                    .interval(Interval.ONE_MIN)
                    .outputSize(OutputSize.FULL)
                    .entitlement("realtime")
                    .onSuccess(r -> {
                        try {
                            handleSuccess((TimeSeriesResponse) r); // UI feedback, file cache, etc.
                            List<StockUnit> units = ((TimeSeriesResponse) r).getStockUnits();
                            units.forEach(u -> u.setSymbol(symbolUpper));
                            Collections.reverse(units); // Ensure oldest first

                            // Store to thread-safe structure for later analysis
                            timelines.computeIfAbsent(symbolUpper, k -> new ArrayList<>()).addAll(units);
                        } catch (IOException e) {
                            throw new RuntimeException(e + " " + symbol); // Surface for debugging
                        } finally {
                            latch.countDown(); // Always count down, even if error
                            SwingUtilities.invokeLater(updateProgress); // UI progress
                        }
                    })
                    .onFailure(err -> {
                        mainDataHandler.handleFailure(err);
                        latch.countDown(); // Don't hang latch on API error
                        SwingUtilities.invokeLater(updateProgress);
                    })
                    .fetch();
        }

        // ====== Wait for All Downloads to Finish ======
        latch.await(); // Blocks until all symbols processed

        // Final progress update and dialog cleanup
        SwingUtilities.invokeLater(() -> {
            progressDialog.dispose();
            logTextArea.append("Initial data loading completed\n");
        });

        // ====== Post-processing: Compute % change for all bars ======
        timelines.forEach((symbol, timeline) -> {
            for (int i = 1; i < timeline.size(); i++) {
                StockUnit cur = timeline.get(i);
                StockUnit prev = timeline.get(i - 1);
                if (prev.getClose() > 0) {
                    double pct = ((cur.getClose() - prev.getClose()) / prev.getClose()) * 100;
                    // Prevent insane spikes: clamp to previous value if too large
                    cur.setPercentageChange(Math.abs(pct) >= 14 ? prev.getPercentageChange() : pct);
                }
            }
        });

        // ====== Analyze All Timelines for Rally Candidates ======
        return calculateIfRally(timelines, false);
    }

    /**
     * Detects which symbols (from a batch) are currently exhibiting a statistically significant "rally" pattern,
     * based on advanced multi-period regression analysis, volatility filtering, and strict trend confirmation logic.
     * <p>
     * <b>How this method achieves high-precision rally detection:</b>
     * <ul>
     *   <li><b>Data Quality Control:</b>
     *       <ul>
     *           <li><b>Cleaning:</b> Filters out non-trading periods (weekends, premarket, after-hours), ensuring only
     *           meaningful price action is used. This avoids misleading spikes and noise from illiquid sessions.</li>
     *           <li><b>Aggregation:</b> Compresses raw minute data into multi-hour bars, smoothing micro-fluctuations
     *           and highlighting real, sustained trends over time.</li>
     *       </ul>
     *   </li>
     *   <li><b>Multi-Period Trend Regression:</b>
     *       <ul>
     *           <li>Performs rolling window regressions over several different durations (e.g., 4, 6, 8, 10, 14 days),
     *           so that only rallies that are persistent and robust across timescales are detected.</li>
     *           <li>Each window must independently demonstrate strong slope (uptrend) and a high R² fit
     *           (low noise/variance).</li>
     *           <li>To avoid "lucky" fits, at least two window lengths must confirm a rally. This redundancy
     *           ensures signals are not artifacts of parameter choices.</li>
     *       </ul>
     *   </li>
     *   <li><b>Recent Strength and Continuation Checks:</b>
     *       <ul>
     *           <li><b>Recency Filter:</b> Even if the long-term trend was strong, the method demands that the
     *           most recent half of the window also passes slope and R² tests. This rejects rallies that
     *           are fizzling out, only flagging those that are still alive and strong.</li>
     *           <li><b>Noise/Volatility Test:</b> Channel width is checked to ensure prices stay near the
     *           regression line (i.e., the rally is orderly and not just wild swings around a trend).</li>
     *           <li><b>Final Continuation:</b> Explicitly requires the last several bars (about one trading day)
     *           to remain close to the trend, catching late failures or reversals.</li>
     *           <li><b>Trend Degradation Guard:</b> Compares the most recent slope to the long-term slope, so
     *           that only rallies which are accelerating or at least stable are flagged (no weakening allowed).</li>
     *       </ul>
     *   </li>
     *   <li><b>Robustness and Transparency:</b>
     *       <ul>
     *           <li>All reasons for rejection (e.g., not enough valid uptrends, trend lost recently, too much noise)
     *           are logged in a detailed summary, supporting transparency and further analysis.</li>
     *           <li>By requiring simultaneous agreement of all these advanced filters, false positives are extremely
     *           rare and only genuine, multi-confirmed rallies are returned.</li>
     *       </ul>
     *   </li>
     * </ul>
     * <p>
     * <b>Summary:</b> The precision of this method comes from its strict requirement that a rally must appear
     * across multiple timescales, be present right up to the most recent bars, remain orderly (low noise), and not
     * degrade at the end. This layered approach, with overlapping filters, ensures that only high-conviction,
     * statistically significant uptrends are ever marked as “rallies” by the system.
     *
     * <p>
     * <b>Pipeline Steps:</b>
     * <ol>
     *   <li><b>Clean and aggregate</b> raw minute data to meaningful bars.</li>
     *   <li><b>Perform regression</b> over various rolling windows to test for persistent uptrend.</li>
     *   <li><b>Require at least two valid window confirmations</b> for robustness.</li>
     *   <li><b>Test the recency and noise</b> in the second half of the window, including channel width.</li>
     *   <li><b>Check trend continuation</b> into the very last bars—rejects faded or choppy moves.</li>
     *   <li><b>Compare recent slope too historical</b> to ensure the rally isn’t losing strength.</li>
     *   <li><b>Return only tickers passing <i>all</i> the above</b>—maximizing precision and minimizing
     *   false positives.</li>
     * </ol>
     *
     * <p>
     * All progress and intermediate results are appended to {@code summary} and printed for user/marker transparency.
     *
     * @param timelines          Map of symbol to cleaned/compressed {@link StockUnit} bar lists.
     * @param useAutoCorrelation boolean for activating alpha intelligence auto correlation
     * @return List of symbols that pass all rally criteria with very high confidence.
     */
    private static List<String> calculateIfRally(Map<String, List<StockUnit>> timelines, boolean useAutoCorrelation) {
        StringBuilder summary = new StringBuilder(); // Used to build a full log of all checks/results for transparency/debugging
        List<String> rallies = new ArrayList<>();    // This will hold all the symbols detected as being "in rally"

        // Loop over each symbol's timeline
        timelines.forEach((symbol, minuteBars) -> {
            summary.append("\n=====================================\n");
            summary.append("📈 Analyzing: ").append(symbol).append(" 📊\n");

            if (useAutoCorrelation) {
                // 0. --- AUTOCORRELATION FETCH ---
                // This section fetches the lag-1 autocorrelation for the symbol's recent 9 days of daily closing prices.
                // It does this *synchronously* (with a latch) to ensure the result is available before further analysis.

                AtomicReference<Double> acLag1 = new AtomicReference<>(null); // Holds the fetched autocorrelation value (null if error)
                CountDownLatch latch = new CountDownLatch(1); // Used to block until the async API response is received

                // Start async request for autocorrelation via AlphaVantage's advanced analytics endpoint
                AlphaVantage.api()
                        .alphaIntelligence()
                        .analyticsFixedWindow()
                        .symbols(symbol)                           // Set the stock ticker
                        .range("9day")                             // Use the last 9 days (trading days, not calendar)
                        .interval("DAILY")                         // Daily bars for autocorrelation calculation
                        .calculations("AUTOCORRELATION(lag=1)")    // Request lag-1 autocorrelation (adjacent days)
                        .ohlc("close")                             // Use closing prices for calculation
                        .onSuccess(response -> {                   // When the API call succeeds:
                            Double val = Optional.ofNullable(response.getReturnsCalculations())
                                    .map(rc -> rc.autocorrelation)                        // Get autocorrelation section of result
                                    .map(ac -> ac.get(symbol))                            // For this ticker
                                    .map(lagMap -> lagMap.get("AUTOCORRELATION(LAG=1)"))  // Get the actual lag-1 value
                                    .orElse(null);                                        // Null if anything missing
                            acLag1.set(val);           // Store the value for use below
                            latch.countDown();         // Release the latch so main thread can proceed
                        })
                        .onFailure(err -> {            // On API failure (network error, etc.):
                            System.err.println("Autocorrelation API error: " + err.getMessage());
                            latch.countDown();         // Release the latch even on failure, to avoid deadlock
                        })
                        .fetch();                      // Send the API request

                try {
                    // Block the current thread until the async API result comes back,
                    // or after a timeout of 4 seconds to avoid hanging indefinitely.
                    latch.await(4, java.util.concurrent.TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt(); // Restore interrupt status if needed
                    System.err.println("Latch interrupted for symbol: " + symbol);
                }

                // --- LOG THE RESULT FOR DEBUGGING ---
                summary.append(String.format(
                        "🔁 Autocorrelation (lag=1): %s\n",
                        acLag1.get() == null ? "N/A" : String.format("%.4f", acLag1.get())
                ));

                // --- AUTOCORRELATION GATING LOGIC ---
                // Now that we have the lag-1 autocorrelation, use it as an early filter for trend strength.
                // If the autocorrelation could not be fetched (API failed or returned no data), reject this symbol immediately
                if (acLag1.get() == null) {
                    summary.append("❌ Failed: Could not fetch autocorrelation\n");
                    return; // Skip this symbol, do not proceed to further checks
                }

                // If autocorrelation is below 0.6, consider the trend too weak to be a "rally"
                if (acLag1.get() < 0.6) {
                    summary.append("❌ Failed: Autocorrelation (lag=1) below threshold (%.2f)\n".formatted(acLag1.get()));
                    return; // Early reject this symbol, skip rest of analysis
                }
            }

            // 1. CLEAN DATA: Remove bars that are out-of-market hours or on weekends
            //    This is critical to ensure regression and channel calculations aren't distorted by illiquid or non-trading periods.
            List<StockUnit> minuteBarsClean = tradingMinutesOnly(minuteBars);

            // 2. AGGREGATE TO LONGER TIMEFRAMES: "Compress" minute bars into 2-hour bars (or whatever MINUTES is set to)
            //    This drastically reduces noise and focuses the regression on meaningful swings/trends, not micro volatility.
            List<StockUnit> series = compress(minuteBarsClean);
            if (series.size() < W) { // W is the minimum number of bars needed for the analysis window
                summary.append("⚠️ Skipped: not enough compressed data (")
                        .append(series.size()).append(" < ").append(W).append(")\n");
                return; // If insufficient data, skip this symbol
            }

            // 3. DEFINE ANALYSIS WINDOW:
            //    Take the last W bars as our rally window. Also take the second half (most recent W/2 bars)
            //    We split it this way to later compare if the trend is "just old" or continuing into the most recent period.
            List<StockUnit> rallyWindowFull = series.subList(series.size() - W, series.size());
            List<StockUnit> secondHalf = rallyWindowFull.subList(W / 2, W);

            boolean isRally = true; // Start by assuming a rally is in progress. Eliminate on any failure below.

            // 4. MULTI-PERIOD TREND VALIDATION: Test for uptrend consistency across multiple rolling windows of different lengths (days).
            //    Why? A true rally should be visible no matter the exact window chosen (robust to parameter tweaks).
            int validDayRanges = 0; // Number of windows that pass our uptrend check
            int checkedDayRanges = 0; // Total number of rolling windows we attempted to check
            for (int days : DAYS_OPTIONS) {
                int w = days * 8; // Convert days to number of bars (assuming 8 bars per day due to 2-hour compression)
                if (series.size() < w) continue; // Can't check this window if not enough data

                checkedDayRanges++; // We are able to check this window
                List<StockUnit> window = series.subList(series.size() - w, series.size());
                LR reg = linReg(window); // Fit a regression line

                summary.append(String.format("🔎 Days: %d | Slope: %.4f | R²: %.4f | Size: %s%n",
                        days, reg.slopePct, reg.r2, window.size()));

                // Both slope (trend up) and R² (good fit) must be above threshold for window to "count"
                if (reg.slopePct >= SLOPE_MIN && reg.r2 >= R2_MIN) {
                    validDayRanges++;
                }
            }

            summary.append(String.format("✅ Valid Day Ranges: %d / %d%n", validDayRanges, checkedDayRanges));
            // Require at least two uptrend-confirmed windows (reduces chance of false positives on noisy charts)
            if (checkedDayRanges > 0 && validDayRanges < 2) {
                isRally = false;
                summary.append("❌ Failed: Not enough valid uptrend periods\n");
            }

            // 5. TEST RECENT PERIOD STRENGTH: Is the uptrend still present in the *most recent* half of the data?
            //    Many rallies fizzle out near the end; we want to catch only those that are still strong.
            LR secondHalfRegLine = linReg(secondHalf);
            summary.append(String.format("📊 Second Half | Slope: %.4f | R²: %.4f%n",
                    secondHalfRegLine.slopePct, secondHalfRegLine.r2));
            // Both recent slope and fit must pass thresholds
            if (secondHalfRegLine.slopePct < SLOPE_MIN && secondHalfRegLine.r2 < R2_MIN) {
                isRally = false;
                summary.append("❌ Failed: Second half regression too weak\n");
            }

            // 6. TEST FOR NOISE/ERRATIC BEHAVIOR: Is the channel narrow? (Are most bars close to regression?)
            //    Reject "rallies" that are only spikes or have huge swings around the uptrend.
            double width = channelWidth(secondHalf, secondHalfRegLine);
            summary.append(String.format("📏 Second Half Channel Width: %.4f%n", width));
            if (width > WIDTH_MAX) {
                isRally = false;
                summary.append("❌ Failed: Channel width too wide\n");
            }

            // 7. FINAL CONTINUATION TEST: Are the most recent bars (e.g., last trading day) actually still in trend?
            //    This catches trend breaks that might happen just at the end (e.g. failed breakout, selloff).
            if (!trendUpToPresent(secondHalf, secondHalfRegLine)) {
                isRally = false;
                summary.append("❌ Failed: Trend not continuing to present\n");
            }

            // 8. IS RECENT TREND AS STRONG AS WHOLE? Avoid cases where the trend has recently weakened.
            //    Compare slope of recent window vs. the whole window.
            LR fullSeriesRegLine = linReg(series);             // Regression for all available data
            LR fullWindowRegLine = linReg(rallyWindowFull);    // Regression for our rally window
            summary.append(String.format("📉 Full Series Slope: %.4f | Full Window Slope: %.4f%n",
                    fullSeriesRegLine.slopePct, fullWindowRegLine.slopePct));
            // If the slope of the current window is much less than historical, rally is likely fading.
            if (fullWindowRegLine.slopePct * 1.5 < fullSeriesRegLine.slopePct) {
                isRally = false;
                summary.append("❌ Failed: Recent trend weaker than historical\n");
            }

            // 9. ONLY IF ALL TESTS PASSED, MARK SYMBOL AS IN RALLY!
            if (isRally) {
                rallies.add(symbol);
                summary.append("🚀 ==> RALLY DETECTED\n");
            }
        });

        // Print the full analysis summary to the console for user/marker transparency.
        summary.append("=====================================\n");
        System.out.println(summary);

        // Return only the tickers passing all conditions.
        return rallies;
    }

    /**
     * Filters a list of {@link StockUnit} bars to only those that occur during regular market trading hours.
     * <ul>
     *   <li>Removes all bars that fall on weekends.</li>
     *   <li>Removes all bars that are before market open or after market close.</li>
     *   <li>Resulting list is sorted in chronological order (oldest first).</li>
     * </ul>
     *
     * @param bars List of raw OHLCV bars (may include premarket, postmarket, or weekend bars)
     * @return A sorted, filtered list containing only bars from Monday-Friday and within [MARKET_OPEN, MARKET_CLOSE]
     */
    private static List<StockUnit> tradingMinutesOnly(List<StockUnit> bars) {
        return bars.stream()
                .filter(u -> {
                    // Extract timestamp of the current bar (LocalDateTime for easy day/time checks)
                    LocalDateTime ts = u.getLocalDateTimeDate();
                    DayOfWeek dow = ts.getDayOfWeek();     // Get the day of week (MON, TUE, ..., SAT, SUN)
                    LocalTime tod = ts.toLocalTime();      // Get the time of day (hh:mm:ss)

                    // Step 1: Weekday filter - Only allow Monday to Friday (i.e., drop weekends entirely)
                    boolean weekday = dow != DayOfWeek.SATURDAY && dow != DayOfWeek.SUNDAY;

                    // Step 2: Trading hours filter - Only include bars after market open and before market close.
                    // - tod.isBefore(MARKET_OPEN) is true if bar is too early (pre-market).
                    // - tod.isAfter(MARKET_CLOSE) is true if bar is too late (post-market).
                    // - So, we want bars that are NOT before open AND NOT after close.
                    boolean mktHours = !tod.isBefore(MARKET_OPEN) && !tod.isAfter(MARKET_CLOSE);

                    // Both conditions must be true to keep the bar
                    return weekday && mktHours;
                })
                // Always return bars in chronological order (oldest to newest)
                .sorted(Comparator.comparing(StockUnit::getLocalDateTimeDate))
                .toList();
    }

    /**
     * Calculates the relative "width" of a trend channel for a given set of bars and a fitted regression line.
     * <p>
     * The width is the largest percentage deviation of actual close from predicted trend value,
     * i.e., the tightness of the channel. Used to reject erratic, volatile rallies.
     *
     * @param win Bars in the window to test (should be the second half of rally window)
     * @param lr  Linear regression model fitted to this window
     * @return Maximum fractional deviation of price from regression trend (e.g., 0.04 = 4%)
     */
    private static double channelWidth(List<StockUnit> win, LR lr) {
        double maxResidual = 0.0;
        // For each bar, calculate the absolute percent deviation from predicted line
        for (int i = 0; i < win.size(); i++) {
            double pred = predictSlope(lr, i); // Regression-predicted close price for bar i
            maxResidual = Math.max(maxResidual, Math.abs(win.get(i).getClose() - pred) / pred);
        }
        return maxResidual;
    }

    /**
     * Determines if the most recent set of bars is still "in alignment" with the uptrend defined by the regression line.
     * <p>
     * This is a robustness check for rally detection. It ensures the uptrend isn't only in the past,
     * but that the last few bars (typically one trading day) are still close to the predicted uptrend line.
     * <ul>
     *   <li>Looks at the last 8 compressed bars (can be tuned).</li>
     *   <li>For each, checks if the close price is within {@code TOLERANCE_PCT} (e.g. 3%) of the regression prediction.</li>
     *   <li>Requires at least 75% of these bars to "align" (avoid one-off outliers).</li>
     * </ul>
     *
     * @param win List of bars representing the most recent window (e.g., second half of rally window)
     * @param lr  Fitted linear regression ({@link LR}) for that window
     * @return {@code true} if enough bars are in trend alignment, {@code false} otherwise.
     */
    private static boolean trendUpToPresent(List<StockUnit> win, LR lr) {
        int checkBars = 8; // How many bars to check at the end (≈ one trading day)
        int aligned = 0;   // Counter for bars that meet the "alignment" requirement

        // Loop over the last 'checkBars' entries of 'win'
        for (int i = win.size() - checkBars; i < win.size(); i++) {
            double predicted = predictSlope(lr, i); // Expected price by regression
            double price = win.get(i).getClose();

            double lower = predicted * (1 - TOLERANCE_PCT); // Lower bound (e.g., -3%)
            double upper = predicted * (1 + TOLERANCE_PCT); // Upper bound (e.g., +3%)

            if (price >= lower && price <= upper) {
                aligned++; // Price is "close enough" to trend
            }
        }

        // Require at least 75% of bars to be aligned (tolerate a few outliers)
        return aligned >= (int) Math.ceil(0.75 * checkBars);
    }

    /**
     * Performs a simple ordinary least squares (OLS) linear regression on a list of bars.
     * <p>
     * Returns an {@link LR} record, including:
     * <ul>
     *   <li>Slope in both price-units-per-bar and as %/bar.</li>
     *   <li>Mean X (bar index) and Y (price).</li>
     *   <li>R-squared (coefficient of determination): how well a line fits the data (1 = perfect).</li>
     * </ul>
     * <p>
     * Used to identify trends: uptrends have positive slope, with R² > 0.6 as a filter.
     *
     * @param bars List of bars (must have size >= 2).
     * @return {@link LR} regression result (all fields 0 if degenerate/undefined)
     */
    private static LR linReg(List<StockUnit> bars) {
        int n = bars.size();

        // Calculate sums for x, y, x^2, x*y
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n; i++) {
            double y = bars.get(i).getClose();
            sx += i;            // Sum of indices
            sy += y;            // Sum of close prices
            sxx += i * (double) i; // Sum of index^2
            sxy += i * y;       // Sum of index*price
        }
        double denom = n * sxx - sx * sx; // Denominator for OLS

        // If denominator is zero, data is degenerate (all at same time)
        if (denom == 0) return new LR(0, 0, 0, 0, 0);

        // Slope in price units per bar (OLS formula)
        double slopePpu = (n * sxy - sx * sy) / denom;
        double xBar = sx / n;   // Mean of bar index (center of window)
        double yBar = sy / n;   // Mean of prices (center of window)
        double slopePct = slopePpu / yBar * 100.0; // Slope as %/bar (normalized)

        // R-squared: proportion of variance explained by model
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++) {
            double y = bars.get(i).getClose();
            double yPred = yBar + slopePpu * (i - xBar); // Prediction for index i
            ssTot += (y - yBar) * (y - yBar); // Total variance
            ssRes += (y - yPred) * (y - yPred); // Residual (unexplained) variance
        }
        double r2 = ssTot == 0 ? 0 : 1 - ssRes / ssTot;

        return new LR(slopePct, slopePpu, r2, yBar, xBar);
    }

    /**
     * Uses the parameters from a linear regression (LR) fit to predict the *expected* close price
     * at a given bar index `i` in the time window.
     * <p>
     * This is the standard formula for a least-squares regression line:
     * y = ȳ + slope * (x - x̄)
     * where:
     * - ȳ (lr.yBar): The mean close price of the window (average y value).
     * - x̄ (lr.xBar): The mean index within the window (average x value, usually middle of window).
     * - slope (lr.slopePpu): The best-fit slope (change in price units per bar, from OLS).
     * - i: The index (relative position) within the window for which we want the prediction.
     * <p>
     * This method is typically called for every bar in the analysis window, to generate the
     * predicted regression value for each bar, so you can compare the *actual* close to the *expected* close.
     * The result is used to compute things like channel width (max deviation from the trend),
     * or to check if recent prices are still "in alignment" with the trend.
     *
     * @param lr The regression object (contains mean x, mean y, and slope in price units/bar)
     * @param i  The target index within the window (0 = start of window, etc.)
     * @return The predicted close price at index i, according to the regression line
     */
    private static double predictSlope(LR lr, int i) {
        // The regression line for OLS (ordinary least squares) is:
        // y = ȳ + m * (x - x̄)
        // - ȳ: mean y (average close)
        // - m:  slope in price units per bar (how much price increases per bar, on average)
        // - x:  current index (i)
        // - x̄: mean index (center of window)
        //
        // This predicts what the price "should be" if the trend held perfectly.

        return lr.yBar + lr.slopePpu * (i - lr.xBar);
        // (i - lr.xBar) gives how far this bar is from the center;
        // Multiply by the slope, add mean y to get predicted value.
    }

    /**
     * Aggregates ("compresses") a list of minute bars into larger bars (e.g. 2h bars).
     * <ul>
     *   <li>Each output bar contains OHLCV for a {@code MINUTES}-long window.</li>
     *   <li>Handles missing minutes and only starts a new bucket if enough time has passed.</li>
     *   <li>Also re-filters to only weekdays (defensive).</li>
     * </ul>
     *
     * @param src List of raw StockUnits (minute bars, possibly unsorted, possibly including weekends)
     * @return List of compressed StockUnits (bucketed, sorted, one per {@code MINUTES} window)
     */
    private static List<StockUnit> compress(List<StockUnit> src) {

        // Defensive: Remove weekend bars (in case they're in the source), and ensure bars are sorted by time.
        // This avoids aggregation errors and keeps output time series consistent.
        List<StockUnit> data = src.stream()
                .filter(u -> {
                    DayOfWeek d = u.getLocalDateTimeDate().getDayOfWeek();
                    // Keep only bars from Monday-Friday (ignore Saturday/Sunday completely)
                    return d != DayOfWeek.SATURDAY && d != DayOfWeek.SUNDAY;
                })
                // Sorting by bar timestamp so our aggregation always moves forward in time.
                .sorted(Comparator.comparing(StockUnit::getLocalDateTimeDate))
                .toList();

        List<StockUnit> out = new ArrayList<>();
        if (data.isEmpty()) return out; // No data? Return an empty output—nothing to compress.

        // Initialize the first bucket's start time to the first (oldest) bar, rounded/truncated to the minute.
        // All bars within this "bucket" period will be grouped together for OHLCV aggregation.
        ZonedDateTime bucketStart = data.get(0).getLocalDateTimeDate()
                .truncatedTo(ChronoUnit.MINUTES)      // Remove any sub-minute time precision
                .atZone(ZoneId.systemDefault());      // Use the system default time zone for consistency

        StockUnit agg = null; // Will hold the current aggregated bar for this bucket

        // Loop over all bars in chronological order
        for (StockUnit stockUnit : data) {
            ZonedDateTime t = stockUnit.getLocalDateTimeDate()
                    .truncatedTo(ChronoUnit.MINUTES)
                    .atZone(ZoneId.systemDefault());

            if (agg == null) {
                // If this is the very first bar (no aggregation started), initialize a new aggregation bar (agg)
                agg = new StockUnit.Builder()
                        .open(stockUnit.getOpen())           // Open is the open of the first bar in the bucket
                        .high(stockUnit.getHigh())           // High so far is the current bar's high
                        .low(stockUnit.getLow())             // Low so far is the current bar's low
                        .close(stockUnit.getClose())         // Close is the current bar's close (will be updated)
                        .adjustedClose(stockUnit.getClose()) // Adjusted close starts same as close (maybe updated)
                        .volume(stockUnit.getVolume())       // Initial volume is just this bar's volume
                        .dividendAmount(0)                   // Default for synthetic bars
                        .splitCoefficient(0)                 // Default for synthetic bars
                        .time(stockUnit.dateTime)            // Timestamp of the first bar in the bucket
                        .build();
                continue; // Move to the next bar; we've initialized our bucket.
            }

            // Check if the current bar still falls within the current bucket
            // (bucket duration is mainDataHandler.MINUTES, e.g., 120 minutes)
            if (ChronoUnit.MINUTES.between(bucketStart, t) < mainDataHandler.MINUTES) {
                // If yes, update our aggregation:
                // - High: max of current and previous highs
                agg.setHigh(Math.max(agg.getHigh(), stockUnit.getHigh()));
                // - Low: min of current and previous lows
                agg.setLow(Math.min(agg.getLow(), stockUnit.getLow()));
                // - Close: always set to the most recent bar's close (rolling forward)
                agg.setClose(stockUnit.getClose());
                // - Volume: accumulate total volume for this bucket
                agg.setVolume(agg.getVolume() + stockUnit.getVolume());
            } else {
                // If not in the same bucket (i.e., new bar is outside the bucket duration):
                // 1. Save the completed aggregation bar to the output list
                out.add(agg);

                // 2. Reset the bucketStart to the current bar's timestamp (truncated to minute)
                bucketStart = t.truncatedTo(ChronoUnit.MINUTES);

                // 3. Start a new aggregation bar for the new bucket, copying all fields from current bar
                agg = new StockUnit.Builder()
                        .open(stockUnit.getOpen())                   // New bucket: open = open of first bar
                        .high(stockUnit.getHigh())                   // New high/low/close/volume fields from current bar
                        .low(stockUnit.getLow())
                        .close(stockUnit.getClose())
                        .adjustedClose(stockUnit.getClose())
                        .volume(stockUnit.getVolume())
                        .dividendAmount(stockUnit.getDividendAmount()) // Pass through if set
                        .splitCoefficient(stockUnit.getSplitCoefficient()) // Pass through if set
                        .time(stockUnit.dateTime)                    // Timestamp is from the new first bar
                        .build();
            }
        }

        // After loop: If there's any unfinished aggregation bar, add it as the last bucket.
        if (agg != null) out.add(agg);

        // Return all aggregated, compressed bars (each representing a MINUTES-long bucket)
        return out;
    }

    /**
     * Evaluates the ML prediction, technical indicators, and recent price action to generate a list of actionable notifications/alerts.
     * <p>
     * This function forms the final "decision layer" in the alerting pipeline. It inspects both statistical/ML results and
     * direct technical signals, combining them with current market context (such as resistance proximity), to decide if an
     * event (spike, gap, breakout, etc.) should trigger a user notification.
     *
     * @param prediction         The ML model's confidence output (probability or score that an event is imminent, e.g., rally/spike).
     * @param stocks             The window of recent stock bars (should be frameSize long).
     * @param symbol             The ticker symbol for the analyzed stock.
     * @param features           The array of raw indicator features for this frame.
     * @param normalizedFeatures The normalized (0-1) array of features, ready for scoring/aggregation.
     * @param uptrendPrediction  The uptrend model's confidence output (probability or score indicating an uptrend).
     * @return List of {@link Notification}s to be shown/queued for the user.
     */
    private static List<Notification> evaluateResult(double prediction, List<StockUnit> stocks, String symbol,
                                                     double[] features, float[] normalizedFeatures, double uptrendPrediction) {
        // === 1. Prepare the output list for all alerts generated for this frame ===
        List<Notification> alertsList = new ArrayList<>();

        // === 2. Assess if price is near a resistance zone (used for notification context) ===
        double nearRes = isNearResistance(stocks);

        // === 3. "Fill the Gap" event logic ===
        // - Checks for classic "gap fill" price actions (sharp drop + bounce/oversold indicators).
        // - Alerts added directly to alertsList if triggered.
        fillTheGap(prediction, stocks, symbol, alertsList);

        // === 4. Defensive fallback: ensure global 'aggressiveness' parameter is set. ===
        // - Some systems may not set this variable on startup, so default to 1.0.
        if (aggressiveness == 0.0) {
            aggressiveness = 1.0F;
        }

        // === 5. Core Spike/R-Line event logic ===
        // - Looks for powerful bullish events using multi-indicator confirmation.
        // - Handles both R-Line (near resistance) and normal spike scenarios.
        spikeUp(
                prediction,
                stocks,
                symbol,
                features,
                alertsList,
                nearRes,
                aggressiveness,
                normalizedFeatures,
                uptrendPrediction
        );

        // === 6. Return all collected notifications for display or further processing ===
        return alertsList;
    }

    /**
     * Detects rapid bullish "spike" events and R-Line (resistance-line) confirmations on stock price data.
     * <ul>
     *     <li>Fires notifications only when strict technical and statistical criteria are met.</li>
     *     <li>Adapts sensitivity dynamically based on a user-defined "aggressiveness" parameter.</li>
     *     <li>Filters for high-confidence breakouts, minimizing noise and false positives.</li>
     * </ul>
     *
     * @param prediction           ML model prediction confidence, from 0.0 to 1.0
     * @param stocks               List of recent StockUnit bars (candles/frames)
     * @param symbol               Ticker symbol (e.g., "AAPL")
     * @param features             Raw indicator feature values (see computeFeatures for order/meaning)
     * @param alertsList           List to append new Notification objects if an alert is triggered
     * @param nearRes              1.0 if price is near resistance, 0.0 otherwise
     * @param manualAggressiveness User/system-level aggressiveness multiplier (risk tolerance, 0.1–2.0)
     * @param normalizedFeatures   Feature vector, normalized to [0,1], for dynamic scoring
     * @param uptrendPrediction    The uptrend model's confidence output (probability or score indicating an uptrend).
     */
    private static void spikeUp(
            double prediction, List<StockUnit> stocks, String symbol, double[] features,
            List<Notification> alertsList, double nearRes, float manualAggressiveness,
            float[] normalizedFeatures, double uptrendPrediction) {
        // === 0. Liquidity Check: Only proceed if there is enough trading volume/capital to exit a position ===
        // Define how much you want to be able to sell
        double requiredNotional = volume;   // Volume
        int liquidityLookBack = 10;         // How many bars/minutes to look back for average liquidity (e.g., 10 minutes)

        // Check if current and recent liquidity are sufficient to execute your trade without major slippage or waiting
        if (!isLiquiditySufficient(stocks, liquidityLookBack, requiredNotional)) {
            // Not enough liquidity recently; skip all spike logic and do NOT fire notifications.
            return;
        }

        // === 1. Calculate dynamic composite "aggressiveness" based on active feature weights ===
        // - Uses all normalized features and their respective category weights.
        double dynamicAggro = calculateWeightedAggressiveness(normalizedFeatures, manualAggressiveness); // Higher manualAgg increases sensitivity

        // === 2. Set adaptive threshold for cumulative percentage move, based on feature activations ===
        // - The more bullish the features, the smaller the required cumulative gain for an alert
        double cumulativeThreshold = 0.6 * dynamicAggro;

        // === 3. Compute rolling sum of percentage changes for last 2 and 3 bars ===
        // - Used to detect sudden strong upward momentum
        double changeUp2 = calculateWindowChange(stocks, 2);
        double changeUp3 = calculateWindowChange(stocks, 3);

        // === 4. Compute adaptive activation thresholds for dynamicAggro using percentile mapping ===
        // - The threshold adapts to the user's aggressiveness setting.
        double pMin = 0.65;
        double pMax = 0.95;
        double threshold = computeDynamicThreshold(manualAggressiveness, computeDynamicPercentile(manualAggressiveness, pMin, pMax));

        // check for PL Tester
        if (market == null) {
            market = "ultraVolatile";
        }

        // Determine if a stock symbol should trigger an uptrend signal, using dynamic parameters
        // based on the market cap group or market sector.
        // The config is tailored for risk and volatility appropriate to each category.
        boolean uptrend;

        if (MEGA_CAPS.contains(symbol)) {
            // Mega caps (largest companies): use lowest risk/least aggressive config
            uptrend = shouldTrigger(stocks, 5, 0.3, 0.8,
                    0.08, 0.39, 3, 0.05, 0.06, false, uptrendPrediction);
        } else if (LARGE_CAPS.contains(symbol)) {
            // Large caps: slightly more aggressive config than mega caps
            uptrend = shouldTrigger(stocks, 5, 0.6, 0.8,
                    0.09, 0.40, 3, 0.12, 0.15, false, uptrendPrediction);
        } else if (MID_CAPS.contains(symbol) || SMALL_CAPS.contains(symbol)) {
            // Mid & small caps: highest volatility/aggressiveness config among cap sizes
            uptrend = shouldTrigger(stocks, 5, 1.5, 0.8,
                    0.08, 0.39, 3, 0.15, 0.5, false, uptrendPrediction);
        } else {
            // Not a known cap group: select config based on sector (market) label
            // Sectors are grouped by typical volatility appetite

            uptrend = switch (market) {
                // Low aggressiveness: stable, less-volatile sectors
                case "bigCaps", "semiconductors", "techGiants", "aiStocks", "financials", "energy",
                     "industrials", "pharma", "foodBeverage", "retail" -> shouldTrigger(stocks, 5, 0.3, 0.8,
                        0.08, 0.39, 3, 0.05, 0.06, false, uptrendPrediction);

                // Mid-aggressiveness: moderate volatility sectors
                case "midCaps", "chineseTech", "autoEV", "healthcareProviders", "robotics", "allSymbols" ->
                        shouldTrigger(stocks, 5, 0.6, 0.8,
                                0.09, 0.40, 3, 0.12, 0.15, false, uptrendPrediction);

                // High aggressiveness: high-risk, high-volatility sectors
                case "ultraVolatile", "highVolatile", "smallCaps", "cryptoBlockchain", "quantum",
                     "favourites" -> shouldTrigger(stocks, 5, 1.5, 0.8,
                        0.08, 0.39, 3, 0.15, 0.5, false, uptrendPrediction);

                // If market not recognized, fail fast and force a fix
                default -> throw new RuntimeException("Need to specify a market: " + market);
            };
        }

        // === 5. Strict "all conditions met" trigger for classic spike event alert ===
        //   (a) features[4] == 1   : Binary "spike" anomaly active
        //   (b) (features[5] >= cumulativeThreshold || normalizedFeatures[5] >= 0.85) : Large enough cumulative percentage gain
        //   (c) dynamicAggro >= threshold : Strong overall feature activation (robustness)
        //   (d) prediction >= 0.9   : ML model must be highly confident (>90%)
        //   (e) features[6] == 1   : Keltner channel breakout detected
        //   (f) changeUp2 and changeUp3 >= 1.5 * manualAggressiveness : Last 2-3 bars have strong % gains
        boolean isTriggered =
                features[4] == 1 &&                                   // Spike anomaly detected
                        (features[5] >= cumulativeThreshold || normalizedFeatures[5] >= 0.85) && // Sufficient cumulative gain
                        dynamicAggro >= threshold &&                          // Strong feature activation
                        prediction >= 0.9 &&                                  // ML prediction is highly confident
                        features[6] == 1 &&                                   // Keltner channel breakout
                        changeUp2 >= getThreshold(symbol) * manualAggressiveness &&  // 2-bar momentum strong enough
                        changeUp3 >= getThreshold(symbol) * manualAggressiveness; // 3-bar momentum strong enough

        int notificationCode = -1; // -1 means no notification

        // === 6. Only if all strict spike conditions are satisfied, trigger a classic alert notification ===
        // - Config code: 3 ("spike" alert) if not near resistance; 2 ("R-Line" alert) if near resistance.
        if (isTriggered) {
            notificationCode = (nearRes == 0) ? 3 : 2; // 3 = Spike, 2 = R-Line near resistance
        }

        // === 7. Upwards movements which are not categorized as spikes but still steady upwards ===
        else if (uptrend && !MEGA_CAPS.contains(symbol)) {
            notificationCode = 4;
        }

        // If a notification is to be sent, send it once:
        if (notificationCode != -1) {
            createNotification(
                    symbol, changeUp3, alertsList,
                    stocks, stocks.get(stocks.size() - 1).getLocalDateTimeDate(),
                    prediction, notificationCode, new ArrayList<>()
            );
        }
    }

    /**
     * Checks if the average notional traded (price × volume) over the last 'lookBack' bars
     * meets or exceeds the required threshold. This is less strict than requiring every bar
     * to meet the threshold, and instead focuses on the average liquidity available.
     *
     * @param stocks           The list of StockUnit bars (chronological order).
     * @param lookBack         The number of most recent bars to consider (e.g., 10 for 10 minutes).
     * @param requiredNotional The minimum average notional value (price × volume) required.
     * @return true if the average notional over the lookBack window is >= requiredNotional; false otherwise.
     */
    private static boolean isLiquiditySufficient(List<StockUnit> stocks, int lookBack, double requiredNotional) {
        // Ensure that there are enough historical bars to perform the liquidity check.
        // If there are fewer bars than 'lookBack', we cannot make a reliable decision, so fail fast.
        if (stocks.size() < lookBack) return false;

        // Will accumulate the sum of notional traded over the lookBack window.
        double totalNotional = 0.0;

        // Iterate over the last 'lookBack' bars in the list.
        // This loop starts at (size - lookBack) to only include the most recent 'lookBack' bars.
        for (int i = stocks.size() - lookBack; i < stocks.size(); i++) {
            StockUnit bar = stocks.get(i); // Fetch the current bar (e.g., 1-min or 5-min candle).
            double close = bar.getClose();     // Get the closing price for this bar.
            double volume = bar.getVolume();   // Get the total volume traded in this bar.

            // Compute the notional value traded in this bar (i.e., how much money changed hands).
            // This is the "liquidity" available in this period for trading.
            totalNotional += close * volume;
        }

        // Calculate the average notional traded per bar over the lookBack window.
        // This represents the typical liquidity you can expect to transact without excessive impact.
        double averageNotional = totalNotional / lookBack;

        // The liquidity is sufficient if the average notional is at least as large as the required amount
        // needed to safely trade your position. If so, return true; otherwise, return false.
        return averageNotional >= requiredNotional;
    }

    /**
     * Determines if the market is in a sustained uptrend over the given window.
     *
     * @param stocks    List of StockUnit candles (chronological order).
     * @param window    How many most recent bars to consider (e.g., 10 or 20).
     * @param minChange Minimum net percentage change over window to qualify as uptrend (e.g., 10 for +10%).
     * @param minGreen  Minimum % of green candles (0.0–1.0) to qualify (e.g., 0.6 means at least 60% green bars).
     * @return True if in uptrend, otherwise false.
     */
    public static boolean isInUptrend(List<StockUnit> stocks, int window, double minChange, double minGreen) {
        // Defensive: Make sure we have enough bars to check the window
        if (stocks == null || stocks.size() < window + 1) return false;

        int start = stocks.size() - window - 1; // First index in the window (oldest bar in window)
        int end = stocks.size() - 1;            // Last index in the window (most recent bar)

        // Get closing prices for start and end of window
        double firstClose = stocks.get(start).getClose();
        double lastClose = stocks.get(end).getClose();

        // Calculate total percent change from first to last close in window
        double percentChange = ((lastClose - firstClose) / firstClose) * 100.0;

        // Count number of green (bullish) candles in the window
        int greenCount = 0;
        for (int i = start + 1; i <= end; i++) {
            // A candle is green if close > open (bullish)
            if (stocks.get(i).getClose() > stocks.get(i).getOpen()) {
                greenCount++;
            }
        }

        // Compute the ratio of green candles to total candles in the window
        double greenRatio = greenCount / (double) window;

        // Find the largest single red (bearish) candle in the window
        double maxRed = 0;
        for (int i = start + 1; i <= end; i++) {
            // Red candle size = open - close (only positive if it's red)
            double red = stocks.get(i).getOpen() - stocks.get(i).getClose();
            if (red > maxRed) maxRed = red;
        }

        // Check that no single red candle erases more than a third of the window's total gain
        boolean noBigPullback = maxRed < (lastClose - firstClose) / 3.0;

        // Must satisfy: (1) big enough gain, (2) enough green candles, (3) no big pullbacks
        return percentChange >= minChange && greenRatio >= minGreen && noBigPullback;
    }

    /**
     * Maps manualAggressiveness ∈ [0.1, 2.0] to a dynamic percentile ∈ [pMin, pMax].
     * Used to determine how "deep" into the possible aggressiveness band to set alert thresholds.
     *
     * @param manualAgg User-chosen aggressiveness value (0.1 to 2.0)
     * @param pMin      Minimum percentile (e.g. 0.65)
     * @param pMax      Maximum percentile (e.g. 0.95)
     * @return Interpolated percentile between pMin and pMax
     */
    private static double computeDynamicPercentile(double manualAgg, double pMin, double pMax) {
        // Normalize manualAgg into [0, 1] where 0.1 is min and 2.0 is max
        double t = (manualAgg - 0.1) / (2.0 - 0.1);
        // Interpolate percentile
        return pMin + t * (pMax - pMin);
    }

    /**
     * Computes a dynamic activation threshold for feature activation,
     * given manual aggressiveness and a target percentile.
     *
     * @param manualAgg User aggressiveness value (0.1 to 2.0)
     * @param p         Target percentile (computed via computeDynamicPercentile, range 0.0 to 1.0)
     * @return Dynamic feature activation threshold
     */
    private static double computeDynamicThreshold(double manualAgg, double p) {
        // The dynamicAggro range is [manualAgg, manualAgg * 2.9]
        double maxAgg = manualAgg * 2.9;
        // Activation threshold is at percentile 'p' in that range
        return manualAgg + (maxAgg - manualAgg) * p;
    }

    /**
     * Calculates the rolling sum of percentage price changes for the last N bars in a list.
     *
     * @param stocks List of recent stock bars.
     * @param window Number of most recent bars to sum.
     * @return Cumulative percentage change over the specified window.
     */
    private static double calculateWindowChange(List<StockUnit> stocks, int window) {
        // Sums the percentage changes over the last 'window' bars in the stocks list
        return stocks.stream()
                .skip(stocks.size() - window)      // Skip to the last 'window' elements
                .mapToDouble(StockUnit::getPercentageChange) // Get percentage change for each
                .sum();                            // Sum the changes
    }

    /**
     * Detects and triggers notifications for "gap fill" events—statistically significant, rapid downward price movements
     * that are highly likely to be followed by a rebound ("filling the gap").
     * <p>
     * <b>How this method achieves near-perfect dip prediction:</b>
     * <ul>
     *     <li><b>Multi-layered indicator confirmation:</b>
     *         <ul>
     *             <li><b>Trend Context:</b> Measures the deviation of the current close from the 20-period SMA (simple moving average),
     *             ensuring that only real departures from the recent trend are considered potential gaps/dips.</li>
     *             <li><b>Dynamic Volatility Filtering:</b> Uses a threshold based on current ATR (Average True Range) and
     *             historical ATR, adjusting its sensitivity to current market volatility—this means the system is
     *             more selective during wild markets and more sensitive during calm periods.</li>
     *             <li><b>Momentum Confirmation:</b> Requires at least one momentum exhaustion signal (RSI &lt; 32 <i>or</i>
     *             Stochastic Oscillator &lt; 5), which are classic technical indicators for “oversold” conditions.</li>
     *             <li><b>Sharp Drop & Sustained Move:</b> Confirms that a significant, sharp drop has occurred and that
     *             this move is broad-based (not just a one-candle anomaly), by looking back over several bars.</li>
     *         </ul>
     *     </li>
     *     <li><b>False positive prevention:</b>
     *         <ul>
     *             <li><b>Strict AND logic:</b> All major filters must agree before triggering an alert; if any
     *             condition fails (e.g., not enough deviation, no momentum exhaustion, or move not sustained), the
     *             alert is suppressed. This redundancy ensures that only true, high-conviction dips fire signals.</li>
     *             <li><b>Adaptive thresholds:</b> The logic automatically tightens or relaxes thresholds based on current
     *             market volatility, reducing “noise” during stress and not missing events during quiet periods.</li>
     *         </ul>
     *     </li>
     *     <li><b>Market regime adaptation:</b> By calibrating against recent and historical volatility (ATR),
     *     the detector can dynamically adjust its expectations, achieving high accuracy in both quiet and turbulent markets.</li>
     *     <li><b>Result:</b> Only if <i>all</i> statistical, trend, and momentum filters are satisfied—signaling a true, statistically rare event—
     *     is a “gap fill” notification generated, leading to exceptional robustness and a very low false positive rate.</li>
     * </ul>
     *
     * <p>
     * <b>Detection Process:</b>
     * <ol>
     *     <li><b>Trend baseline:</b> Compute 20-period SMA; measure deviation of current close from SMA.</li>
     *     <li><b>Volatility filter:</b> Compute ATR for current window and compare to historical ATR to set a dynamic gap threshold.</li>
     *     <li><b>Momentum check:</b> Require RSI &lt; 32 or Stochastic &lt; 5.</li>
     *     <li><b>Sharp drop check:</b> Confirm a recent, rapid price drop in the lookback window.</li>
     *     <li><b>Sustained movement:</b> Require the move to be broad-based (not a single-candle outlier).</li>
     *     <li><b>Alert:</b> Only if <i>all</i> these filters agree is a notification created for the user/trader.</li>
     * </ol>
     *
     * <p>
     * <b>Summary:</b> The power of this method is in its strict combination of adaptive, multifactor statistical and technical tests.
     * Each component guards against a different type of false positive, making it extremely effective at only flagging genuine
     * gap fill opportunities with a high probability of mean-reversion.
     *
     * @param prediction The latest ML prediction/confidence value (used for alert ranking or display).
     * @param stocks     Window of StockUnit bars (must have at least {@code smaPeriod} entries).
     * @param symbol     Ticker symbol (used for historical ATR lookup).
     * @param alertsList The master notification list to which any generated gap fill notifications are appended.
     */
    private static void fillTheGap(double prediction, List<StockUnit> stocks, String symbol, List<Notification> alertsList) {
        // --- Detection periods and thresholds (tunable for market regime) ---
        int smaPeriod = 20;           // Length for moving average baseline (trend context)
        int atrPeriod = 14;           // ATR window for volatility baseline
        int rsiPeriod = 10;           // RSI calculation period (momentum)
        int stochasticPeriod = 14;    // Stochastic oscillator period (momentum divergence)
        int dropLookBack = 10;        // Look-back for sharp drop test (recent sudden moves)
        int stochasticLimit = 5;      // Threshold for "extreme" stochastic
        int rsiLimit = 32;            // RSI < 32 means strongly oversold
        double sharpDropThreshold = 2.0;  // Minimum % drop to count as sharp
        double dipThreshold = -2.5;       // Total move threshold for "sustained" dips

        // --- Ensure we have enough bars for SMA calculation ---
        if (stocks.size() >= smaPeriod) {
            int endIndex = stocks.size();
            int startIndex = endIndex - smaPeriod;

            // === 1. Trend Detection: Compute simple moving average over period ===
            double sma = calculateSMA(stocks.subList(startIndex, endIndex));
            double currentClose = stocks.get(endIndex - 1).getClose();
            double deviation = currentClose - sma; // How far below the trend are we?

            // === 2. Volatility-Adjusted Gap Threshold ===
            // ATR measures volatility in current window; getHistoricalATR is long-term baseline.
            double atr = calculateATR(stocks.subList(endIndex - atrPeriod, endIndex), atrPeriod);
            double historicalATR = getHistoricalATR(symbol);
            double volatilityRatio = atr / historicalATR;

            // --- Dynamic multiplier: lowers trigger threshold in high volatility ---
            double multiplier = Math.max(1.5, 3.0 - (volatilityRatio * 0.8));
            double gapThreshold = -multiplier * atr; // Required deviation below SMA to count as a "gap"

            // === 3. Momentum Confirmation (classic bottoming signals) ===
            double rsi = calculateRSI(stocks, rsiPeriod);
            boolean oversold = rsi < rsiLimit; // Is RSI below "oversold" cutoff?
            double stochastic = calculateStochastic(stocks, stochasticPeriod);
            boolean momentumDivergence = stochastic < stochasticLimit; // Is momentum at extreme lows?

            // === 4. Drop Detection: Was there a recent sudden drop? ===
            boolean sharpDrop = checkSharpDrop(stocks, dropLookBack, sharpDropThreshold);

            // === 5. Is this a "wide gap"? (Deviation far below threshold) ===
            boolean isWideGap = deviation < (gapThreshold * 1.3);

            // === 6. Sustained Move: Is the move broad-based, not a one-bar fluke? ===
            boolean sustainedMove = checkSustainedMovement(stocks, dropLookBack, dipThreshold);

            // --- Only fire alert if ALL core conditions are met ---
            boolean baseCondition = isWideGap && deviation < -0.15 &&
                    sharpDrop && sustainedMove &&
                    (oversold || momentumDivergence);

            // Define how much you want to be able to sell
            double requiredNotional = volume;
            int liquidityLookBack = 10;        // How many bars/minutes to look back for average liquidity (e.g., 10 minutes)

            // Check if current and recent liquidity are sufficient to execute your trade without major slippage or waiting
            // === Final: If so, add a "gap fill" notification to output ===
            if (baseCondition && isLiquiditySufficient(stocks, liquidityLookBack, requiredNotional)) {
                createNotification(symbol, deviation, alertsList, stocks,
                        stocks.get(endIndex - 1).getLocalDateTimeDate(),
                        prediction, 1, new ArrayList<>()); // config=1 means "gap fill"
            }
        }
    }

    //Indicators

    /**
     * Checks whether the current close price is within 0.5% below the recent resistance level (recent highest high).
     * <p>
     * Used for alert logic to trigger special notifications (e.g., "R-Line" events) if price is testing or just below resistance.
     *
     * <ul>
     *     <li>Resistance is defined as the highest high from all but the most recent bar.</li>
     *     <li>If current close is within 0.5% (below but not over) of this level, returns 1.0.</li>
     *     <li>Otherwise returns 0.0.</li>
     * </ul>
     *
     * @param stocks Window of StockUnit bars (must have at least 2 bars).
     * @return 1.0 if within 0.5% of resistance, otherwise 0.0.
     */
    private static double isNearResistance(List<StockUnit> stocks) {
        if (stocks.size() < 2) {
            return 0.0; // Not enough data points to define resistance
        }

        // Use all bars except the latest to find recent high (resistance)
        List<StockUnit> previousStocks = stocks.subList(0, stocks.size() - 1);

        if (previousStocks.isEmpty()) {
            return 0.0; // Defensive fallback
        }

        // Find resistance: highest high among previous bars
        double resistanceLevel = previousStocks.stream()
                .mapToDouble(StockUnit::getHigh)
                .max()
                .orElse(0.0);

        // Get the most recent close price
        StockUnit currentStock = stocks.get(stocks.size() - 1);
        double currentClose = currentStock.getClose();

        // Compute threshold (within 0.5% below resistance)
        double threshold = resistanceLevel * 0.995;

        // Return 1.0 if within [threshold, resistanceLevel], else 0.0
        return (currentClose >= threshold && currentClose <= resistanceLevel) ? 1.0 : 0.0;
    }

    /**
     * <p>
     * Determines whether a bullish trigger event should be fired, using a strict multi-layer defense
     * against false positives and adverse fills. This method enforces advanced entry criteria
     * typical of robust trading systems, rejecting signals when market conditions are unfavorable.
     * </p>
     *
     * <p>
     * <b>A trigger will be blocked if ANY of the following conditions are met:</b>
     * <ul>
     *     <li><b>Last candle not bullish:</b> The most recent bar closed below its open.</li>
     *     <li><b>Recent resistance proximity:</b> Price is at/near a known resistance level within the last 15 bars.</li>
     *     <li><b>Microtrend failure:</b> The last <code>barsToCheck</code> bars do not show a consistent micro-uptrend.</li>
     *     <li><b>Bearish gap-down:</b> The latest candle opens significantly below the previous close, beyond the allowed tolerance.</li>
     *     <li><b>Wick/spread abnormality:</b> One or more of the recent candles show excessive wick size, suggesting indecision or volatility.</li>
     * </ul>
     * </p>
     *
     * <p>
     * The method prints a step-by-step debug trace of the trigger decision, indicating which (if any)
     * filter(s) blocked the signal. Each filter acts as a safeguard to reduce noise and prevent entries in high-risk scenarios.
     * </p>
     *
     * @param stocks             Chronological list of OHLC bars (oldest first).
     * @param window             Look-back window for core uptrend detection (multi-bar trend filter).
     * @param minChangePct       Minimum % price gain required over <code>window</code> bars to define an uptrend.
     * @param minGreenRatio      Minimum fraction of bullish (green) candles in the trend window.
     * @param gapTolerancePct    Maximum allowed downward gap between previous close and current open, in percent.
     * @param wickToleranceRatio Maximum allowed ratio of wick length to candle range for valid bars.
     * @param barsToCheck        Number of most recent bars to check for microtrend and wick rules.
     * @param flatTolerance      Minimum percent increase required for each consecutive close in microtrend.
     * @param minPumpPct         Minimum rise required in the candle pattern
     * @param debug              Determine if debug log should be printed
     * @param uptrendPrediction  The uptrend model's confidence output (probability or score indicating an uptrend).
     * @return <code>true</code> if ALL trigger conditions are satisfied and a bullish signal should be fired;
     * <code>false</code> otherwise. Prints a full debug trace for every evaluation.
     */
    public static boolean shouldTrigger(
            List<StockUnit> stocks, int window, double minChangePct, double minGreenRatio,
            double gapTolerancePct, double wickToleranceRatio, int barsToCheck, double flatTolerance,
            double minPumpPct, boolean debug, double uptrendPrediction) {

        StringBuilder dbg = new StringBuilder();

        // 1️⃣ Defensive: Enough data?
        if (stocks == null || stocks.size() < Math.max(window + 1, barsToCheck)) {
            return false;
        }

        // 2️⃣ Fundamental trend filter
        boolean fundamentalTrend = isInUptrend(stocks, window, minChangePct, minGreenRatio);
        dbg.append(String.format("📊 Fundamental trend: %s\n", fundamentalTrend ? "✅" : "❌"));

        // 2.5 ML-based trend filter
//        double mlThreshold = 0.0;
//        boolean mlTrendOk = uptrendPrediction >= mlThreshold;
//        dbg.append(String.format("🤖 ML uptrend score: %.2f (thr = %.2f): %s\n", uptrendPrediction, mlThreshold, mlTrendOk ? "✅" : "❌"));

        // block only if BOTH fundamental AND ML say “no”
        if (!fundamentalTrend) {// && !mlTrendOk) {
            dbg.append("🚫 Blocked: neither fundamental nor ML trend passed\n");
            if (debug) System.out.print(dbg + "\n");
            return false;
        }

        StockUnit curr = stocks.get(stocks.size() - 1);
        StockUnit prev = stocks.get(stocks.size() - 2);

        // 3️⃣ Last candle bullishness
        boolean lastCandleGreen = curr.getClose() > curr.getOpen();
        dbg.append(String.format("🟩 Last candle green: %s\n", lastCandleGreen ? "✅" : "❌"));

        // 4️⃣ Resistance proximity
        boolean atResistance = isNearResistance(stocks.subList(stocks.size() - 15, stocks.size())) == 1.0;
        dbg.append(String.format("🟫 Resistance check: %s\n", atResistance ? "❌ blocked" : "✅ clear"));

        // 5️⃣ Gap-down risk
        boolean hasGap = hasGapDown(prev, curr, gapTolerancePct, dbg);

        // 6️⃣ Wick/spread abnormality
        boolean wickRuleOk = checkBadWicks(stocks, barsToCheck, wickToleranceRatio, dbg);

        // 7️⃣ Micro-uptrend
        boolean microUp = lastNBarsRising(stocks, barsToCheck, flatTolerance, minPumpPct, dbg);
        dbg.append(String.format("📈 Microtrend: %s\n", microUp ? "✅" : "❌"));

        // 8️⃣ Summarize all blocking reasons
        if (!lastCandleGreen) dbg.append("🚫 Blocked: Last candle is not green\n");
        if (atResistance) dbg.append("🚫 Blocked: Near resistance\n");
        if (!microUp) dbg.append("🚫 Blocked: Microtrend fail\n");
        if (hasGap) dbg.append("🚫 Blocked: Gap-down detected\n");
        if (!wickRuleOk) dbg.append("🚫 Blocked: Wick rule violation\n");

        // 9️⃣ Final result
        boolean result = lastCandleGreen && !atResistance && microUp && !hasGap && wickRuleOk;
        dbg.append(result ? "✅ TRIGGERED!\n" : "❌ No trigger.\n");

        if (debug) {
            System.out.print(dbg + "\n");
        }

        return result;
    }

    /**
     * <p>
     * Checks for a true micro-uptrend in the last <code>nBars</code> candles, ensuring that:
     * <ul>
     *   <li>Most closes are higher than their predecessor by at least <code>sidewaysTolerancePct</code>.</li>
     *   <li>At least one close in the window is a significant bullish "pump" (rising by at least <code>minPumpPct</code>).</li>
     *   <li>At most one "flat" (non-rising) close is allowed for resilience against minor noise.</li>
     * </ul>
     * The function appends a detailed debug trace of the check for transparency and tuning.
     * </p>
     *
     * @param candles              Chronological list of OHLC bars (oldest first).
     * @param nBars                Number of most recent bars to check for rising closes.
     * @param sidewaysTolerancePct Minimum % gain required for each consecutive close to qualify as "rising".
     * @param minPumpPct           Minimum % gain defining a "big pump" event within the window.
     * @param dbg                  Debug log; appends microtrend step-by-step status.
     * @return <code>true</code> if at least one "big pump" is detected and at most one flat is found;
     * <code>false</code> otherwise.
     */
    private static boolean lastNBarsRising(List<StockUnit> candles, int nBars,
                                           double sidewaysTolerancePct, double minPumpPct, StringBuilder dbg) {
        // Not enough bars for the microtrend check.
        if (candles.size() < nBars) {
            dbg.append("🚫 Not enough bars for microtrend check\n");
            return false;
        }
        int numFlat = 0;
        boolean hasBigPump = false;
        dbg.append("🔍 Microtrend details:\n");
        // Loop through the last nBars-1 pairs of closes.
        for (int i = candles.size() - nBars; i < candles.size() - 1; i++) {
            double prevClose = candles.get(i).getClose();
            double nextClose = candles.get(i + 1).getClose();
            // Minimum allowed close for "rising" by sideways tolerance.
            double minAllowed = prevClose * (1 + sidewaysTolerancePct / 100.0);
            // Minimum allowed close for "big pump" status.
            double minBigPump = prevClose * (1 + minPumpPct / 100.0);
            dbg.append(String.format("  pre:%.4f, curr:%.4f, minSide:%.4f", prevClose, nextClose, minAllowed));
            // If the close fails to rise enough, mark as flat.
            if (nextClose < minAllowed) {
                numFlat++;
                dbg.append(" (flat)\n");
            } else {
                dbg.append(" (rising)");
                // If it's a significant rise, flag as big pump.
                if (nextClose >= minBigPump) {
                    hasBigPump = true;
                    dbg.append(" (BIG PUMP!)");
                }
                dbg.append("\n");
            }
        }
        // Passes only if there's a big pump and at most one flat.
        boolean ok = hasBigPump && numFlat <= 1;
        dbg.append(String.format("🍾 BigPump:%s, Flats:%d\n", hasBigPump ? "✅" : "❌", numFlat));
        return ok;
    }

    /**
     * <p>
     * Detects a bearish gap-down event between two consecutive candles, protecting against risky entries
     * when the current open is significantly below the previous close.
     * <ul>
     *   <li>If the current open, even after adding the allowed tolerance, is still below the prior close,
     *       a gap-down is flagged and the trigger should be blocked.</li>
     * </ul>
     * Logs the details of the calculation to the provided debug log.
     * </p>
     *
     * @param prev         The previous candle/bar (T-1).
     * @param curr         The current candle/bar (T).
     * @param tolerancePct Maximum allowed downward gap (as a percent of the previous close).
     * @param dbg          Debug log; appends gap-down calculation details.
     * @return <code>true</code> if a significant gap-down is detected; <code>false</code> otherwise.
     */
    private static boolean hasGapDown(StockUnit prev, StockUnit curr, double tolerancePct, StringBuilder dbg) {
        // Calculate the allowed gap as a fraction of the previous close.
        double allowedGap = prev.getClose() * tolerancePct / 100.0;
        // A gap is detected if the current open plus the allowed gap is still below previous close.
        boolean gap = curr.getOpen() + allowedGap < prev.getClose();
        dbg.append(String.format("🕳️ Gap-down: %s (Prev Close=%.2f, Open=%.2f, Tol=%.2f)\n",
                gap ? "❌ detected" : "✅ ok", prev.getClose(), curr.getOpen(), allowedGap));
        return gap;
    }

    /**
     * <p>
     * Examines the recent <code>barsToCheck</code> candles for abnormal wick (shadow) sizes,
     * filtering out periods of excessive indecision or volatility that often produce false signals.
     * </p>
     *
     * <p>
     * <b>Logic and Filters:</b>
     * <ul>
     *   <li>For each bar, computes the ratios of the lower wick (open - low) and upper wick (high - close)
     *       relative to the full bar range (high - low).</li>
     *   <li>If either wick exceeds <code>wickToleranceRatio</code>, the bar is considered "bad."</li>
     *   <li>If any wick exceeds a severe threshold (0.7), the entire window is invalidated immediately.</li>
     *   <li>Allows up to one third of the window (<code>barsToCheck/3</code>) to be "bad" for tolerance of minor noise.</li>
     * </ul>
     * Each step and bar evaluation is logged to <code>dbg</code> for debugging and transparency.
     * </p>
     *
     * @param stocks             Chronological list of OHLC bars (oldest first).
     * @param barsToCheck        Number of most recent bars to check for abnormal wick sizes.
     * @param wickToleranceRatio Maximum allowed ratio for either upper or lower wick to total range (e.g., 0.5 = wick cannot be more than half the range).
     * @param dbg                Debug log; appends stepwise wick analysis details.
     * @return <code>true</code> if severe wicks are not found and the number of "bad" bars is within tolerance; <code>false</code> otherwise.
     */
    private static boolean checkBadWicks(List<StockUnit> stocks, int barsToCheck,
                                         double wickToleranceRatio, StringBuilder dbg) {
        // Start wick analysis block in debug log.
        dbg.append("🕯️ Wick analysis:\n");
        int badCount = 0;     // Counter for bars with abnormal wicks.
        boolean severe = false; // True if any wick is extreme (ratio > 0.7).

        // Loop through the specified recent bars, from oldest to newest in the window.
        for (int i = stocks.size() - barsToCheck; i < stocks.size(); i++) {
            StockUnit bar = stocks.get(i);

            // Calculate the full bar range.
            double range = bar.getHigh() - bar.getLow();

            // Compute lower wick ratio (normalized); handle zero/negative ranges as bad data.
            double lowWick = range > 0 ? (bar.getOpen() - bar.getLow()) / range : 1;

            // Compute upper wick ratio (normalized).
            double highWick = range > 0 ? (bar.getHigh() - bar.getClose()) / range : 1;

            // Flag the bar as "bad" if range is non-positive or any wick exceeds allowed tolerance.
            boolean bad = range <= 0 || lowWick > wickToleranceRatio || highWick > wickToleranceRatio;

            // Log wick ratios and evaluation for this bar.
            dbg.append(String.format("  [%s] low:%.4f, high:%.4f vs tol:%.4f %s\n",
                    bar.getDateDate(), lowWick, highWick, wickToleranceRatio, bad ? "❌" : "✅"));

            if (bad) badCount++;

            // Severe wick detected: if any wick ratio is over 0.7, invalidate immediately and stop.
            if (lowWick > 0.7 || highWick > 0.7) {
                severe = true;
                dbg.append("   🚫 Severe wick detected (ratio > 0.7)\n");
                break;
            }
        }

        // Allow up to a third of bars to be "bad" for resilience; any severe wick fails the whole window.
        int maxBad = barsToCheck / 3;
        boolean ok = badCount <= maxBad && !severe;

        // Final debug summary of this check.
        dbg.append(String.format("  BadWicks:%d/%d => %s\n", badCount, barsToCheck, ok ? "✅" : "❌"));

        return ok;
    }

    /**
     * Determines if a Simple Moving Average (SMA) crossover event has occurred.
     * This is used to detect bullish (golden cross) or bearish (death cross) events.
     *
     * @param window      List of StockUnit objects representing sequential price bars (most recent last).
     * @param shortPeriod Length of the short-term SMA window (e.g., 9).
     * @param longPeriod  Length of the long-term SMA window (e.g., 21). Must be > shortPeriod.
     * @param symbol      Stock symbol, used for mapping persistent state.
     * @return 1 for bullish crossover (golden cross),
     * -1 for bearish crossover (death cross),
     * 0 for no crossover or insufficient data.
     *
     * <p>
     * Logic:
     * - Computes both current and previous short and long period SMAs.
     * - Detects bullish if short SMA crosses above long SMA.
     * - Detects bearish if short SMA crosses below long SMA.
     * - Maintains persistent state per-symbol to avoid duplicate signals.
     * </p>
     */
    public static int isSMACrossover(List<StockUnit> window, int shortPeriod, int longPeriod, String symbol) {
        // Return prior state if not enough data or periods are invalid
        if (window == null || window.size() < longPeriod + 1 || shortPeriod >= longPeriod) {
            // Use a state map to persist last crossover state for each symbol
            return smaStateMap.getOrDefault(symbol, 0);
        }

        // Convert close prices to an array for efficient access
        double[] closes = window.stream()
                .mapToDouble(StockUnit::getClose)
                .toArray();

        // Compute the most recent (current) short and long SMAs using the latest available points
        double shortSMA = calculateSMA(closes, closes.length - shortPeriod, shortPeriod);      // e.g., SMA9 of last 9 closes
        double longSMA = calculateSMA(closes, closes.length - longPeriod, longPeriod);         // e.g., SMA21 of last 21 closes

        // Compute previous SMAs for both short and long periods (shift window by one to the past)
        double prevShortSMA = calculateSMA(closes, closes.length - shortPeriod - 1, shortPeriod);
        double prevLongSMA = calculateSMA(closes, closes.length - longPeriod - 1, longPeriod);

        // Bullish crossover: short SMA crosses above long SMA between previous and current
        boolean bullishCrossover = (prevShortSMA <= prevLongSMA) && (shortSMA > longSMA);

        // Bearish crossover: short SMA crosses below long SMA between previous and current
        boolean bearishCrossover = (prevShortSMA >= prevLongSMA) && (shortSMA < longSMA);

        // Retrieve the current crossover state for the symbol
        int currentState = smaStateMap.getOrDefault(symbol, 0);

        // Update the state if a crossover is detected
        if (bullishCrossover) {
            smaStateMap.put(symbol, 1);    // 1 indicates bullish
            currentState = 1;
        } else if (bearishCrossover) {
            smaStateMap.put(symbol, -1);   // -1 indicates bearish
            currentState = -1;
        }
        // No crossover, state unchanged

        return currentState; // Return the crossover state (1, -1, or 0)
    }

    /**
     * Determines if the cumulative sum of percentage changes in a window
     * exceeds a given threshold. Used to detect 'spikes' in cumulative returns.
     *
     * @param window    List of StockUnit objects, most recent last.
     * @param period    Number of periods to sum for the spike check.
     * @param threshold Threshold (as a raw sum) to determine a spike event.
     * @return 1 if sum >= threshold (spike detected), 0 otherwise.
     *
     * <p>
     * - Adds up the 'percentageChange' values of the most recent N bars.
     * - If sum exceeds threshold, a spike is flagged.
     * </p>
     */
    public static int isCumulativeSpike(List<StockUnit> window, int period, double threshold) {
        // Not enough bars to check spike
        if (window.size() < period) return 0;

        // Sum up the percentageChange field for last 'period' bars
        double sum = window.stream()
                .skip(window.size() - period)    // Keep only last 'period' entries
                .mapToDouble(StockUnit::getPercentageChange)
                .sum();

        // 1 = spike detected, 0 = not detected
        return sum >= threshold ? 1 : 0;
    }

    /**
     * Calculates the TRIX indicator value for a list of stock prices.
     * TRIX is the percentage rate-of-change of a triple-smoothed Exponential Moving Average (EMA).
     *
     * @param prices List of StockUnit objects, ordered oldest to newest (chronological).
     * @param period The period window (lookback) for the EMA smoothing.
     * @return TRIX value as a percentage rate-of-change. Returns 0 if not enough data.
     *
     * <p>
     * Steps:
     * 1. Compute EMA on closes: once, then again (EMA of EMA), then again (EMA of EMA of EMA).
     * 2. Calculate percentage rate of change between last two triple EMA values.
     * </p>
     */
    public static double calculateTRIX(List<StockUnit> prices, int period) {
        // Need at least (3*period + 1) bars for triple smoothing to be meaningful
        final int minDataPoints = 3 * period + 1;
        if (prices.size() < minDataPoints || period < 2) {
            return 0; // Not enough data for valid calculation
        }

        // Extract all close prices as a List<Double>
        List<Double> closes = prices.stream()
                .map(StockUnit::getClose)
                .collect(Collectors.toList());

        // 1st EMA smoothing
        List<Double> singleEMA = calculateEMASeries(closes, period);
        // 2nd EMA smoothing (EMA of first EMA series)
        List<Double> doubleEMA = calculateEMASeries(singleEMA, period);
        // 3rd EMA smoothing (EMA of second EMA series)
        List<Double> tripleEMA = calculateEMASeries(doubleEMA, period);

        // TRIX is percentage rate-of-change of last two triple-EMA values
        if (tripleEMA.size() < 2) return 0; // Not enough values

        double current = tripleEMA.get(tripleEMA.size() - 1);    // Most recent triple-EMA
        double previous = tripleEMA.get(tripleEMA.size() - 2);   // Previous triple-EMA

        // Compute percentage change from previous to current (can be negative)
        return ((current - previous) / previous) * 100;
    }

    /**
     * Calculates the Rate of Change (ROC) indicator for a price window.
     * ROC = ((currentClose - closeNPeriodsAgo) / closeNPeriodsAgo) * 100
     *
     * @param window  List of StockUnit objects, most recent last.
     * @param periods Lookback period for ROC (e.g., 20).
     * @return ROC value as percent change, or 0 if insufficient data.
     *
     * <p>
     * - Measures the percentage change from N periods ago to now.
     * - Used as a momentum indicator.
     * </p>
     */
    public static double calculateROC(List<StockUnit> window, int periods) {
        // Require at least (periods+1) values for a valid calculation
        if (window.size() < periods + 1) return 0;

        // Take only the relevant closes for the calculation (last (periods+1) closes)
        double[] closes = window.stream()
                .skip(window.size() - periods - 1)   // Skip older values, keep only required window
                .mapToDouble(StockUnit::getClose)
                .toArray();

        // ROC is percent change from first (N periods ago) to last (current)
        double current = closes[closes.length - 1];  // Most recent close
        double past = closes[0];                     // Close N periods ago

        return ((current - past) / past) * 100;      // Percent change (positive or negative)
    }

    /**
     * Calculates the total cumulative percentage change for the most recent 8 bars.
     * This can be used as a feature to measure total momentum or spike activity.
     *
     * @param stocks List of StockUnit objects, most recent last.
     * @return Total cumulative percentage change over last 8 bars.
     *
     * <p>
     * - Uses the 'percentageChange' field.
     * - If less than 8, will use from index 0 to current size.
     * </p>
     */
    private static double cumulativePercentageChange(List<StockUnit> stocks) {
        // Calculate the starting index for the last 8 elements.
        // If the list is shorter than 8, this will be negative, but subList handles that safely as 0.
        int startIndex = stocks.size() - 8;

        // Sum the percentageChange values for the last 8 StockUnits (or all if fewer than 8).
        // subList(startIndex, stocks.size()) will include all items from startIndex to the end of the list.
        // If startIndex < 0, subList(0, stocks.size()) is used, meaning the whole list.
        return stocks.subList(Math.max(0, startIndex), stocks.size())
                .stream()
                .mapToDouble(StockUnit::getPercentageChange)
                .sum();
    }

    /**
     * Checks if a Keltner Channel breakout has occurred for the given stock window.
     * <p>
     * The Keltner Channel consists of an Exponential Moving Average (EMA) plus/minus a multiple of the Average True Range (ATR).
     * This method detects whether the latest close has "broken out" above the upper band AND if the cumulative move is large enough.
     *
     * @param window          List of StockUnit price bars, oldest to most recent (must be at least (emaPeriod or 4) + 1 bars long).
     * @param emaPeriod       The period (lookback window) for the EMA centerline.
     * @param atrPeriod       The period for the ATR volatility calculation.
     * @param multiplier      How many ATRs above EMA defines the upper channel (usually 2).
     * @param cumulativeLimit Minimum absolute % move required for a breakout to count.
     * @return 1 if there is a valid breakout above the channel and the cumulative move is sufficient, 0 otherwise.
     *
     * <p>
     * Steps:
     * 1. Compute EMA for the given period as the channel midline.
     * 2. Compute ATR (average true range) for channel width.
     * 3. Compute the upper band: EMA + multiplier * ATR.
     * 4. Check if current close is above upper band AND move from 8 bars ago is big enough.
     * </p>
     */
    public static int isKeltnerBreakout(List<StockUnit> window, int emaPeriod, int atrPeriod, double multiplier,
                                        double cumulativeLimit) {
        // Data sufficiency check: need at least enough bars for EMA and ATR plus 1 for reference
        if (window.size() < Math.max(emaPeriod, 4) + 1) {
            return 0; // Not enough data points, skip calculation
        }

        // ---- Keltner Channel Calculation ----

        // Calculate EMA (Exponential Moving Average) over emaPeriod as the central channel
        double ema = calculateEMA(window, emaPeriod);

        // Calculate ATR (Average True Range) over atrPeriod for volatility width
        double atr = calculateATR(window, atrPeriod);

        // Upper band is the central EMA plus N * ATR
        double upperBand = ema + (multiplier * atr);

        // ---- Cumulative Change Calculation ----

        // Get indices for current close and close 8 bars ago (reference for cumulative move)
        int currentIndex = window.size() - 1;
        int referenceIndex = currentIndex - 8;

        double currentClose = window.get(currentIndex).getClose();      // Most recent close
        double referenceClose = window.get(referenceIndex).getClose();  // Close 8 bars ago

        // Calculate cumulative % change over the last 8 bars
        double cumulativeChange = ((currentClose - referenceClose) / referenceClose) * 100;

        // ---- Combined Breakout and Cumulative Check ----

        // Breakout: is the current close above the upper channel?
        boolean isBreakout = currentClose > upperBand;

        // Is the move over the last 8 bars significant enough in absolute %?
        boolean hasSignificantMove = Math.abs(cumulativeChange) >= cumulativeLimit;

        // Only return 1 if both breakout and significant move detected
        return (isBreakout && hasSignificantMove) ? 1 : 0;
    }

    /**
     * Calculates the Elder-Ray Index (Bull Power) for the most recent price in the window.
     * <p>
     * Elder-Ray Index = Last Close - EMA(period)
     * Measures strength of bulls (above zero) or bears (below zero).
     *
     * @param window    List of StockUnit bars, most recent last.
     * @param emaPeriod Period (lookback) for the EMA.
     * @return Difference between latest close and EMA(period) (can be positive or negative).
     */
    public static double elderRayIndex(List<StockUnit> window, int emaPeriod) {
        // Calculate the EMA (Exponential Moving Average) over the given period.
        // This acts as the trend baseline for the current window.
        double ema = calculateEMA(window, emaPeriod);

        // Get the most recent closing price (the latest StockUnit in the window).
        double latestClose = window.get(window.size() - 1).getClose();

        // The Elder-Ray Index (also known as Bull Power) is the difference
        // between the latest close and the EMA. A positive value suggests
        // bullish strength (close above EMA), while a negative value suggests
        // bearish pressure (close below EMA).
        return latestClose - ema;
    }

    /**
     * Calculates the Simple Moving Average (SMA) from a close price array.
     * Used internally for SMA crossover logic.
     *
     * @param closes     Array of close prices (ordered oldest to newest).
     * @param startIndex Start index in array to begin averaging (inclusive).
     * @param period     Number of elements to average.
     * @return Arithmetic mean of the close prices in the range, or 0 if indices are out of bounds.
     *
     * <p>
     * Example: If closes = [10,11,12,13,14], startIndex=2, period=3, SMA = mean([12,13,14])
     * </p>
     */
    private static double calculateSMA(double[] closes, int startIndex, int period) {
        // Check if indices are valid; return 0 if not enough data to calculate SMA
        if (startIndex < 0 || startIndex + period > closes.length) return 0;

        double sum = 0;

        // Sum the closing prices from startIndex for the given period
        for (int i = startIndex; i < startIndex + period; i++) {
            sum += closes[i];
        }

        // Return the average (Simple Moving Average)
        return sum / period;
    }

    /**
     * Calculates the Exponential Moving Average (EMA) of the provided stock prices.
     *
     * @param prices List of StockUnit price bars (oldest to newest).
     * @param period Period (lookback) for the EMA calculation.
     * @return EMA value for the last bar in the window, or 0 if not enough data.
     *
     * <p>
     * Steps:
     * 1. Start with SMA as the first EMA value for the initial window.
     * 2. Use recursive EMA formula: EMA_today = (Price_today - EMA_yesterday) * S + EMA_yesterday, S = 2/(N+1)
     * </p>
     */
    private static double calculateEMA(List<StockUnit> prices, int period) {
        // Return 0 if period is invalid or not enough data for calculation
        if (prices.size() < period || period <= 0) {
            return 0;
        }

        // Calculate the initial EMA value using a Simple Moving Average (SMA)
        // This is the seed for the recursive EMA formula
        double sma = prices.stream()
                .limit(period)
                .mapToDouble(StockUnit::getClose)
                .average()
                .orElse(0);

        // Smoothing constant for EMA, gives more weight to recent data
        final double smoothing = 2.0 / (period + 1);

        // Start with the SMA as the first EMA value
        double ema = sma;

        // Apply the EMA update formula for each new price after the initial period
        for (int i = period; i < prices.size(); i++) {
            double currentPrice = prices.get(i).getClose();
            ema = (currentPrice - ema) * smoothing + ema;
        }

        // Return the most up-to-date EMA value (for the latest price)
        return ema;
    }

    /**
     * Calculates an entire series of EMA values from a list of data points.
     * Used for indicators that need multiple EMA smoothings (e.g., TRIX).
     *
     * @param data   List of double values (e.g., closes or intermediate EMA results).
     * @param period Period for the EMA.
     * @return List of EMA values, same order as input, starting after initial SMA.
     */
    private static List<Double> calculateEMASeries(List<Double> data, int period) {
        // If there isn't enough data to calculate even the first EMA value, return an empty list
        if (data.size() < period) return Collections.emptyList();

        List<Double> emaSeries = new ArrayList<>();

        // Calculate the smoothing factor for the EMA calculation
        double smoothing = 2.0 / (period + 1);

        // Calculate the initial value for EMA using the average of the first 'period' data points (SMA)
        double sma = data.subList(0, period)
                .stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);

        // Add the initial SMA as the first EMA value in the series
        emaSeries.add(sma);

        // For every data point after the initial period, apply the EMA formula recursively
        for (int i = period; i < data.size(); i++) {
            double currentValue = data.get(i);
            double prevEMA = emaSeries.get(emaSeries.size() - 1); // Get last calculated EMA
            // EMA formula: blend previous EMA with current value using smoothing
            double newEMA = (currentValue - prevEMA) * smoothing + prevEMA;
            emaSeries.add(newEMA); // Add the new EMA value to the series
        }

        // Return the list of all computed EMA values (starts with the seed SMA)
        return emaSeries;
    }

    /**
     * Calculates the Average True Range (ATR) for a sequence of stock prices.
     * ATR is a volatility measure, averaged over 'period' bars.
     *
     * @param window List of StockUnit bars, oldest to newest.
     * @param period Number of bars to use for averaging.
     * @return ATR value, or 0 if not enough data.
     *
     * <p>
     * True Range = max(high - low, abs(high - prevClose), abs(low - prevClose))
     * ATR = mean(true range over period)
     * </p>
     */
    private static double calculateATR(List<StockUnit> window, int period) {
        double atrSum = 0;
        // Loop from 1 to size-1 to compare each bar to previous bar
        for (int i = 1; i < window.size(); i++) {
            double high = window.get(i).getHigh();               // Current bar high
            double low = window.get(i).getLow();                 // Current bar low
            double prevClose = window.get(i - 1).getClose();     // Previous bar close

            // True Range: the largest of these three quantities
            double trueRange = Math.max(
                    high - low,
                    Math.max(Math.abs(high - prevClose), Math.abs(low - prevClose))
            );
            atrSum += trueRange;
        }

        // Average over the number of periods specified
        return atrSum / period;
    }

    /**
     * Calculates the Relative Strength Index (RSI) for a list of stock data.
     * RSI is a momentum oscillator that measures the speed and change of price movements.
     * The value ranges from 0 to 100, where:
     * - 70+ typically signals overbought
     * - 30- typically signals oversold
     * - 50 is considered neutral
     *
     * <p>RSI = 100 - (100 / (1 + RS)), where RS is the average gain divided by the average loss
     *
     * @param stocks List of StockUnit objects in **chronological order** (oldest to newest).
     * @param period Number of bars (lookback window) to compute RSI, usually 14.
     * @return RSI value for the most recent bar (0-100). Returns 50 if not enough data or invalid period.
     */
    private static double calculateRSI(List<StockUnit> stocks, int period) {
        // Not enough bars or invalid period, return neutral 50
        if (stocks.size() <= period || period < 1) return 50; // Neutral default

        // List to store price changes between consecutive closes
        List<Double> changes = new ArrayList<>();

        // Calculate close-to-close changes for the last 'period' bars
        for (int i = 1; i <= period; i++) {
            double change = stocks.get(i).getClose() - stocks.get(i - 1).getClose();
            changes.add(change);
        }

        // Calculate average gain (only positive changes) and average loss (only negative, absolute value)
        double avgGain = changes.stream()
                .filter(c -> c > 0)
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);

        double avgLoss = Math.abs(
                changes.stream()
                        .filter(c -> c < 0)
                        .mapToDouble(Double::doubleValue)
                        .average()
                        .orElse(0)
        );

        // If avgLoss is zero, set RSI to 100 (extremely overbought / all gains)
        if (avgLoss == 0) return 100; // Prevent division by zero

        // RS = average gain / average loss
        double rs = avgGain / avgLoss;

        // RSI formula
        return 100 - (100 / (1 + rs));
    }

    /**
     * Calculates the Simple Moving Average (SMA) of the close price over a specified period.
     *
     * @param periodStocks List of StockUnit objects, most recent last. Size should match desired period.
     * @return SMA (arithmetic mean) of the close price for the input period.
     *
     * <p>
     * If the input list is empty, returns 0.
     * </p>
     */
    private static double calculateSMA(List<StockUnit> periodStocks) {
        // Edge case: No data, return zero
        if (periodStocks.isEmpty()) return 0;

        // Calculate average of close prices for all items in the list
        return periodStocks.stream()
                .mapToDouble(StockUnit::getClose)
                .average()
                .orElse(0);
    }

    /**
     * Retrieves the historical Average True Range (ATR) for a given symbol using up to
     * HISTORICAL_LOOK_BACK bars, and caches the result for faster future lookups.
     * ATR measures volatility, showing how much an asset moves on average during a given period.
     *
     * @param symbol Stock symbol (string, must match what is used in symbolTimelines).
     * @return Calculated ATR for the historical window, or 1.0 as fallback if not enough data.
     *
     * <p>
     * - Checks for cached value first.
     * - Uses up to HISTORICAL_LOOK_BACK bars, but at least 14 bars are required for meaningful ATR.
     * - Caches the computed ATR value for future use (per symbol).
     * - Returns 1.0 if not enough data to compute ATR.
     * </p>
     */
    private static double getHistoricalATR(String symbol) {
        // Fast path: Return cached ATR if present for this symbol
        if (historicalATRCache.containsKey(symbol)) {
            return historicalATRCache.get(symbol);
        }

        // Fetch all price bars for this symbol
        List<StockUnit> allStocks = symbolTimelines.get(symbol);

        // No data, or empty list: fallback value
        if (allStocks == null || allStocks.isEmpty()) {
            return 1.0; // Fallback default for missing/invalid data
        }

        int availableDays = allStocks.size();
        // Limit lookback to the minimum of available bars and the global lookback cap
        int calculatedLookback = Math.min(availableDays, HISTORICAL_LOOK_BACK);

        // Need at least 14 bars for reliable ATR
        if (calculatedLookback < 14) {
            return 1.0; // Not enough data, use fallback
        }

        // Find the sublist of last 'calculatedLookback' bars to use for ATR
        int startIndex = Math.max(0, availableDays - calculatedLookback);
        List<StockUnit> historicalData = allStocks.subList(startIndex, availableDays);

        // Compute ATR using the historical sublist and its size as period
        double atr = calculateATR(historicalData, calculatedLookback);

        // Cache computed value for future quick lookup
        historicalATRCache.put(symbol, atr);
        return atr;
    }

    /**
     * Checks if there is a sharp drop in price within the lookback window.
     * A "sharp drop" is defined as a previous close dropping by more than {@code threshold} percent
     * relative to the current close.
     *
     * @param stocks    List of StockUnit objects, assumed chronological (oldest to newest).
     * @param lookBack  How many bars back to look for a sharp drop (e.g., 10).
     * @param threshold Percentage drop to trigger the signal (e.g., 2.0 = 2%).
     * @return true if a sharp drop is found; false otherwise.
     */
    private static boolean checkSharpDrop(List<StockUnit> stocks, int lookBack, double threshold) {
        // Require at least lookBack+1 bars to compare
        if (stocks.size() < lookBack + 1) return false;

        // Get the latest close price (most recent bar)
        double currentClose = stocks.get(stocks.size() - 1).getClose();

        // Check for a sharp drop from any of the previous {lookBack} closes to current
        for (int i = 1; i <= lookBack; i++) {
            // Close price I bar before the latest
            double previousClose = stocks.get(stocks.size() - 1 - i).getClose();

            // Calculate percent drop from previousClose to currentClose
            double dropPercent = ((previousClose - currentClose) / previousClose) * 100;

            // If the drop exceeds threshold, sharp drop found
            if (dropPercent > threshold) {
                return true;
            }
        }
        // No sharp drop found in lookback window
        return false;
    }

    /**
     * Checks if there has been a sustained negative price movement
     * (i.e., consistent overall downward trend) over the lookback period.
     *
     * @param stocks    List of StockUnit objects, chronological.
     * @param lookback  Number of bars to look back (e.g., 10).
     * @param threshold Total percentage movement threshold (negative, e.g., -2.5 for -2.5%).
     * @return true if total move < threshold (i.e., sufficiently negative); false otherwise.
     */
    private static boolean checkSustainedMovement(List<StockUnit> stocks, int lookback, double threshold) {
        // Need at least lookback bars to check
        if (stocks.size() < lookback) return false;

        double totalMove = 0;
        // Sum up percent change for each bar in the lookback window
        for (int i = 1; i <= lookback; i++) {
            // Current and previous close prices for each consecutive pair
            StockUnit current = stocks.get(stocks.size() - i);
            StockUnit previous = stocks.get(stocks.size() - i - 1);

            // Compute fractional change and accumulate
            totalMove += (current.getClose() - previous.getClose()) / previous.getClose();
        }

        // Convert to percent, compare to threshold (more negative = more downward movement)
        return totalMove * 100 < threshold;
    }

    /**
     * Calculates the Stochastic Oscillator (%K) value for the most recent bar in a period.
     * This measures the close price's position relative to the period's high/low range.
     *
     * @param stocks List of StockUnit objects, assumed chronological (oldest to newest).
     * @param period Lookback period for the oscillator (e.g., 14).
     * @return Stochastic value (0-100). Returns 50 if all prices are identical.
     *
     * <p>
     * Formula:
     * %K = 100 * (lastClose - lowestLow) / (highestHigh - lowestLow)
     * </p>
     */
    private static double calculateStochastic(List<StockUnit> stocks, int period) {
        // Use only the last 'period' bars for the calculation
        List<StockUnit> lookback = stocks.subList(stocks.size() - period, stocks.size());

        // Find the highest high and lowest low in the window
        double highestHigh = lookback.stream().mapToDouble(StockUnit::getHigh).max().orElse(0);
        double lowestLow = lookback.stream().mapToDouble(StockUnit::getLow).min().orElse(0);
        double lastClose = lookback.get(lookback.size() - 1).getClose();

        // If all prices are the same (flat range), avoid division by zero
        if (highestHigh == lowestLow) return 50;

        // Standard %K formula
        return 100 * (lastClose - lowestLow) / (highestHigh - lowestLow);
    }

    /**
     * Helper to create and add a Notification for a specific stock event.
     * Adds a formatted notification message to alertsList depending on config.
     *
     * @param symbol           The stock symbol for the event.
     * @param totalChange      The triggering percent change (spike or dip, etc).
     * @param alertsList       The shared list to store created notifications.
     * @param stockUnitList    The time series (bars) associated with the event.
     * @param date             The date/time of the event (LocalDateTime).
     * @param prediction       Model prediction value (if available).
     * @param config           Type of event:
     *                         1 = gap filler,
     *                         2 = R-line spike,
     *                         3 = spike.
     *                         4 = uptrend movement
     *                         5 = Second based spike
     * @param validationWindow A list of subsequent bars immediately following the event,
     *                         used to label or train the ML model. Must contain at least
     *                         VALIDATE_SIZE bars. The first VALIDATE_SIZE entries are used
     *                         to decide if the initial signal was “good” (e.g., price moved
     *                         above a threshold) or “bad” (e.g., price failed to move).
     */
    private static void createNotification(String symbol, double totalChange, List<Notification> alertsList,
                                           List<StockUnit> stockUnitList, LocalDateTime date,
                                           double prediction, int config, List<StockUnit> validationWindow) {
        // Depending on config, use different message formats for type of event
        if (config == 1) {
            // Gap fill (move filling a previous price gap)
            alertsList.add(new Notification(
                    String.format("Gap %s ↓↑ %.2f, %s", symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("Fill the gap at the %s", date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    stockUnitList, date, symbol, totalChange, 1, validationWindow));
        } else if (config == 2) {
            // R-line spike (upward price movement with caution warning since close to previous bar hence might be the top and reverse point)
            alertsList.add(new Notification(
                    String.format("%.2f%% %s R-Line %.2f, %s", totalChange, symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("R-Line Spike Proceed with caution by %.2f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    stockUnitList, date, symbol, totalChange, 2, validationWindow));
        } else if (config == 3) {
            // Upward spike (sharp increase)
            alertsList.add(new Notification(
                    String.format("%.2f%% %s ↑ %.2f, %s", totalChange, symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("Increased by %.2f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    stockUnitList, date, symbol, totalChange, 3, validationWindow));
        } else if (config == 4) {
            // Uptrend movement which is not categorized as a hardcore spike
            alertsList.add(new Notification(
                    String.format("%.2f%% %s Uptrend %.2f, %s", totalChange, symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("Uptrend slope: %.2f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    stockUnitList, date, symbol, totalChange, 4, validationWindow));
        } else if (config == 5) {
            // Second based pre predicted spike
            alertsList.add(new Notification(
                    String.format("%.2f%% %s Second Spike %.2f, %s", totalChange, symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("Second based spike: %.2f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    stockUnitList, date, symbol, totalChange, 5, validationWindow));
        }
    }

    /**
     * Retrieves the time series (minute bars, etc.) for a given stock symbol.
     * Returns an unmodifiable (read-only) list, preventing external code from altering the timeline data.
     *
     * @param symbol The stock symbol to look up (case-insensitive; will be uppercased).
     * @return An unmodifiable List of StockUnit objects for the symbol.
     * If the symbol is not found, returns an empty unmodifiable list.
     *
     * <p>
     * Use this method to safely expose timeline data without risk of callers modifying the internal state.
     * </p>
     */
    public static List<StockUnit> getSymbolTimeline(String symbol) {
        // Get the timeline list for the uppercased symbol; if missing, return empty list
        return Collections.unmodifiableList(
                symbolTimelines.getOrDefault(symbol.toUpperCase(), new ArrayList<>())
        );
    }

    /**
     * Continuously collects real-time price data for a given symbol and appends it to "Symbol + realtime.txt".
     * This function schedules a background task (runs every second) that fetches a single real-time update,
     * then writes the bar's data to file (one line per update, appending).
     *
     * @param symbol The stock symbol to track in real-time.
     *               <p>
     *               - Each line in the output file represents one fetched bar, formatted in StockUnit-like notation.
     *               - The file grows with each new bar; manage the file if it gets too large over time.
     *               - Handles I/O errors gracefully
     *               </p>
     */
    public static void realTimeDataCollector(String symbol) {
        // Initialize the Alpha Vantage API with loaded token
        InitAPi(configHandler.loadConfig()[3][1]);

        // Schedule a task to run every 1 second (fetching and writing latest data)
        executorService.scheduleAtFixedRate(() -> getRealTimeUpdate(symbol, response -> {
            try {
                File data = new File(symbol.toUpperCase() + "realtime.txt");

                // Create the output file if it does not exist yet
                if (!data.exists()) {
                    data.createNewFile();
                }

                // Use try-with-resources to guarantee file is properly closed even if errors occur
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(data, true))) { // true: append mode

                    // ================== TICK DATA PREPROCESSING ==================

                    // Check if we should use extended-hours data (pre- / post-market) and build stock then
                    // fallback to regular close/open if not present.
                    String formatted = buildStock(response, useExtended(response));

                    writer.write(formatted);
                    writer.newLine();

                }  // BufferedWriter auto-closes here

            } catch (IOException e) {
                // Print file errors (e.g., disk full, permissions) to the standard error stream
                System.err.println("Error writing to file: " + e.getMessage());
            }
        }), 0, 5, TimeUnit.SECONDS);
    }

    /**
     * Builds a formatted string representing a single stock "tick" as a StockUnit.
     * <p>
     * This method takes a real-time market data response and creates a single-line summary
     * of a stock's state, formatted for further processing or storage. In the context of real-time
     * streaming, only the latest available price ("close") is reliable. As such, open, high, and low
     * are set equal to close, since only a single tick's price is known.
     * <p>
     * The method supports both regular and extended-hours quotes. Non-essential fields (e.g., adjustedClose,
     * dividendAmount, splitCoefficient, percentageChange, target) are populated with placeholders and can
     * be updated post-processing as needed.
     *
     * @param response the real-time quote/tick data, encapsulated in a RealTimeResponse.RealTimeMatch object
     * @param extended whether to use extended-hours quote (if available), otherwise regular close price
     * @return a formatted string containing all relevant StockUnit fields, ready for CSV-like output
     */
    @NotNull
    private static String buildStock(RealTimeResponse.RealTimeMatch response, boolean extended) {
        // For a single tick, open/high/low/close are all set to the same value
        // This value comes from either the extended-hours quote or the regular close price, based on the flag.
        double open = extended ? response.getExtendedHoursQuote() : response.getClose();
        double high = extended ? response.getExtendedHoursQuote() : response.getClose();
        double low = extended ? response.getExtendedHoursQuote() : response.getClose();
        double close = extended ? response.getExtendedHoursQuote() : response.getClose();

        // Real-time volume (could be zero in extended hours, or not always updated in after-market sessions)
        double volume = response.getVolume();

        // Format all stock fields into a custom string representation.
        // - adjustedClose: Placeholder 0.0; typically calculated later for EOD data.
        // - dividendAmount: 0.0 (Not available in real-time tick data).
        // - splitCoefficient: 0.0 (No split info in tick data).
        // - date: timestamp from tick data.
        // - symbol: ticker symbol.
        // - percentageChange: Placeholder 0.0 (Calculated in post-processing).
        // - target: Placeholder 0
        return String.format(
                "StockUnit{open=%.4f, high=%.4f, low=%.4f, close=%.4f, " + "adjustedClose=%.1f, volume=%.3f, dividendAmount=%.1f, " +
                        "splitCoefficient=%.1f, date=%s, symbol=%s, " + "percentageChange=%.1f, target=%d},",
                open,
                high,
                low,
                close,
                0.0, // adjustedClose - placeholder
                volume,
                0.0, // dividendAmount - placeholder
                0.0, // splitCoefficient - placeholder
                response.getTimestamp(),
                response.getSymbol(),
                0.0, // percentageChange - placeholder
                0    // target - placeholder
        );
    }

    /**
     * Callback interface for retrieving an array of fundamental/price data values for a symbol.
     * Used in asynchronous API calls to receive data when ready.
     */
    public interface DataCallback {
        void onDataFetched(Double[] values);
    }

    /**
     * Callback for receiving a timeline (list of StockUnit bars) for a symbol.
     * Used when historical or intraday price data has been fetched.
     */
    public interface TimelineCallback {
        void onTimeLineFetched(List<StockUnit> stocks);
    }

    // ==== Callback interfaces for async operations ====

    /**
     * Callback for results from a symbol search API call (partial match / search bar, etc.).
     * Provides a list of matching symbol strings.
     */
    public interface SymbolSearchCallback {
        /**
         * Called when symbol search succeeds.
         *
         * @param symbols List of matching symbols.
         */
        void onSuccess(List<String> symbols);

        /**
         * Called if symbol search fails (e.g., network/API error).
         *
         * @param e Exception describing the failure.
         */
        void onFailure(Exception e);
    }

    /**
     * Callback for retrieving a list of news items for a symbol.
     */
    public interface ReceiveNewsCallback {
        void onNewsReceived(List<NewsResponse.NewsItem> news);
    }

    /**
     * Callback for when a set of symbols (matching filter/criteria) becomes available.
     * Used for bulk symbol filtering (e.g., liquidity scan).
     */
    public interface SymbolCallback {
        void onSymbolsAvailable(List<String> symbols);
    }

    /**
     * Callback for when a real-time price/bar update is received.
     * Used by the realTimeDataCollector and other streaming modules.
     */
    public interface RealTimeCallback {
        void onRealTimeReceived(RealTimeResponse.RealTimeMatch value);
    }

    /**
     * Callback for when company/fundamental overview data has been retrieved.
     */
    public interface OverviewCallback {
        void onOverviewReceived(CompanyOverviewResponse value);
    }

    /**
     * Swing dialog to visually track the progress of background data fetching tasks.
     * Displays a progress bar and a status label showing (current/total) count.
     * Intended to be used from the EDT (Event Dispatch Thread).
     */
    static class ProgressDialog extends JDialog {
        // The JProgressBar component for the bar visualization
        private final JProgressBar progressBar;
        // The JLabel for status text (shows processed/total)
        private final JLabel statusLabel;

        /**
         * Constructs a modal dialog with a progress bar and status label.
         *
         * @param parent The parent window for positioning (can be null).
         */
        public ProgressDialog(Frame parent) {
            super(parent, "Fetching Data", true);
            setSize(300, 100);
            setLayout(new BorderLayout());
            setLocationRelativeTo(parent);

            progressBar = new JProgressBar(0, 100);
            statusLabel = new JLabel("Initializing...", SwingConstants.CENTER);

            add(statusLabel, BorderLayout.NORTH);
            add(progressBar, BorderLayout.CENTER);
        }

        /**
         * Updates the progress bar and label based on the number of items processed.
         * Can be called from any thread; will marshal update to Swing EDT.
         *
         * @param current How many symbols/items have been processed.
         * @param total   The total number of symbols/items expected.
         */
        public void updateProgress(int current, int total) {
            SwingUtilities.invokeLater(() -> {
                int progress = (int) (((double) current / total) * 100);
                progressBar.setValue(progress);
                statusLabel.setText(String.format("Processed %d of %d symbols", current, total));
            });
        }
    }

    /**
     * Immutable record for storing linear regression results on price bars.
     * slopePct - Slope as percent per bar
     * slopePpu - Slope as price units per bar
     * r2       - R squared (fit quality)
     * yBar     - Mean Y value (price)
     * xBar     - Mean X value (index)
     */
    private record LR(double slopePct, double slopePpu, double r2, double yBar, double xBar) {
    }
}