package org.crecker;

import com.crazzyghost.alphavantage.AlphaVantage;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.parameters.Interval;
import com.crazzyghost.alphavantage.parameters.OutputSize;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.crazzyghost.alphavantage.timeseries.response.TimeSeriesResponse;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import static org.crecker.mainDataHandler.CACHE_DIR;
import static org.crecker.mainDataHandler.symbolTimelines;

public class dataTester {

    /**
     * Pulls new intraday data for a given stock symbol from the Alpha Vantage API.
     * Writes the data to cache for later testing/training use.
     *
     * @param symbol the stock ticker (e.g., "AAPL") for which to fetch data
     */
    public static void getData(String symbol) {
        String apiKey = "2NN1RGFV3V34ORCZ"; // Alpha Vantage API key

        // Build the API config (timeout set to 10 seconds)
        Config cfg = Config.builder()
                .key(apiKey)
                .timeOut(10)
                .build();

        // Initialise the Alpha Vantage API singleton
        AlphaVantage.api().init(cfg);

        // Build and execute the intraday timeSeries API call for the given symbol
        AlphaVantage.api()
                .timeSeries()
                .intraday()
                .forSymbol(symbol)
                .interval(Interval.ONE_MIN)            // Request 1-minute intervals
                .outputSize(OutputSize.FULL)           // Request full available output (not just latest)
                .entitlement("realtime")                // allow real time requests
                .onSuccess(e -> {                      // On success, handle the response
                    try {
                        handleSuccess((TimeSeriesResponse) e); // Process and save data to file
                    } catch (IOException ex) {
                        ex.printStackTrace();                 // Print error if unable to write to file
                        throw new RuntimeException(ex);
                    }
                })
                .onFailure(mainDataHandler::handleFailure)     // Delegate failure to main handler
                .fetch();                                     // Fire off the request
    }

    /**
     * Handles a successful API response.
     * <p>
     * Writes the received time series data to disk for caching.
     *
     * @param response AlphaVantage time series response object
     * @throws IOException if file write fails
     */
    public static void handleSuccess(TimeSeriesResponse response) throws IOException {
        // Save the stock data to cache in reverse chronological order (new to old)
        BufferedWriter bufferedWriter = getBufferedWriter(response);
        bufferedWriter.close(); // Close stream to release file handle
    }

    /**
     * Gets a BufferedWriter that writes the stock data to a file named after the stock symbol.
     *
     * @param response API response containing metadata and stock data units
     * @return BufferedWriter (already written to, just needs closing)
     * @throws IOException if any file IO operation fails
     */
    private static BufferedWriter getBufferedWriter(TimeSeriesResponse response) throws IOException {
        // Ensure the cache directory exists, create if it does not
        File cacheDir = new File(CACHE_DIR);
        if (!cacheDir.exists()) cacheDir.mkdirs();

        // Construct filename as "<SYMBOL>.txt" in uppercase for consistency
        File data = new File(cacheDir, response.getMetaData().getSymbol().toUpperCase() + ".txt");

        // If the file does not exist, create it; this ensures we don't try to write to a non-existent file
        if (!data.exists()) {
            data.createNewFile();
        }

        // Open the file for writing using BufferedWriter for efficient output
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(data));
        // Convert the list of StockUnits to a string array representation and write it as one line
        bufferedWriter.write(Arrays.toString(response.getStockUnits().toArray()));
        bufferedWriter.flush(); // Ensure all data is physically written to disk
        return bufferedWriter;
    }

    /**
     * Parses a StockUnit from its string representation as returned by toString().
     *
     * @param stockUnitString The string representation of a StockUnit (as written in cache)
     * @return StockUnit object reconstructed from the string data
     */
    public static StockUnit parseStockUnit(String stockUnitString) {
        // Remove "StockUnit{" prefix from the string (assuming default toString() output)
        stockUnitString = stockUnitString.replace("StockUnit{", "").trim();

        // Split the string into its individual fields, separated by commas
        String[] attributes = stockUnitString.split(", ");

        // Each attribute is of form "field=value", so we split by '=' and parse each value accordingly
        double open = Double.parseDouble(attributes[0].split("=")[1]);
        double high = Double.parseDouble(attributes[1].split("=")[1]);
        double low = Double.parseDouble(attributes[2].split("=")[1]);
        double close = Double.parseDouble(attributes[3].split("=")[1]);
        double adjustedClose = Double.parseDouble(attributes[4].split("=")[1]);
        double volume = Double.parseDouble(attributes[5].split("=")[1]);
        double dividendAmount = Double.parseDouble(attributes[6].split("=")[1]);
        double splitCoefficient = Double.parseDouble(attributes[7].split("=")[1]);
        String dateTime = attributes[8].split("=")[1]; // The time stamp

        // Use StockUnit.Builder for type safety and immutability
        return new StockUnit.Builder()
                .open(open)
                .high(high)
                .low(low)
                .close(close)
                .adjustedClose(adjustedClose)
                .volume(volume)
                .dividendAmount(dividendAmount)
                .splitCoefficient(splitCoefficient)
                .time(dateTime)
                .build();
    }

    /**
     * Calculates and updates the percentage price change between consecutive stock points
     * for each stock symbol's timeline.
     * <p>
     * If the calculated percentage change is too large (>=14%), it reuses the previous value.
     * <p>
     * Updates each StockUnit in-place with the calculated change.
     */
    public static void calculateStockPercentageChange() {
        // Iterate over all timelines (per symbol)
        symbolTimelines.forEach((symbol, timeline) -> {
            // Ignore symbols with too little data to compare (less than 2 points)
            if (timeline.size() < 2) {
                return;
            }

            // Calculate percentage change for each consecutive pair of StockUnits
            for (int i = 1; i < timeline.size(); i++) {
                StockUnit current = timeline.get(i);   // Current time step
                StockUnit previous = timeline.get(i - 1); // Previous time step

                // Avoid division by zero and nonsensical changes
                if (previous.getClose() > 0) {
                    double change = ((current.getClose() - previous.getClose()) / previous.getClose()) * 100;
                    // If the computed change is an outlier (>= 14%), fallback to previous change (e.g., for bad ticks)
                    change = Math.abs(change) >= 14 ? previous.getPercentageChange() : change;
                    current.setPercentageChange(change); // Save calculated value to the object
                }
            }
        });
    }
}