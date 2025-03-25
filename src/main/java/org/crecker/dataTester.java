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

import static org.crecker.mainDataHandler.symbolTimelines;

public class dataTester {
    //method for pulling new data from server for tests and training
    public static void getData(String symbol) {
        String apiKey = "2NN1RGFV3V34ORCZ";

        // Configure the API client
        Config cfg = Config.builder()
                .key(apiKey)
                .timeOut(10) // Timeout in seconds
                .build();

        // Initialize the Alpha Vantage API
        AlphaVantage.api().init(cfg);

        AlphaVantage.api()
                .timeSeries()
                .intraday()
                .forSymbol(symbol)
                .interval(Interval.ONE_MIN)
                .outputSize(OutputSize.FULL)
                .onSuccess(e -> {
                    try {
                        handleSuccess((TimeSeriesResponse) e);
                    } catch (IOException ex) {
                        ex.printStackTrace();
                        throw new RuntimeException(ex);
                    }
                })
                .onFailure(mainDataHandler::handleFailure)
                .fetch();
    }

    public static void handleSuccess(TimeSeriesResponse response) throws IOException {
        // This generates some test data since we don't have unlimited API access
        BufferedWriter bufferedWriter = getBufferedWriter(response); //in reversed format (new to old)
        bufferedWriter.close(); // Close the BufferedWriter to free system resources
    }

    private static BufferedWriter getBufferedWriter(TimeSeriesResponse response) throws IOException {
        File data = new File(response.getMetaData().getSymbol().toUpperCase() + ".txt"); // Create output file

        // Check if the file already exists
        if (!data.exists()) {
            // If the file does not exist, create a new file
            data.createNewFile(); // May throw IOException if it fails
        }

        // Initialize BufferedWriter to write to the file
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(data)); // Create a BufferedWriter to write to the file
        bufferedWriter.write(Arrays.toString(response.getStockUnits().toArray())); // Write the Stock units data to the file as a string
        bufferedWriter.flush(); // Flush the writer to ensure all data is written to the file
        return bufferedWriter;
    }

    public static StockUnit parseStockUnit(String stockUnitString) {
        // Remove "StockUnit{" from the beginning of the string
        stockUnitString = stockUnitString.replace("StockUnit{", "").trim();

        // Split the Stock unit attributes by commas
        String[] attributes = stockUnitString.split(", ");

        // Parse each attribute
        double open = Double.parseDouble(attributes[0].split("=")[1]);
        double high = Double.parseDouble(attributes[1].split("=")[1]);
        double low = Double.parseDouble(attributes[2].split("=")[1]);
        double close = Double.parseDouble(attributes[3].split("=")[1]);
        double adjustedClose = Double.parseDouble(attributes[4].split("=")[1]);
        double volume = Double.parseDouble(attributes[5].split("=")[1]);
        double dividendAmount = Double.parseDouble(attributes[6].split("=")[1]);
        double splitCoefficient = Double.parseDouble(attributes[7].split("=")[1]);
        String dateTime = attributes[8].split("=")[1];

        // Use the Builder to create the StockUnit object
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

    public static void calculateStockPercentageChange() {
        symbolTimelines.forEach((symbol, timeline) -> {
            if (timeline.size() < 2) {
                return;
            }

            for (int i = 1; i < timeline.size(); i++) {
                StockUnit current = timeline.get(i);
                StockUnit previous = timeline.get(i - 1);

                if (previous.getClose() > 0) {
                    double change = ((current.getClose() - previous.getClose()) / previous.getClose()) * 100;
                    change = Math.abs(change) >= 14 ? previous.getPercentageChange() : change;
                    current.setPercentageChange(change);
                }
            }

        });
    }
}