package org.crecker;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Map;

public class csvDataGen {
    private static final String CSV_FILE = "features_data.csv";
    private static final Object headerLock = new Object(); // Lock object for synchronization

    public static void writeHeaderIfNeeded(Map<String, Map<String, Double>> indicatorToIndex) throws IOException {
        File file = new File(CSV_FILE);
        if (!file.exists()) {
            synchronized (headerLock) { // Ensure atomic check-and-write
                if (!file.exists()) { // Double-check inside the synchronized block
                    try (BufferedWriter writer = new BufferedWriter(new FileWriter(file, true))) {
                        String header = "Date,Prediction," + String.join(",", indicatorToIndex.keySet());
                        writer.write(header);
                        writer.newLine();
                    }
                }
            }
        }
    }

    // Write a row of features to the CSV
    public static void appendFeaturesToCSV(float[] features, Date date, double prediction) throws IOException {
        // Format the date to a readable string format (e.g., "yyyy-MM-dd HH:mm:ss")
        String formattedDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(date);

        // Round each feature to 6 decimal places and convert to string
        String row = Arrays.stream(toDoubleArray(features))
                .mapToObj(f -> String.format("%.6f", f))
                .reduce((a, b) -> a + "," + b)
                .orElse("");

        // Prepend the date to the row
        row = formattedDate + "," + String.format("%.6f", prediction) + "," + row;

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(CSV_FILE, true))) {
            writer.write(row);
            writer.newLine();
        }
    }

    // Convert float[] to double[]
    private static double[] toDoubleArray(float[] features) {
        double[] doubles = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            doubles[i] = features[i];
        }
        return doubles;
    }

    public static void saveFeaturesToCSV(float[] features, Map<String, Map<String, Double>> indicatorToIndex, Date dateDate, double prediction) {
        try {
            // Write header only once if file doesn't exist
            writeHeaderIfNeeded(indicatorToIndex);
            // Append feature row
            appendFeaturesToCSV(features, dateDate, prediction);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
