package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.CandlestickRenderer;
import org.jfree.chart.ui.Layer;
import org.jfree.data.xy.DefaultOHLCDataset;
import org.jfree.data.xy.OHLCDataItem;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.crecker.pLTester.dumpNotifications;

/**
 * A Swing-based UI for manually labeling notifications as “Good” or “Bad”
 * by displaying a candlestick chart for each notification’s historical and future stock windows.
 */
public class NotificationLabelingUI extends JFrame {
    /**
     * The ordered list of notifications to present to the user.
     */
    private final List<Notification> notifications;
    /**
     * Index of the currently displayed notification (0-based).
     */
    private int currentIndex = 0;

    // UI components
    /**
     * Displays “Notification i of N” at the top.
     */
    private final JLabel counterLabel = new JLabel();
    /**
     * Holds the JFreeChart chart showing price data.
     */
    private final transient ChartPanel chartPanel;
    /**
     * Button for marking the current notification as “Good” (target = 1).
     */
    private final JButton goodButton;
    /**
     * Button for marking the current notification as “Bad” (target = 0).
     */
    private final JButton badButton;

    /**
     * Constructor.
     *
     * @param notifications The ordered list of {@link Notification} objects to label.
     */
    public NotificationLabelingUI(List<Notification> notifications) {
        super("Notification Labeling Suite"); // Set window title
        this.notifications = notifications;   // Store reference to list

        // Basic JFrame setup:
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);         // Exit on close
        setLayout(new BorderLayout());                         // Use BorderLayout for arrangement
        setPreferredSize(new Dimension(1000, 700));            // Suggest window size

        // 1) Top: counter “Notification i of N”
        counterLabel.setHorizontalAlignment(JLabel.CENTER);     // Center-align the text
        add(counterLabel, BorderLayout.NORTH);                  // Add to top region

        // 2) Center: placeholder for chartPanel; initialize with null chart
        chartPanel = new ChartPanel(null);                      // Create chart panel (no chart yet)
        add(chartPanel, BorderLayout.CENTER);                   // Add to center region

        // 3) South: button panel with “Good” (green) and “Bad” (red)
        JPanel buttonPanel = new JPanel();                      // Container for buttons
        goodButton = new JButton("Good");                       // Create “Good” button
        goodButton.setBackground(new Color(0x00AA00));          // Dark green background
        goodButton.setForeground(Color.WHITE);                  // White text
        goodButton.setOpaque(true);                             // Make background visible
        goodButton.setBorderPainted(false);                     // Remove default border

        badButton = new JButton("Bad");                         // Create “Bad” button
        badButton.setBackground(new Color(0xCC0000));           // Dark red background
        badButton.setForeground(Color.WHITE);                    // White text
        badButton.setOpaque(true);                              // Make background visible
        badButton.setBorderPainted(false);                      // Remove default border

        buttonPanel.add(goodButton);                             // Add “Good” to panel
        buttonPanel.add(badButton);                              // Add “Bad” to panel
        add(buttonPanel, BorderLayout.SOUTH);                    // Add panel to bottom region

        // Wire up button actions:
        goodButton.addActionListener(e -> labelCurrentNotification(1)); // Label as Good
        badButton.addActionListener(e -> labelCurrentNotification(0));  // Label as Bad

        pack();                          // Pack components to preferred sizes
        setLocationRelativeTo(null);     // Center the window on screen

        // Render the first chart & counter immediately
        updateUIForCurrent();
    }

    /**
     * Updates the counter label and the chartPanel for the notification at {@code currentIndex}.
     * <p>
     * If all notifications have been labeled (currentIndex >= size), disables buttons and shows a “finished” message.
     * </p>
     */
    private void updateUIForCurrent() {
        // If we’ve reached or passed the end of the list:
        if (currentIndex >= notifications.size()) {
            // Display finished message and disable UI elements
            counterLabel.setText("All notifications labeled!"); // Show final text
            chartPanel.setChart(null);                           // Clear any existing chart
            goodButton.setEnabled(false);                        // Disable Good button
            badButton.setEnabled(false);                         // Disable Bad button
            return;                                               // Done
        }

        // Otherwise, get the current notification to display:
        Notification notif = notifications.get(currentIndex);

        // Update counter text to “Notification X of N”
        counterLabel.setText(String.format(
                "Notification %d of %d",
                currentIndex + 1,           // 1-based index for user
                notifications.size()        // Total count
        ));

        // Create a candlestick chart for this notification using historical and future windows
        JFreeChart chart = createCandlestickChart(
                notif.getStockUnitList(),           // Historical list of StockUnit
                notif.getValidationWindow()         // Future‐window list (up to 10 bars)
        );
        chartPanel.setChart(chart);             // Set chart into the panel for display
    }

    /**
     * Called when the user clicks “Good” or “Bad” to label the current Notification.
     * <p>
     * Sets the current notification’s target, writes it to disk (via dumpNotifications),
     * and advances to the next notification.
     * </p>
     *
     * @param targetValue 1 indicates “Good”, 0 indicates “Bad”.
     */
    private void labelCurrentNotification(int targetValue) {
        // Retrieve the current notification object
        Notification notif = notifications.get(currentIndex);
        // Assign the user’s label to the notification
        notif.setTarget(targetValue);

        try {
            // Append this single notification to the persistent store (e.g., CSV or JSON file)
            dumpNotifications(new ArrayList<>(Collections.singleton(notif)));
        } catch (IOException ex) {
            ex.printStackTrace();
            // Note: We choose to continue even if writing fails, but could show an error dialog
        }

        // Advance to the next notification index
        currentIndex++;
        // Refresh UI for the newly current notification (or end state)
        updateUIForCurrent();
    }

    /**
     * Creates a candlestick chart with two distinct background regions:
     * <ul>
     *     <li>Historical region (stockUnits) shaded light gray</li>
     *     <li>Future region (validationWindow) shaded pale yellow</li>
     * </ul>
     *
     * @param historical   The list of {@link StockUnit} representing the past window (chronological).
     * @param futureWindow The list of {@link StockUnit} representing the next up to 10 bars (future period).
     * @return A {@link JFreeChart} instance showing the combined OHLC data.
     */
    private JFreeChart createCandlestickChart(
            List<StockUnit> historical,
            List<StockUnit> futureWindow
    ) {
        // 1) Convert both historical and futureStock lists into a single array of OHLCDataItem
        int totalBars = historical.size() + futureWindow.size();   // Total bars to plot
        OHLCDataItem[] dataItems = new OHLCDataItem[totalBars];    // Allocate array

        // Populate dataItems with historical bars first
        for (int i = 0; i < historical.size(); i++) {
            StockUnit u = historical.get(i);                   // Get ith historical StockUnit
            double open = u.getOpen();                         // Opening price
            double high = u.getHigh();                         // Highest price
            double low = u.getLow();                           // Lowest price
            double close = u.getClose();                       // Closing price
            double volume = u.getVolume();                     // Traded volume
            // Convert the LocalDateTime in StockUnit to java.util.Date (date-only precision)
            dataItems[i] = new OHLCDataItem(
                    u.getDateDate(), open, high, low, close, volume
            );
        }
        // Populate dataItems with future bars immediately following the historical ones
        for (int j = 0; j < futureWindow.size(); j++) {
            StockUnit u = futureWindow.get(j);                 // Get jth future StockUnit
            double open = u.getOpen();                         // Opening price
            double high = u.getHigh();                         // Highest price
            double low = u.getLow();                           // Lowest price
            double close = u.getClose();                       // Closing price
            double volume = u.getVolume();                     // Traded volume
            dataItems[historical.size() + j] = new OHLCDataItem(
                    u.getDateDate(), open, high, low, close, volume
            );
        }

        // 2) Build a DefaultOHLCDataset from our dataItems
        String seriesKey = "Price Series";                      // Label for the dataset
        DefaultOHLCDataset dataset = new DefaultOHLCDataset(seriesKey, dataItems);

        // 3) Create the candlestick chart (no legend, null title)
        JFreeChart chart = ChartFactory.createCandlestickChart(
                null,                // No chart title
                "Date",              // X-axis label
                "Price",             // Y-axis label
                dataset,             // The OHLC dataset we just created
                false                // Do not show a legend
        );

        // 4) Customize the plot: add colored background intervals
        XYPlot plot = (XYPlot) chart.getPlot();                // Extract plot from chart
        plot.setBackgroundPaint(Color.WHITE);                  // Set plain white background

        // Configure Y-axis (price axis) to auto-range, excluding zero if needed
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);                          // Auto-scale the range
        rangeAxis.setAutoRangeIncludesZero(false);             // Don’t force zero in range

        // a) Mark historical region with light gray
        // Determine start and end Dates for historical region
        java.util.Date histStart = dataItems[0].getDate();                              // First bar’s date
        java.util.Date histEnd = dataItems[historical.size() - 1].getDate();           // Last historical bar’s date
        IntervalMarker histMarker = new IntervalMarker(
                histStart.getTime(),      // Lower bound in ms since epoch
                histEnd.getTime()         // Upper bound in ms since epoch
        );
        histMarker.setPaint(new Color(0x777777));   // Light gray paint
        histMarker.setAlpha(0.5f);                  // 50% opacity
        plot.addDomainMarker(histMarker, Layer.BACKGROUND); // Add behind the candlesticks

        // b) Mark future region with pale yellow, only if futureWindow is non-empty
        if (!futureWindow.isEmpty()) {
            java.util.Date futStart = dataItems[historical.size()].getDate();          // First future bar’s date
            java.util.Date futEnd = dataItems[totalBars - 1].getDate();                // Last bar’s date
            IntervalMarker futMarker = new IntervalMarker(
                    futStart.getTime(),    // Lower bound for future window
                    futEnd.getTime()       // Upper bound for future window
            );
            futMarker.setPaint(new Color(0xFFFACD));  // Pale yellow (“LemonChiffon”)
            futMarker.setAlpha(0.5f);                 // 50% opacity
            plot.addDomainMarker(futMarker, Layer.BACKGROUND); // Add behind the candlesticks
        }

        // 5) Tweak date‐axis formatting for X-axis
        DateAxis domainAxis = (DateAxis) plot.getDomainAxis(); // Get the date axis
        domainAxis.setAutoRange(true);                          // Auto-scale date range
        domainAxis.setLowerMargin(0.01);                        // Small left margin so bars don’t butt against edge
        domainAxis.setUpperMargin(0.01);                        // Small right margin
        // Optionally set a custom date format, e.g.:
        // domainAxis.setDateFormatOverride(new SimpleDateFormat("HH:mm"));

        // 6) Adjust candlestick renderer properties
        CandlestickRenderer renderer = (CandlestickRenderer) plot.getRenderer();
        renderer.setAutoWidthMethod(CandlestickRenderer.WIDTHMETHOD_SMALLEST);
        // Automatically compute bar width as small as possible based on data spacing

        // Return the fully constructed and styled chart
        return chart;
    }
}