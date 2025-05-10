package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jetbrains.annotations.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.CandlestickRenderer;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.time.ohlc.OHLCItem;
import org.jfree.data.time.ohlc.OHLCSeries;
import org.jfree.data.time.ohlc.OHLCSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.net.URI;
import java.time.LocalDateTime;
import java.util.List;

import static org.crecker.mainUI.getTicker;
import static org.crecker.mainUI.useCandles;

public class Notification {
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;
    private final String title;
    private final String content;
    private final List<StockUnit> stockUnitList;
    private final LocalDateTime localDateTime;
    private final String symbol;
    private final double change;
    private final int config;
    private final Color color;
    private final TimeSeries timeSeries;
    private final OHLCSeries ohlcSeries;
    JLabel percentageChange;
    private JFrame notificationFrame; // Frame for the notification
    private ChartPanel chartPanel;

    public Notification(String title, String content, List<StockUnit> stockUnitList, LocalDateTime localDateTime, String symbol, double change, int config) {
        this.title = title;
        this.content = content;
        this.stockUnitList = stockUnitList;
        this.localDateTime = localDateTime;
        this.symbol = symbol;
        this.change = change;
        this.config = config;

        /*
          config 0 dip
          config 1 gap filler
          config 2 R-line spike
          config 3 spike
         */
        if (config == 0) {
            this.color = new Color(255, 0, 0);         // Bright Red
        } else if (config == 1) {
            this.color = new Color(255, 140, 0);       // Deep Orange
        } else if (config == 2) {
            this.color = new Color(0, 128, 255);       // Sky Blue
        } else if (config == 3) {
            this.color = new Color(34, 177, 76);       // Leaf Green
        } else {
            this.color = new Color(128, 0, 128);       // Royal Purple
        }

        this.ohlcSeries = new OHLCSeries(symbol + " OHLC");
        processOHLCData(stockUnitList);

        this.timeSeries = new TimeSeries(symbol + " Price");
        processTimeSeriesData(stockUnitList);
    }

    private static void addFirstMarker(XYPlot plot, double xPosition) {
        // Clear previous markers if any
        if (marker1 != null) {
            plot.removeDomainMarker(marker1);
        }
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Create and add the first marker
        marker1 = new ValueMarker(xPosition);
        marker1.setPaint(Color.GREEN);  // First marker in green
        marker1.setStroke(new BasicStroke(2.5f));  // Customize thickness
        plot.addDomainMarker(marker1);
    }

    @NotNull
    private static XYPlot getXyPlot(JFreeChart chart) {
        XYPlot plot = chart.getXYPlot();

        // Configure candlestick renderer
        CandlestickRenderer renderer = new CandlestickRenderer();
        renderer.setAutoWidthMethod(CandlestickRenderer.WIDTHMETHOD_SMALLEST);
        renderer.setUpPaint(Color.GREEN);    // Color for "up" candles (close >= open)
        renderer.setDownPaint(Color.RED);    // Color for "down" candles (close < open)
        renderer.setUseOutlinePaint(true);
        renderer.setDrawVolume(false); // Prevents forced 0 baseline
        plot.setRenderer(renderer);
        return plot;
    }

    public Color getColor() {
        return color;
    }

    public double getChange() {
        return change;
    }

    public String getSymbol() {
        return symbol;
    }

    public LocalDateTime getLocalDateTime() {
        return localDateTime;
    }

    public String getTitle() {
        return title;
    }

    public String getContent() { //get the content
        return content;
    }

    public List<StockUnit> getStockUnitList() {
        return stockUnitList;
    }

    public int getConfig() {
        return config;
    }

    public ChartPanel getChartPanel() {
        return chartPanel;
    }

    public TimeSeries getTimeSeries() {
        return timeSeries;
    }

    public OHLCSeries getOHLCSeries() {
        return ohlcSeries;
    }

    private void processOHLCData(List<StockUnit> stockUnits) {
        for (StockUnit unit : stockUnits) {
            ohlcSeries.add(new OHLCItem(
                    new Second(unit.getDateDate()),
                    unit.getOpen(),
                    unit.getHigh(),
                    unit.getLow(),
                    unit.getClose()
            ));
        }
    }

    private void processTimeSeriesData(List<StockUnit> stockUnits) {
        for (StockUnit unit : stockUnits) {
            Second period = new Second(unit.getDateDate());
            timeSeries.add(period, unit.getClose());
        }
    }

    public void addDataPoint(StockUnit unit) {
        if (unit.getLocalDateTimeDate().isAfter(this.localDateTime)) {
            Second period = new Second(unit.getDateDate());

            ohlcSeries.add(new OHLCItem(
                    period,
                    unit.getOpen(),
                    unit.getHigh(),
                    unit.getLow(),
                    unit.getClose()
            ));

            timeSeries.addOrUpdate(period, unit.getClose());
            updateUI();
        }
    }

    private void updateUI() {
        SwingUtilities.invokeLater(() -> chartPanel.repaint());
    }

    public void showNotification() {
        // Create the notification window
        notificationFrame = new JFrame(title);
        notificationFrame.setSize(700, 500); // Increased size
        notificationFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        notificationFrame.setLocationRelativeTo(null);
        notificationFrame.setAlwaysOnTop(true);

        // Create the text area with content, enable line wrapping
        JTextArea textArea = new JTextArea(content);
        textArea.setEditable(false);
        textArea.setLineWrap(true);
        textArea.setWrapStyleWord(true);

        // Add the text area to a scroll pane
        JScrollPane scrollPane = new JScrollPane(textArea);

        // Main panel to hold everything
        JPanel mainPanel = new JPanel(new BorderLayout());

        // Chart panel setup
        chartPanel = createChart();
        chartPanel.setPreferredSize(new Dimension(600, 320)); // Slightly larger chart

        // Bottom panel with extra features
        JPanel bottomPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));

        percentageChange = new JLabel("Percentage Change");

        // Open in-app chart
        JButton openRealTime = new JButton("Open in Realtime SuperChart");

        openRealTime.addActionListener(e -> mainUI.getInstance().handleStockSelection(this.symbol));

        // Hidden link shown as button
        String tickerCode = getTicker(symbol);
        String url = "https://app.trading212.com/?lang=de&ticker=" + tickerCode;

        JButton openWebPortal = new JButton("Open in Web Portal");
        openWebPortal.setForeground(Color.BLUE);
        openWebPortal.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        openWebPortal.setFocusPainted(false);
        openWebPortal.setBorderPainted(false);
        openWebPortal.setContentAreaFilled(false);

        openWebPortal.addActionListener(e -> {
            try {
                Desktop.getDesktop().browse(new URI(url));
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        });

        // Add buttons/labels
        bottomPanel.add(openWebPortal);
        bottomPanel.add(percentageChange);
        bottomPanel.add(openRealTime);

        // Assemble final layout
        mainPanel.add(scrollPane, BorderLayout.NORTH);
        mainPanel.add(chartPanel, BorderLayout.CENTER);
        mainPanel.add(bottomPanel, BorderLayout.SOUTH);

        // Add the main panel to the notification window
        notificationFrame.add(mainPanel);
        notificationFrame.validate();
        notificationFrame.repaint();

        // Make the notification visible
        notificationFrame.setVisible(true);
    }

    private ChartPanel createChart() {
        if (useCandles) {
            return createOHLCChart();
        } else {
            return createTimeSeriesChart();
        }
    }

    private ChartPanel createTimeSeriesChart() {
        TimeSeriesCollection dataset = new TimeSeriesCollection(timeSeries);
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                symbol + " Price Chart",
                "Date",
                "Price",
                dataset,
                true,
                true,
                false
        );

        // Customize the plot
        XYPlot plot = chart.getXYPlot();
        plot.getDomainAxis().setAutoRange(true);
        plot.getRangeAxis().setAutoRange(true);
        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        // Create the chart panel and enable zoom and pan features
        ChartPanel chartPanel = new ChartPanel(chart);

        // Add mouse listener for marker placement and shaded region
        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());
                ValueAxis xAxis = plot.getDomainAxis();
                ValueAxis yAxis = plot.getRangeAxis();

                // Convert mouse position to data coordinates
                double x = xAxis.java2DToValue(p.getX(), chartPanel.getScreenDataArea(), plot.getDomainAxisEdge());
                double y = yAxis.java2DToValue(p.getY(), chartPanel.getScreenDataArea(), plot.getRangeAxisEdge());

                if (Double.isNaN(point1X)) {
                    // First point selected, set the first marker
                    point1X = x;
                    point1Y = y;
                    addFirstMarker(plot, point1X);
                } else {
                    // Second point selected, set the second marker and shaded region
                    point2X = x;
                    point2Y = y;
                    addSecondMarkerAndShade(plot);

                    // Reset points for next selection
                    point1X = Double.NaN;
                    point1Y = Double.NaN;
                    point2X = Double.NaN;
                    point2Y = Double.NaN;
                }
            }
        });

        return chartPanel;
    }

    private ChartPanel createOHLCChart() {
        OHLCSeriesCollection dataset = new OHLCSeriesCollection();
        dataset.addSeries(ohlcSeries);

        JFreeChart chart = ChartFactory.createCandlestickChart(
                symbol + " OHLC Chart",
                "Date",
                "Price",
                dataset,
                true
        );

        XYPlot plot = getXyPlot(chart);

        // Set auto-range for axes and exclude zero from the range
        plot.getDomainAxis().setAutoRange(true);
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);
        rangeAxis.setAutoRangeIncludesZero(false); // Exclude zero from auto-range
        rangeAxis.configure(); // Force recalculation of the axis range

        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        ChartPanel chartPanel = new ChartPanel(chart);

        // Add mouse listener for marker placement and shaded region
        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());
                ValueAxis xAxis = plot.getDomainAxis();
                ValueAxis yAxis = plot.getRangeAxis();

                // Convert mouse position to data coordinates
                double x = xAxis.java2DToValue(p.getX(), chartPanel.getScreenDataArea(), plot.getDomainAxisEdge());
                double y = yAxis.java2DToValue(p.getY(), chartPanel.getScreenDataArea(), plot.getRangeAxisEdge());

                if (Double.isNaN(point1X)) {
                    // First point selected, set the first marker
                    point1X = x;
                    point1Y = y;
                    addFirstMarker(plot, point1X);
                } else {
                    // Second point selected, set the second marker and shaded region
                    point2X = x;
                    point2Y = y;
                    addSecondMarkerAndShade(plot);

                    // Reset points for next selection
                    point1X = Double.NaN;
                    point1Y = Double.NaN;
                    point2X = Double.NaN;
                    point2Y = Double.NaN;
                }
            }
        });

        return chartPanel;
    }

    private void addSecondMarkerAndShade(XYPlot plot) {
        // Clear previous markers and shaded region if they exist
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Calculate the percentage difference between the two y-values
        double percentageDiff = ((point2Y - point1Y) / point1Y) * 100;

        // Determine the color of the second marker based on percentage difference
        Color markerColor = (percentageDiff >= 0) ? Color.GREEN : Color.RED;
        Color shadeColor = (percentageDiff >= 0) ? new Color(100, 200, 100, 50) : new Color(200, 100, 100, 50);

        // Create and add the second marker
        marker2 = new ValueMarker(point2X);
        marker2.setPaint(markerColor);
        marker2.setStroke(new BasicStroke(1.5f)); // Customize thickness
        plot.addDomainMarker(marker2);

        // Create and add the shaded region between the two markers
        shadedRegion = new IntervalMarker(Math.min(point1X, point2X), Math.max(point1X, point2X));
        shadedRegion.setPaint(shadeColor);  // Translucent green or red shade
        plot.addDomainMarker(shadedRegion);

        percentageChange.setText(String.format("Percentage Change: %.3f%%", percentageDiff));
    }

    public void closeNotification() {
        if (notificationFrame != null) {
            notificationFrame.dispose();
        }
    }

    @Override
    public String toString() {
        return title; // Display the title in the list
    }
}