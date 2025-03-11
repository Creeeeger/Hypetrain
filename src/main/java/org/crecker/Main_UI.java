package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jetbrains.annotations.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.util.*;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class Main_UI extends JFrame {
    public static JTextArea logTextArea;
    public static Main_UI gui;
    static int vol;
    static float hyp;
    static boolean shouldSort, useRealtime;
    static String sym, key;
    static String selected_stock = "Select a Stock"; //selected_stock is the Stock to show in the chart bar
    static JPanel symbol_panel, chart_tool_panel, hype_panel, chartPanel;
    static JTextField searchField;
    static JButton tenMinutesButton, thirtyMinutesButton, oneHourButton, oneDayButton, threeDaysButton, oneWeekButton, twoWeeksButton, oneMonthButton;
    static JLabel openLabel, highLabel, lowLabel, volumeLabel, peLabel, mktCapLabel, fiftyTwoWkHighLabel, fiftyTwoWkLowLabel, pegLabel;
    static DefaultListModel<String> stockListModel;
    static Map<String, Color> stockColors;
    static ChartPanel chartDisplay; // This will hold the chart
    static TimeSeries timeSeries;

    static JList<Notification> notificationList;
    static DefaultListModel<Notification> notificationListModel;
    static Notification currentNotification; // Track currently opened notification

    static DefaultListModel<News> NewsListModel;
    static News CurrentNews;

    static List<StockUnit> stocks;
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;
    private static Date startDate;
    private final ScheduledExecutorService executorService = Executors.newSingleThreadScheduledExecutor();
    private String currentStockSymbol = null; // Track the currently displayed stock symbol

    public Main_UI() {
        // Setting layout for the frame (1 row, 4 columns)
        setLayout(new BorderLayout());
        BorderFactory.createTitledBorder("Stock monitor");

        // Menu bar
        setJMenuBar(createMenuBar());

        //Panels
        symbol_panel = create_symbol_panel();
        chart_tool_panel = create_chart_tool_panel();
        hype_panel = create_hype_panel();

        // Add sections to the frame using BorderLayout
        add(symbol_panel, BorderLayout.WEST);
        add(chart_tool_panel, BorderLayout.CENTER);
        add(hype_panel, BorderLayout.EAST);
    }

    public static void setValues() {
        String[][] setting_data = config_handler.load_config();
        vol = Integer.parseInt(setting_data[0][1]);
        hyp = Float.parseFloat(setting_data[1][1]);
        sym = setting_data[2][1];
        shouldSort = Boolean.parseBoolean(setting_data[3][1]);
        key = setting_data[4][1];
        useRealtime = Boolean.parseBoolean(setting_data[5][1]);
    }

    public static void main(String[] args) {
        gui = new Main_UI();
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setSize(1900, 1000); // Width and height of the window
        gui.setVisible(true);
        gui.setTitle("Hype train");

        updateStockInfoLabels(0, 0, 0, 0, 0, 0, 0, 0, 0); //initially fill up the Stock data section

        // Dynamically construct the file path
        Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
        File config = configPath.toFile();
        if (!config.exists()) {
            config_handler.create_config();
            setValues();
            load_table(sym);

            if (!key.isEmpty()) {
                Main_data_handler.InitAPi(key); //comment out when not testing api to save tokens
            } else {
                throw new RuntimeException("You need to add a key in the settings menu!!");
            }

            Settings_handler gui_Setting = new Settings_handler(vol, hyp, sym = create_sym_array(), shouldSort, key, useRealtime);
            gui_Setting.setVisible(true);
            gui_Setting.setSize(500, 500);
            gui_Setting.setAlwaysOnTop(true);

            logTextArea.append("New config created\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        } else {
            logTextArea.append("Load config\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

            setValues();

            if (!key.isEmpty()) {
                Main_data_handler.InitAPi(key); //comment out when not testing api to save tokens
            } else {
                throw new RuntimeException("You need to add a key in the settings menu!!");
            }

            load_table(sym);
            logTextArea.append("Config loaded\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        }
    }

    // Utility method to refresh all components in a container
    public static void refreshAllComponents(Container container) {
        SwingUtilities.invokeLater(() -> {
            for (Component component : container.getComponents()) {
                component.revalidate();
                component.repaint();
                if (component instanceof Container) {
                    refreshAllComponents((Container) component);  // Recursively refresh nested containers
                }
            }
        });
    }

    public static void save_config(String[][] data) {
        config_handler.save_config(data);
        logTextArea.append("Config saved successfully\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }

    public static void load_table(String config) {
        // Split the string into individual entries
        stockColors.clear();
        stockListModel.clear();

        try {
            config = config.substring(1, config.length() - 1); // Remove outer brackets
            String[] entries = config.split("],\\[");

            // Create a 2D array to hold the Stock symbol and corresponding Color object
            Object[][] stockArray = new Object[entries.length][2]; // 2D array: [stockSymbol, Color]

            // Iterate through each entry and populate the 2D array
            for (int i = 0; i < entries.length; i++) {
                // Split by "," to separate the Stock symbol and color part
                String[] parts = entries[i].split(",java.awt.Color\\[r=");
                String stockSymbol = parts[0]; // Get the Stock symbol
                String colorString = parts[1]; // Get the color part (e.g., "102,g=205,b=170]")

                // Parse the RGB values
                String[] rgbParts = colorString.replace("]", "").split(",g=|,b=");
                int r = Integer.parseInt(rgbParts[0]);
                int g = Integer.parseInt(rgbParts[1]);
                int b = Integer.parseInt(rgbParts[2]);

                // Create a Color object from the RGB values
                Color color = new Color(r, g, b);

                // Add the Stock symbol and color to the 2D array
                stockArray[i][0] = stockSymbol;
                stockArray[i][1] = color;
            }

            for (Object[] objects : stockArray) {
                stockListModel.addElement(objects[0].toString());
                stockColors.put(objects[0].toString(), (Color) objects[1]);
            }
        } catch (Exception e) {
            e.printStackTrace();
            logTextArea.append("No elements saved before\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        }
    }

    public static String create_sym_array() {
        StringBuilder symBuilder = new StringBuilder();

        for (Map.Entry<String, Color> entry : stockColors.entrySet()) {
            String stockSymbol = entry.getKey(); // Get the key (Stock symbol)
            Color color = entry.getValue();      // Get the value (color)

            symBuilder.append("[").append(stockSymbol).append(",").append(color).append("],");
        }

        // Remove the trailing comma if the StringBuilder is not empty
        if (!symBuilder.isEmpty()) {
            symBuilder.setLength(symBuilder.length() - 1); // Remove the last comma
        }
        return symBuilder.toString();
    }

    public static void updateStockInfoLabels(double open, double high, double low, double volume,
                                             double peRatio, double pegRatio, double fiftyTwoWkHigh, double fiftyTwoWkLow, double marketCap) {
        openLabel.setText("Open: " + String.format("%.2f", open));
        highLabel.setText("High: " + String.format("%.2f", high));
        lowLabel.setText("Low: " + String.format("%.2f", low));
        volumeLabel.setText("Vol: " + String.format("%.0f", volume));
        peLabel.setText("P/E: " + String.format("%.2f", peRatio));
        pegLabel.setText("P/E/G: " + String.format("%.0f", pegRatio));
        fiftyTwoWkHighLabel.setText("52W H: " + String.format("%.2f", fiftyTwoWkHigh));
        fiftyTwoWkLowLabel.setText("52W L: " + String.format("%.2f", fiftyTwoWkLow));
        mktCapLabel.setText("Mkt Cap: " + String.format("%.2f", marketCap));
    }

    public static String[][] getValues() {
        return new String[][]{
                {"volume", String.valueOf(vol)},
                {"hype_strength", String.valueOf(hyp)},
                {"symbols", sym = create_sym_array()},
                {"sort", String.valueOf(shouldSort)},
                {"key", key},
                {"realtime", String.valueOf(useRealtime)},
        };
    }

    public static void addNotification(String title, String content, TimeSeries timeSeries,
                                       LocalDateTime localDateTime, String symbol, double change) {
        SwingUtilities.invokeLater(() -> notificationListModel.addElement(new Notification(title, content, timeSeries, localDateTime, symbol, change)));
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
        marker1.setStroke(new BasicStroke(1.5f));  // Customize thickness
        plot.addDomainMarker(marker1);
    }

    private static void addSecondMarkerAndShade(XYPlot plot) {
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

        logTextArea.append(String.format("Percentage Change: %.3f%%\n", percentageDiff));
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }

    public static void load_config() {
        setValues();

        load_table(sym);
        Main_UI.refreshAllComponents(gui.getContentPane());
        Main_data_handler.InitAPi(key); //comment out when not testing api to save tokens

        logTextArea.append("Config reloaded\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }

    @NotNull
    private static JButton getJButton(JList<String> stockList) {
        JButton removeButton = new JButton("-");
        removeButton.addActionListener(e -> {
            // Get the selected value
            String selectedValue = stockList.getSelectedValue();
            if (selectedValue != null) {
                // Remove the selected value from the list model
                stockColors.remove(selectedValue);
                stockListModel.removeElement(selectedValue);

                sym = create_sym_array(); //Create the symbol array
            }
        });
        return removeButton;
    }

    public JPanel create_symbol_panel() {
        // Create a panel with BorderLayout
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(250, 0)); // Set fixed width of 150px

        // Create a search field at the top
        searchField = new JTextField();
        searchField.setBorder(BorderFactory.createTitledBorder("Search"));

        // Create a list for possible search results
        DefaultListModel<String> searchListModel = new DefaultListModel<>();
        JList<String> searchList = new JList<>(searchListModel);
        searchList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        JScrollPane searchScrollPane = new JScrollPane(searchList);

        searchScrollPane.setPreferredSize(new Dimension(125, 0));

        // Create a scrollable list of Stock items using DefaultListModel
        stockListModel = new DefaultListModel<>();
        JList<String> stockList = new JList<>(stockListModel);
        stockList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Define fixed colors for each Stock symbol using a HashMap
        stockColors = new HashMap<>();

        // Create a custom ListCellRenderer to apply fixed colors and round borders
        stockList.setCellRenderer((list, value, index, isSelected, cellHasFocus) -> {
            JLabel label = new JLabel(value, JLabel.CENTER); // Center text

            // Set fixed background color from the map
            Color fixedColor = stockColors.getOrDefault(value, Color.LIGHT_GRAY); // Default to light gray if no color is mapped
            label.setOpaque(true);
            label.setBackground(fixedColor);

            // Set black round border
            label.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2, true));

            // Handle selection styling without changing the background color
            if (isSelected) {
                label.setForeground(Color.WHITE); // Change text color on selection
            } else {
                label.setForeground(Color.BLACK); // Default text color when not selected
            }

            return label;
        });

        // Add a ListSelectionListener to handle Stock selection events
        stockList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) { // Prevent duplicate executions during list update
                selected_stock = stockList.getSelectedValue().toUpperCase().trim(); // Get selected Stock symbol

                // Fetch Stock data asynchronously
                Main_data_handler.get_Info_Array(selected_stock, values -> {
                    if (values != null && values.length >= 9) {
                        // Handle null values by assigning default value using parallel processing
                        Arrays.setAll(values, i -> values[i] == null ? 0.00 : values[i]);

                        // Update Stock info labels with safely processed values
                        updateStockInfoLabels(values[0], values[1], values[2], values[3],
                                values[4], values[5], values[6], values[7], values[8]);

                    } else {
                        // Handle incomplete data case
                        SwingUtilities.invokeLater(() -> {
                            logTextArea.append("Received null or incomplete data.\n");
                            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                        });

                        // Log exception for debugging
                        throw new RuntimeException("Received null or incomplete data.");
                    }
                });

                Main_data_handler.get_timeline(selected_stock, values -> {
                    // Extract original close values to avoid interference from concurrent modifications
                    List<Double> originalCloses = new ArrayList<>(values.size());
                    for (StockUnit stock : values) {
                        originalCloses.add(stock.getClose());
                    }

                    // Process each element in parallel based on original data
                    IntStream.range(1, values.size()).parallel().forEach(i -> {
                        double currentOriginalClose = originalCloses.get(i);
                        double previousOriginalClose = originalCloses.get(i - 1);
                        if (Math.abs((currentOriginalClose - previousOriginalClose) / previousOriginalClose) >= 0.1) {
                            values.get(i).setClose(previousOriginalClose);
                        }
                    });

                    // After processing, update on EDT
                    SwingUtilities.invokeLater(() -> {
                        stocks = values;
                        refreshChartData(1);
                    });
                });

                Main_data_handler.receive_News(selected_stock, values -> {
                    // Clear the news list and update UI in the Event Dispatch Thread
                    SwingUtilities.invokeLater(() -> {
                        NewsListModel.clear();
                        for (com.crazzyghost.alphavantage.news.response.NewsResponse.NewsItem value : values) {
                            addNews(value.getTitle(), value.getSummary(), value.getUrl());
                        }
                    });
                });

                if (useRealtime) {
                    SwingUtilities.invokeLater(this::startRealTimeUpdates);
                }
            }
        });

        // Add the Stock list into a scroll pane
        JScrollPane stockScrollPane = new JScrollPane(stockList);

        // Create the "-" button for removing selected items
        JButton removeButton = getJButton(stockList);

        // Create a sub-panel for the button at the bottom
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        buttonPanel.add(removeButton);

        JButton addButton = new JButton("+");
        buttonPanel.add(addButton);
        addButton.addActionListener(e -> {
            String selectedSymbol = searchList.getSelectedValue();
            if (selectedSymbol != null && !stockListModel.contains(selectedSymbol)) {
                // Add the selected symbol to the Stock list
                stockListModel.addElement(selectedSymbol);
                stockColors.put(selectedSymbol, generateRandomColor()); // Assign a random color or use another logic
                sym = create_sym_array();
            }
        });

        // Add the search field to the top, the scrollable Stock list to the center, and the button panel to the bottom
        panel.add(searchField, BorderLayout.NORTH);
        panel.add(stockScrollPane, BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);

        // Add the search list for suggestions (optional)
        panel.add(searchScrollPane, BorderLayout.EAST);

        // Optional: Add a border with title to the panel
        panel.setBorder(BorderFactory.createTitledBorder("Stock Symbols"));

        // Add a document listener to update the search list dynamically
        searchField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                updateSearchList();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                updateSearchList();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                updateSearchList();
            }

            private void updateSearchList() {
                String searchText = searchField.getText().trim().toUpperCase();
                searchListModel.clear();

                if (!searchText.isEmpty()) {
                    // Use the async findMatchingSymbols method with a callback
                    Main_data_handler.findMatchingSymbols(searchText, new Main_data_handler.SymbolSearchCallback() {
                        @Override
                        public void onSuccess(List<String> matchedSymbols) {
                            // Ensure the UI update happens on the Event Dispatch Thread (EDT)
                            SwingUtilities.invokeLater(() -> searchListModel.addAll(matchedSymbols));
                        }

                        @Override
                        public void onFailure(Exception e) {
                            // Handle the error, e.g., show a message or log the error
                            SwingUtilities.invokeLater(e::printStackTrace);
                        }
                    });
                }
            }
        });
        return panel;
    }

    public void startRealTimeUpdates() {
        if (useRealtime) {
            executorService.scheduleAtFixedRate(() -> {
                if (useRealtime) {
                    // Asynchronous fetching of real-time stock data
                    Main_data_handler.getRealTimeUpdate(selected_stock, value -> {
                        if (value != null) {
                            SwingUtilities.invokeLater(() -> {
                                try {
                                    if (!timeSeries.isEmpty()) {
                                        Date date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(value.getTimestamp());

                                        // Update axis properly for live updates
                                        XYPlot plot = chartDisplay.getChart().getXYPlot();
                                        DateAxis axis = (DateAxis) plot.getDomainAxis();
                                        axis.setRange(startDate, date);
                                        updateYAxisRange(plot, startDate, date);

                                        // Get the latest closing price for the new data point
                                        double latestClose = value.getClose();

                                        // Update the time series with the new timestamp and closing price
                                        timeSeries.addOrUpdate(new Minute(date), latestClose);

                                        chartPanel.repaint();
                                    } else {
                                        System.out.println("TimeSeries is empty.");
                                    }
                                } catch (Exception e) {
                                    e.printStackTrace();
                                    logTextArea.append("Error updating chart: " + e.getMessage() + "\n");
                                    logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                                }
                            });
                        } else {
                            logTextArea.append("No real-time updates available\n");
                            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                        }
                    });
                }
            }, 0, 1, TimeUnit.SECONDS);
        }
    }

    private Color generateRandomColor() {
        Random rand = new Random();
        int red = rand.nextInt(128) + 128;   // Ensures red is between 128 and 255
        int green = rand.nextInt(128) + 128; // Ensures green is between 128 and 255
        int blue = rand.nextInt(128) + 128;  // Ensures blue is between 128 and 255
        return new Color(red, green, blue);
    }

    public JPanel create_chart_tool_panel() {
        JPanel mainPanel = new JPanel(new BorderLayout());

        // First Row - Chart and News
        JPanel firstRowPanel = new JPanel(new BorderLayout());

        // Left side of the first row - Chart Panel with Buttons above
        chartPanel = new JPanel(new BorderLayout());

        // Buttons for time range selection (Day, Week, Month)
        JPanel buttonPanel = new JPanel();
        tenMinutesButton = new JButton("10 Minutes");
        thirtyMinutesButton = new JButton("30 Minutes");
        oneHourButton = new JButton("1 Hour");
        oneDayButton = new JButton("1 Day");
        threeDaysButton = new JButton("3 Days");
        oneWeekButton = new JButton("1 Week");
        twoWeeksButton = new JButton("2 Weeks");
        oneMonthButton = new JButton("1 Month");

        tenMinutesButton.setEnabled(false);
        thirtyMinutesButton.setEnabled(false);
        oneHourButton.setEnabled(false);
        oneDayButton.setEnabled(false);
        threeDaysButton.setEnabled(false);
        oneWeekButton.setEnabled(false);
        twoWeeksButton.setEnabled(false);
        oneMonthButton.setEnabled(false);

        //add day buttons
        buttonPanel.add(tenMinutesButton);
        buttonPanel.add(thirtyMinutesButton);
        buttonPanel.add(oneHourButton);
        buttonPanel.add(oneDayButton);
        buttonPanel.add(threeDaysButton);
        buttonPanel.add(oneWeekButton);
        buttonPanel.add(twoWeeksButton);
        buttonPanel.add(oneMonthButton);

        // Adding Action Listeners with lambda expressions
        tenMinutesButton.addActionListener(e -> refreshChartData(6));
        thirtyMinutesButton.addActionListener(e -> refreshChartData(7));
        oneHourButton.addActionListener(e -> refreshChartData(8));
        oneDayButton.addActionListener(e -> refreshChartData(1));
        threeDaysButton.addActionListener(e -> refreshChartData(2));
        oneWeekButton.addActionListener(e -> refreshChartData(3));
        twoWeeksButton.addActionListener(e -> refreshChartData(4));
        oneMonthButton.addActionListener(e -> refreshChartData(5));

        // Placeholder for JFreeChart (replace with actual chart code)
        JPanel chartPlaceholder = new JPanel(new BorderLayout());
        chartPlaceholder.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        chartPlaceholder.setPreferredSize(new Dimension(600, 400));

        // Initialize the chart
        timeSeries = new TimeSeries(selected_stock + " price");
        chartDisplay = createChart(timeSeries, selected_stock + " price Chart");

        // Add a titled border to the chart panel
        chartPanel.setBorder(BorderFactory.createTitledBorder("Stock price Chart"));
        chartPanel.add(buttonPanel, BorderLayout.NORTH);
        chartPanel.add(chartPlaceholder, BorderLayout.CENTER);
        chartPanel.add(chartDisplay);

        // Right side of the first row - Company News
        // Initialize the News list model
        NewsListModel = new DefaultListModel<>();
        JScrollPane newsScrollPane = getNewsScrollPane();
        newsScrollPane.setPreferredSize(new Dimension(200, 400));

        // Add a titled border to the news section
        newsScrollPane.setBorder(BorderFactory.createTitledBorder("Company News"));

        // Add chartPanel (80%) and newsScrollPane (20%) to the firstRowPanel
        firstRowPanel.add(chartPanel, BorderLayout.CENTER);
        firstRowPanel.add(newsScrollPane, BorderLayout.EAST);

        // Second Row - Lists of Stock information
        JPanel secondRowPanel = new JPanel(new GridLayout(1, 4));

        // First column - Open, High, Low
        JPanel openHighLowPanel = new JPanel(new GridLayout(3, 1));
        openLabel = new JLabel("Open: ");
        highLabel = new JLabel("High: ");
        lowLabel = new JLabel("Low: ");

        openHighLowPanel.add(openLabel);
        openHighLowPanel.add(highLabel);
        openHighLowPanel.add(lowLabel);

        // Second column - Volume, P/E, P/E/G
        JPanel volumePEMktCapPanel = new JPanel(new GridLayout(3, 1));
        volumeLabel = new JLabel("Vol: ");
        peLabel = new JLabel("P/E: ");
        pegLabel = new JLabel("P/E/G: ");

        volumePEMktCapPanel.add(volumeLabel);
        volumePEMktCapPanel.add(peLabel);
        volumePEMktCapPanel.add(pegLabel);

        // Third column - 52W High, 52W Low, MKT CAP
        JPanel rangeAndAvgVolPanel = new JPanel(new GridLayout(3, 1));
        fiftyTwoWkHighLabel = new JLabel("52W H: ");
        fiftyTwoWkLowLabel = new JLabel("52W L: ");
        mktCapLabel = new JLabel("Mkt Cap: ");

        rangeAndAvgVolPanel.add(fiftyTwoWkHighLabel);
        rangeAndAvgVolPanel.add(fiftyTwoWkLowLabel);
        rangeAndAvgVolPanel.add(mktCapLabel);

        // Add all columns to the secondRowPanel
        secondRowPanel.add(openHighLowPanel);
        secondRowPanel.add(volumePEMktCapPanel);
        secondRowPanel.add(rangeAndAvgVolPanel);

        // Add a titled border to the Stock info section
        secondRowPanel.setBorder(BorderFactory.createTitledBorder("Stock Information"));

        // Add first and second rows to the main panel
        mainPanel.add(firstRowPanel, BorderLayout.CENTER);
        mainPanel.add(secondRowPanel, BorderLayout.SOUTH);

        return mainPanel;
    }

    @NotNull
    private JScrollPane getNewsScrollPane() {
        JList<News> NewsList = new JList<>(NewsListModel);
        NewsList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Add mouse listener to handle clicks
        NewsList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) { // Open on double click
                    int index = NewsList.locationToIndex(e.getPoint());
                    if (index != -1) {
                        News clickedNews = NewsListModel.getElementAt(index);
                        openNews(clickedNews);
                    }
                }
            }
        });

        return new JScrollPane(NewsList);
    }

    public void refreshChartData(int choice) {
        // Handle stock symbol change
        if (!selected_stock.equals(currentStockSymbol)) {
            currentStockSymbol = selected_stock;
            timeSeries.clear();
            chartDisplay.getChart().setTitle(selected_stock + " price Chart"); // Update chart title

            try {
                if (stocks != null) {
                    stocks.parallelStream().forEach(stock -> {
                        Calendar cal = Calendar.getInstance();
                        cal.setTime(stock.getDateDate());
                        int year = cal.get(Calendar.YEAR);
                        int month = cal.get(Calendar.MONTH) + 1; // Months are 0-based
                        int day = cal.get(Calendar.DAY_OF_MONTH);
                        int hour = cal.get(Calendar.HOUR_OF_DAY);
                        int minuteValue = cal.get(Calendar.MINUTE);
                        Minute minute = new Minute(minuteValue, hour, day, month, year);

                        synchronized (timeSeries) { // Ensure thread safety when adding to timeSeries
                            timeSeries.add(minute, stock.getClose());
                        }
                    });
                }
            } catch (Exception e) {
                e.printStackTrace();
                logTextArea.append("Error loading data: " + e.getMessage() + "\n");
            }
        }

        // Calculate time range based on selection
        long duration = getDurationMillis(choice);
        Date endDate = timeSeries.getItemCount() > 0
                ? timeSeries.getTimePeriod(timeSeries.getItemCount() - 1).getEnd()
                : new Date();
        startDate = new Date(endDate.getTime() - duration);

        // Update chart axis
        XYPlot plot = chartDisplay.getChart().getXYPlot();
        DateAxis axis = (DateAxis) plot.getDomainAxis();
        axis.setRange(startDate, endDate);

        // Update the Y-axis range
        updateYAxisRange(plot, startDate, endDate);

        chartDisplay.repaint();
        tenMinutesButton.setEnabled(true);
        thirtyMinutesButton.setEnabled(true);
        oneHourButton.setEnabled(true);
        oneDayButton.setEnabled(true);
        threeDaysButton.setEnabled(true);
        oneWeekButton.setEnabled(true);
        twoWeeksButton.setEnabled(true);
        oneMonthButton.setEnabled(true);
    }

    // Method to update Y-axis dynamically based on data
    private void updateYAxisRange(XYPlot plot, Date startDate, Date endDate) {
        // Determine the min and max Y-values within the date range
        double minY = Double.MAX_VALUE;
        double maxY = -Double.MAX_VALUE;

        for (int i = 0; i < timeSeries.getItemCount(); i++) {
            Date date = timeSeries.getTimePeriod(i).getEnd();
            if (!date.before(startDate) && !date.after(endDate)) {
                double closeValue = timeSeries.getValue(i).doubleValue();
                minY = Math.min(minY, closeValue);
                maxY = Math.max(maxY, closeValue);
            }
        }

        // Set the new range for the Y-axis
        if (minY != Double.MAX_VALUE && maxY != -Double.MAX_VALUE) {
            ValueAxis yAxis = plot.getRangeAxis();
            yAxis.setRange(minY - (maxY - minY) * 0.1, maxY + (maxY - minY) * 0.1 + 0.1); // Add a little margin + offset to avoid errors
        }
    }

    private long getDurationMillis(int choice) {
        return switch (choice) {
            case 1 -> TimeUnit.DAYS.toMillis(1);    // 1 day
            case 2 -> TimeUnit.DAYS.toMillis(3);    // 3 days
            case 3 -> TimeUnit.DAYS.toMillis(7);    // 1 week
            case 4 -> TimeUnit.DAYS.toMillis(14);   // 2 weeks
            case 5 -> TimeUnit.DAYS.toMillis(30);   // ~1 month
            case 6 -> TimeUnit.MINUTES.toMillis(10);   // 10 Min
            case 7 -> TimeUnit.MINUTES.toMillis(30);   // 30 Min
            case 8 -> TimeUnit.HOURS.toMillis(1);   // 1 Hour
            default -> throw new IllegalArgumentException("Invalid time range");
        };
    }

    private ChartPanel createChart(TimeSeries timeSeries, String chartName) {
        // Wrap the TimeSeries in a TimeSeriesCollection
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                chartName, // Chart title
                "Date",    // X-axis Label
                "Price",   // Y-axis Label
                dataset,   // The dataset
                true,      // Show legend
                true,      // Show tooltips
                false      // Show URLs
        );

        // Customize the plot
        XYPlot plot = chart.getXYPlot();
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

    public JPanel create_hype_panel() {
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(300, 0)); // Set fixed width of 200px
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        JLabel notifications = new JLabel("Hype notifications");
        panel.add(notifications);

        // Initialize the notification list model
        notificationListModel = new DefaultListModel<>();
        notificationList = new JList<>(notificationListModel);
        notificationList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Add mouse listener to handle clicks
        notificationList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (shouldSort) {
                    sort_notifications();
                }

                if (e.getClickCount() == 2) { // Open on double click
                    int index = notificationList.locationToIndex(e.getPoint());
                    if (index != -1) {
                        Notification clickedNotification = notificationListModel.getElementAt(index);
                        openNotification(clickedNotification);
                    }
                }
            }
        });

        // Custom ListCellRenderer for the notifications
        notificationList.setCellRenderer((list, value, index, isSelected, cellHasFocus) -> {
            JLabel label = new JLabel(value.getTitle(), JLabel.CENTER);

            // Set the notification color as background
            label.setOpaque(true);
            label.setBackground(new Color(33, 255, 13));

            // Set border and text color based on selection
            label.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2, true));
            label.setForeground(isSelected ? Color.WHITE : Color.BLACK);

            return label;
        });

        // Wrap the JList in a JScrollPane
        JScrollPane scrollPane = new JScrollPane(notificationList);
        scrollPane.setPreferredSize(new Dimension(200, 100)); // Set the preferred size of the scroll pane
        scrollPane.setMaximumSize(new Dimension(Integer.MAX_VALUE, Integer.MAX_VALUE)); // Allow it to expand fully
        panel.add(scrollPane);

        // Add a logging area above the test button
        logTextArea = new JTextArea(3, 20); // Set rows to control initial visible lines
        logTextArea.setEditable(false);
        logTextArea.setLineWrap(true); // Enable line wrapping
        logTextArea.setWrapStyleWord(true); // Wrap at word boundaries for cleaner appearance

        JScrollPane logScrollPane = new JScrollPane(logTextArea);
        logScrollPane.setPreferredSize(new Dimension(200, 150)); // Set preferred size for height
        logScrollPane.setMaximumSize(new Dimension(Integer.MAX_VALUE, 50)); // Set maximum height to restrict growth

        JLabel logLabel = new JLabel("Hype log window");
        panel.add(logLabel);
        panel.add(logScrollPane);

        // Add flexible space between the log window and the rest
        panel.add(Box.createRigidArea(new Dimension(0, 10))); // Small vertical spacing
        panel.add(Box.createVerticalGlue()); // Flexible space to push the log window up

        panel.add(logScrollPane);
        panel.add(Box.createRigidArea(new Dimension(0, 10))); // Vertical spacing
        panel.setBorder(BorderFactory.createTitledBorder("Notifications"));

        return panel;
    }

    // Open notification and close previous one
    private void openNotification(Notification notification) {
        // Close the currently opened notification if it exists
        if (currentNotification != null) {
            currentNotification.closeNotification();
        }
        // Show the new notification
        notification.showNotification();
        currentNotification = notification; // Update the current notification reference
    }

    // Method to add a News
    public void addNews(String title, String content, String url) { //Method to add News to the panel
        NewsListModel.addElement(new News(title, content, url));
    }

    // Open News and close previous one
    private void openNews(News news) {
        // Close the currently opened News if it exists
        if (CurrentNews != null) {
            CurrentNews.closeNews();
        }

        news.showNews();
        CurrentNews = news;
    }

    // Create the menu bar
    private JMenuBar createMenuBar() {
        JMenuBar menuBar = new JMenuBar();

        // Create menus
        JMenu file = new JMenu("File");
        JMenu settings = new JMenu("Settings");
        JMenu hype_mode_menu = new JMenu("Hype mode");
        JMenu Notifications = new JMenu("Notifications");

        //JMenuItems File
        JMenuItem load = new JMenuItem("Load the config (manually again)");
        JMenuItem import_c = new JMenuItem("Import config");
        JMenuItem export_c = new JMenuItem("Export config");
        JMenuItem save = new JMenuItem("Save the config");
        JMenuItem exit = new JMenuItem("Exit (saves)");

        //Settings
        JMenuItem setting_handler = new JMenuItem("Open settings");

        //Hype mode
        JMenuItem activate_hype_mode = new JMenuItem("Activate hype mode");

        //Notifications
        JMenuItem clear = new JMenuItem("Clear Notifications");
        JMenuItem sort = new JMenuItem("Sort Notifications");

        //add it to the menus File
        file.add(load);
        file.add(import_c);
        file.add(export_c);
        file.add(save);
        file.add(exit);

        //Settings
        settings.add(setting_handler);

        //HypeMode
        hype_mode_menu.add(activate_hype_mode);

        //Notifications
        Notifications.add(clear);
        Notifications.add(sort);

        // Add menus to the menu bar
        menuBar.add(file);
        menuBar.add(settings);
        menuBar.add(hype_mode_menu);
        menuBar.add(Notifications);

        load.addActionListener(new event_Load());
        save.addActionListener(new event_save());
        exit.addActionListener(new event_exit());
        import_c.addActionListener(new event_import());
        export_c.addActionListener(new event_export());
        setting_handler.addActionListener(new event_settings());
        activate_hype_mode.addActionListener(new event_activate_hype_mode());
        clear.addActionListener(e -> notificationListModel.clear());
        sort.addActionListener(new event_sort_notifications());

        return menuBar;
    }

    public void sort_notifications() {
        // Convert the model's elements to a List using Collections.list
        List<Notification> notifications = Collections.list(notificationListModel.elements());

        // Sort notifications in descending order of percentage change using a concise comparator
        notifications.sort(Comparator.comparingDouble(Notification::getChange).reversed());

        // Clear the model and add all sorted notifications back
        notificationListModel.clear();
        notificationListModel.addAll(notifications);
    }

    public static class event_export implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Define the source file (config.xml in the root directory)
            Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
            File configFile = configPath.toFile();

            // Check if the config file exists
            if (!configFile.exists()) {
                JOptionPane.showMessageDialog(null, "config.xml not found in the project root!");
                return;
            }

            // Create a JFileChooser for exporting files
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setDialogTitle("Export Configuration");

            // Show the save dialog (for saving the file)
            int userSelection = fileChooser.showSaveDialog(null);

            // If the user approves file selection
            if (userSelection == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();

                try {
                    // Copy the config.xml to the selected directory, overwriting if necessary
                    Files.copy(configFile.toPath(), selectedFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    JOptionPane.showMessageDialog(null, "Configuration exported successfully!");
                } catch (IOException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(null, "Error exporting configuration file: " + ex.getMessage());
                }
            }
        }
    }

    public static class event_activate_hype_mode implements ActionListener {
        private static final ExecutorService executorService = Executors.newSingleThreadExecutor();

        @Override
        public void actionPerformed(ActionEvent e) {
            // Run the Hype Mode in a dedicated background thread
            executorService.submit(() -> {
                try {
                    Main_data_handler.start_Hype_Mode(vol, hyp);
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            });
        }
    }

    public static class event_settings implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            Settings_handler gui = new Settings_handler(vol, hyp, sym, shouldSort, key, useRealtime);
            gui.setSize(500, 500);
            gui.setAlwaysOnTop(true);
            gui.setTitle("Config handler ");
            gui.setVisible(true);
        }
    }

    public static class event_save implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            save_config(getValues());
        }
    }

    public static class event_exit implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            save_config(getValues()); //Save them in case user forgot
            System.out.println("Exit application");
            System.exit(0); // Exit the application
        }
    }

    public static class event_import implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Create a JFileChooser for importing files
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setDialogTitle("Import Configuration");

            // Show the open dialog (for choosing a file)
            int userSelection = fileChooser.showOpenDialog(null);

            // If a file was selected
            if (userSelection == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();

                // Define the target file in the root directory with the name "config.xml"
                Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
                File configFile = configPath.toFile();

                try {
                    // Copy and rename the file, overwriting if it exists
                    Files.copy(selectedFile.toPath(), configFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

                    load_config();

                    JOptionPane.showMessageDialog(null, "Configuration imported and saved as config.xml successfully!");

                } catch (IOException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(null, "Error processing configuration file: " + ex.getMessage());
                }
            }
        }
    }

    public static class event_Load implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            load_config();
        }
    }

    public class event_sort_notifications implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            sort_notifications();
        }
    }
}