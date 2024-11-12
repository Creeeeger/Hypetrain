package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.IntervalMarker;
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
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.*;

public class Main_UI extends JFrame {
    public static JTextArea logTextArea;
    static int vol;
    static float hyp;
    static boolean isSorted;
    static String sym, key;
    static String selected_stock = "Select a Stock"; //selected_stock is the Stock to show in the chart bar
    static String[][] setting_data;
    static JPanel symbol_panel, chart_tool_panel, hype_panel, chartPanel;
    static JMenuBar menuBar;
    static JMenu file, settings, hype_mode_menu, Notifications;
    static JMenuItem load, save, exit, setting_handler, activate_hype_mode, clear, sort, import_c, export_c;
    static JTextField searchField;
    static JLabel openLabel, highLabel, lowLabel, volumeLabel, peLabel, mktCapLabel, fiftyTwoWkHighLabel, fiftyTwoWkLowLabel, pegLabel;
    static JButton removeButton, addButton, oneDayButton, threeDaysButton, oneWeekButton, twoWeeksButton, oneMonthButton;
    static DefaultListModel<String> stockListModel;
    static Map<String, Color> stockColors;
    static ChartPanel chartDisplay; // This will hold the chart
    static TimeSeries timeSeries; // Your time series data

    static DefaultListModel<Notification> notificationListModel;
    static JList<Notification> notificationList;
    static Notification currentNotification; // Track currently opened notification

    static DefaultListModel<News> NewsListModel;
    static JList<News> NewsList;
    static News CurrentNews;

    static List<StockUnit> stocks;

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
        setting_data = config_handler.load_config();
        vol = Integer.parseInt(setting_data[0][1]);
        hyp = Float.parseFloat(setting_data[1][1]);
        sym = setting_data[2][1];
        isSorted = Boolean.parseBoolean(setting_data[3][1]);
        key = setting_data[4][1];
    }

    public static void main(String[] args) {
        Main_UI gui = new Main_UI();
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setSize(1900, 1000); // Width and height of the window
        gui.setVisible(true);
        gui.setTitle("Hype train");

        updateStockInfoLabels(0, 0, 0, 0, 0, 0, 0, 0, 0); //initially fill up the Stock data section

        File config = new File("config.xml");
        if (!config.exists()) {
            config_handler.create_config();
            setValues();

            if (!key.isEmpty()) {
                Main_data_handler.InitAPi(key); //comment out when not testing api to save tokens
            } else {
                throw new RuntimeException("You need to add a key in the settings menu!!");
            }

            refresh(true, true, true, false);

            Settings_handler gui_Setting = new Settings_handler(vol, hyp, sym = create_sym_array(), isSorted, key);
            gui_Setting.setVisible(true);
            gui_Setting.setSize(500, 500);
            gui_Setting.setAlwaysOnTop(true);
            System.out.println("New config created");

        } else {
            System.out.println("Load config");
            setValues();

            if (!key.isEmpty()) {
                Main_data_handler.InitAPi(key); //comment out when not testing api to save tokens
            } else {
                throw new RuntimeException("You need to add a key in the settings menu!!");
            }

            load_table(sym);

            refresh(true, true, true, false);
            System.out.println("Config loaded!");
        }
    }

    public static void refresh(boolean symbols, boolean charts, boolean notification, boolean settings) { //Method for refreshing the ui based on the given panels to refresh
        if (symbols) {
            symbol_panel.revalidate();
            symbol_panel.repaint();
        }

        if (charts) {
            chart_tool_panel.revalidate();
            chart_tool_panel.repaint();
        }

        if (notification) {
            hype_panel.revalidate();
            hype_panel.repaint();
        }

        if (settings) {
            Settings_handler.settingsPanel.revalidate();
            Settings_handler.settingsPanel.repaint();
        }
    }

    public static void load_config() {
        setValues();

        refresh(true, true, true, true);
        Main_data_handler.InitAPi(key); //comment out when not testing api to save tokens

        System.out.println("Config reloaded!");
    }

    public static void save_config(String[][] data) {
        config_handler.save_config(data);
        System.out.println("Config saved successfully");
    }

    public static void load_table(String config) {
        // Split the string into individual entries
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
            System.out.println("No elements saved before");
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
        if (symBuilder.length() > 0) {
            symBuilder.setLength(symBuilder.length() - 1); // Remove the last comma
        }
        return symBuilder.toString();
    }

    public static void updateStockInfoLabels(double open, double high, double low, double volume, double peRatio, double pegRatio, double fiftyTwoWkHigh, double fiftyTwoWkLow, double marketCap) {
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
                {"sort", String.valueOf(isSorted)},
                {"key", key}
        };
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

                    // Update Stock info labels only if values are not null
                    if (values != null && values.length == 9) {

                        for (int i = 0; i < values.length; i++) {
                            if (values[i] == null) { // Check if values[i] is null
                                values[i] = 0.00; // Assign a default value
                            }
                        }

                        updateStockInfoLabels(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]);
                    } else {
                        System.out.println("Received null or incomplete data.");
                    }
                });

                //Fetch timeLine aSync
                Main_data_handler.get_timeline(selected_stock, values -> {
                    for (int i = 1; i < values.size(); i++) {
                        double current_close = values.get(i).getClose();

                        // Ensure there is a previous Stock entry to compare with
                        double previous_close = values.get(i - 1).getClose();

                        // Check for a 10% dip or peak
                        if (Math.abs((current_close - previous_close) / previous_close) >= 0.1) {
                            // Replace the current close with the previous close using the setter method
                            values.get(i).setClose(previous_close);
                        }
                    }

                    stocks = values;

                    // Refresh the chart data for the selected Stock
                    refreshChartData(1);
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
            }
        });

        // Add the Stock list into a scroll pane
        JScrollPane stockScrollPane = new JScrollPane(stockList);

        // Create the "-" button for removing selected items
        removeButton = new JButton("-");
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

        // Create a sub-panel for the button at the bottom
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        buttonPanel.add(removeButton);

        addButton = new JButton("+");
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
                            SwingUtilities.invokeLater(() -> {
                                for (String symbol : matchedSymbols) {
                                    searchListModel.addElement(symbol);
                                }
                            });
                        }

                        @Override
                        public void onFailure(Exception e) {
                            // Handle the error, e.g., show a message or log the error
                            SwingUtilities.invokeLater(() -> System.err.println("Failed to load symbols: " + e.getMessage()));
                        }
                    });
                }
            }
        });

        return panel;
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
        oneDayButton = new JButton("1 Day");
        threeDaysButton = new JButton("3 Days");
        oneWeekButton = new JButton("1 Week");
        twoWeeksButton = new JButton("2 Weeks");
        oneMonthButton = new JButton("1 Month");

        //add day buttons
        buttonPanel.add(oneDayButton);
        buttonPanel.add(threeDaysButton);
        buttonPanel.add(oneWeekButton);
        buttonPanel.add(twoWeeksButton);
        buttonPanel.add(oneMonthButton);

        // Adding Action Listeners with lambda expressions
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
        NewsList = new JList<>(NewsListModel);
        NewsList.setVisibleRowCount(10); // Set the visible row count
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

        JScrollPane newsScrollPane = new JScrollPane(NewsList);
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

    public void refreshChartData(int choice) {
        //Create a new time series
        timeSeries = new TimeSeries(selected_stock + " price");

        // Clear the existing data in the TimeSeries
        timeSeries.clear();

        // Define an array to hold the iteration limits for each case
        int[] limits = {960, 2880, 6720, 9600, 19200}; // Limits for cases 1 to 5

        // Check if the choice is valid
        if (choice < 1 || choice > 5) {
            throw new RuntimeException("A case must be selected");
        }

        // Get the limit based on the choice
        int limit = limits[choice - 1]; // Adjust for zero-based index

        try {
            // Populate the time series with Stock data
            for (int i = 0; i < limit; i++) {
                String timestamp = stocks.get(i).getDate();
                double closingPrice = stocks.get(i).getClose(); // Assuming getClose() returns closing price

                // Add the data to the TimeSeries
                timeSeries.add(new Minute(Main_data_handler.convertToDate(timestamp)), closingPrice);
            }
        } catch (Exception e) {
            System.out.println("No data received: " + e.getMessage());
        }

        // Create a new chart with the updated title
        ChartPanel newChartDisplay = createChart(timeSeries, selected_stock + " Price Chart");

        // Remove the old chart
        chartPanel.remove(chartDisplay); // Remove the old chart
        chartDisplay = newChartDisplay; // Update the reference to the new chart
        chartPanel.add(chartDisplay, BorderLayout.CENTER); // Add the new chart

        chartPanel.revalidate(); // Refresh the panel
        chartPanel.repaint(); // Repaint the panel to show new chart
    }

    private ChartPanel createChart(TimeSeries timeSeries, String chartName) {
        // Wrap the TimeSeries in a TimeSeriesCollection
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                chartName, // Chart title
                "Date", // X-axis Label
                "Price", // Y-axis Label
                dataset, // The dataset
                true, // Show legend
                true, // Show tooltips
                false // Show URLs
        );

        // Customizing the plot
        XYPlot plot = chart.getXYPlot();
        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        return new ChartPanel(chart);
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
        notificationList.setVisibleRowCount(10); // Set the visible row count
        notificationList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Add mouse listener to handle clicks
        notificationList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (isSorted) {
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

        // Test button to add test notifications
        JButton button = new JButton("add (test)");
        panel.add(button);
        button.addActionListener(new event_addNotification());

        return panel;
    }

    // Method to add a notification
    public void addNotification(String title, String content, TimeSeries timeSeries) { //Method to add a new notification to the panel
        Notification newNotification = new Notification(title, content, timeSeries);
        notificationListModel.addElement(newNotification);
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
        News newNews = new News(title, content, url);
        NewsListModel.addElement(newNews);
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
        menuBar = new JMenuBar();

        // Create menus
        file = new JMenu("File");
        settings = new JMenu("Settings");
        hype_mode_menu = new JMenu("Hype mode");
        Notifications = new JMenu("Notifications");

        //JMenuItems File
        load = new JMenuItem("Load the config (manually again)");
        import_c = new JMenuItem("Import config");
        export_c = new JMenuItem("Export config");
        save = new JMenuItem("Save the config");
        exit = new JMenuItem("Exit (saves)");

        //Settings
        setting_handler = new JMenuItem("Open settings");

        //Hype mode
        activate_hype_mode = new JMenuItem("Activate hype mode");

        //Notifications
        clear = new JMenuItem("Clear Notifications");
        sort = new JMenuItem("Sort Notifications");

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

    // Helper method to extract percentage value from the notification title
    public float extractPercentage(String title) {
        int percentIndex = title.indexOf("%");
        if (percentIndex != -1) {
            try {
                // Parse the substring into a float value
                return Float.parseFloat(title.substring(0, percentIndex));
            } catch (NumberFormatException e) {
                e.printStackTrace(); // Handle invalid format gracefully
            }
        }
        return 0.0f; // Default to 0 if extraction fails
    }

    public void sort_notifications() {
        // Convert the notification list model to a List for easier manipulation
        List<Notification> notifications = new ArrayList<>();

        for (int i = 0; i < notificationListModel.size(); i++) {
            notifications.add(notificationListModel.getElementAt(i));
        }

        // Sort the notifications based on percentage change
        notifications.sort((n1, n2) -> {
            // Extract the percentage from each notification title
            float percent1 = extractPercentage(n1.getTitle());
            float percent2 = extractPercentage(n2.getTitle());

            // Sort in descending order of percentage
            return Float.compare(percent2, percent1);
        });

        // Clear the model and repopulate it with sorted notifications
        notificationListModel.clear();
        for (Notification notification : notifications) {
            notificationListModel.addElement(notification);
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
                File configFile = new File("config.xml");

                try {
                    // Copy and rename the file, overwriting if it exists
                    Files.copy(selectedFile.toPath(), configFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

                    load_config();

                    JOptionPane.showMessageDialog(null, "Configuration imported and saved as config.xml successfully!");

                } catch (IOException ex) {
                    JOptionPane.showMessageDialog(null, "Error processing configuration file: " + ex.getMessage());
                    ex.printStackTrace();
                }
            }
        }
    }

    public static class event_export implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Define the source file (config.xml in the root directory)
            File configFile = new File("config.xml");

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
                    JOptionPane.showMessageDialog(null, "Error exporting configuration file: " + ex.getMessage());
                    ex.printStackTrace();
                }
            }
        }
    }

    public static class event_activate_hype_mode implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            Main_data_handler.start_Hype_Mode(vol, hyp);
        }
    }

    public static class event_settings implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            Settings_handler gui = new Settings_handler(vol, hyp, sym, isSorted, key);
            gui.setSize(500, 500);
            gui.setAlwaysOnTop(true);
            gui.setTitle("Config handler ");
            gui.setVisible(true);
        }
    }

    public static class event_Load implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            load_config();
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

    public class event_sort_notifications implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            sort_notifications();
        }
    }

    //test method
    public class event_addNotification implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                List<Notification> notifications = data_tester.Main_data_puller();

                for (Notification notification : notifications) {
                    addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries()); //add notification sample
                }

            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        }
    }
}