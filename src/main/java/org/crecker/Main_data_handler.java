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
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.List;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.addNotification;
import static org.crecker.Main_UI.logTextArea;

public class Main_data_handler {
    public static ArrayList<timeframe> matchList = new ArrayList<>();
    public static ArrayList<percents> percentList = new ArrayList<>();
    public static JLabel percentageChange;
    static JFreeChart chart;
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;

    public static void main(String[] args) { //use the method to generate lists for training the algorithm
        String apiKey = "2NN1RGFV3V34ORCZ"; //SIKE NEW KEY

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
                .forSymbol("NVDA")
                .interval(Interval.ONE_MIN)
                .outputSize(OutputSize.FULL)
                .onSuccess(e -> {
                    try {
                        handleSuccess((TimeSeriesResponse) e);
                    } catch (IOException ex) {
                        throw new RuntimeException(ex);
                    }
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void handleSuccess(TimeSeriesResponse response) throws IOException {
        // This generates some test data since we don't have unlimited API access
        BufferedWriter bufferedWriter = getBufferedWriter(response); //in reversed format (new to old)
        bufferedWriter.close(); // Close the BufferedWriter to free system resources

        // Create a TimeSeries object for plotting
        TimeSeries timeSeries = new TimeSeries(response.getMetaData().getSymbol().toUpperCase() + " Stock Price");

        // Get Stock units
        List<StockUnit> stocks = response.getStockUnits();

        Collections.reverse(stocks); //reverse (old to new)

        // Populate the time series with Stock data
        for (StockUnit stock : stocks) {
            String timestamp = stock.getDate();
            double closingPrice = stock.getClose(); // Assuming getClose() returns closing price

            // Add the data to the TimeSeries
            timeSeries.add(new Minute(convertToDate(timestamp)), closingPrice);
        }

        // Plot the data
        plotData(timeSeries, response.getMetaData().getSymbol().toUpperCase() + " Stock prices", "Date", "Price");
    }

    private static BufferedWriter getBufferedWriter(TimeSeriesResponse response) throws IOException {
        File data = new File(response.getMetaData().getSymbol().toUpperCase() + ".txt"); // Create a File object for the output file named "NVDA.txt"

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

    public static void plotData(TimeSeries timeSeries, String chart_name, String X_axis, String Y_axis) {
        // Wrap the TimeSeries in a TimeSeriesCollection, which implements XYDataset
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        chart = ChartFactory.createTimeSeriesChart(
                chart_name, // Chart title
                X_axis, // X-axis Label
                Y_axis, // Y-axis Label
                dataset, // The dataset (now a TimeSeriesCollection)
                true, // Show legend
                true, // Show tooltips
                false // Show URLs
        );

        // Customizing the plot
        XYPlot plot = chart.getXYPlot();

        // Add a light red shade below Y=0
        plot.addRangeMarker(new IntervalMarker(Double.NEGATIVE_INFINITY, 0.0, new Color(255, 200, 200, 100)));

        // Add a light green shade above Y=0
        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        // Add a black line at y = 0
        ValueMarker zeroLine = new ValueMarker(0.0);
        zeroLine.setPaint(Color.BLACK); // Set the color to black
        zeroLine.setStroke(new BasicStroke(1.0f)); // Set the stroke for the line
        plot.addRangeMarker(zeroLine);

        // Enable zoom and pan features on the chart panel
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setMouseWheelEnabled(true); // Zoom with mouse wheel
        chartPanel.setZoomAroundAnchor(true);  // Zoom on the point where the mouse is anchored
        chartPanel.setRangeZoomable(true);     // Allow zooming on the Y-axis
        chartPanel.setDomainZoomable(true);    // Allow zooming on the X-axis

        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());
                XYPlot plot = chart.getXYPlot();
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

        // Add buttons for controlling the scale
        JPanel controlPanel = getjPanel(chartPanel);

        // Create the frame to display the chart
        JFrame frame = new JFrame("Stock Data");
        frame.setLayout(new BorderLayout());
        frame.add(chartPanel, BorderLayout.CENTER);
        frame.add(controlPanel, BorderLayout.SOUTH); // Add control buttons to the frame
        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
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
        percentageChange.setText(String.format("Percentage Change: %.3f%%", percentageDiff));
    }

    @NotNull
    private static JPanel getjPanel(ChartPanel chartPanel) {
        JButton autoRangeButton = new JButton("Auto Range");
        autoRangeButton.addActionListener(e -> chartPanel.restoreAutoBounds()); // Reset to original scale

        JButton zoomInButton = new JButton("Zoom In");
        zoomInButton.addActionListener(e -> chartPanel.zoomInBoth(0.5, 0.5)); // Zoom in by 50%

        JButton zoomOutButton = new JButton("Zoom Out");
        zoomOutButton.addActionListener(e -> chartPanel.zoomOutBoth(0.5, 0.5)); // Zoom out by 50%

        JPanel controlPanel = new JPanel();
        percentageChange = new JLabel("Percentage Change");

        controlPanel.add(autoRangeButton);
        controlPanel.add(zoomInButton);
        controlPanel.add(zoomOutButton);
        controlPanel.add(percentageChange);
        return controlPanel;
    }

    //Start of main code
    public static void InitAPi(String token) {
        // Configure the API client
        Config cfg = Config.builder()
                .key(token)
                .timeOut(10) // Timeout in seconds
                .build();

        // Initialize the Alpha Vantage API
        AlphaVantage.api().init(cfg);
    }

    public static void get_timeline(String symbol_name, TimelineCallback callback) {
        List<StockUnit> stocks = new ArrayList<>(); // Directly use a List<StockUnit>

        AlphaVantage.api()
                .timeSeries()
                .intraday()
                .forSymbol(symbol_name)
                .interval(Interval.ONE_MIN)
                .outputSize(OutputSize.FULL)
                .onSuccess(e -> {
                    TimeSeriesResponse response = (TimeSeriesResponse) e;
                    stocks.addAll(response.getStockUnits()); // Populate the list

                    callback.onTimeLineFetched(stocks); // Call the callback with the Stock list
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void get_Info_Array(String symbol_name, DataCallback callback) {
        Double[] data = new Double[9];

        // Fetch fundamental data
        AlphaVantage.api()
                .fundamentalData()
                .companyOverview()
                .forSymbol(symbol_name)
                .onSuccess(e -> {
                    CompanyOverviewResponse overview_response = (CompanyOverviewResponse) e;
                    CompanyOverview response = overview_response.getOverview();

                    if (response != null) {
                        data[4] = response.getPERatio();
                        data[5] = response.getPEGRatio();
                        data[6] = response.getFiftyTwoWeekHigh();
                        data[7] = response.getFiftyTwoWeekLow();
                        data[8] = Double.valueOf(response.getMarketCapitalization());
                    } else {
                        System.out.println("Company overview response is null.");
                    }
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();

        AlphaVantage.api()
                .timeSeries()
                .quote()
                .forSymbol(symbol_name)
                .onSuccess(e -> {
                    QuoteResponse response = (QuoteResponse) e;

                    if (response != null) {
                        data[0] = response.getOpen();
                        data[1] = response.getHigh();
                        data[2] = response.getLow();
                        data[3] = response.getVolume();
                    } else {
                        System.out.println("Quote response is null.");
                    }

                    // Call the callback with the fetched data
                    callback.onDataFetched(data);
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static Date convertToDate(String timestamp) {
        // Convert timestamp to Date
        // Assuming timestamp format is "yyyy-MM-dd HH:mm:ss"
        try {
            return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(timestamp);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            return new Date();
        }
    }

    public static Date convertToDate_Simple(String timestamp) {
        // Convert timestamp to Date
        // Assuming timestamp format is "yyyy-MM-dd HH:mm:ss" to "yyyy-MM-dd" date
        try {
            return new SimpleDateFormat("yyyy-MM-dd").parse(timestamp);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            return new Date();
        }
    }

    public static void handleFailure(AlphaVantageException error) {
        System.out.println("error: " + error.getMessage());
    }

    public static void findMatchingSymbols(String searchText, SymbolSearchCallback callback) {
        List<String> allSymbols = new ArrayList<>();

        AlphaVantage.api()
                .Stocks()
                .setKeywords(searchText)
                .onSuccess(e -> {
                    List<StockResponse.StockMatch> list = e.getMatches();

                    for (StockResponse.StockMatch stockResponse : list) {
                        allSymbols.add(stockResponse.getSymbol());
                    }
                    // Filter and invoke the callback on success
                    List<String> filteredSymbols = allSymbols.stream()
                            .filter(symbol -> symbol.toUpperCase().startsWith(searchText.toUpperCase()))
                            .collect(Collectors.toList());
                    callback.onSuccess(filteredSymbols);
                })
                .onFailure(failure -> {
                    // Handle failure and invoke the failure callback
                    Main_data_handler.handleFailure(failure);
                    callback.onFailure(new RuntimeException("API call failed"));
                })
                .fetch();
    }

    public static void receive_News(String Symbol, ReceiveNewsCallback callback) {
        AlphaVantage.api()
                .News()
                .setTickers(Symbol)
                .setSort("LATEST")
                .setLimit(12)
                .onSuccess(e -> callback.onNewsReceived(e.getNewsItems()))
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void start_Hype_Mode(int tradeVolume, float hypeStrength) {
        String[] stockSymbols = { //List of theoretical trade-able symbols
                "AEM", "CVNA", "VRT", "ODFL", "EW", "KHC", "VLO", "CTVA", "HES", "TCOM",
                "MCHP", "LNG", "CBRE", "GLW", "FERG", "ACGL", "SNOW", "IT", "LVS", "DDOG",
                "LULU", "AME", "DFS", "EA", "GIS", "YUM", "IRM", "VRSK", "IDXX", "HSY",
                "BKR", "SYY", "NGG", "BNS", "NXPI", "COF", "EPD", "WDAY", "RSG", "AJG",
                "FTNT", "ADSK", "AFL", "PCG", "PSA", "DHI", "TTD", "SLB", "MET", "ROP",
                "GM", "SE", "TSN", "MKC", "EXPE", "TRU", "MDB", "MT", "ZTO", "BBY",
                "ARE", "OMC", "LPLA", "AER", "CLX", "PFG", "ULTA", "VRSN", "LUV", "DGX",
                "APTV", "DKNG", "UTHR", "AKAM", "FTAI", "POOL", "MGM", "CAVA", "TTEK", "SWK",
                "GGG", "DUOL", "BMRN", "XPO", "AES", "TXRH", "DOCU", "GMAB", "CF", "USFD",
                "REG", "SWKS", "TFC", "TRV", "NSC", "ET", "OKE", "APP", "URI", "PSX",
                "GWW", "O", "COIN", "AZO", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
                "TSM", "BRK/B", "TSLA", "LLY", "AVGO", "WMT", "JPM", "XOM", "UNH", "V",
                "NVO", "ORCL", "MA", "HD", "PG", "COST", "JNJ", "ABBV", "BAC", "NFLX",
                "KO", "CRM", "SAP", "ASML", "CVX", "MRK", "TMUS", "AMD", "BABA", "SMFG",
                "TM", "NVS", "PEP", "AZN", "LIN", "WFC", "CSCO", "ADBE", "MCD", "TMO",
                "PM", "ABT", "IBM", "NOW", "MS", "QCOM", "GE", "AXP", "CAT", "TXN",
                "ISRG", "FMX", "DHR", "RY", "VZ", "DIS", "PDD", "NEE", "AMGN", "INTU",
                "RTX", "HSBC", "GS", "UBER", "PFE", "HDB", "CMCSA", "T", "UL", "ARM",
                "AMAT", "SPGI", "LOW", "TTE", "BLK", "BKNG", "PGR", "UNP", "SNY", "SYK",
                "HON", "LMT", "SCHW", "TJX", "BSX", "KKR", "ANET", "VRTX", "C", "BX",
                "MUFG", "COP", "RACE", "MU", "PANW", "NKE", "ADP", "UPS", "CB", "MDT",
                "ADI", "FI", "BUD", "SBUX", "DE", "UBS", "GILD", "IBN", "MMC", "PLD",
                "SONY", "BMY", "AMT", "REGN", "SHOP", "PLTR", "SO", "LRCX", "INTC", "TD",
                "ELV", "ICE", "BA", "HCA", "MDLZ", "INFY", "SHW", "KLAC", "DUK", "RELX",
                "SCCO", "TT", "PBR", "CI", "EQIX", "ABNB", "ENB", "MO", "PYPL", "MCO",
                "CEG", "CTAS", "BP", "CMG", "GD", "WM", "APH", "RIO", "ZTS", "BN",
                "APO", "CME", "AON", "PH", "GEV", "WELL", "CL", "GSK", "BTI", "MSI",
                "SNPS", "ITW", "SPOT", "USB", "TDG", "NOC", "PNC", "TRI", "DEO", "CRWD",
                "CNQ", "MAR", "EQNR", "ECL", "CP", "CVS", "MRVL", "MMM", "APD", "CNI",
                "TGT", "ORLY", "CDNS", "BDX", "EOG", "BMO", "FCX", "CARR", "FDX", "MCK",
                "JD", "CSX"
        };
        logTextArea.append(String.format("Activating hype mode for auto Stock scanning, Settings: %s Volume, %s Hype, %s Stocks to scan\n", tradeVolume, hypeStrength, stockSymbols.length));
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        List<String> possibleSymbols = new ArrayList<>(); //get the symbols based on the config

        if (tradeVolume > 300000) {
            File file = new File(tradeVolume + ".txt");

            if (file.exists()) {
                // Load symbols from file if it exists
                try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        possibleSymbols.add(line);
                    }
                } catch (IOException e) {
                    System.out.println(e.getMessage());
                }

                logTextArea.append("Loaded symbols from file\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                hypeModeFinder(possibleSymbols);
            } else {
                logTextArea.append("Started getting possible symbols\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                // If file does not exist, call API and write symbols to file
                get_available_symbols(tradeVolume, stockSymbols, result -> {
                    try (FileWriter writer = new FileWriter(file)) {
                        for (String s : result) {
                            String symbol = s.toUpperCase();
                            possibleSymbols.add(symbol);  // Add to possibleSymbols
                            writer.write(symbol + System.lineSeparator());  // Write to file
                        }
                    } catch (IOException e) {
                        System.out.println(e.getMessage());
                    }

                    logTextArea.append("Finished getting possible symbols\n");
                    logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                    hypeModeFinder(possibleSymbols);
                });
            }
        } else {
            // For tradeVolume <= 300000, directly copy symbols to the list and process
            possibleSymbols.addAll(Arrays.asList(stockSymbols));

            logTextArea.append("Use pre set symbols\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

            hypeModeFinder(possibleSymbols);
        }
    }

    public static void get_available_symbols(int tradeVolume, String[] possibleSymbols, SymbolCallback callback) {
        List<String> actualSymbols = new ArrayList<>();

        for (int i = 0; i < possibleSymbols.length; i++) {
            int finalI = i;
            AlphaVantage.api()
                    .fundamentalData()
                    .companyOverview()
                    .forSymbol(possibleSymbols[i])
                    .onSuccess(e -> {
                        long market_cap = ((CompanyOverviewResponse) e).getOverview().getMarketCapitalization();
                        long outstanding_Shares = ((CompanyOverviewResponse) e).getOverview().getSharesOutstanding();

                        AlphaVantage.api()
                                .timeSeries()
                                .daily()
                                .forSymbol(possibleSymbols[finalI])
                                .outputSize(OutputSize.COMPACT)
                                .onSuccess(tsResponse -> {
                                    double close = ((TimeSeriesResponse) tsResponse).getStockUnits().get(0).getClose();
                                    long volume = ((TimeSeriesResponse) tsResponse).getStockUnits().get(0).getVolume();

                                    // Check conditions and add to actual symbols
                                    if (tradeVolume < market_cap) {
                                        if (((double) tradeVolume / close) < volume) {
                                            if (((long) tradeVolume / close) < outstanding_Shares) {
                                                actualSymbols.add(possibleSymbols[finalI]);
                                            }
                                        }
                                    }

                                    // Check if all symbols have been processed
                                    if (finalI == possibleSymbols.length - 1) {
                                        callback.onSymbolsAvailable(actualSymbols); // Call the callback when all are done
                                    }
                                })
                                .onFailure(Main_data_handler::handleFailure)
                                .fetch();
                    })
                    .onFailure(Main_data_handler::handleFailure)
                    .fetch();
        }
    }

    public static void hypeModeFinder(List<String> symbols) {
        logTextArea.append("Started pulling data from server\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        while (true) {
            try {
                List<RealTimeResponse.RealTimeMatch> matches = new ArrayList<>();
                // CountDownLatch to wait for all API calls to finish
                CountDownLatch latch = new CountDownLatch((int) Math.ceil(symbols.size() / 100.0));

                for (int i = 0; i < Math.ceil(symbols.size() / 100.0); i++) {
                    String symbolsBatch = String.join(",", symbols.subList(i * 100, Math.min((i + 1) * 100, symbols.size()))).toUpperCase();
                    AlphaVantage.api()
                            .Realtime()
                            .setSymbols(symbolsBatch)
                            .onSuccess(response -> {
                                matches.addAll(response.getMatches());
                                latch.countDown(); // Decrement the latch count when a batch completes
                            })
                            .onFailure(Main_data_handler::handleFailure)
                            .fetch();
                }

                // Wait for all async calls to complete
                latch.await(); // This will block until all counts down to 0

                // Once all async calls are completed, process the matches
                process_data(matches);

                // Wait for 5 seconds before repeating the function
                Thread.sleep(5000);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt(); // Restore interrupted status
                logTextArea.append("Error occurred during data pull\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                break; // Exit the loop if interrupted
            }
        }
    }

    public static void process_data(List<RealTimeResponse.RealTimeMatch> matches) {
        // Create a new timeframe with all the matches from this batch
        timeframe frame = new timeframe(new ArrayList<>(matches)); // Wrap matches in a new ArrayList

        if (matchList.isEmpty() || matchList.size() == 1) {
            matchList.add(frame); // Add the timeframe to matchList
        } else {
            if (matchList.get(matchList.size() - 1).matches.size() == matches.size()) {
                matchList.add(frame); // Add the timeframe to matchList
            } else {
                System.out.println("matchSize doesn't match: " + matchList.get(matchList.size() - 1).matches.size() + " vs. " + matches.size());
            }
        }

        calculatePercentageChange();
    }

    public static void calculatePercentageChange() {
        // Create a new list for the percentage changes associated with the current timeframe
        List<percent_unit> percentBatch = new ArrayList<>();

        // Check if matchList has more than one batch of data
        if (matchList.size() > 2) {
            // Iterate through the matches of the last timeframe added to matchList
            for (int i = 0; i < matchList.get(matchList.size() - 1).matches.size(); i++) {
                String date = matchList.get(matchList.size() - 1).matches.get(i).getTimestamp();

                // Get the current close price and the previous close price
                double currentClose = matchList.get(matchList.size() - 1).matches.get(i).getClose();
                double previousClose = matchList.get(matchList.size() - 2).matches.get(i).getClose();

                // Calculate the percentage change between consecutive time points
                double percentageChange = ((currentClose - previousClose) / previousClose) * 100;  // Calculate percentage change

                // Create a new percent_unit object and add it to the batch
                percent_unit unit = new percent_unit(date, percentageChange);
                percentBatch.add(unit);
            }

            // Create a percents object with the collected batch of percent_unit objects
            percents percentObj = new percents(new ArrayList<>(percentBatch));

            // Add the percents object to percentList (store the batch of percentage changes)
            percentList.add(percentObj);

            calculatePercents();
        } else {
            logTextArea.append("Not enough percentage data available.\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        }

        logTextArea.append("New percentages got calculated\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }

    public static void calculatePercents() {
        hardcoreCrash(20);

        //!!!Implement the percentage change method
    }

    public static void hardcoreCrash(int entries) {
        logTextArea.append("Checking for crashes\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        // Ensure we have enough frames to compare
        if (percentList.size() >= entries) {
            // Initialize changes list with 0 for each stock
            List<Double> changes = new ArrayList<>(Collections.nCopies(percentList.get(0).units.size(), 0.0));

            // Loop over the frames, starting from the (entries)th last frame to compare with the previous frames
            for (int i = percentList.size() - entries; i < percentList.size(); i++) {
                percents currentFrame = percentList.get(i);

                // Loop through all the stocks (units) in the current frame
                for (int j = 0; j < currentFrame.units.size(); j++) {
                    double currentPercentage = currentFrame.units.get(j).percentage;
                    changes.set(j, changes.get(j) + currentPercentage); // Sum up the percentage changes for each stock

                    // Define a crash threshold
                    if (changes.get(j) < -6.0) {
                        // Report the crash
                        addNotification(matchList.get(0).matches.get(j).getSymbol() + " Crash!", String.valueOf(changes.get(j)), null, new Color(255, 99, 71));
                    }
                }
            }
        }
    }

    //Interfaces
    public interface DataCallback {
        void onDataFetched(Double[] values);
    }

    public interface TimelineCallback {
        void onTimeLineFetched(List<StockUnit> stocks);
    }

    public interface SymbolSearchCallback {
        void onSuccess(List<String> symbols);

        void onFailure(Exception e);
    }

    public interface ReceiveNewsCallback {
        void onNewsReceived(List<NewsResponse.NewsItem> news);
    }

    public interface SymbolCallback {
        void onSymbolsAvailable(List<String> symbols);
    }

    // OOP Models
    public static class timeframe {
        ArrayList<RealTimeResponse.RealTimeMatch> matches;

        // Constructor to initialize matches
        public timeframe(ArrayList<RealTimeResponse.RealTimeMatch> matches) {
            this.matches = matches;
        }
    }

    public static class percents {
        ArrayList<percent_unit> units;

        public percents(ArrayList<percent_unit> unit) {
            this.units = unit;
        }
    }

    public static class percent_unit {
        String date;
        Double percentage;

        public percent_unit(String date, Double percentage) {
            this.date = date;
            this.percentage = percentage;
        }
    }
}

//TODO
//!!!Implement the percentage change method