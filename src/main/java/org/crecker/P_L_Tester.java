package org.crecker;

import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.Objects;

import static org.crecker.Main_UI.addNotification;
import static org.crecker.Main_data_handler.*;
import static org.crecker.data_tester.*;

public class P_L_Tester {
    public static void main(String[] args) {
        // getNewStocks();
        PLAnalysis();
    }

    private static void getNewStocks() {
        for (String s : Arrays.asList("SMCI", "IONQ", "WOLF", "MARA", "APP")) {
            getData(s);
        }
    }

    public static void PLAnalysis() {
        //variables
        double capital = 130000;  //capital variable
        double dipLevel = -1.2;
        int fee = 0;
        int stock = 0;
        int calls = 0;
        LocalDateTime lastClose = null;
        double capitalOriginal = capital;  //capital variable
        double revenue; //revenue counter
        String[] fileNames = {"MARA.txt", "IONQ.txt", "SMCI.txt", "WOLF.txt"}; //add more files

        prepData(fileNames);

        for (double i = 1; i > -1; i -= 0.05) {
            //Reset variables
            capital = 130000;
            dipLevel = i;
            calls = 0;
            lastClose = null;
            capitalOriginal = capital;
            revenue = 0;

            //looping through the notifications
            for (Notification currentEvent : notificationsForPLAnalysis) {
                // Extract notification date and format it
                Date notificationDate = currentEvent.getTimeSeries()
                        .getTimePeriod(currentEvent.getTimeSeries().getItemCount() - 1)
                        .getEnd();

                Calendar calendar = Calendar.getInstance();
                calendar.setTime(notificationDate);
                calendar.set(Calendar.SECOND, 0); // Set seconds to 00
                String outputDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(calendar.getTime());

                // Extract stock name
                String stockName = currentEvent.getTitle()
                        .substring(currentEvent.getTitle().indexOf("%") + 2, currentEvent.getTitle().indexOf(" stock"));

                // Find the stock in the array
                int stockInArray = -1;
                for (int j = 0; j < stockList.get(4).stockUnits.size(); j++) {
                    if (Objects.equals(stockList.get(4).stockUnits.get(j).getSymbol(), stockName)) {
                        stockInArray = j;
                        break;
                    }
                }

                // Handle case when stock is not found
                if (stockInArray == -1) {
                    System.out.println("Stock not found: " + stockName);
                    continue;
                }

                // Find the matching date in the array
                int dateInArray = -1;
                for (int j = 4; j < stockList.size(); j++) {
                    if (Objects.equals(stockList.get(j).stockUnits.get(stockInArray).getDate(), outputDate)) {
                        dateInArray = j;
                        break;
                    }
                }

                // Next dip method
                if (!(stockList.get(dateInArray + 1).stockUnits.get(stockInArray).getPercentageChange() <= dipLevel)) {

                    if (lastClose != null && lastClose.isAfter(LocalDateTime.parse(stockList.get(dateInArray + 1).stockUnits.get(stockInArray).getDate(), DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")))) {
                        continue;
                    }

                    calls++;

                    while (dateInArray < stockList.size() - 2) {  // Ensure we don't go out of bounds

                        double percentageChange = 0;
                        try {
                            percentageChange = stockList.get(dateInArray + 2).stockUnits.get(stockInArray).getPercentageChange();
                        } catch (Exception ignored) {
                        }

                        // Update capital based on the percentage change
                        capital = capital * (1 + (percentageChange / 100));

                        // Print the change and relevant details
                          //    System.out.printf("%.2f%% %.2f %s %s %s%n", percentageChange, (capital - capitalOriginal), currentEvent.getContent(), stockList.get(dateInArray + 2).stockUnits.get(stockInArray).getDate(), stockName);

                        // If the percentage change is above the dip level, continue processing
                        if (percentageChange <= dipLevel) {
                            lastClose = LocalDateTime.parse(stockList.get(dateInArray + 2).stockUnits.get(stockInArray).getDate(), DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                            break;  // Exit the loop if the percentage change is no longer above dipLevel
                        }

                        // Move to the next data point
                        dateInArray++;
                    }

                   //   getNext5Minutes(capital, dateInArray, stockInArray);

                    //  createNotification(currentEvent);

                    // Subtract trading fee
                    capital = capital - fee;
                    //  System.out.println();
                }
            }

            revenue = (capital - capitalOriginal) * 0.75;
            if(!(revenue / calls < 400)){
                System.out.printf("%.2f%n", i);
                System.out.printf("Total Revenue %s€\n", String.format("%.2f", revenue)); // print out results
                System.out.println("Calls:" + calls);
                System.out.printf("Rev per call: %.2f \n", revenue / calls);
                System.out.println();
            }
        }
        //     createTimeline(fileNames, stock);
    }

    private static void prepData(String[] fileNames) {
        // Calculation of spikes, Process data for each file
        for (String fileName : fileNames) {
            try {
                processStockDataFromFile(fileName, fileName.substring(0, fileName.indexOf(".")), 20000);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        data_tester.calculateStockPercentageChange();
        calculateSpikes();
    }

    private static void createNotification(Notification currentEvent) {
        try {
            addNotification(currentEvent.getTitle(), currentEvent.getContent(), currentEvent.getTimeSeries(), currentEvent.getColor());
        } catch (Exception ignored) {
        }
    }

    private static void createTimeline(String[] fileNames, int stock) {
        TimeSeries timeSeries = null;
        try {
            timeSeries = new TimeSeries(fileNames[stock]);
        } catch (Exception ignored) {
        }

        for (int i = 2; i < stockList.size(); i++) {
            try {
                String timestamp = stockList.get(i).stockUnits.get(stock).getDate();
                double closingPrice = stockList.get(i).stockUnits.get(stock).getClose(); // Assuming getClose() returns closing price
                double prevClosingPrice = stockList.get(i - 1).stockUnits.get(stock).getClose(); // Assuming getClose() returns closing price

                if (Math.abs((closingPrice - prevClosingPrice) / prevClosingPrice * 100) > 14) {
                    closingPrice = prevClosingPrice;
                }

                // Add the data to the TimeSeries
                if (timeSeries != null) {
                    timeSeries.addOrUpdate(new Minute(convertToDate(timestamp)), closingPrice);
                }
            } catch (Exception e) {
                break;
            }
        }

        try {
            plotData(timeSeries, stockList.get(4).stockUnits.get(stock).getSymbol() + " price change", "Date", "price");
        } catch (Exception ignored) {
        }
    }

    private static void getNext5Minutes(double capital, int dateInArray, int stockInArray) {
        double simulatedCapital = capital; // Local variable to simulate the capital changes

        for (int i = 1; i <= 5 && (dateInArray + i < stockList.size()); i++) {
            double futurePercentageChange = 0;
            try {
                futurePercentageChange = stockList.get(dateInArray + 2 + i).stockUnits.get(stockInArray).getPercentageChange();
            } catch (Exception e) {
                continue;
            }
            String futureDate = stockList.get(dateInArray + 2 + i).stockUnits.get(stockInArray).getDate();

            // Calculate simulated capital based on the percentage change
            simulatedCapital = simulatedCapital * (1 + (futurePercentageChange / 100));

            // Print event details
            System.out.printf("\u001B[33mEvent %d: %.2f%% on %s\u001B[0m%n", i, futurePercentageChange, futureDate);
        }

        // Calculate and display the difference in simulated capital
        double profitOrLoss = simulatedCapital - capital;
        System.out.printf("\u001B[32mPotential Profit/Loss from next 5 events: %.2f\u001B[0m%n", profitOrLoss);
    }
}