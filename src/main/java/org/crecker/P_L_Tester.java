package org.crecker;

import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Objects;

import static org.crecker.Main_UI.addNotification;
import static org.crecker.Main_data_handler.*;
import static org.crecker.data_tester.plotData;
import static org.crecker.data_tester.processStockDataFromFile;

public class P_L_Tester { //profit loss function for determining the efficiency of the algorithm
    public static void main(String[] args) {
        //   getData("SNOW");
        PLAnalysis();
    }

    public static void PLAnalysis() {
        //variables
        double capital = 120000;  //capital variable
        double dipLevel = -0.40;
        int fee = 0;
        int stock = 0;

        double capitalOriginal = capital;  //capital variable
        double revenue; //revenue counter
        String[] fileNames = {"IREN.txt", "PLTR.txt", "SMCI.txt", "TSLA.txt", "WOLF.txt", "MSTR.txt", "SNOW.txt", "NVDA.txt"}; //add more files

        // Calculation of spikes, Process data for each file
        for (String fileName : fileNames) {
            try {
                processStockDataFromFile(fileName, fileName.substring(0, fileName.indexOf(".")), 15000);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        data_tester.calculateStockPercentageChange();
        calculateSpikes();

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

            double initialPercentageChange = stockList.get(dateInArray + 1).stockUnits.get(stockInArray).getPercentageChange();
            double changeInNotification = Double.parseDouble(currentEvent.getContent().substring(currentEvent.getContent().indexOf("by ") + 3, currentEvent.getContent().indexOf("%")));

            // Next dip method
            if (!(initialPercentageChange <= dipLevel) && (changeInNotification < 0 || changeInNotification >= 1.5)) {
                while (dateInArray < stockList.size() - 2) {  // Ensure we don't go out of bounds
                    double percentageChange = stockList.get(dateInArray + 2).stockUnits.get(stockInArray).getPercentageChange();

                    // Update capital based on the percentage change
                    capital = capital * (1 + (percentageChange / 100));

                    // Print the change and relevant details
                    System.out.printf("%.2f%% %.2f %s %s %s%n", percentageChange, (capital - capitalOriginal), currentEvent.getContent(), stockList.get(dateInArray + 2).stockUnits.get(stockInArray).getDate(), stockName);

                    // If the percentage change is above the dip level, continue processing
                    if (percentageChange <= dipLevel) {
                        break;  // Exit the loop if the percentage change is no longer above dipLevel
                    }

                    // Move to the next data point
                    dateInArray++;
                }
                try {
                    addNotification(currentEvent.getTitle(), currentEvent.getContent(), currentEvent.getTimeSeries(), currentEvent.getColor());
                } catch (Exception ignored) {
                }

                // Subtract trading fee
                capital = capital - fee;
                System.out.println();
            }
        }

        revenue = (capital - capitalOriginal) * 0.75;
        System.out.printf("Total Revenue %s€\n", String.format("%.2f", revenue)); // print out results

        for (Notification forPLAnalysis : notificationsForPLAnalysis) {
            System.out.println(forPLAnalysis.getTitle() + " " + forPLAnalysis.getContent());
        }

        TimeSeries timeSeries = new TimeSeries("stock");
        for (int i = 2; i < stockList.size(); i++) {
            try {
                String timestamp = stockList.get(i).stockUnits.get(stock).getDate();
                double closingPrice = stockList.get(i).stockUnits.get(stock).getClose(); // Assuming getClose() returns closing price
                double prevClosingPrice = stockList.get(i - 1).stockUnits.get(stock).getClose(); // Assuming getClose() returns closing price

                if (Math.abs((closingPrice - prevClosingPrice) / prevClosingPrice * 100) > 14) {
                    closingPrice = prevClosingPrice;
                }

                // Add the data to the TimeSeries
                timeSeries.addOrUpdate(new Minute(convertToDate(timestamp)), closingPrice);
            } catch (Exception e) {
                break;
            }
        }
        try {
            plotData(timeSeries, stockList.get(4).stockUnits.get(stock).getSymbol() + " price change", "Date", "price");
        } catch (Exception ignored) {
        }
    }
}