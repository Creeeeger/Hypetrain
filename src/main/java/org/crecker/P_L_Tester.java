package org.crecker;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;

import static org.crecker.Main_data_handler.*;
import static org.crecker.data_tester.calculateStockPercentageChange;
import static org.crecker.data_tester.processStockDataFromFile;

public class P_L_Tester { //profit loss function for determining the efficiency of the algorithm
    public static void main(String[] args) {
        //variables
        double capital = 120000;  //capital variable
        double dipLevel = 0.0;
        int fee = 0;
        double capitalOriginal = capital;  //capital variable
        double revenue; //revenue counter
        String[] fileNames = {"NVDA.txt", "PLTR.txt", "SMCI.txt", "TSLA.txt", "TSM.txt", "WOLF.txt", "MSTR.txt", "SNOW.txt"}; //add more files

        // Calculation of spikes
        // Process data for each file
        for (String fileName : fileNames) {
            try {
                processStockDataFromFile(fileName, fileName.substring(0, fileName.indexOf(".")));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        calculateStockPercentageChange();
        calculateSpikes();

        //sorting of notifications
        List<Notification> sortedNotifications = new ArrayList<>(notificationsForPLAnalysis);

        sortedNotifications.sort((n1, n2) -> {
            Date date1 = n1.getTimeSeries().getTimePeriod(n1.getTimeSeries().getItemCount() - 1).getEnd();
            Date date2 = n2.getTimeSeries().getTimePeriod(n2.getTimeSeries().getItemCount() - 1).getEnd();
            return date1.compareTo(date2); // Compare dates to sort from old to new
        });

        System.out.println(notificationsForPLAnalysis.size()); // print out amount of symbols

        //looping through the notifications
        for (int i = 0; i < sortedNotifications.size(); i++) {
            Notification currentEvent = sortedNotifications.get(i);

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
            while (dateInArray < stockList.size() - 1) {  // Ensure we don't go out of bounds
                double percentageChange = stockList.get(dateInArray + 1).stockUnits.get(stockInArray).getPercentageChange();

                // Print the change and relevant details
                System.out.printf("%.2f%% %s %s%n", percentageChange, currentEvent.getContent(), stockList.get(dateInArray).stockUnits.get(stockInArray).getDate());

                // Update capital based on the percentage change
                capital = capital * (1 + (percentageChange / 100));

                // If the percentage change is above the dip level, continue processing
                if (percentageChange <= dipLevel) {
                    break;  // Exit the loop if the percentage change is no longer above dipLevel
                }

                // Move to the next data point
                dateInArray++;
            }

            // Subtract trading fee and print the result
            capital = capital - fee;
            System.out.println();
        }

        revenue = (capital - capitalOriginal) * 0.75;
        System.out.printf("Total Revenue %s€\n", String.format("%.2f", revenue)); // print out results
    }
}