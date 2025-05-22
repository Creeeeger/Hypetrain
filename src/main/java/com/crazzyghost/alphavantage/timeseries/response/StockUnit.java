package com.crazzyghost.alphavantage.timeseries.response;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.Date;

/**
 * Represents a single entry (unit) of stock time series data, including prices, volume, splits, and meta-data.
 * <p>
 * Instances should be created via the {@link Builder} to ensure immutability of core data fields.
 */
public class StockUnit {

    /**
     * Date and time of the data point, as a string (e.g., "2024-05-16 17:00:00")
     */
    public final String dateTime;
    /**
     * Opening price for the period
     */
    private final double open;
    /**
     * Adjusted closing price (includes splits/dividends)
     */
    private final double adjustedClose;
    /**
     * Dividend amount distributed for the period
     */
    private final double dividendAmount;
    /**
     * Split coefficient applied in the period
     */
    private final double splitCoefficient;
    /**
     * Highest price during the period (modifiable for derived datasets)
     */
    private double high;
    /**
     * Lowest price during the period (modifiable for derived datasets)
     */
    private double low;
    /**
     * Trading volume during the period (modifiable for derived datasets)
     */
    private double volume;
    /**
     * Stock symbol (e.g., "AAPL"). Can be modified if needed
     */
    private String symbol;

    /**
     * Percentage price change for the period (optional; used for analytics/features)
     */
    private double percentageChange;

    /**
     * Closing price for the period (modifiable for derived datasets)
     */
    private double close;

    /**
     * Custom target variable (optional; for supervised ML/analytics tasks)
     */
    private int target;

    /**
     * Constructs a StockUnit from a Builder instance.
     *
     * @param builder Builder containing all fields.
     */
    public StockUnit(Builder builder) {
        this.open = builder.open;
        this.high = builder.high;
        this.low = builder.low;
        this.close = builder.close;
        this.adjustedClose = builder.adjustedClose;
        this.volume = builder.volume;
        this.dividendAmount = builder.dividendAmount;
        this.splitCoefficient = builder.splitCoefficient;
        this.dateTime = builder.dateTime;
        this.symbol = builder.symbol;
        this.percentageChange = builder.percentageChange;
        this.target = builder.target;
    }

    /**
     * @return The opening price
     */
    public double getOpen() {
        return open;
    }

    /**
     * @return The highest price
     */
    public double getHigh() {
        return high;
    }

    /**
     * @param high Set the highest price (for derived/updated datasets)
     */
    public void setHigh(double high) {
        this.high = high;
    }

    /**
     * @return The lowest price
     */
    public double getLow() {
        return low;
    }

    /**
     * @param low Set the lowest price (for derived/updated datasets)
     */
    public void setLow(double low) {
        this.low = low;
    }

    /**
     * @return The closing price
     */
    public double getClose() {
        return close;
    }

    /**
     * @param close Set the closing price (for derived/updated datasets)
     */
    public void setClose(double close) {
        this.close = close;
    }

    /**
     * @return Adjusted closing price (includes splits/dividends)
     */
    public double getAdjustedClose() {
        return adjustedClose;
    }

    /**
     * @return Volume of trades
     */
    public double getVolume() {
        return volume;
    }

    /**
     * @param volume Set the volume (for derived/updated datasets)
     */
    public void setVolume(double volume) {
        this.volume = volume;
    }

    /**
     * @return Custom target (for analytics/ML tasks)
     */
    public int getTarget() {
        return target;
    }

    /**
     * @param target Set custom target variable
     */
    public void setTarget(int target) {
        this.target = target;
    }

    /**
     * @return Dividend amount
     */
    public double getDividendAmount() {
        return dividendAmount;
    }

    /**
     * @return Split coefficient
     */
    public double getSplitCoefficient() {
        return splitCoefficient;
    }

    /**
     * Parses the dateTime string to {@link LocalDateTime}.
     * Supports both with and without milliseconds.
     *
     * @return LocalDateTime representation, truncated to seconds if milliseconds are present.
     */
    public LocalDateTime getLocalDateTimeDate() {
        DateTimeFormatter formatterWithMillis = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
        DateTimeFormatter formatterWithoutMillis = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        try {
            return LocalDateTime.parse(dateTime, formatterWithMillis).truncatedTo(ChronoUnit.SECONDS);
        } catch (Exception e) {
            return LocalDateTime.parse(dateTime, formatterWithoutMillis);
        }
    }

    /**
     * Parses the dateTime string to {@link Date} (legacy Java API).
     *
     * @return Date representation.
     * @throws RuntimeException if parsing fails.
     */
    public Date getDateDate() {
        try {
            return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(this.dateTime);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @return Stock symbol (e.g., "AAPL")
     */
    public String getSymbol() {
        return symbol;
    }

    /**
     * @param symbol Set the stock symbol (for derived/updated datasets)
     */
    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    /**
     * @return Percentage price change for the period
     */
    public double getPercentageChange() {
        return percentageChange;
    }

    /**
     * @param percentageChange Set the percentage price change
     */
    public void setPercentageChange(double percentageChange) {
        this.percentageChange = percentageChange;
    }

    /**
     * Returns a detailed string representation of all fields for debugging/logging.
     *
     * @return String with all values
     */
    @Override
    public String toString() {
        return "\n" + "StockUnit{" +
                "open=" + open +
                ", high=" + high +
                ", low=" + low +
                ", close=" + close +
                ", adjustedClose=" + adjustedClose +
                ", volume=" + volume +
                ", dividendAmount=" + dividendAmount +
                ", splitCoefficient=" + splitCoefficient +
                ", date=" + dateTime +
                ", symbol=" + symbol +
                ", percentageChange=" + percentageChange +
                ", target=" + target +
                '}';
    }

    /**
     * Builder pattern for constructing immutable {@link StockUnit} instances.
     * Allows flexible setting of each field before creation.
     */
    public static class Builder {
        double open;
        double high;
        double low;
        double close;
        double adjustedClose;
        double volume;
        double dividendAmount;
        double splitCoefficient;
        String dateTime;
        String symbol;
        double percentageChange;
        int target;

        /**
         * @param open Opening price
         */
        public Builder open(double open) {
            this.open = open;
            return this;
        }

        /**
         * @param high Highest price
         */
        public Builder high(double high) {
            this.high = high;
            return this;
        }

        /**
         * @param low Lowest price
         */
        public Builder low(double low) {
            this.low = low;
            return this;
        }

        /**
         * @param close Closing price
         */
        public Builder close(double close) {
            this.close = close;
            return this;
        }

        /**
         * @param close Adjusted closing price (alias)
         */
        public Builder adjustedClose(double close) {
            this.adjustedClose = close;
            return this;
        }

        /**
         * @param dividendAmount Dividend amount
         */
        public Builder dividendAmount(double dividendAmount) {
            this.dividendAmount = dividendAmount;
            return this;
        }

        /**
         * @param volume Trading volume
         */
        public Builder volume(double volume) {
            this.volume = volume;
            return this;
        }

        /**
         * @param splitCoefficient Split coefficient
         */
        public Builder splitCoefficient(double splitCoefficient) {
            this.splitCoefficient = splitCoefficient;
            return this;
        }

        /**
         * @param dateTime Date/time string ("yyyy-MM-dd HH:mm:ss" or "yyyy-MM-dd HH:mm:ss.SSS")
         */
        public Builder time(String dateTime) {
            this.dateTime = dateTime;
            return this;
        }

        /**
         * @param symbol Stock symbol
         */
        public Builder symbol(String symbol) {
            this.symbol = symbol;
            return this;
        }

        /**
         * @param percentageChange Percentage price change
         */
        public Builder percentageChange(double percentageChange) {
            this.percentageChange = percentageChange;
            return this;
        }

        /**
         * @param target Custom target value (for analytics/ML tasks)
         */
        public Builder target(int target) {
            this.target = target;
            return this;
        }

        /**
         * @return New immutable {@link StockUnit} instance with all fields set
         */
        public StockUnit build() {
            return new StockUnit(this);
        }
    }
}
