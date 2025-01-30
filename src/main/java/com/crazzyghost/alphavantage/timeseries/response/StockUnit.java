package com.crazzyghost.alphavantage.timeseries.response;


import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.Date;

public class StockUnit {

    private final double open;
    private final double high;
    private final double low;
    private final double adjustedClose;
    private final double volume;
    private final double dividendAmount;
    private final double splitCoefficient;
    private final String dateTime;
    private String symbol;
    private double percentageChange;
    private double close;


    private StockUnit(Builder builder) {
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
    }

    public double getOpen() {
        return open;
    }

    public double getHigh() {
        return high;
    }

    public double getLow() {
        return low;
    }

    public double getClose() {
        return close;
    }

    public void setClose(double close) {
        this.close = close;
    }

    public double getAdjustedClose() {
        return adjustedClose;
    }

    public double getVolume() {
        return volume;
    }

    public double getDividendAmount() {
        return dividendAmount;
    }

    public double getSplitCoefficient() {
        return splitCoefficient;
    }

    public LocalDateTime getLocalDateTimeDate() {
        DateTimeFormatter formatterWithMillis = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
        DateTimeFormatter formatterWithoutMillis = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        try {
            return LocalDateTime.parse(dateTime, formatterWithMillis).truncatedTo(ChronoUnit.SECONDS);
        } catch (Exception e) {
            return LocalDateTime.parse(dateTime, formatterWithoutMillis);
        }
    }

    public Date getDateDate() {
        try {
            return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(this.dateTime);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public double getPercentageChange() {
        return percentageChange;
    }

    public void setPercentageChange(double percentageChange) {
        this.percentageChange = percentageChange;
    }

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
                '}';
    }

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

        public Builder open(double open) {
            this.open = open;
            return this;
        }

        public Builder high(double high) {
            this.high = high;
            return this;
        }

        public Builder low(double low) {
            this.low = low;
            return this;
        }

        public Builder close(double close) {
            this.close = close;
            return this;
        }

        public Builder adjustedClose(double close) {
            this.adjustedClose = close;
            return this;
        }

        public Builder dividendAmount(double dividendAmount) {
            this.dividendAmount = dividendAmount;
            return this;
        }

        public Builder volume(double volume) {
            this.volume = volume;
            return this;
        }

        public Builder splitCoefficient(double splitCoefficient) {
            this.splitCoefficient = splitCoefficient;
            return this;
        }

        public Builder time(String dateTime) {
            this.dateTime = dateTime;
            return this;
        }

        public Builder symbol(String symbol) {
            this.symbol = symbol;
            return this;
        }

        public Builder percentageChange(double percentageChange) {
            this.percentageChange = percentageChange;
            return this;
        }

        public StockUnit build() {
            return new StockUnit(this);
        }
    }
}
