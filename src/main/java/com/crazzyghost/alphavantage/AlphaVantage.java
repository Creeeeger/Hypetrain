/*
 *
 * Copyright (c) 2020 Sylvester Sefa-Yeboah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.crazzyghost.alphavantage;

import com.crazzyghost.alphavantage.alphaIntelligence.AlphaIntelligence;
import com.crazzyghost.alphavantage.cryptocurrency.Crypto;
import com.crazzyghost.alphavantage.economicindicator.EconomicIndicator;
import com.crazzyghost.alphavantage.exchangerate.ExchangeRate;
import com.crazzyghost.alphavantage.forex.Forex;
import com.crazzyghost.alphavantage.fundamentaldata.FundamentalData;
import com.crazzyghost.alphavantage.indicator.Indicator;
import com.crazzyghost.alphavantage.news.News;
import com.crazzyghost.alphavantage.realtime.RealTime;
import com.crazzyghost.alphavantage.sector.Sector;
import com.crazzyghost.alphavantage.stock.Stock;
import com.crazzyghost.alphavantage.technicalindicator.TechnicalIndicator;
import com.crazzyghost.alphavantage.timeseries.TimeSeries;

/**
 * Client interface of library.
 * The API is accessed through this class.
 * Exposes a singleton instance for interaction
 *
 * @author Sylvester Sefa-Yeboah
 * @since 1.0.0
 */
public class AlphaVantage {

    /**
     * Singleton instance of AlphaVantage client.
     */
    private static AlphaVantage INSTANCE;

    /**
     * API configuration holding API key and network settings.
     */
    private Config config;

    /**
     * Private constructor for singleton pattern.
     * Use {@link #api()} to get the instance.
     */
    private AlphaVantage() {
    }

    /**
     * Returns the singleton instance of AlphaVantage client interface.
     * <p>
     * Use this method to access the API client. The instance should be initialized with {@link #init(Config)} before making requests.
     *
     * @return Singleton {@link AlphaVantage} instance.
     */
    public static AlphaVantage api() {
        if (INSTANCE == null) {
            INSTANCE = new AlphaVantage();
        }
        return INSTANCE;
    }

    /**
     * Initializes the client with the given configuration.
     * <p>
     * This should be called before using any API methods to ensure API key and settings are set.
     *
     * @param config {@link Config} object containing API key and settings.
     */
    public void init(Config config) {
        this.config = config;
    }

    /**
     * Provides access to Stock Time Series endpoints.
     * <p>
     * Use this method to make requests for daily, weekly, monthly prices, and intraday time series data.
     *
     * @return {@link TimeSeries} instance for time series requests.
     */
    public TimeSeries timeSeries() {
        return new TimeSeries(config);
    }

    /**
     * Provides access to Foreign Exchange (Forex/FX) endpoints.
     * <p>
     * Use this method to get real-time and historical FX rates.
     *
     * @return {@link Forex} instance for FX requests.
     */
    public Forex forex() {
        return new Forex(config);
    }

    /**
     * Provides access to digital/physical exchange rate endpoints.
     * <p>
     * Use this to retrieve conversion rates between digital currencies and/or fiat currencies.
     *
     * @return {@link ExchangeRate} instance for exchange rate requests.
     */
    public ExchangeRate exchangeRate() {
        return new ExchangeRate(config);
    }

    /**
     * Provides access to digital currency (cryptocurrency) endpoints.
     * <p>
     * Use this to retrieve crypto prices, market cap, and more.
     *
     * @return {@link Crypto} instance for cryptocurrency requests.
     */
    public Crypto crypto() {
        return new Crypto(config);
    }

    /**
     * (Deprecated) Provides access to legacy technical indicator endpoints.
     * <p>
     * Prefer using {@link #technicalIndicator()} for updated endpoints.
     *
     * @return {@link Indicator} instance for legacy indicator requests.
     * @deprecated Use {@link #technicalIndicator()} instead.
     */
    @Deprecated
    public Indicator indicator() {
        return new Indicator(config);
    }

    /**
     * Provides access to technical indicator endpoints.
     * <p>
     * Use this to retrieve moving averages, RSI, MACD, and other technical indicators.
     *
     * @return {@link TechnicalIndicator} instance for technical indicator requests.
     */
    public TechnicalIndicator technicalIndicator() {
        return new TechnicalIndicator(config);
    }

    /**
     * Provides access to sector performance endpoints.
     * <p>
     * Use this to retrieve performance data across different market sectors.
     *
     * @return {@link Sector} instance for sector performance requests.
     */
    public Sector sector() {
        return new Sector(config);
    }

    /**
     * Provides access to company fundamental data endpoints.
     * <p>
     * Use this to retrieve earnings, balance sheet, income statements, and other fundamental metrics.
     *
     * @return {@link FundamentalData} instance for fundamental data requests.
     */
    public FundamentalData fundamentalData() {
        return new FundamentalData(config);
    }

    /**
     * Provides access to economic indicator endpoints.
     * <p>
     * Use this to retrieve GDP, unemployment, CPI, and other macroeconomic indicators.
     *
     * @return {@link EconomicIndicator} instance for economic indicator requests.
     */
    public EconomicIndicator economicIndicator() {
        return new EconomicIndicator(config);
    }

    /**
     * Provides access to stock search and symbol lookup endpoints.
     * <p>
     * Use this to search for companies or stock symbols.
     *
     * @return {@link Stock} instance for stock symbol search and lookup.
     */
    public Stock Stocks() {
        return new Stock(config);
    }

    /**
     * Provides access to news endpoints.
     * <p>
     * Use this to retrieve the latest market news and articles.
     *
     * @return {@link News} instance for news requests.
     */
    public News News() {
        return new News(config);
    }

    /**
     * Provides access to real-time bulk quote endpoints.
     * <p>
     * Use this to get live prices and market data.
     *
     * @return {@link RealTime} instance for real-time quote requests.
     */
    public RealTime Realtime() {
        return new RealTime(config);
    }

    /**
     * Provides access to Alpha Intelligence endpoints.
     * <p>
     * Use this to get advanced analytics, news sentiment, and AI-powered insights.
     *
     * @return {@link AlphaIntelligence} instance for Alpha Intelligence requests.
     */
    public AlphaIntelligence alphaIntelligence() {
        return new AlphaIntelligence(config);
    }
}