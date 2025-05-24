package com.crazzyghost.alphavantage.parameters;

/**
 * Enum representing all Alpha Vantage API function types.
 */
public enum Function {

    // ------------------- Stock Time Series Functions -------------------
    /**
     * Intraday stock prices
     */
    TIME_SERIES_INTRADAY,
    /**
     * Extended intraday stock prices (historical data)
     */
    TIME_SERIES_INTRADAY_EXTENDED,
    /**
     * Daily stock prices
     */
    TIME_SERIES_DAILY,
    /**
     * Daily adjusted stock prices
     */
    TIME_SERIES_DAILY_ADJUSTED,
    /**
     * Weekly stock prices
     */
    TIME_SERIES_WEEKLY,
    /**
     * Weekly adjusted stock prices
     */
    TIME_SERIES_WEEKLY_ADJUSTED,
    /**
     * Monthly stock prices
     */
    TIME_SERIES_MONTHLY,
    /**
     * Monthly adjusted stock prices
     */
    TIME_SERIES_MONTHLY_ADJUSTED,
    /**
     * Real-time global quote
     */
    GLOBAL_QUOTE,

    // ------------------- Exchange Rate Functions -------------------
    /**
     * Real-time currency exchange rates
     */
    CURRENCY_EXCHANGE_RATE,

    // ------------------- Forex (FX) Functions -------------------
    /**
     * Intraday forex data
     */
    FX_INTRADAY,
    /**
     * Daily forex data
     */
    FX_DAILY,
    /**
     * Weekly forex data
     */
    FX_WEEKLY,
    /**
     * Monthly forex data
     */
    FX_MONTHLY,

    // ------------------- Digital Currency Functions -------------------
    /**
     * Daily digital currency data
     */
    DIGITAL_CURRENCY_DAILY,
    /**
     * Weekly digital currency data
     */
    DIGITAL_CURRENCY_WEEKLY,
    /**
     * Monthly digital currency data
     */
    DIGITAL_CURRENCY_MONTHLY,
    /**
     * Crypto rating info
     */
    CRYPTO_RATING,

    // ------------------- Technical Indicator Functions -------------------
    /**
     * Simple Moving Average
     */
    SMA,
    /**
     * Exponential Moving Average
     */
    EMA,
    /**
     * Weighted Moving Average
     */
    WMA,
    /**
     * Double Exponential Moving Average
     */
    DEMA,
    /**
     * Triple Exponential Moving Average
     */
    TEMA,
    /**
     * Triangular Moving Average
     */
    TRIMA,
    /**
     * Kaufman's Adaptive Moving Average
     */
    KAMA,
    /**
     * MESA Adaptive Moving Average
     */
    MAMA,
    /**
     * Volume Weighted Average Price
     */
    VWAP,
    /**
     * Triple Exponential Moving Average (T3)
     */
    T3,
    /**
     * Moving Average Convergence Divergence
     */
    MACD,
    /**
     * Extended MACD
     */
    MACDEXT,
    /**
     * Stochastic Oscillator
     */
    STOCH,
    /**
     * Fast Stochastic Oscillator
     */
    STOCHF,
    /**
     * Relative Strength Index
     */
    RSI,
    /**
     * Stochastic RSI
     */
    STOCHRSI,
    /**
     * Williams %R
     */
    WILLR,
    /**
     * Average Directional Index
     */
    ADX,
    /**
     * Average Directional Movement Index Rating
     */
    ADXR,
    /**
     * Absolute Price Oscillator
     */
    APO,
    /**
     * Percentage Price Oscillator
     */
    PPO,
    /**
     * Momentum
     */
    MOM,
    /**
     * Balance of Power
     */
    BOP,
    /**
     * Commodity Channel Index
     */
    CCI,
    /**
     * Chande Momentum Oscillator
     */
    CMO,
    /**
     * Rate of Change
     */
    ROC,
    /**
     * Rate of Change Ratio
     */
    ROCR,
    /**
     * Aroon Indicator
     */
    AROON,
    /**
     * Aroon Oscillator
     */
    AROONOSC,
    /**
     * Money Flow Index
     */
    MFI,
    /**
     * Triple Exponential Average (TRIX)
     */
    TRIX,
    /**
     * Ultimate Oscillator
     */
    ULTOSC,
    /**
     * Directional Movement Index
     */
    DX,
    /**
     * Minus Directional Indicator
     */
    MINUS_DI,
    /**
     * Plus Directional Indicator
     */
    PLUS_DI,
    /**
     * Minus Directional Movement
     */
    MINUS_DM,
    /**
     * Plus Directional Movement
     */
    PLUS_DM,
    /**
     * Bollinger Bands
     */
    BBANDS,
    /**
     * Midpoint over period
     */
    MIDPOINT,
    /**
     * Mid-price over period
     */
    MIDPRICE,
    /**
     * Parabolic SAR
     */
    SAR,
    /**
     * True Range
     */
    TRANGE,
    /**
     * Average True Range
     */
    ATR,
    /**
     * Normalized ATR
     */
    NATR,
    /**
     * Chaikin A/D Line
     */
    AD,
    /**
     * Chaikin A/D Oscillator
     */
    ADOSC,
    /**
     * On Balance Volume
     */
    OBV,
    /**
     * Hilbert Transform - Trendline
     */
    HT_TRENDLINE,
    /**
     * Hilbert Transform - SineWave
     */
    HT_SINE,
    /**
     * Hilbert Transform - Trend vs Cycle Mode
     */
    HT_TRENDMODE,
    /**
     * Hilbert Transform - Dominant Cycle Period
     */
    HT_DCPERIOD,
    /**
     * Hilbert Transform - Dominant Cycle Phase
     */
    HT_DCPHASE,
    /**
     * Hilbert Transform - Phasor Components
     */
    HT_PHASOR,

    // ------------------- Sector Performance -------------------
    /**
     * Sector performance data
     */
    SECTOR,

    // ------------------- Fundamental Data -------------------
    /**
     * Company overview
     */
    OVERVIEW,
    /**
     * Company income statement
     */
    INCOME_STATEMENT,
    /**
     * Company balance sheet
     */
    BALANCE_SHEET,
    /**
     * Company cash flow statement
     */
    CASH_FLOW,
    /**
     * Company earnings
     */
    EARNINGS,
    /**
     * Listing status
     */
    LISTING_STATUS,

    // ------------------- Economic Indicators -------------------
    /**
     * Real GDP data
     */
    REAL_GDP,
    /**
     * Real GDP per capita
     */
    REAL_GDP_PER_CAPITA,
    /**
     * Treasury yield rates
     */
    TREASURY_YIELD,
    /**
     * Federal funds rate
     */
    FEDERAL_FUNDS_RATE,
    /**
     * Consumer Price Index
     */
    CPI,
    /**
     * Inflation rate
     */
    INFLATION,
    /**
     * Inflation expectation
     */
    INFLATION_EXPECTATION,
    /**
     * Consumer sentiment
     */
    CONSUMER_SENTIMENT,
    /**
     * Retail sales data
     */
    RETAIL_SALES,
    /**
     * Durable goods orders
     */
    DURABLES,
    /**
     * Unemployment rate
     */
    UNEMPLOYMENT,
    /**
     * Nonfarm payroll data
     */
    NONFARM_PAYROLL,

    // ------------------- Utility/Other Functions -------------------
    /**
     * Stock symbol search
     */
    SYMBOL_SEARCH,
    /**
     * News sentiment analysis
     */
    NEWS_SENTIMENT,
    /**
     * Real-time bulk quotes
     */
    REALTIME_BULK_QUOTES,
    /**
     * Fixed window analytics
     */
    ANALYTICS_FIXED_WINDOW,
    /**
     * Insider transaction data
     */
    INSIDER_TRANSACTIONS
}