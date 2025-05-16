package com.crazzyghost.alphavantage.news.response;

import com.crazzyghost.alphavantage.parser.Parser;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Represents the response from the Alpha Vantage News Sentiment API.
 * It contains a list of news items and potentially an error message if parsing fails.
 */
public class NewsResponse {

    /**
     * Error message if the API response could not be parsed properly
     */
    private final String errorMessage;

    /**
     * List of news items returned by the API
     */
    private final List<NewsItem> newsItems;

    /**
     * Private constructor used by the parser or builder to create an instance.
     *
     * @param newsItems the list of news items (never null, empty if none)
     * @param error     an error message if parsing failed, otherwise null
     */
    private NewsResponse(List<NewsItem> newsItems, String error) {
        this.errorMessage = error;
        this.newsItems = newsItems != null ? newsItems : new ArrayList<>();
    }

    /**
     * Static factory method to create a NewsResponse from a generic map.
     * This delegates parsing to the NewsParser class.
     *
     * @param stringObjectMap the raw response map from the API
     * @return a parsed NewsResponse object
     */
    public static NewsResponse of(Map<String, Object> stringObjectMap) {
        Parser<NewsResponse> parser = new NewsParser();
        return parser.parse(stringObjectMap);
    }

    /**
     * Returns the list of news items parsed from the response.
     *
     * @return list of NewsItem objects
     */
    public List<NewsItem> getNewsItems() {
        return newsItems;
    }

    /**
     * Returns the error message if there was a problem parsing the response.
     * Null if no error occurred.
     *
     * @return error message or null
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Returns a string representation of the news items for easy debugging.
     */
    @Override
    public String toString() {
        return newsItems.toString();
    }

    /**
     * Represents a single news article with its metadata and sentiment information.
     */
    public static class NewsItem {

        private final String title;
        private final String url;
        private final String timePublished;
        private final List<String> authors;
        private final String summary;
        private final String source;
        private final double overallSentimentScore;
        private final String overallSentimentLabel;
        private final List<TickerSentiment> tickerSentiment;

        /**
         * Constructs a NewsItem.
         *
         * @param title                 headline of the news article
         * @param url                   URL link to the full article
         * @param timePublished         publication timestamp in ISO format
         * @param authors               list of authors (empty list if none)
         * @param summary               brief summary of the article
         * @param source                news source/publisher
         * @param overallSentimentScore numerical sentiment score for the article
         * @param overallSentimentLabel sentiment label (e.g., Positive, Negative, Neutral)
         * @param tickerSentiment       list of ticker-specific sentiment data
         */
        public NewsItem(String title, String url, String timePublished, List<String> authors, String summary,
                        String source, double overallSentimentScore, String overallSentimentLabel, List<TickerSentiment> tickerSentiment) {
            this.title = title;
            this.url = url;
            this.timePublished = timePublished;
            this.authors = authors != null ? authors : new ArrayList<>();
            this.summary = summary;
            this.source = source;
            this.overallSentimentScore = overallSentimentScore;
            this.overallSentimentLabel = overallSentimentLabel;
            this.tickerSentiment = tickerSentiment != null ? tickerSentiment : new ArrayList<>();
        }

        /**
         * Returns a simple string representation combining title, summary and sentiment label.
         */
        @Override
        public String toString() {
            return title + " - " + summary + " (" + overallSentimentLabel + ")";
        }

        // Getters for selected fields

        public String getTitle() {
            return title;
        }

        public String getSummary() {
            return summary;
        }

        public String getUrl() {
            return url;
        }
    }

    /**
     * Represents sentiment information related to a particular ticker symbol within a news article.
     */
    public static class TickerSentiment {

        private final String ticker;
        private final double relevanceScore;
        private final double sentimentScore;
        private final String sentimentLabel;

        /**
         * Constructs a TickerSentiment object.
         *
         * @param ticker         the ticker symbol this sentiment relates to (e.g., "AAPL")
         * @param relevanceScore how relevant this ticker is in the article (0-1 scale)
         * @param sentimentScore numerical sentiment score for the ticker
         * @param sentimentLabel qualitative sentiment label for the ticker
         */
        public TickerSentiment(String ticker, double relevanceScore, double sentimentScore, String sentimentLabel) {
            this.ticker = ticker;
            this.relevanceScore = relevanceScore;
            this.sentimentScore = sentimentScore;
            this.sentimentLabel = sentimentLabel;
        }

        /**
         * Returns a string representation of ticker sentiment for display.
         */
        @Override
        public String toString() {
            return ticker + ": " + sentimentLabel + " (" + sentimentScore + ")";
        }
    }

    /**
     * Parser implementation that converts a generic map into a NewsResponse instance.
     * This class handles extracting news items and ticker sentiment from the raw API JSON.
     */
    public static class NewsParser extends Parser<NewsResponse> {

        @SuppressWarnings("unchecked")
        @Override
        public NewsResponse parse(Map<String, Object> stringObjectMap) {
            List<NewsItem> newsItems = new ArrayList<>();
            String errorMessage = null;

            // Return error response if API returned an empty JSON
            if (stringObjectMap.isEmpty()) {
                return onParseError("Empty JSON returned by the API, no news items found.");
            }

            try {
                // Extract the list of news articles under "feed" key
                List<Map<String, Object>> feed = (List<Map<String, Object>>) stringObjectMap.get("feed");
                if (feed != null) {
                    for (Map<String, Object> item : feed) {
                        // Parse the list of ticker sentiments for this news item
                        List<TickerSentiment> tickerSentiments = new ArrayList<>();
                        List<Map<String, Object>> tickerSentimentList = (List<Map<String, Object>>) item.get("ticker_sentiment");
                        if (tickerSentimentList != null) {
                            for (Map<String, Object> tickerSentimentMap : tickerSentimentList) {
                                tickerSentiments.add(new TickerSentiment(
                                        (String) tickerSentimentMap.get("ticker"),
                                        Double.parseDouble(tickerSentimentMap.get("relevance_score").toString()),
                                        Double.parseDouble(tickerSentimentMap.get("ticker_sentiment_score").toString()),
                                        (String) tickerSentimentMap.get("ticker_sentiment_label")
                                ));
                            }
                        }

                        // Create NewsItem instance from the extracted data
                        newsItems.add(new NewsItem(
                                (String) item.get("title"),
                                (String) item.get("url"),
                                (String) item.get("time_published"),
                                (List<String>) item.get("authors"),
                                (String) item.get("summary"),
                                (String) item.get("source"),
                                Double.parseDouble(item.get("overall_sentiment_score").toString()),
                                (String) item.get("overall_sentiment_label"),
                                tickerSentiments
                        ));
                    }
                }
            } catch (ClassCastException | NullPointerException e) {
                // If any parsing errors occur, capture a generic error message
                errorMessage = "Error parsing news sentiment results.";
            }

            return new NewsResponse(newsItems, errorMessage);
        }

        /**
         * Returns a NewsResponse representing a parsing error with the given message.
         *
         * @param error error message describing the parse failure
         * @return a NewsResponse containing the error
         */
        @Override
        public NewsResponse onParseError(String error) {
            return new NewsResponse(null, error);
        }
    }
}