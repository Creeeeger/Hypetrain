package com.crazzyghost.alphavantage.alphaIntelligence.response;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Represents the response from the Alpha Vantage Insider Transactions endpoint.
 * <p>
 * Contains a list of insider transactions for a specified symbol, or an error message if present.
 *
 * <h2>Usage Example:</h2>
 * <pre>
 * Map&lt;String, Object&gt; responseData = // ... API response map
 * InsiderTransactionsResponse response = InsiderTransactionsResponse.of(responseData);
 * if (response.getErrorMessage() != null) {
 *     System.err.println("Error: " + response.getErrorMessage());
 * } else {
 *     for (InsiderTransactionsResponse.Transaction txn : response.getTransactions()) {
 *         System.out.println("Executive: " + txn.executive + ", Shares: " + txn.shares);
 *     }
 * }
 * </pre>
 */
@SuppressWarnings("unchecked")
public class InsiderTransactionsResponse {

    /**
     * API error message, if any.
     */
    private String errorMessage;

    /**
     * List of insider transactions returned by the API.
     */
    private final List<Transaction> transactions = new ArrayList<>();

    /**
     * Represents a single insider transaction.
     * <p>
     * Fields are populated directly from the API response map.
     */
    public static class Transaction {
        /**
         * Date of the transaction (ISO format, e.g., "2023-01-01").
         */
        public String transactionDate;
        /**
         * Ticker symbol of the security (e.g., "AAPL").
         */
        public String ticker;
        /**
         * Name of the executive involved in the transaction.
         */
        public String executive;
        /**
         * Title or role of the executive (e.g., "CEO").
         */
        public String executiveTitle;
        /**
         * Type of security (e.g., "Common Stock").
         */
        public String securityType;
        /**
         * Indicates if it was an acquisition or disposal ("acquisition" or "disposal").
         */
        public String acquisitionOrDisposal;
        /**
         * Number of shares involved in the transaction (as a String, may need conversion).
         */
        public String shares;
        /**
         * Price per share at which the transaction occurred (as a String).
         */
        public String sharePrice;

        /**
         * Constructs a Transaction from the given map.
         *
         * @param map Map containing transaction field values, as returned by the API.
         */
        public Transaction(Map<String, Object> map) {
            this.transactionDate = (String) map.getOrDefault("transaction_date", null);
            this.ticker = (String) map.getOrDefault("ticker", null);
            this.executive = (String) map.getOrDefault("executive", null);
            this.executiveTitle = (String) map.getOrDefault("executive_title", null);
            this.securityType = (String) map.getOrDefault("security_type", null);
            this.acquisitionOrDisposal = (String) map.getOrDefault("acquisition_or_disposal", null);
            this.shares = (String) map.getOrDefault("shares", null);
            this.sharePrice = (String) map.getOrDefault("share_price", null);
        }
    }

    /**
     * Empty constructor.
     */
    public InsiderTransactionsResponse() {
    }

    /**
     * Factory method to create an InsiderTransactionsResponse from a deserialized API response map.
     *
     * @param data The API response map (parsed from JSON).
     * @return A populated InsiderTransactionsResponse.
     */
    public static InsiderTransactionsResponse of(Map<String, Object> data) {
        InsiderTransactionsResponse response = new InsiderTransactionsResponse();
        // Check for error message in the response
        if (data.containsKey("Error Message")) {
            response.errorMessage = (String) data.get("Error Message");
        }
        // Parse transaction data if present
        if (data.containsKey("data")) {
            List<Object> dataList = (List<Object>) data.get("data");
            for (Object obj : dataList) {
                if (obj instanceof Map) {
                    response.transactions.add(new Transaction((Map<String, Object>) obj));
                }
            }
        }
        return response;
    }

    /**
     * Gets the error message returned by the API, if any.
     *
     * @return the error message, or null if the request was successful.
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Gets the list of insider transactions included in the response.
     *
     * @return an unmodifiable list of Transaction objects.
     */
    public List<Transaction> getTransactions() {
        return transactions;
    }
}