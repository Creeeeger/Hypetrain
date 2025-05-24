package com.crazzyghost.alphavantage.alphaIntelligence.request;

import com.crazzyghost.alphavantage.parameters.Function;

/**
 * Abstract base class for all Alpha Intelligence request types.
 * <p>
 * Encapsulates the required Alpha Vantage function parameter
 * and provides an extensible builder for creating concrete request objects.
 */
public abstract class AlphaIntelligenceRequest {
    /**
     * The Alpha Vantage function to be called for this request.
     */
    public Function function;

    /**
     * Abstract builder for AlphaIntelligenceRequest subclasses.
     * Uses generics to support method chaining in derived builders.
     *
     * @param <T> the type of the concrete builder (for fluent chaining)
     */
    public abstract static class Builder<T extends Builder<T>> {
        /**
         * Function to be assigned to the built request.
         */
        protected Function function;

        /**
         * Set the Alpha Vantage API function for the request.
         *
         * @param function The API function to be used.
         * @return this builder instance for chaining
         */
        public T function(Function function) {
            this.function = function;
            return (T) this;
        }

        /**
         * Build and return the configured request instance.
         * <p>
         * Must be implemented by concrete subclasses.
         *
         * @return The built AlphaIntelligenceRequest object.
         */
        public abstract AlphaIntelligenceRequest build();
    }
}