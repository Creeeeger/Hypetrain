# Runtime Flows

## App Startup

Entry point: `org.crecker.mainUI.main`.

```mermaid
flowchart TD
    Start[mainUI.main] --> Look[Install FlatMacDarkLaf]
    Look --> Cache[Ensure cache directory exists]
    Cache --> ConfigExists{config.xml exists?}
    ConfigExists -- no --> Create[configHandler.createConfig]
    Create --> Set1[setValues]
    Set1 --> UI1[Create mainUI frame]
    UI1 --> Settings[Open settings dialog]
    Settings --> Init1[Init Alpha Vantage if key is set]
    ConfigExists -- yes --> Set2[setValues]
    Set2 --> UI2[Create mainUI frame]
    UI2 --> Init2[Init Alpha Vantage if key is set]
    Init1 --> Finish[refresh chart, fetch ticker map, inspect ONNX parameters]
    Init2 --> Finish
    Finish --> RT{realtime enabled?}
    RT -- yes --> Notifications[Start notification updater]
    RT -- no --> Idle[Wait for user action]
```

Startup side effects:

- Creates `cache/` if missing.
- Creates `config.xml` if missing.
- Loads watchlist symbols/colors into the left panel.
- Calls `RallyPredictor.setParameters()` to inspect ONNX input shapes.
- Starts notification updating if `realtime=true`.

## Selecting A Stock

Triggered by clicking a symbol in the watchlist or by opening a notification in the realtime chart.

```mermaid
flowchart TD
    Select[User selects symbol] --> Normalize[Uppercase and trim symbol]
    Normalize --> Fetch[mainUI.fetchTimeLine]
    Fetch --> AV[mainDataHandler.getTimeline]
    AV --> Intraday[Alpha Vantage intraday 1-minute full]
    Intraday --> Clean[Clamp extreme wick spikes]
    Clean --> Stocks[Update selected stock list]
    Stocks --> Chart[refreshChartData]
    Chart --> Info[Fetch latest quote and fundamentals]
    Info --> News[Fetch latest news]
    Chart --> Realtime{realtime chart enabled?}
    Realtime -- yes --> Poll[Start per-symbol realtime update task]
    Realtime -- no --> Done[Display static chart]
```

The chart can be line or candlestick depending on `candle` in config. Time range buttons aggregate data into
minute/hour/day periods inside `mainUI`.

## Activating Hype Mode

Menu: `Hype mode -> Activate hype mode`.

```mermaid
flowchart TD
    Click[Activate hype mode] --> Worker[Single background executor]
    Worker --> Start[mainDataHandler.startHypeMode]
    Start --> Regime[Read market regime from config]
    Regime --> Symbols[Get symbols from stockCategoryMap]
    Symbols --> CacheFile{market_volume txt exists?}
    CacheFile -- yes --> Load[Load cached tradable symbols]
    CacheFile -- no --> Filter[getAvailableSymbols]
    Filter --> Fundamentals[Fetch market cap and shares outstanding]
    Fundamentals --> Daily[Fetch recent daily bars]
    Daily --> Liquidity[Apply liquidity/tradability checks]
    Liquidity --> Write[Write market_volume txt]
    Load --> Finder[hypeModeFinder]
    Write --> Finder
```

`getAvailableSymbols` filters candidates by:

- Trade volume relative to market cap.
- Trade volume relative to average daily volume.
- Shares to trade relative to shares outstanding.

## Hype Mode Initial Load

Method: `mainDataHandler.hypeModeFinder`.

```mermaid
flowchart TD
    Finder[hypeModeFinder symbols] --> Progress[Open progress dialog]
    Progress --> ForEach[For each symbol]
    ForEach --> Cache{cache/SYMBOL.txt exists?}
    Cache -- yes --> Read[processStockDataFromFile]
    Cache -- no --> Download[Alpha Vantage full intraday]
    Download --> Save[dataTester.handleSuccess writes cache]
    Download --> Store[Reverse oldest-to-newest and store]
    Read --> Latch[CountDownLatch]
    Store --> Latch
    Latch --> AllDone{all symbols loaded?}
    AllDone --> Percent[calculateStockPercentageChange false]
    Percent --> IndicatorRanges[precomputeIndicatorRanges true]
    IndicatorRanges --> FeatureRanges[precomputeFeatureRanges true]
    FeatureRanges --> Loop[Enter live polling loop]
```

## Live Minute Polling Loop

There are two paths controlled by `useParallelFetch` in `mainDataHandler`.

The current default is `true`, so the app uses bulk real-time endpoint batching.

```mermaid
flowchart TD
    Loop[Every minute loop] --> Seed{firstTimeComputed?}
    Seed -- no --> ComputeFirst[computeFirstTime: fetch compact intraday and seed model buffers]
    Seed -- yes --> SkipSeed[Skip seed]
    ComputeFirst --> Second[secondFramework lifecycle check]
    SkipSeed --> Second
    Second --> Bulk[getRealTimeMatches in batches of 100]
    Bulk --> Append[Append latest StockUnit to symbolTimelines]
    Append --> Percent[calculateStockPercentageChange true]
    Percent --> SpikeScan[calculateSpikesInRally frameSize]
    SpikeScan --> Trim[trimSymbolTimelines 300]
    Trim --> Sleep[Sleep until next full minute]
    Sleep --> Loop
```

The older serial path calls `fetchSymbolData` for each symbol and waits with a latch.

## Second Framework

The second framework is enabled by `secondFrameWork=true` in `config.xml`. It is a faster, second-level scanner that
runs beside the minute loop.

```mermaid
flowchart TD
    Toggle{useSecondFramework?} -- false --> Stop[Interrupt SecondFrameworkThread]
    Toggle -- true --> Running{thread already alive?}
    Running -- yes --> Keep[Keep existing thread]
    Running -- no --> Start[Start SecondFrameworkThread]
    Start --> Poll[Every 5 seconds fetch real-time batches]
    Poll --> Process[processStockData]
    Process --> Prepare[prepareMatches into realTimeTimelines]
    Prepare --> Changes[calculateChangesForSeconds]
    Changes --> Eval[evaluateSeconds]
    Eval --> Recent[Keep last 55 seconds]
    Recent --> Spike{pctChange threshold and 70 percent green bars?}
    Spike -- no --> Poll
    Spike -- yes --> Aggregate[Aggregate seconds into one minute-like bar]
    Aggregate --> Advanced[advancedProcessing]
    Advanced --> Notification[Add config 5 second-based notification]
    Notification --> Poll
```

## Signal And Notification Delivery

```mermaid
flowchart TD
    Window[30-bar frame] --> Features[computeFeatures and normalize]
    Features --> SpikeONNX[spike_predictor.onnx rolling prediction]
    Features --> UpFeatures[compute slope features]
    UpFeatures --> UpONNX[uptrendML.onnx rolling prediction]
    SpikeONNX --> Eval[evaluateResult]
    UpONNX --> Eval
    Eval --> Gap[fillTheGap]
    Eval --> Spike[spikeUp]
    Gap --> AlertList[Notification list]
    Spike --> AlertList
    AlertList --> UI[mainUI.addNotification]
    UI --> History[Prune recent notification history]
    UI --> List[Add to hype panel]
    UI --> System[terminal-notifier or Windows tray]
    UI --> PushCut[curl POST to PushCut endpoint]
```

Notification config codes:

| Code | Meaning                |
|------|------------------------|
| `1`  | Gap fill               |
| `2`  | R-line near resistance |
| `3`  | Spike                  |
| `4`  | Uptrend                |
| `5`  | Second-based spike     |

## News And Company Overview

When a stock is selected:

- `mainDataHandler.receiveNews` fetches latest Alpha Vantage news for the ticker.
- News entries are shown in the news list and can be opened in a separate window.
- The company overview button calls `mainDataHandler.getCompanyOverview` and displays the description.

