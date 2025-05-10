# 🚂 Hype Train — System Ruleset

> ### "Buy Fear — Sell Euphoria."
> Stay mentally stable. Not every hype is a win. Most hype is bait.

---

## 🧠 Core Principles

- Only triggers when **others are panicking** — look for *high volume flushes* and forced liquidations.
- FOMO is auto-filtered — **no entry** without a confirmed **retracement** signal.
- This system survives because it *waits*.

---

## 💀 If You Miss an Entry

### DO NOT CHASE.

Patience is survival.

> *Wait for tomorrow's setup.*

#### Penalty:

Watch *The Big Short* on loop until market close.

---

## ☠️ After 2 Consecutive Losses

> Trading is a performance sport. Mindset > Setup.

Mandatory:

- 48 hour *no-trading* break.
- Backtest 50 historical setups from your strategy library.
- Rebuild confidence through data — not hope.

---

## 🧨 Kill FOMO Forever

### Entry Paradox:

> *The harder it feels to buy, the better the trade.*

---

### Visualize Sellers:

That volume spike?  
Someone's *life savings* just got liquidated.

---

### Institutional Mirror:

Ask yourself:
> *"What would Citadel do?"*  
> Then do the opposite.

---

## Final Reminder:

> FOMO & Confirmation Bias = Account Killers.

They lead to:

- Impulsive entries
- Chasing garbage setups
- Ignoring risk
- Breaking rules

Discipline is alpha.  
Obey the system — or be the exit liquidity.
---

# 📈 Stock Tracker – Feature Overview

A fully-featured real-time and historical stock analytics tool built for fast, intelligent market monitoring.

---

## 🔍 Symbol Search & Management

- **Live Symbol Search** – real-time suggestions via API
- **Watchlist Control** – add or remove tracked symbols
- **Pre-defined & Custom Watchlists** – e.g., “Volume > 90,000 €” with ability to create your own sets

---

## 📊 Real-Time & Historical Data

- **Multiple Time Ranges**:
    - `1 min`, `3 min`, `5 min`, `10 min`, `30 min`, `1 h`, `4 h`, `1 day`, `3 days`, `1 week`, `2 weeks`, `1 month`
- **Chart Types** – Toggle between:
    - Candlestick (OHLC)
    - Line Chart
- **Auto Refresh** – 1-second interval updates for live data

---

## 📈 Price Analytics

- **Percentage Change Calculator** – select any two points on the chart (with shaded region)
- **Key Metrics Displayed**:
    - Volume
    - P/E Ratio
    - P/E/G Ratio
    - 52-Week High/Low
    - Market Cap

---

## 📰 News & Company Overview

- **Live Company News** – with two-line previews
- **Quick Article Access** – double-click to open full articles
- **Company Overview Dialog** – summary of the selected company

---

## 🔔 Notifications & “Hype” Alerts

- **In-App Notifications** – auto-expire after 20 minutes or duplicate
- **System Notifications** – support for macOS & Windows
- **Hype Mode** – scan for rallying stocks
- **Check for Rallies** – open a popup with potential candidates
- **Notification Sorting** – by date or % change

---

## ⚙️ Configuration Management

- **Settings Import/Export** – via XML
- **Persisted Preferences**:
    - Volume threshold
    - Symbol list
    - Sort preferences
    - API key
    - Real-time toggle
    - Aggressiveness level
    - Candle view toggle
- **Cross-Device Config Reuse** – port settings easily between machines

---

## 🔎 Volume Filtering & Customization

- **Volume Filter** – only track stocks where full volume is open
- **High-Volume Stock Filters** – pre-defined threshold (> 90,000 €)

---

## 🧠 Custom AI Support

- **AI Hook** – integrate your own trained prediction engine

---

## 🖥️ UI & Usability

- **Responsive Swing UI** – with three panels:
    - Symbols
    - Chart
    - Hype
- **Visual Features**:
    - Color-coded symbol entries
    - Rounded and resizable titled borders
    - Clean layout and menu actions for all core features

---

## 🎯 Interactive Chart & Annotation Tools

- **Two-Point %-Change Measurement** – click any two spots to drop start/end markers, shade the interval, and instantly
  compute percentage change.
- **Custom Markers & Shading** – first marker in green; second marker colored by direction (green/up, red/down);
  translucent region fill.
- **Ad-hoc Analysis Mode** – markers reset automatically after each measurement so you can experiment freely.

---

## 🔄 Live Data Streaming & UI Refresh

- **Real-Time Tick Ingestion** – new ticks after the notification timestamp are auto-appended to both OHLC and line
  series.
- **Auto-Repaint Swing UI** – chart panel repaints on the Event Dispatch Thread to ensure lag-free updates.

---

## 🔔 Enhanced Notification Window

- **Always-On-Top Popup** – stays above all other windows; disposes automatically after timeout or via manual close.
- **Rich Scrollable Content Pane** – word-wrapped text area for arbitrary message lengths.
- **Quick-Action Buttons**
    - “Open in Web Portal” → launches your trading-platform URL
    - “Open in Realtime SuperChart” → jumps back to main UI for deeper drill-downs

---

## 🌈 Configurable Alert Styling

- **Color-Coded Alert Types** –
    - Dips: Bright Red
    - Gap-Fillers: Deep Orange
    - R-Line Spikes: Sky Blue
    - Big Spikes: Leaf Green
    - Default Catch-All: Royal Purple
- **Threshold-Driven Hype Alerts** – tie each config to custom volatility/volume thresholds so only meaningful events
  trigger.

---

## 🚀 Performance & Threading

- **Efficient Updates** – `ScheduledExecutorService` for periodic refresh
- **Async API Calls** – non-blocking background threads
- **Data Caching** – faster chart redraws via cached aggregations

## Indicators in use

1. Simple Moving Average (SMA) Crossovers
2. TRIX Indicator
3. Rate of Change (ROC)
4. Cumulative Percentage Change
5. Percentage Change Threshold
6. Keltner Channels
7. Elder-Ray Index Approximation

## Python env. instructions

- Open project in pycharm
- Install dependencies which throw errors or dependency install command inside the file
- Run main.py

## Model knowledge & research

### Buffer size

- Feed in a time period buffer: get less aggressive values and a more smoothened output
- Feed in temp buffer of repeated values: get aggressive switches for the Time an event occurs
- No matter what size single buffer is more efficient and accurate hence the one event buffer is used.

### Window size for training

- Short windows of 5 Minutes give clustered data without clear separation. This is because the LSTM architecture can't
  detect patterns in that short time.
- 10 Minutes are still to short to separate any events
- 20 Minutes gives good data but yet not good enough to ensure that noise is classified as low outcome
- 30 Minutes do the job well but still many BS events are captured and high ranked so 40 minutes is a better choice
  since filtering works better there
- 35-40 Minutes give a good frame with clear separation
- Due to random IDKKKKK sometimes the range from 25-35 is the best: Maybe I get onto something and I can stabilize it,
  who knows...

### Weighting of classes

- the weights of classes added loads of noise to the result so it got removed

### Batch size

- 64 has just turned out to be the best with the least noise

### Bollinger Bands

- Synthetic value for higher stability as it turned out

---

## 📈 A. Trend Following (Your Current Setup)

### ⏱ 1-Minute Timeframe

- ✅ Works for ~85% of SMA/EMA/TRIX strategies
- ✅ Survives spread + slippage
- ❌ Misses entry/exit precision by ~0.3%

### ⚡ 1-Second Timeframe

- ❌ 72% false crossovers due to noise
- ✅ Captures an extra ~0.8% in strong trend moves
- 💀 Requires 10x more stop-loss recalibrations

---

## 🔄 B. Mean Reversion (Your Spike Detection)

### ⏱ 1-Minute Timeframe

- ✅ Reliable for 2–5% retracements
- ❌ Arrives late on V-shaped recoveries

### ⚡ 1-Second Timeframe

- ✅ Front-runs ~90% of minute-based strategies
- 💀 ~40% whipsaw rate without Level 2 data

---

### 📈 Different Times Require Different Measures

To let **HypeTrain** operate effectively under varying market conditions, I've implemented a preset weighting table.  
These weightings adapt HypeTrain's behavior to match different environments like Bull/Bear Markets or during high
volatility phases.

| **Category** | **Bull Market** | **Bear Market** | **High Volatility** | **Scraper Mode** |
|--------------|:---------------:|:---------------:|:-------------------:|:----------------:|
| **TREND**    |      0.30       |      0.15       |        0.20         |       0.10       |
| **MOMENTUM** |      0.40       |      0.25       |        0.35         |       0.10       |
| **STATS**    |      0.15       |      0.30       |        0.25         |       0.45       |
| **ADVANCED** |      0.15       |      0.30       |        0.20         |       0.35       |

> 🛠️ *These dynamic weight adjustments allow HypeTrain to stay sharp across all market terrains.*

### Miscellaneous

- To restore advanced charts, use commit [
  `6b1ef75`](https://github.com/Creeeeger/Hypetrain/commit/6b1ef7522d2d5fb9324d92e6447ab9268cc39274).

# Disclaimer

The information provided by this program is intended solely for educational and informational purposes. By utilizing
this program, you acknowledge and agree to the following terms:

### 1. No Guarantee of Success

The strategies and recommendations offered herein do not guarantee any specific outcome or financial success. Please be
aware that trading and investing involve inherent risks, and you may incur financial losses.

### 2. Personal Responsibility

Any decisions made based on the information provided are your sole responsibility. It is strongly advised that you
conduct your own research and consider consulting a qualified financial professional before engaging in any trading or
investment activities.

### 3. Liability Waiver

The creators, developers, and distributors of this program disclaim any liability for financial losses or damages that
may arise from your use of the program. By using this program, you agree to release and hold harmless all parties
involved from any claims or damages.

### 4. Risk Awareness

Engaging in trading and investing carries significant risks, including the potential loss of your initial investment.
You should only invest what you can afford to lose.

---

**By using this program, you confirm that you have read, understood, and agree to this disclaimer.**