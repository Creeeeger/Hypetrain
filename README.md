# Hype train

---

- Get notified when the next hype comes in
- Hop onto the hype Train
- damn boy make that cash
- Donate 1% of your earnings to me
- Now new without OOP models but hashmaps for insane performance


- search for symbols
- check percentage change
- receive the latest news about a stock
- Time history about a stock
- Add your custom symbols to your watch list
- Have an insight into the stock before you buy over the notification window
- Re-use the config when using on another machine (import / export config)
- get notified when the hype comes in
- set in settings your volume to only check for stocks where you can buy full volume to make maximum profit
- pre-defined set of stocks for volumes of 200-300k €
- Use of custom Trained AI for additional support for prediction

## Indicators in use

1. Simple Moving Average (SMA) Crossovers
2. TRIX Indicator
3. Rate of Change (ROC)
4. Bollinger Bands
5. Cumulative Percentage Change
6. Percentage Change Threshold
7. Keltner Channels
8. Elder-Ray Index Approximation

## Python env. instructions

- Open project in pycharm
- Install dependencies which throw errors
- Run main.py

## Model knowledge

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
- Due to random IDKKKKK sometimes the range from 25-35 is the best: Maybe I get onto something and I can stabilize it, who knows...

### Weighting of classes

- the weights of classes added loads of noise to the result so it got removed

### Batch size

- 64 has just turned out to be the best with the least noise

---

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