import json

# File paths for input and output JSONL (newline-delimited JSON) files.
INPUT_PATH = "notifications.json"
OUTPUT_PATH = "notifications1.json"

# Open the input file for reading and the output file for writing.
# Using 'with' ensures both files are properly closed even if an error occurs.
with open(INPUT_PATH, 'r') as fin, open(OUTPUT_PATH, 'w') as fout:
    # Process the file line by line to avoid loading everything into memory at once.
    for line in fin:
        # Remove leading/trailing whitespace and skip empty lines.
        line = line.strip()
        if not line:
            continue

        # Attempt to parse the JSON object from the current line.
        # If the line is malformed, log an error and skip it.
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print("Malformed JSON line, skipping.")
            continue

        # Retrieve the 'lookbackWindow' list of bar dictionaries; default to empty list if missing.
        lookback = obj.get("lookbackWindow", [])

        # Build a list of all 'volume' values in the lookback window.
        # Use bar.get("volume", 0) to safely handle missing volume keys.
        volumes = [bar.get("volume", 0) for bar in lookback]

        # Only proceed with normalization if we actually have volume data.
        if volumes:
            # Compute the minimum and maximum volumes in this window.
            min_v = min(volumes)
            max_v = max(volumes)
            range_v = max_v - min_v  # Range of values for scaling

            if range_v > 0:
                # Standard min-max normalization: (value - min) / (max - min)
                # This scales all volumes into the [0.0, 1.0] interval.
                for bar in lookback:
                    raw_vol = bar.get("volume", 0)
                    bar["volume"] = (raw_vol - min_v) / range_v
            else:
                # If all volumes are identical (range == 0), map them all to 0.0
                # This prevents division by zero and retains relative uniformity.
                for bar in lookback:
                    bar["volume"] = 0.0

        # Serialize the modified object back to a JSON string (compact form)
        # and write it as a single line in the output file.
        fout.write(json.dumps(obj))
        fout.write("\n")

# After processing all lines, notify the user that the output is complete.
print(f"Normalized volumes written to {OUTPUT_PATH}")
