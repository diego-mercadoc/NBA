import pandas as pd

# **Step 1: Convert your existing CSV to naive datetimes**
# Run this code snippet once in your Python environment to update nba_games_all.csv

df = pd.read_csv('nba_games_all.csv')
# Force them all to parse as UTC first (to avoid errors on any offset)
df['Date'] = pd.to_datetime(df['Date'], utc=True)

# If you do NOT want time zones at all, remove (localize to None):
df['Date'] = df['Date'].dt.tz_localize(None)

# Now all Date entries are tz-naive
df.to_csv('nba_games_all.csv', index=False)
print("Re-saved nba_games_all.csv with purely naive datetimes!")


