# NBA Data Scraper

A Python tool for scraping NBA game and team statistics from Basketball-Reference, with comprehensive data validation, feature engineering, and betting insights.

## Overview

This scraper is designed to handle the complexities of NBA season data, including:
- Season transitions (October-June spanning two calendar years)
- Special cases like the COVID-affected 2020-21 season
- Current season's played, scheduled, and future games
- Advanced statistics for betting analysis

## Features

### Data Collection
- Scrapes game-by-game results across multiple NBA seasons
- Handles both single-page and monthly-page formats
- Collects advanced team statistics
- Proper season year mapping (e.g., 2024-25 season)
- Smart rate limiting with minimal delays

### Data Processing
- Validates data completeness and accuracy
- Normalizes team names across different formats
- Handles data cleaning and formatting
- Tracks scraping progress for resume capability

### Advanced Features
- Rolling statistics (last 5 games performance)
- Days of rest between games
- Team winning/losing streaks
- Point differentials
- Betting insights and analysis
- Quarter-by-quarter performance stats

### Current Season Handling
- Tracks played vs scheduled games
- Identifies future scheduled games (through April 13, 2025)
- Maintains data accuracy for partial seasons

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/NBA.git
cd NBA
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
Run the scraper:
```bash
python nba_scraper.py
```

### Configuration
Modify seasons in `main()`:
```python
scraper = NBADataScraper(start_season=2021, end_season=2025)
```

### Output Files
- `nba_games_all.csv`: Complete games dataset
- `nba_team_stats_all.csv`: Team statistics
- `scraper_progress.json`: Progress tracking

## Data Fields

### Games Data
- Basic: Date, Teams, Points
- Advanced: Rest Days, Streaks
- Rolling Stats: Points, Differentials (5-game window)
- Current Season: Future Game Flags

### Team Stats
- Standard: Wins, Losses
- Advanced: ORtg, DRtg, Pace, SRS
- Quarter Performance: Period-by-period scoring

## Validation

The scraper includes comprehensive validation:
- Expected game counts per season
- Duplicate detection and removal
- Team name consistency
- Data completeness checks
- Season transition validation

## Rate Limiting

Optimized for speed while respecting server limits:
- Base wait: 0.5 seconds
- Random jitter: 0-1 seconds
- Monthly cooldown: 1-2 seconds

## Progress Tracking

- Saves progress after each month
- Allows resuming interrupted scrapes
- Tracks completed seasons

## Betting Analysis

- Current game predictions
- Team form analysis
- Streak tracking
- Point differential trends
- Schedule advantage metrics

## Development

### Project Structure
```
NBA/
├── nba_scraper.py     # Main scraper code
├── requirements.txt   # Dependencies
├── .cursorrules      # Project context rules
├── README.md         # Documentation
└── data/
    ├── nba_games_all.csv
    └── nba_team_stats_all.csv
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Notes

- Handles COVID-affected 2020-21 season differently
- Current season (2024-25) has special handling
- Data updates respect Basketball-Reference's terms
- Betting insights are for informational purposes

## Requirements

- Python 3.8+
- pandas
- numpy
- requests
- lxml

## License

MIT License - see LICENSE file for details 