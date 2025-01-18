import pandas as pd
import calendar
from datetime import datetime, timedelta
import time
import random
import numpy as np
import logging
import requests
from requests.exceptions import RequestException, HTTPError
import json  # Add at top with other imports
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_scraper.log'),
        logging.StreamHandler()
    ]
)

class NBADataScraper:
    """
    A class to scrape and process NBA game data and team statistics from Basketball-Reference.
    
    Features:
    - Scrapes game results and advanced team statistics
    - Calculates rest days between games for each team
    - Tracks winning/losing streaks
    - Computes rolling statistics (points scored, allowed, differential)
    - Validates data completeness and accuracy
    """
    
    def __init__(self, start_season, end_season):
        self.start_season = start_season
        self.end_season = end_season
        self.min_request_interval = 0.5  # Reduced from 30 to 0.5 seconds
        self.max_retries = 5
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.last_request_time = 0
        self.progress_file = 'scraper_progress.json'
        self.load_progress()
        
        # Expanded team name mapping
        self.team_name_map = {
            "LA Clippers": "Los Angeles Clippers",
            "LA Lakers": "Los Angeles Lakers",
            "NY Knicks": "New York Knicks",
            "GS Warriors": "Golden State Warriors",
            "SA Spurs": "San Antonio Spurs",
            "UTAH": "Utah Jazz",
            "PHO": "Phoenix Suns",
            "PHI": "Philadelphia 76ers",
            "BRK": "Brooklyn Nets",
            "CHO": "Charlotte Hornets"
        }
    
    def load_progress(self):
        """Load scraping progress from file."""
        try:
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
            logging.info(f"Loaded progress from {self.progress_file}")
        except FileNotFoundError:
            self.progress = {
                'last_season': None,
                'last_month': None,
                'completed_seasons': []
            }
            self.save_progress()

    def save_progress(self):
        """Save scraping progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
        logging.info(f"Saved progress to {self.progress_file}")

    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests with randomization."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        base_wait = 0.5  # Reduced from previous value
        
        # Add smaller random jitter between 0-1 seconds
        jitter = random.uniform(0, 1)
        total_wait = base_wait + jitter
        
        if elapsed < total_wait:
            wait_time = total_wait - elapsed
            logging.info(f"Rate limiting: waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def _make_request(self, url, retries=0):
        """Make a request with exponential backoff retry logic and randomization."""
        if retries >= self.max_retries:
            raise Exception(f"Max retries ({self.max_retries}) exceeded for URL: {url}")
        
        try:
            self._wait_for_rate_limit()
            response = self.session.get(url)
            response.raise_for_status()
            
            # Reduced delay after successful request (0.5-1 seconds)
            time.sleep(random.uniform(0.5, 1))
            
            dfs = pd.read_html(StringIO(response.text))
            return dfs
        except (requests.exceptions.RequestException, HTTPError) as e:
            # Base wait time with randomization
            base_wait = (2 ** retries) * self.min_request_interval
            jitter = random.uniform(0, base_wait * 0.1)  # 10% jitter
            wait_time = base_wait + jitter
            
            logging.warning(f"Request failed ({str(e)}), waiting {wait_time:.2f} seconds before retry {retries + 1}/{self.max_retries}")
            time.sleep(wait_time)
            return self._make_request(url, retries + 1)
    
    def validate_season_data(self, df, season):
        """
        Validate season data for completeness and accuracy.
        
        Args:
            df (pd.DataFrame): DataFrame containing season's game data
            season (int): NBA season year (e.g., 2024 for 2023-24 season)
            
        Returns:
            bool: True if data passes validation, False otherwise
        """
        if df is None or df.empty:
            logging.warning(f"No data found for season {season}")
            return False
        
        # Get current date for validation
        current_date = datetime.now()
        season_start_year = season - 1
        season_end_year = season
        
        # Define expected games for different seasons
        expected_games = {
            2021: 1080,  # COVID-shortened season (72 games per team)
            2022: 1230,  # Regular 82-game season
            2023: 1230,
            2024: 1230,
            2025: None   # Current season, partial data expected
        }
        
        # Check if we're in current season
        is_current_season = (
            (current_date.year == season_start_year and current_date.month >= 10) or
            (current_date.year == season_end_year and current_date.month <= 6)
        )
        
        # For current season
        if is_current_season:
            # Convert dates if needed
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Split into played and scheduled games
            played_games = df[df['Date'] <= current_date]
            scheduled_games = df[
                (df['Date'] > current_date) & 
                (df['Date'] <= pd.Timestamp('2025-04-13'))
            ]
            
            logging.info(f"Current season ({season_start_year}-{season_end_year}) status:")
            logging.info(f"- Played games: {len(played_games)}")
            logging.info(f"- Scheduled future games: {len(scheduled_games)}")
            logging.info(f"- Total games: {len(df)}")
            
            # Basic validation for current season
            if len(played_games) < 100:  # Minimum games for season start
                logging.warning("Unusually low number of played games")
                return False
            
            return True
            
        # For completed seasons
        elif season in expected_games and expected_games[season] is not None:
            actual_games = len(df)
            if abs(actual_games - expected_games[season]) > 50:  # Allow some flexibility
                logging.warning(
                    f"Unexpected number of games for season {season}. "
                    f"Found {actual_games}, expected {expected_games[season]}"
                )
                return False
        
        # Check for duplicate games
        duplicates = df.duplicated(subset=['Date', 'Away_Team', 'Home_Team'], keep=False)
        if duplicates.any():
            logging.warning(f"Found {duplicates.sum()} duplicate games in season {season}")
            df = df.drop_duplicates(subset=['Date', 'Away_Team', 'Home_Team'], keep='first')
        
        logging.info(f"Validation complete for season {season}")
        return True
    
    def get_monthly_games(self, season, month):
        """
        Scrape games for a specific month and season from Basketball-Reference.
        
        Args:
            season (int): NBA season year (e.g., 2024 for 2023-24 season)
            month (str or None): Month name in lowercase, or None for full season page
            
        Returns:
            pd.DataFrame or None: DataFrame containing game data if successful, None otherwise
        """
        # Build URL based on season
        if month is None:
            url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
        else:
            url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games-{month}.html"
        
        try:
            logging.info(f"Fetching data from {url}")
            dfs = self._make_request(url)
            if dfs:
                df = dfs[0]
                # Check if this looks like a games table
                required_cols = ['Home/Neutral', 'Visitor/Neutral', 'PTS', 'PTS.1']
                if not all(col in df.columns for col in required_cols):
                    logging.warning(f"Unexpected table format at {url}")
                    return None
                
                # Clean column names
                df.rename(columns={
                    'Visitor/Neutral': 'Away_Team',
                    'Home/Neutral': 'Home_Team',
                    'PTS': 'Away_Points',
                    'PTS.1': 'Home_Points',
                    'Start (ET)': 'Start_Time'
                }, inplace=True)
                
                # Add season info
                df['Season'] = season
                df['Month'] = month if month else ''
                
                # Normalize team names
                df['Away_Team'] = df['Away_Team'].replace(self.team_name_map)
                df['Home_Team'] = df['Home_Team'].replace(self.team_name_map)
                
                # Convert dates and handle season transitions
                df['Date'] = pd.to_datetime(df['Date'])
                
                # For current season (2024-25), mark future and scheduled games
                current_date = datetime.now()
                if season == 2025:
                    df['Is_Future'] = df['Date'] > current_date
                    df['Is_Scheduled'] = df['Date'] <= pd.Timestamp('2025-04-13')
                    df['Is_Played'] = ~df['Is_Future']
                    
                    future_count = df['Is_Future'].sum()
                    scheduled_count = df['Is_Scheduled'].sum()
                    played_count = df['Is_Played'].sum()
                    
                    logging.info(f"Current season status for {month if month else 'full season'} {season}:")
                    logging.info(f"- Played games: {played_count}")
                    logging.info(f"- Future scheduled games: {scheduled_count}")
                    logging.info(f"- Future unscheduled games: {future_count - scheduled_count}")
                
                logging.info(f"Successfully scraped {len(df)} games for {month if month else 'full season'} {season}")
                return df

        except Exception as e:
            if "404 Client Error" in str(e):
                logging.warning(f"Page not found: {url}")
            else:
                logging.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def clean_games_data(self, games_df):
        """
        Clean and enhance the games dataset with additional metrics.
        
        Args:
            games_df (pd.DataFrame): Raw games DataFrame
            
        Returns:
            pd.DataFrame or None: Cleaned and enhanced DataFrame if successful, None if input is invalid
            
        Enhancements:
        - Converts dates to datetime format
        - Cleans numeric data (points)
        - Removes future/unplayed games
        - Adds game outcome columns
        - Calculates rest days between games
        - Adds winning/losing streaks
        """
        if games_df is None or games_df.empty:
            logging.warning("Received empty or None games DataFrame")
            return None

        logging.info("Starting data cleaning process...")
        
        # Convert date and handle timezones
        games_df['Date'] = pd.to_datetime(games_df['Date'])
        logging.info("Converted dates to datetime format")

        # Clean numeric data
        games_df['Away_Points'] = pd.to_numeric(games_df['Away_Points'], errors='coerce')
        games_df['Home_Points'] = pd.to_numeric(games_df['Home_Points'], errors='coerce')
        logging.info("Converted points to numeric format")

        # Remove future games (no points)
        initial_rows = len(games_df)
        games_df = games_df.dropna(subset=['Away_Points', 'Home_Points'])
        removed_rows = initial_rows - len(games_df)
        if removed_rows > 0:
            logging.info(f"Removed {removed_rows} future/invalid games")

        # Add game outcome columns
        games_df['Home_Win'] = (games_df['Home_Points'] > games_df['Away_Points']).astype(int)
        games_df['Point_Diff'] = games_df['Home_Points'] - games_df['Away_Points']
        logging.info("Added game outcome columns")

        # Calculate days of rest
        logging.info("Calculating rest days...")
        games_df = self.add_rest_days(games_df)

        # Add winning/losing streaks
        logging.info("Calculating team streaks...")
        games_df = self.add_streaks(games_df)

        logging.info(f"Data cleaning complete. Final dataset contains {len(games_df)} games")
        return games_df
    
    def add_rest_days(self, df):
        """Calculate days of rest for each team"""
        df = df.sort_values('Date')
        
        for team_type in ['Home', 'Away']:
            rest_days = []
            for team in df[f'{team_type}_Team'].unique():
                team_games = df[df[f'{team_type}_Team'] == team].copy()
                team_games['Days_Rest'] = (team_games['Date'] - team_games['Date'].shift(1)).dt.days
                rest_days.extend(team_games['Days_Rest'].tolist())
            df[f'{team_type}_Rest_Days'] = rest_days
        
        return df
    
    def add_streaks(self, df):
        """Calculate winning and losing streaks for each team"""
        df = df.sort_values('Date')
        df['Home_Streak'] = 0
        df['Away_Streak'] = 0
        
        for team in df['Home_Team'].unique():
            # Get all games for the team (both home and away)
            team_games = df[(df['Home_Team'] == team) | (df['Away_Team'] == team)].copy()
            team_games = team_games.sort_values('Date')
            
            # Calculate streak
            current_streak = 0
            for idx, row in team_games.iterrows():
                if row['Home_Team'] == team:
                    won = row['Home_Win'] == 1
                else:
                    won = row['Home_Win'] == 0
                
                if won:
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                else:
                    current_streak = current_streak - 1 if current_streak < 0 else -1
                
                # Update the streak in the original dataframe
                if row['Home_Team'] == team:
                    df.loc[idx, 'Home_Streak'] = current_streak
                else:
                    df.loc[idx, 'Away_Streak'] = current_streak
        
        return df
    
    def get_all_games(self, recent_only=False):
        """
        Get all games data for specified seasons.
        
        Args:
            recent_only (bool): If True, only scrape the current season's games
        """
        all_games = []
        seasons = range(self.start_season, self.end_season + 1)
        
        if recent_only:
            seasons = [self.end_season]
            logging.info("Recent only mode - scraping only current season")
        
        for season in seasons:
            if season in self.progress['completed_seasons'] and not recent_only:
                logging.info(f"Skipping completed season {season}")
                continue
                
            logging.info(f"\nScraping season {season}...")
            
            try:
                season_games = self.get_season_games(season)
                if season_games is not None and isinstance(season_games, pd.DataFrame):
                    all_games.append(season_games)
                    
                    if not recent_only:
                        self.progress['completed_seasons'].append(season)
                        self.save_progress()
                
                # Add cooldown between seasons
                if season != seasons[-1]:
                    cooldown = random.uniform(1, 2)  # Reduced from previous values
                    logging.info(f"Season complete - cooling down for {cooldown:.2f} seconds")
                    time.sleep(cooldown)
                    
            except Exception as e:
                logging.error(f"Error scraping season {season}: {str(e)}")
                self.progress['last_season'] = season
                self.save_progress()
                raise e
        
        if all_games:
            try:
                return pd.concat(all_games, ignore_index=True)
            except Exception as e:
                logging.error(f"Error concatenating game data: {str(e)}")
                return None
        return None

    def get_season_games(self, season):
        """
        Get all games for a specific season with progress tracking.
        
        Args:
            season (int): NBA season year (e.g., 2024 for 2023-24 season)
            
        Returns:
            pd.DataFrame: DataFrame containing all games for the season
        """
        # Define season-specific month ranges with proper year mapping
        season_months = {
            2021: {  # COVID season
                'start_year': 2020,
                'months': [
                    ('december', 2020),
                    ('january', 2021), ('february', 2021), ('march', 2021),
                    ('april', 2021), ('may', 2021), ('june', 2021), ('july', 2021)
                ]
            },
            2022: {  # Regular season
                'start_year': 2021,
                'months': [
                    ('october', 2021), ('november', 2021), ('december', 2021),
                    ('january', 2022), ('february', 2022), ('march', 2022),
                    ('april', 2022), ('may', 2022), ('june', 2022)
                ]
            },
            2023: {
                'start_year': 2022,
                'months': [
                    ('october', 2022), ('november', 2022), ('december', 2022),
                    ('january', 2023), ('february', 2023), ('march', 2023),
                    ('april', 2023), ('may', 2023), ('june', 2023)
                ]
            },
            2024: {
                'start_year': 2023,
                'months': [
                    ('october', 2023), ('november', 2023), ('december', 2023),
                    ('january', 2024), ('february', 2024), ('march', 2024),
                    ('april', 2024), ('may', 2024), ('june', 2024)
                ]
            },
            2025: {  # Current season
                'start_year': 2024,
                'months': [
                    ('october', 2024), ('november', 2024), ('december', 2024),
                    ('january', 2025), ('february', 2025), ('march', 2025),
                    ('april', 2025)  # Only scheduled through April 13, 2025
                ]
            }
        }
        
        if season not in season_months:
            logging.error(f"Invalid season year: {season}")
            return None
            
        season_games = []
        season_config = season_months[season]
        
        # Try full season page first
        try:
            full_season_df = self.get_monthly_games(season, None)
            if full_season_df is not None and not full_season_df.empty:
                logging.info(f"Successfully scraped full season page for {season}")
                return full_season_df
        except Exception as e:
            logging.warning(f"Could not scrape full season page: {str(e)}")
        
        # If full season page fails, try month by month
        for month, year in season_config['months']:
            if (self.progress['last_season'] == season and 
                self.progress['last_month'] == month):
                logging.info(f"Resuming from {month} {year}")
            
            try:
                month_games = self.get_monthly_games(season, month)
                if month_games is not None:
                    # Add the correct year to the dates
                    month_games['Date'] = pd.to_datetime(month_games['Date'])
                    month_games['Date'] = month_games['Date'].apply(
                        lambda x: x.replace(year=year)
                    )
                    season_games.append(month_games)
                    self.progress['last_month'] = month
                    self.save_progress()
                
                # Reduced cooldown between months
                if month != season_config['months'][-1][0]:
                    cooldown = random.uniform(1, 2)  # Reduced from 30-60 to 1-2 seconds
                    logging.info(f"Month complete - cooling down for {cooldown:.2f} seconds")
                    time.sleep(cooldown)
                    
            except Exception as e:
                logging.error(f"Error scraping {month} {year}: {str(e)}")
                self.progress['last_season'] = season
                self.progress['last_month'] = month
                self.save_progress()
                raise e
        
        if season_games:
            season_df = pd.concat(season_games, ignore_index=True)
            # Remove duplicates before validation
            season_df = season_df.drop_duplicates(subset=['Date', 'Away_Team', 'Home_Team'], keep='first')
            if self.validate_season_data(season_df, season):
                return season_df
        return None
    
    def get_team_stats(self, season):
        """
        Scrape advanced team stats for a specific season.
        
        Args:
            season (int): NBA season year
            
        Returns:
            pd.DataFrame: DataFrame with team stats if successful, None otherwise
            
        Stats include:
        - Basic: Wins, Losses
        - Advanced: ORtg, DRtg, Pace, SRS
        - Four Factors: eFG%, TOV%, ORB%, FT/FGA
        - Quarter/Half Performance:
          * First Half scoring trends
          * Quarter-by-quarter scoring
          * Strong/weak periods
        """
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
        try:
            logging.info(f"Fetching team stats from {url}")
            dfs = self._make_request(url)
            
            # The advanced stats are in table 10
            if len(dfs) > 10:
                df = dfs[10]
                
                # Clean up the column names
                df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
                
                # Rename the columns we want
                col_map = {
                    'Team': 'Team',
                    'ORtg': 'ORtg',
                    'DRtg': 'DRtg',
                    'Pace': 'Pace',
                    'W': 'Wins',
                    'L': 'Losses',
                    'SRS': 'SRS',
                    'eFG%': 'eFG_Pct',
                    'TOV%': 'TOV_Pct',
                    'ORB%': 'ORB_Pct',
                    'FT/FGA': 'FT_Rate'
                }
                
                # Map the actual column names to our desired names
                actual_cols = {
                    'Team': [col for col in df.columns if 'Team' in col][0],
                    'ORtg': [col for col in df.columns if 'ORtg' in col][0],
                    'DRtg': [col for col in df.columns if 'DRtg' in col][0],
                    'Pace': [col for col in df.columns if 'Pace' in col][0],
                    'W': [col for col in df.columns if col.endswith('W')][0],
                    'L': [col for col in df.columns if col.endswith('L')][0],
                    'SRS': [col for col in df.columns if 'SRS' in col][0],
                    'eFG%': [col for col in df.columns if 'eFG%' in col][0],
                    'TOV%': [col for col in df.columns if 'TOV%' in col][0],
                    'ORB%': [col for col in df.columns if 'ORB%' in col][0],
                    'FT/FGA': [col for col in df.columns if 'FT/FGA' in col][0]
                }
                
                rename_map = {actual_cols[old]: new for old, new in col_map.items()}
                df = df.rename(columns=rename_map)
                
                # Remove League Average and duplicate rows
                df = df[~df['Team'].str.contains('League Average|Division', na=False)]
                
                # Add season info
                df['Season'] = season
                
                # Normalize team names
                df['Team'] = df['Team'].replace(self.team_name_map)
                
                # Select and reorder columns
                keep_cols = [
                    'Team', 'Season', 'Wins', 'Losses', 
                    'ORtg', 'DRtg', 'Pace', 'SRS',
                    'eFG_Pct', 'TOV_Pct', 'ORB_Pct', 'FT_Rate'
                ]
                df = df[keep_cols]
                
                # Convert numeric columns
                numeric_cols = [
                    'Wins', 'Losses', 'ORtg', 'DRtg', 'Pace', 'SRS',
                    'eFG_Pct', 'TOV_Pct', 'ORB_Pct', 'FT_Rate'
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Add quarter and half performance stats
                df = self.add_quarter_stats(df, season)
                
                logging.info(f"Successfully scraped team stats for season {season}")
                return df
            
            logging.warning(f"Could not find team stats table at {url}")
            return None
        except Exception as e:
            logging.error(f"Error scraping team stats for {season}: {str(e)}")
            return None

    def add_quarter_stats(self, df, season):
        """
        Add quarter and half-time performance statistics with improved reliability.
        
        Args:
            df (pd.DataFrame): Team stats DataFrame
            season (int): NBA season year
            
        Returns:
            pd.DataFrame: DataFrame with added quarter/half stats
            
        Added statistics:
        - First half scoring average (for/against)
        - Quarter-by-quarter scoring trends
        - Strong/weak periods identification
        """
        try:
            # Initialize quarter stats columns
            quarter_cols = [
                'FirstHalf_Points_For', 'FirstHalf_Points_Against', 'FirstHalf_Margin',
                'Q1_Points_For', 'Q1_Points_Against', 'Q1_Margin',
                'Q2_Points_For', 'Q2_Points_Against', 'Q2_Margin',
                'Q3_Points_For', 'Q3_Points_Against', 'Q3_Margin',
                'Q4_Points_For', 'Q4_Points_Against', 'Q4_Margin'
            ]
            
            for col in quarter_cols:
                df[col] = np.nan
            
            # Get quarter-by-quarter scoring data month by month
            months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']
            
            for month in months:
                url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games-{month}.html"
                logging.info(f"Fetching quarter stats from {url}")
                
                try:
                    games_dfs = self._make_request(url)
                    if not games_dfs:
                        continue
                        
                    games_df = games_dfs[0]
                    
                    # Check if quarter columns exist
                    quarter_patterns = {
                        1: ['1st', 'Q1', '1Q'],
                        2: ['2nd', 'Q2', '2Q'],
                        3: ['3rd', 'Q3', '3Q'],
                        4: ['4th', 'Q4', '4Q']
                    }
                    
                    # Find matching column names
                    quarter_cols_found = {}
                    for q, patterns in quarter_patterns.items():
                        for pattern in patterns:
                            home_col = next((col for col in games_df.columns if f"{pattern}" in col and "Home" in col), None)
                            away_col = next((col for col in games_df.columns if f"{pattern}" in col and "Away" in col), None)
                            if home_col and away_col:
                                quarter_cols_found[q] = (home_col, away_col)
                                break
                    
                    if not quarter_cols_found:
                        logging.warning(f"No quarter columns found for {month} {season}")
                        continue
                    
                    # Process each team's quarter performance
                    for team in df['Team'].unique():
                        # Get home and away games
                        home_mask = games_df['Home/Neutral'] == team
                        away_mask = games_df['Visitor/Neutral'] == team
                        home_games = games_df[home_mask]
                        away_games = games_df[away_mask]
                        
                        if len(home_games) == 0 and len(away_games) == 0:
                            continue
                        
                        # Calculate quarter stats
                        for quarter, (home_col, away_col) in quarter_cols_found.items():
                            try:
                                # Convert to numeric, handling any non-numeric values
                                home_games[home_col] = pd.to_numeric(home_games[home_col], errors='coerce')
                                home_games[away_col] = pd.to_numeric(home_games[away_col], errors='coerce')
                                away_games[home_col] = pd.to_numeric(away_games[home_col], errors='coerce')
                                away_games[away_col] = pd.to_numeric(away_games[away_col], errors='coerce')
                                
                                # Calculate averages
                                points_for = (
                                    (home_games[home_col].mean() * len(home_games) +
                                     away_games[away_col].mean() * len(away_games))
                                    / (len(home_games) + len(away_games))
                                )
                                
                                points_against = (
                                    (home_games[away_col].mean() * len(home_games) +
                                     away_games[home_col].mean() * len(away_games))
                                    / (len(home_games) + len(away_games))
                                )
                                
                                # Update DataFrame
                                df.loc[df['Team'] == team, f'Q{quarter}_Points_For'] = points_for
                                df.loc[df['Team'] == team, f'Q{quarter}_Points_Against'] = points_against
                                df.loc[df['Team'] == team, f'Q{quarter}_Margin'] = points_for - points_against
                                
                            except Exception as e:
                                logging.warning(f"Error processing Q{quarter} for {team}: {str(e)}")
                                continue
                        
                        # Calculate first half stats
                        try:
                            q1_for = df.loc[df['Team'] == team, 'Q1_Points_For'].iloc[0]
                            q2_for = df.loc[df['Team'] == team, 'Q2_Points_For'].iloc[0]
                            q1_against = df.loc[df['Team'] == team, 'Q1_Points_Against'].iloc[0]
                            q2_against = df.loc[df['Team'] == team, 'Q2_Points_Against'].iloc[0]
                            
                            if not np.isnan([q1_for, q2_for, q1_against, q2_against]).any():
                                df.loc[df['Team'] == team, 'FirstHalf_Points_For'] = q1_for + q2_for
                                df.loc[df['Team'] == team, 'FirstHalf_Points_Against'] = q1_against + q2_against
                                df.loc[df['Team'] == team, 'FirstHalf_Margin'] = (q1_for + q2_for) - (q1_against + q2_against)
                        
                        except Exception as e:
                            logging.warning(f"Error calculating first half stats for {team}: {str(e)}")
                            continue
                    
                except Exception as e:
                    logging.warning(f"Error processing {month} {season}: {str(e)}")
                    continue
            
            logging.info("Successfully added quarter and half performance stats")
            return df
            
        except Exception as e:
            logging.error(f"Error adding quarter stats: {str(e)}")
            return df
    
    def get_all_team_stats(self):
        """Scrape team stats for all specified seasons"""
        all_stats = []
        for season in range(self.start_season, self.end_season + 1):
            print(f"\nScraping team stats for season {season}...")
            df = self.get_team_stats(season)
            if df is not None:
                all_stats.append(df)
            time.sleep(random.uniform(1, 3))
        
        if all_stats:
            return pd.concat(all_stats, ignore_index=True)
        return None
    
    def compute_rolling_stats(self, games_df, window=5):
        """Compute rolling statistics for teams"""
        games_df = games_df.copy()
        
        # Sort by date
        games_df.sort_values('Date', inplace=True)
        
        # Initialize columns for rolling stats
        stats = ['Points_Scored', 'Points_Allowed', 'Point_Diff']
        for stat in stats:
            games_df[f'Home_{stat}_Roll{window}'] = np.nan
            games_df[f'Away_{stat}_Roll{window}'] = np.nan
        
        # Compute rolling stats for each team
        for team in games_df['Home_Team'].unique():
            # Home games
            mask_home = games_df['Home_Team'] == team
            games_df.loc[mask_home, f'Home_Points_Scored_Roll{window}'] = (
                games_df.loc[mask_home, 'Home_Points']
                .rolling(window, min_periods=1).mean()
            )
            games_df.loc[mask_home, f'Home_Points_Allowed_Roll{window}'] = (
                games_df.loc[mask_home, 'Away_Points']
                .rolling(window, min_periods=1).mean()
            )
            games_df.loc[mask_home, f'Home_Point_Diff_Roll{window}'] = (
                games_df.loc[mask_home, 'Point_Diff']
                .rolling(window, min_periods=1).mean()
            )
            
            # Away games
            mask_away = games_df['Away_Team'] == team
            games_df.loc[mask_away, f'Away_Points_Scored_Roll{window}'] = (
                games_df.loc[mask_away, 'Away_Points']
                .rolling(window, min_periods=1).mean()
            )
            games_df.loc[mask_away, f'Away_Points_Allowed_Roll{window}'] = (
                games_df.loc[mask_away, 'Home_Points']
                .rolling(window, min_periods=1).mean()
            )
            games_df.loc[mask_away, f'Away_Point_Diff_Roll{window}'] = (
                -games_df.loc[mask_away, 'Point_Diff']
                .rolling(window, min_periods=1).mean()
            )
        
        return games_df

    def get_player_stats(self, season):
        """
        Scrape player performance statistics for props betting.
        
        Args:
            season (int): NBA season year
            
        Returns:
            pd.DataFrame: DataFrame with player stats if successful, None otherwise
            
        Stats include:
        - Basic: Points, Rebounds, Assists per game
        - Advanced: Usage Rate, Minutes, Efficiency
        - Situational: Home/Away splits, Rest day performance
        - Trends: Last 5/10 games performance
        """
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
        try:
            logging.info(f"Fetching player stats from {url}")
            dfs = pd.read_html(url)
            
            if dfs:
                df = dfs[0]
                
                # Clean up the column names
                df.columns = df.columns.str.replace('%', 'Pct').str.replace('/', '_per_')
                
                # Remove rows for players who changed teams (marked with TOT)
                df = df[df['Tm'] != 'TOT']
                
                # Add season info
                df['Season'] = season
                
                # Calculate per-minute stats
                if 'MP' in df.columns:
                    for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
                        if stat in df.columns:
                            df[f'{stat}_per_36'] = df[stat] * 36 / df['MP']
                
                # Get usage rate and advanced stats
                advanced_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
                logging.info(f"Fetching advanced stats from {advanced_url}")
                
                adv_dfs = pd.read_html(advanced_url)
                if adv_dfs:
                    adv_df = adv_dfs[0]
                    adv_df.columns = adv_df.columns.str.replace('%', 'Pct')
                    
                    # Remove duplicate rows
                    adv_df = adv_df[adv_df['Tm'] != 'TOT']
                    
                    # Select relevant columns
                    adv_cols = ['Player', 'Tm', 'USGPct', 'ORtg', 'DRtg']
                    adv_df = adv_df[adv_cols]
                    
                    # Merge with main stats
                    df = df.merge(adv_df, on=['Player', 'Tm'], how='left')
                
                # Get game logs for recent performance
                df['Recent_Form'] = df.apply(
                    lambda x: self.get_player_recent_form(season, x['Player']),
                    axis=1
                )
                
                logging.info(f"Successfully scraped stats for {len(df)} players")
                return df
            
            logging.warning(f"Could not find player stats table at {url}")
            return None
            
        except Exception as e:
            logging.error(f"Error scraping player stats: {str(e)}")
            return None
    
    def get_player_recent_form(self, season, player_name):
        """
        Get player's recent performance data for props betting.
        
        Args:
            season (int): NBA season year
            player_name (str): Player's name
            
        Returns:
            dict: Dictionary containing recent performance metrics
        """
        try:
            # Convert player name to URL format
            player_url = player_name.lower().replace(' ', '-')
            # Get first 5 chars of last name + first 2 of first name + 01 (BR's format)
            player_id = f"{player_url.split('-')[-1][:5]}{player_url.split('-')[0][:2]}01"
            
            url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{season}"
            logging.info(f"Fetching game log for {player_name}")
            
            dfs = pd.read_html(url)
            if dfs:
                games_df = dfs[7]  # Regular season game log
                if not games_df.empty:
                    # Get last 10 games
                    recent_games = games_df.tail(10)
                    
                    # Calculate averages
                    stats = {
                        'Last10_PTS': recent_games['PTS'].mean(),
                        'Last10_REB': recent_games['TRB'].mean(),
                        'Last10_AST': recent_games['AST'].mean(),
                        'Last10_MIN': recent_games['MP'].mean(),
                        'Last5_PTS': recent_games.tail(5)['PTS'].mean(),
                        'Last5_REB': recent_games.tail(5)['TRB'].mean(),
                        'Last5_AST': recent_games.tail(5)['AST'].mean(),
                        'Last5_MIN': recent_games.tail(5)['MP'].mean(),
                        'Home_PTS': games_df[games_df['Unnamed: 5'] == '@']['PTS'].mean(),
                        'Away_PTS': games_df[games_df['Unnamed: 5'] != '@']['PTS'].mean()
                    }
                    return stats
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting recent form for {player_name}: {str(e)}")
            return None

    def get_player_matchup_stats(self, season, player_name, opponent):
        """
        Get player's performance stats against a specific opponent.
        
        Args:
            season (int): NBA season year
            player_name (str): Player's name
            opponent (str): Opponent team name
            
        Returns:
            dict: Dictionary containing matchup-specific performance metrics
        """
        try:
            # Convert player name to URL format
            player_url = player_name.lower().replace(' ', '-')
            # Get first 5 chars of last name + first 2 of first name + 01 (BR's format)
            player_id = f"{player_url.split('-')[-1][:5]}{player_url.split('-')[0][:2]}01"
            
            url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{season}"
            logging.info(f"Fetching matchup stats for {player_name} vs {opponent}")
            
            dfs = pd.read_html(url)
            if dfs:
                games_df = dfs[7]  # Regular season game log
                if not games_df.empty:
                    # Filter games against the opponent
                    opp_games = games_df[games_df['Opp'] == opponent]
                    
                    if len(opp_games) > 0:
                        # Calculate matchup stats
                        stats = {
                            'Games_Played': len(opp_games),
                            'Points_Avg': opp_games['PTS'].mean(),
                            'Points_Max': opp_games['PTS'].max(),
                            'Points_Min': opp_games['PTS'].min(),
                            'Rebounds_Avg': opp_games['TRB'].mean(),
                            'Assists_Avg': opp_games['AST'].mean(),
                            'Minutes_Avg': opp_games['MP'].mean(),
                            'Last_Meeting': {
                                'Date': opp_games.iloc[-1]['Date'],
                                'Points': opp_games.iloc[-1]['PTS'],
                                'Rebounds': opp_games.iloc[-1]['TRB'],
                                'Assists': opp_games.iloc[-1]['AST']
                            }
                        }
                        
                        # Add performance trends
                        if len(opp_games) >= 2:
                            points_trend = opp_games['PTS'].iloc[-1] - opp_games['PTS'].iloc[-2]
                            stats['Points_Trend'] = points_trend
                            stats['Form_vs_Opponent'] = "🔥" if points_trend > 0 else "❄️"
                        
                        return stats
                    
                    logging.info(f"No games found against {opponent}")
                    return None
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting matchup stats for {player_name} vs {opponent}: {str(e)}")
            return None

    def analyze_player_props(self, player_name, opponent, player_stats=None, recent_form=None):
        """
        Analyze player props betting opportunities.
        
        Args:
            player_name (str): Player's name
            opponent (str): Opponent team name
            player_stats (dict, optional): Player's season stats
            recent_form (dict, optional): Player's recent performance data
            
        Returns:
            dict: Dictionary containing props betting analysis
        """
        try:
            current_season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
            
            # Get matchup-specific stats
            matchup_stats = self.get_player_matchup_stats(current_season, player_name, opponent)
            
            if matchup_stats is None:
                return None
            
            analysis = {
                'Points': {
                    'Season_Avg': player_stats['PTS'] if player_stats else None,
                    'Last_5': recent_form['Last5_PTS'] if recent_form else None,
                    'Vs_Opponent_Avg': matchup_stats['Points_Avg'],
                    'Last_Meeting': matchup_stats['Last_Meeting']['Points'],
                    'Trend': matchup_stats.get('Points_Trend'),
                    'Form': matchup_stats.get('Form_vs_Opponent')
                },
                'Rebounds': {
                    'Season_Avg': player_stats['TRB'] if player_stats else None,
                    'Last_5': recent_form['Last5_REB'] if recent_form else None,
                    'Vs_Opponent_Avg': matchup_stats['Rebounds_Avg'],
                    'Last_Meeting': matchup_stats['Last_Meeting']['Rebounds']
                },
                'Assists': {
                    'Season_Avg': player_stats['AST'] if player_stats else None,
                    'Last_5': recent_form['Last5_AST'] if recent_form else None,
                    'Vs_Opponent_Avg': matchup_stats['Assists_Avg'],
                    'Last_Meeting': matchup_stats['Last_Meeting']['Assists']
                },
                'Minutes': {
                    'Season_Avg': player_stats['MP'] if player_stats else None,
                    'Last_5': recent_form['Last5_MIN'] if recent_form else None,
                    'Vs_Opponent_Avg': matchup_stats['Minutes_Avg']
                }
            }
            
            # Add insights
            insights = []
            
            # Points analysis
            if analysis['Points']['Last_5'] and analysis['Points']['Season_Avg']:
                pts_diff = analysis['Points']['Last_5'] - analysis['Points']['Season_Avg']
                if abs(pts_diff) > 3:
                    trend = "up" if pts_diff > 0 else "down"
                    insights.append(f"Scoring trend: {trend} ({abs(pts_diff):.1f} pts vs season avg)")
            
            if analysis['Points']['Vs_Opponent_Avg'] and analysis['Points']['Season_Avg']:
                matchup_diff = analysis['Points']['Vs_Opponent_Avg'] - analysis['Points']['Season_Avg']
                if abs(matchup_diff) > 3:
                    performance = "better" if matchup_diff > 0 else "worse"
                    insights.append(f"Scores {abs(matchup_diff):.1f} pts {performance} vs {opponent}")
            
            # Minutes analysis
            if analysis['Minutes']['Last_5'] and analysis['Minutes']['Season_Avg']:
                min_diff = analysis['Minutes']['Last_5'] - analysis['Minutes']['Season_Avg']
                if abs(min_diff) > 3:
                    trend = "up" if min_diff > 0 else "down"
                    insights.append(f"Minutes trend: {trend} ({abs(min_diff):.1f} min vs season avg)")
            
            analysis['Insights'] = insights
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing props for {player_name} vs {opponent}: {str(e)}")
            return None

    def get_current_games(self):
        """
        Get games for the current date or next available game date.
        
        Returns:
            pd.DataFrame: DataFrame containing today's games and relevant betting metrics
        """
        try:
            current_date = datetime.now()
            games_df = pd.read_csv('nba_games_all.csv')
            games_df['Date'] = pd.to_datetime(games_df['Date'])
            
            # Get games for today or future games
            future_games = games_df[games_df['Date'] >= current_date].sort_values('Date')
            
            if not future_games.empty:
                next_game_date = future_games.iloc[0]['Date'].date()
                today_games = future_games[future_games['Date'].dt.date == next_game_date]
                
                if not today_games.empty:
                    logging.info(f"\nGames for {next_game_date}:")
                    for _, game in today_games.iterrows():
                        logging.info(f"\n{game['Away_Team']} @ {game['Home_Team']}")
                        logging.info(f"Home Streak: {game['Home_Streak']}, Away Streak: {game['Away_Streak']}")
                        logging.info(f"Home L5 Point Diff: {game['Home_Point_Diff_Roll5']:.1f}")
                        logging.info(f"Away L5 Point Diff: {game['Away_Point_Diff_Roll5']:.1f}")
                
                return today_games
            else:
                logging.info("No upcoming games found in the dataset")
                return None
            
        except Exception as e:
            logging.error(f"Error getting current games: {str(e)}")
            return None

    def get_betting_insights(self, games_df=None):
        """
        Generate betting insights for specified games.
        
        Args:
            games_df (pd.DataFrame, optional): DataFrame of games to analyze. If None, uses today's games.
            
        Returns:
            dict: Dictionary containing betting insights for each game
        """
        try:
            if games_df is None:
                games_df = self.get_current_games()
            
            if games_df is None or games_df.empty:
                return None
            
            insights = {}
            for _, game in games_df.iterrows():
                game_key = f"{game['Away_Team']} @ {game['Home_Team']}"
                
                # Calculate basic metrics
                home_form = "🔥" if game['Home_Streak'] > 0 else "❄️" if game['Home_Streak'] < 0 else "➖"
                away_form = "🔥" if game['Away_Streak'] > 0 else "❄️" if game['Away_Streak'] < 0 else "➖"
                
                # Determine edge based on recent performance
                home_edge = game['Home_Point_Diff_Roll5'] - game['Away_Point_Diff_Roll5']
                edge = "Strong Home" if home_edge > 7 else \
                       "Lean Home" if home_edge > 3 else \
                       "Strong Away" if home_edge < -7 else \
                       "Lean Away" if home_edge < -3 else "Pick'em"
                
                insights[game_key] = {
                    'Home_Form': f"{home_form} ({game['Home_Streak']} streak)",
                    'Away_Form': f"{away_form} ({game['Away_Streak']} streak)",
                    'Home_L5_Diff': f"{game['Home_Point_Diff_Roll5']:.1f}",
                    'Away_L5_Diff': f"{game['Away_Point_Diff_Roll5']:.1f}",
                    'Edge': edge,
                    'Confidence': 'High' if abs(home_edge) > 7 else 'Medium' if abs(home_edge) > 3 else 'Low'
                }
            
            return insights
            
        except Exception as e:
            logging.error(f"Error generating betting insights: {str(e)}")
            return None

def main():
    """
    Main execution function that orchestrates the NBA data scraping process.
    """
    try:
        logging.info("Starting NBA data scraping process...")
        
        # Initialize scraper for multiple seasons (2021-2025)
        scraper = NBADataScraper(start_season=2021, end_season=2025)
        
        # First, try to load existing data
        try:
            existing_games = pd.read_csv('nba_games_all.csv')
            existing_games['Date'] = pd.to_datetime(existing_games['Date'])
            logging.info(f"Loaded {len(existing_games)} existing games")
        except FileNotFoundError:
            existing_games = None
            logging.info("No existing games data found")
        
        # Determine if we need a full scrape or just an update
        if existing_games is not None:
            last_game_date = existing_games['Date'].max()
            current_date = datetime.now()
            days_since_update = (current_date - last_game_date).days
            
            if days_since_update < 1:
                logging.info("Data is up to date, no scraping needed")
                games_df = existing_games
            else:
                logging.info(f"Data is {days_since_update} days old, updating current season only")
                games_df = scraper.get_all_games(recent_only=True)
                if games_df is not None:
                    # Remove any existing games from the current season
                    existing_games = existing_games[existing_games['Season'] != 2025]
                    # Combine with new games
                    games_df = pd.concat([existing_games, games_df], ignore_index=True)
        else:
            # No existing data, do a full scrape
            logging.info("Scraping all seasons...")
            games_df = scraper.get_all_games()
        
        if games_df is not None:
            logging.info(f"Successfully scraped {len(games_df)} games")
            
            # Compute rolling statistics
            logging.info("Computing rolling statistics...")
            games_df = scraper.compute_rolling_stats(games_df, window=5)
            
            # Basic data validation
            logging.info("Validating data...")
            games_per_season = games_df.groupby('Season').size()
            logging.info("\nGames per season:")
            for season, count in games_per_season.items():
                logging.info(f"Season {season}: {count} games")
            
            duplicates = games_df.duplicated(subset=['Date', 'Away_Team', 'Home_Team']).sum()
            if duplicates > 0:
                logging.warning(f"Found {duplicates} duplicate games")
                games_df = games_df.drop_duplicates(subset=['Date', 'Away_Team', 'Home_Team'], keep='first')
            else:
                logging.info("No duplicate games found")
            
            # Save to CSV
            logging.info("Saving games data...")
            games_df.to_csv('nba_games_all.csv', index=False)
            logging.info("Games data saved successfully")
            
            # Get current games and betting insights
            logging.info("\nAnalyzing current games...")
            current_games = scraper.get_current_games()
            if current_games is not None:
                betting_insights = scraper.get_betting_insights(current_games)
                if betting_insights:
                    logging.info("\nBetting Insights:")
                    for game, insights in betting_insights.items():
                        logging.info(f"\n{game}")
                        logging.info(f"Home Form: {insights['Home_Form']}")
                        logging.info(f"Away Form: {insights['Away_Form']}")
                        logging.info(f"Edge: {insights['Edge']} ({insights['Confidence']} confidence)")
        else:
            logging.error("Failed to scrape games data")
            return
        
        # Get team stats only if needed
        team_stats = None  # Initialize to None
        try:
            existing_stats = pd.read_csv('nba_team_stats_all.csv')
            if 2025 not in existing_stats['Season'].unique():
                logging.info("Updating team stats for current season...")
                current_stats = scraper.get_team_stats(2025)
                if current_stats is not None:
                    existing_stats = existing_stats[existing_stats['Season'] != 2025]
                    team_stats = pd.concat([existing_stats, current_stats], ignore_index=True)
                    team_stats.to_csv('nba_team_stats_all.csv', index=False)
            else:
                logging.info("Team stats are up to date")
                team_stats = existing_stats
        except FileNotFoundError:
            logging.info("Scraping all team stats...")
            team_stats = scraper.get_all_team_stats()
            if team_stats is not None:
                team_stats.to_csv('nba_team_stats_all.csv', index=False)
        
        if team_stats is not None:
            logging.info(f"Successfully processed stats for {len(team_stats)} team-seasons")
            
            # Basic validation
            teams_per_season = team_stats.groupby('Season').size()
            logging.info("\nTeams per season:")
            for season, count in teams_per_season.items():
                logging.info(f"Season {season}: {count} teams")
                if count != 30 and season != 2025:  # Current season might be incomplete
                    logging.warning(f"Unexpected number of teams in season {season}: {count} (expected 30)")
        
        logging.info("\nNBA data scraping process completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred during the scraping process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
