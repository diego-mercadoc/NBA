import pandas as pd
import calendar
from datetime import datetime
import time
import random
import numpy as np

class NBADataScraper:
    def __init__(self, start_season=2024, end_season=2024):
        self.start_season = start_season
        self.end_season = end_season
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
    
    def validate_season_data(self, df, season):
        """Validate season data for completeness and accuracy"""
        if df is None or df.empty:
            print(f"Warning: No data found for season {season}")
            return False
        
        # Check number of games (around 1230 for a regular season)
        expected_games = 1230  # 82 games * 30 teams / 2
        actual_games = len(df)
        if abs(actual_games - expected_games) > 100:  # Allow some flexibility for shortened seasons
            print(f"Warning: Unexpected number of games for season {season}. Found {actual_games}, expected ~{expected_games}")
        
        # Check for duplicate games
        duplicates = df.duplicated(subset=['Date', 'Away_Team', 'Home_Team'], keep=False)
        if duplicates.any():
            print(f"Warning: Found {duplicates.sum()} duplicate games in season {season}")
            df = df.drop_duplicates(subset=['Date', 'Away_Team', 'Home_Team'], keep='first')
        
        return True
    
    def get_monthly_games(self, season, month):
        """Scrape games for a specific month and season"""
        # Try single-page approach first for recent seasons
        if month is None:
            url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
        else:
            url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games-{month}.html"
        
        try:
            print(f"Fetching {url}")
            dfs = pd.read_html(url)
            if dfs:
                df = dfs[0]
                # Check if this looks like a games table
                if not all(col in df.columns for col in ['Home/Neutral', 'Visitor/Neutral', 'PTS', 'PTS.1']):
                    print(f"Warning: Unexpected table format at {url}")
                    return None
                
                # Clean column names
                df.rename(columns={
                    'Visitor/Neutral': 'Away_Team',
                    'Home/Neutral': 'Home_Team',
                    'PTS': 'Away_Points',
                    'PTS.1': 'Home_Points',
                    'Start (ET)': 'Start_Time'
                }, inplace=True)
                
                # Add season and month info
                df['Season'] = season
                df['Month'] = month if month else ''
                
                # Normalize team names
                df['Away_Team'] = df['Away_Team'].replace(self.team_name_map)
                df['Home_Team'] = df['Home_Team'].replace(self.team_name_map)
                
                return df
        except Exception as e:
            if "404 Client Error" in str(e):
                print(f"Page not found: {url}")
            else:
                print(f"Error scraping {url}: {str(e)}")
            return None
    
    def clean_games_data(self, games_df):
        """Clean and enhance games data"""
        if games_df is None or games_df.empty:
            return None
        
        # Convert date and handle timezones
        games_df['Date'] = pd.to_datetime(games_df['Date'])
        
        # Clean numeric data
        games_df['Away_Points'] = pd.to_numeric(games_df['Away_Points'], errors='coerce')
        games_df['Home_Points'] = pd.to_numeric(games_df['Home_Points'], errors='coerce')
        
        # Remove future games (no points)
        games_df = games_df.dropna(subset=['Away_Points', 'Home_Points'])
        
        # Add game outcome columns
        games_df['Home_Win'] = (games_df['Home_Points'] > games_df['Away_Points']).astype(int)
        games_df['Point_Diff'] = games_df['Home_Points'] - games_df['Away_Points']
        
        # Calculate days of rest
        games_df = self.add_rest_days(games_df)
        
        # Add winning/losing streaks
        games_df = self.add_streaks(games_df)
        
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
    
    def get_all_games(self):
        """Scrape games for all specified seasons"""
        all_games = []
        months = ["october", "november", "december", "january", 
                 "february", "march", "april", "may", "june"]
        
        for season in range(self.start_season, self.end_season + 1):
            print(f"\nScraping season {season}...")
            season_games = []
            
            # Always try both approaches for completeness
            # First try the season page
            df = self.get_monthly_games(season, None)
            if df is not None:
                season_games.append(df)
            
            # Then try monthly pages
            for month in months:
                print(f"Scraping {month} {season}...")
                df = self.get_monthly_games(season, month)
                if df is not None:
                    season_games.append(df)
                time.sleep(random.uniform(1, 3))
            
            if season_games:
                season_df = pd.concat(season_games, ignore_index=True)
                # Remove duplicates before validation
                season_df = season_df.drop_duplicates(subset=['Date', 'Away_Team', 'Home_Team'], keep='first')
                if self.validate_season_data(season_df, season):
                    all_games.append(season_df)
        
        if all_games:
            games_df = pd.concat(all_games, ignore_index=True)
            return self.clean_games_data(games_df)
        return None
    
    def get_team_stats(self, season):
        """Scrape advanced team stats for a specific season"""
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
        try:
            print(f"Fetching {url}")
            dfs = pd.read_html(url)
            
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
                    'SRS': 'SRS'
                }
                
                # Map the actual column names to our desired names
                actual_cols = {
                    'Team': [col for col in df.columns if 'Team' in col][0],
                    'ORtg': [col for col in df.columns if 'ORtg' in col][0],
                    'DRtg': [col for col in df.columns if 'DRtg' in col][0],
                    'Pace': [col for col in df.columns if 'Pace' in col][0],
                    'W': [col for col in df.columns if col.endswith('W')][0],
                    'L': [col for col in df.columns if col.endswith('L')][0],
                    'SRS': [col for col in df.columns if 'SRS' in col][0]
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
                keep_cols = ['Team', 'Season', 'Wins', 'Losses', 'ORtg', 'DRtg', 'Pace', 'SRS']
                df = df[keep_cols]
                
                # Convert numeric columns
                numeric_cols = ['Wins', 'Losses', 'ORtg', 'DRtg', 'Pace', 'SRS']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            
            print(f"Warning: Could not find team stats table at {url}")
            return None
        except Exception as e:
            print(f"Error scraping team stats for {season}: {str(e)}")
            return None
    
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

def main():
    # Initialize scraper for multiple seasons (e.g., 2021-2024)
    scraper = NBADataScraper(start_season=2021, end_season=2024)
    
    # Get games data
    print("Scraping games data...")
    games_df = scraper.get_all_games()
    if games_df is not None:
        print(f"Found {len(games_df)} games")
        
        # Compute rolling statistics
        print("\nComputing rolling statistics...")
        games_df = scraper.compute_rolling_stats(games_df, window=5)
        
        # Basic data validation
        print("\nValidating data...")
        games_per_season = games_df.groupby('Season').size()
        print("\nGames per season:")
        print(games_per_season)
        
        duplicates = games_df.duplicated(subset=['Date', 'Away_Team', 'Home_Team']).sum()
        print(f"\nDuplicate games found: {duplicates}")
        
        # Save to CSV
        print("\nSaving games data...")
        games_df.to_csv('nba_games_all.csv', index=False)
    
    # Get team stats
    print("\nScraping team stats...")
    team_stats = scraper.get_all_team_stats()
    if team_stats is not None:
        print(f"Found stats for {len(team_stats)} team-seasons")
        
        # Basic validation
        teams_per_season = team_stats.groupby('Season').size()
        print("\nTeams per season:")
        print(teams_per_season)
        
        # Save to CSV
        print("\nSaving team stats...")
        team_stats.to_csv('nba_team_stats_all.csv', index=False)

if __name__ == "__main__":
    main() 
