import pandas as pd
import numpy as np
import logging
import requests
from datetime import datetime, timedelta
import time
import random
from bs4 import BeautifulSoup
import json

class NBAOddsScraper:
    """
    Scraper for NBA betting odds from OddsPortal.
    
    Features:
    - Scrapes moneyline, spread, and totals odds
    - Tracks line movements
    - Aggregates odds from multiple bookmakers
    - Calculates consensus lines and market efficiency
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.base_url = "https://www.oddsportal.com/basketball/usa/nba"
        self.min_request_interval = 2.0
        self.last_request_time = 0
        
        # Bookmakers to track (in order of priority)
        self.bookmakers = [
            "bet365", "pinnacle", "unibet", "williamhill",
            "betway", "draftkings", "fanduel", "caesars"
        ]
    
    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests with randomization."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed + random.uniform(0, 1)
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def get_game_odds(self, game_date=None):
        """
        Get odds for NBA games on a specific date.
        
        Args:
            game_date (datetime, optional): Date to get odds for. Defaults to today.
            
        Returns:
            pd.DataFrame: DataFrame containing odds data
        """
        try:
            if game_date is None:
                game_date = datetime.now()
            
            date_str = game_date.strftime("%Y%m%d")
            url = f"{self.base_url}/results/#{date_str}"
            
            self._wait_for_rate_limit()
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            games_data = []
            
            # Find all game rows
            game_rows = soup.find_all('tr', class_='deactivate')
            
            for row in game_rows:
                try:
                    teams = row.find('td', class_='name-participate').text.strip().split(' - ')
                    if len(teams) != 2:
                        continue
                        
                    away_team, home_team = teams
                    
                    # Get odds for different markets
                    odds_data = {
                        'Date': game_date,
                        'Away_Team': away_team,
                        'Home_Team': home_team,
                        'Moneyline_Away': None,
                        'Moneyline_Home': None,
                        'Spread': None,
                        'Spread_Odds_Away': None,
                        'Spread_Odds_Home': None,
                        'Total': None,
                        'Total_Over_Odds': None,
                        'Total_Under_Odds': None,
                        'Best_Bookie_ML': None,
                        'Best_Bookie_Spread': None,
                        'Best_Bookie_Total': None,
                        'Consensus_Line_Movement': None,
                        'Sharp_Money_Indicator': None
                    }
                    
                    # Get detailed odds page for this game
                    game_link = row.find('a', href=True)['href']
                    self._wait_for_rate_limit()
                    detailed_response = self.session.get(f"https://www.oddsportal.com{game_link}")
                    detailed_soup = BeautifulSoup(detailed_response.text, 'html.parser')
                    
                    # Extract odds from each bookmaker
                    for bookie in self.bookmakers:
                        bookie_row = detailed_soup.find('tr', {'data-bookie': bookie})
                        if bookie_row:
                            # Extract moneyline odds
                            ml_cells = bookie_row.find_all('td', class_='right')
                            if len(ml_cells) >= 2:
                                away_odds = float(ml_cells[0].text.strip())
                                home_odds = float(ml_cells[1].text.strip())
                                
                                if odds_data['Moneyline_Away'] is None or away_odds > odds_data['Moneyline_Away']:
                                    odds_data['Moneyline_Away'] = away_odds
                                    odds_data['Best_Bookie_ML'] = bookie
                                
                                if odds_data['Moneyline_Home'] is None or home_odds > odds_data['Moneyline_Home']:
                                    odds_data['Moneyline_Home'] = home_odds
                    
                    # Get spread and totals from separate tabs
                    markets = {'#ah': 'spread', '#ou': 'totals'}
                    for market_tab, market_type in markets.items():
                        self._wait_for_rate_limit()
                        market_url = f"https://www.oddsportal.com{game_link}{market_tab}"
                        market_response = self.session.get(market_url)
                        market_soup = BeautifulSoup(market_response.text, 'html.parser')
                        
                        if market_type == 'spread':
                            spread_row = market_soup.find('tr', class_='odd')
                            if spread_row:
                                spread = float(spread_row.find('td').text.strip())
                                odds_data['Spread'] = spread
                                
                                spread_odds = spread_row.find_all('td', class_='right')
                                if len(spread_odds) >= 2:
                                    odds_data['Spread_Odds_Away'] = float(spread_odds[0].text.strip())
                                    odds_data['Spread_Odds_Home'] = float(spread_odds[1].text.strip())
                                    odds_data['Best_Bookie_Spread'] = spread_row.get('data-bookie', '')
                        
                        elif market_type == 'totals':
                            totals_row = market_soup.find('tr', class_='odd')
                            if totals_row:
                                total = float(totals_row.find('td').text.strip())
                                odds_data['Total'] = total
                                
                                total_odds = totals_row.find_all('td', class_='right')
                                if len(total_odds) >= 2:
                                    odds_data['Total_Over_Odds'] = float(total_odds[0].text.strip())
                                    odds_data['Total_Under_Odds'] = float(total_odds[1].text.strip())
                                    odds_data['Best_Bookie_Total'] = totals_row.get('data-bookie', '')
                    
                    # Calculate sharp money indicators
                    if odds_data['Moneyline_Away'] and odds_data['Moneyline_Home']:
                        implied_prob_away = 1 / odds_data['Moneyline_Away']
                        implied_prob_home = 1 / odds_data['Moneyline_Home']
                        market_efficiency = (implied_prob_away + implied_prob_home) - 1
                        odds_data['Market_Efficiency'] = market_efficiency
                    
                    games_data.append(odds_data)
                    
                except Exception as e:
                    logging.warning(f"Error processing game row: {str(e)}")
                    continue
            
            if games_data:
                df = pd.DataFrame(games_data)
                logging.info(f"Successfully scraped odds for {len(df)} games on {game_date.date()}")
                return df
            
            logging.warning(f"No odds data found for {game_date.date()}")
            return None
            
        except Exception as e:
            logging.error(f"Error scraping odds: {str(e)}")
            return None
    
    def get_line_movements(self, game_url):
        """
        Get line movement history for a specific game.
        
        Args:
            game_url (str): URL of the game's odds page
            
        Returns:
            dict: Dictionary containing line movement data
        """
        try:
            self._wait_for_rate_limit()
            response = self.session.get(game_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            movements = {
                'moneyline': [],
                'spread': [],
                'total': []
            }
            
            # Extract line movement data from the page
            movement_table = soup.find('table', class_='odds-movements')
            if movement_table:
                rows = movement_table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        timestamp = cells[0].text.strip()
                        market_type = cells[1].text.strip().lower()
                        old_value = cells[2].text.strip()
                        new_value = cells[3].text.strip()
                        
                        if market_type in movements:
                            movements[market_type].append({
                                'timestamp': timestamp,
                                'old_value': old_value,
                                'new_value': new_value
                            })
            
            return movements
            
        except Exception as e:
            logging.error(f"Error getting line movements: {str(e)}")
            return None
    
    def analyze_sharp_action(self, odds_df):
        """
        Analyze odds data for sharp betting patterns.
        
        Args:
            odds_df (pd.DataFrame): DataFrame containing odds data
            
        Returns:
            pd.DataFrame: DataFrame with sharp betting indicators
        """
        try:
            df = odds_df.copy()
            
            # Calculate reverse line movement
            for idx, row in df.iterrows():
                if row['Moneyline_Away'] and row['Moneyline_Home']:
                    # Convert odds to implied probabilities
                    implied_prob_away = 1 / row['Moneyline_Away']
                    implied_prob_home = 1 / row['Moneyline_Home']
                    
                    # Check for reverse line movement
                    movements = self.get_line_movements(row['game_url'])
                    if movements and movements['moneyline']:
                        latest_move = movements['moneyline'][-1]
                        if latest_move['old_value'] and latest_move['new_value']:
                            old_prob = 1 / float(latest_move['old_value'])
                            new_prob = 1 / float(latest_move['new_value'])
                            
                            # Detect steam moves
                            prob_change = new_prob - old_prob
                            df.loc[idx, 'Steam_Move'] = 'Yes' if abs(prob_change) > 0.05 else 'No'
                            df.loc[idx, 'Steam_Direction'] = 'Away' if prob_change > 0 else 'Home' if prob_change < 0 else None
            
            return df
            
        except Exception as e:
            logging.error(f"Error analyzing sharp action: {str(e)}")
            return odds_df  # Return original DataFrame if analysis fails 