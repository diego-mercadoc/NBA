import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
import time
import re

class NBAInjuryTracker:
    """
    Tracks and processes NBA injury data from multiple sources.
    Provides injury impact scores for teams based on missing players.
    """
    
    def __init__(self):
        self.injuries = pd.DataFrame()
        self.player_values = {}  # Cache for player impact values
        self.last_update = None
        self.update_frequency = timedelta(hours=12)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Player impact tiers (simplified version)
        self.player_tiers = {
            'superstar': 0.15,    # e.g., Jokic, Giannis
            'star': 0.10,         # e.g., Booker, Mitchell
            'starter': 0.05,      # Quality starters
            'rotation': 0.025,    # Key rotation players
            'bench': 0.01         # End of bench
        }
        
        # Initialize with mock data for testing
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with mock injury data for testing"""
        mock_injuries = [
            {
                'player': 'Jamal Murray',
                'team': 'Denver Nuggets',
                'status': 'Questionable',
                'details': 'Ankle - Game time decision',
                'source': 'Mock Data',
                'last_update': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'player': 'LeBron James',
                'team': 'Los Angeles Lakers',
                'status': 'Probable',
                'details': 'Rest - Likely to play',
                'source': 'Mock Data',
                'last_update': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'player': 'Jaylen Brown',
                'team': 'Boston Celtics',
                'status': 'Out',
                'details': 'Knee - Will miss next game',
                'source': 'Mock Data',
                'last_update': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'player': 'Devin Booker',
                'team': 'Phoenix Suns',
                'status': 'Doubtful',
                'details': 'Hamstring - Unlikely to play',
                'source': 'Mock Data',
                'last_update': datetime.now().strftime('%Y-%m-%d')
            }
        ]
        self.injuries = pd.DataFrame(mock_injuries)
    
    def update_injuries(self):
        """Update injury data from multiple sources with fallback mechanism"""
        current_time = datetime.now()
        
        # Check if update is needed
        if (self.last_update and 
            current_time - self.last_update < self.update_frequency):
            logging.info("Using cached injury data")
            return
        
        logging.info("Updating injury data...")
        
        # Try multiple sources in order of preference
        sources = [
            (self._scrape_nba_injuries, "NBA.com"),
            (self._scrape_espn_injuries, "ESPN"),
            (self._scrape_rotoworld_injuries, "Rotoworld"),
            (self._scrape_basketball_reference_injuries, "Basketball Reference")
        ]
        
        all_injuries = pd.DataFrame()
        successful_source = None
        
        for scraper, source_name in sources:
            try:
                injuries = scraper()
                if not injuries.empty:
                    logging.info(f"Successfully scraped injuries from {source_name}")
                    all_injuries = pd.concat([all_injuries, injuries])
                    successful_source = source_name
                    break
            except Exception as e:
                logging.warning(f"Failed to scrape {source_name}: {str(e)}")
                continue
        
        if successful_source:
            # Deduplicate and clean the data
            all_injuries = all_injuries.drop_duplicates(subset=['player', 'team'])
            self.injuries = all_injuries
            self.last_update = current_time
            logging.info(f"Found {len(self.injuries)} active injuries from {successful_source}")
        else:
            logging.warning("Could not fetch live injury data. Using mock data for testing.")
            if self.injuries.empty:
                self._initialize_mock_data()
            self.last_update = current_time
    
    def _scrape_nba_injuries(self):
        """Scrape injury data from NBA.com"""
        try:
            url = "https://www.nba.com/players/injuries"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            injuries = []
            
            # Process injury table
            injury_table = soup.find('table', class_='injury-table')
            if injury_table:
                for row in injury_table.find_all('tr')[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        injury = {
                            'player': cols[0].text.strip(),
                            'team': cols[1].text.strip(),
                            'status': cols[2].text.strip(),
                            'details': cols[3].text.strip(),
                            'source': 'NBA.com',
                            'last_update': datetime.now().strftime('%Y-%m-%d')
                        }
                        injuries.append(injury)
            
            time.sleep(2)  # Respect rate limits
            return pd.DataFrame(injuries)
            
        except Exception as e:
            logging.error(f"Error scraping NBA.com injuries: {str(e)}")
            return pd.DataFrame()
    
    def _scrape_basketball_reference_injuries(self):
        """Scrape injury data from Basketball Reference"""
        try:
            url = "https://www.basketball-reference.com/friv/injuries.fcgi"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            injuries = []
            
            # Process injury table
            injury_table = soup.find('table', id='injuries')
            if injury_table:
                for row in injury_table.find_all('tr')[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        injury = {
                            'player': cols[0].text.strip(),
                            'team': cols[1].text.strip(),
                            'status': cols[2].text.strip(),
                            'details': cols[3].text.strip(),
                            'source': 'Basketball Reference',
                            'last_update': datetime.now().strftime('%Y-%m-%d')
                        }
                        injuries.append(injury)
            
            time.sleep(2)  # Respect rate limits
            return pd.DataFrame(injuries)
            
        except Exception as e:
            logging.error(f"Error scraping Basketball Reference injuries: {str(e)}")
            return pd.DataFrame()
    
    def _scrape_espn_injuries(self):
        """Scrape injury data from ESPN"""
        try:
            url = "https://www.espn.com/nba/injuries"
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            injuries = []
            
            # Process each team's injuries
            for team_section in soup.find_all('div', class_='ResponsiveTable'):
                team_name = team_section.find('div', class_='Table__Title').text
                
                for row in team_section.find_all('tr')[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        injury = {
                            'player': cols[0].text.strip(),
                            'team': team_name,
                            'status': cols[1].text.strip(),
                            'details': cols[2].text.strip(),
                            'source': 'ESPN',
                            'last_update': datetime.now().strftime('%Y-%m-%d')
                        }
                        injuries.append(injury)
            
            time.sleep(2)  # Respect rate limits
            return pd.DataFrame(injuries)
            
        except Exception as e:
            logging.error(f"Error scraping ESPN injuries: {str(e)}")
            return pd.DataFrame()
    
    def _scrape_rotoworld_injuries(self):
        """Scrape injury data from Rotoworld"""
        try:
            url = "https://www.nbcsports.com/edge/basketball/nba/player-news"
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            injuries = []
            
            # Process injury news
            for news_item in soup.find_all('div', class_='player-news-item'):
                headline = news_item.find('div', class_='headline')
                if headline and any(kw in headline.text.lower() for kw in ['injury', 'out', 'questionable', 'doubtful']):
                    player_name = news_item.find('span', class_='player-name').text
                    team_name = news_item.find('span', class_='team-name').text
                    
                    injury = {
                        'player': player_name,
                        'team': team_name,
                        'status': self._extract_status(headline.text),
                        'details': news_item.find('div', class_='content').text,
                        'source': 'Rotoworld',
                        'last_update': datetime.now().strftime('%Y-%m-%d')
                    }
                    injuries.append(injury)
            
            time.sleep(2)  # Respect rate limits
            return pd.DataFrame(injuries)
            
        except Exception as e:
            logging.error(f"Error scraping Rotoworld injuries: {str(e)}")
            return pd.DataFrame()
    
    def _extract_status(self, headline):
        """Extract injury status from headline"""
        status_keywords = {
            'out': 'Out',
            'questionable': 'Questionable',
            'doubtful': 'Doubtful',
            'probable': 'Probable',
            'available': 'Available'
        }
        
        headline = headline.lower()
        for keyword, status in status_keywords.items():
            if keyword in headline:
                return status
        return 'Unknown'
    
    def get_team_injury_impact(self, team):
        """Calculate injury impact score for a team"""
        if self.injuries.empty or team not in self.injuries['team'].values:
            return 0.0
        
        team_injuries = self.injuries[self.injuries['team'] == team]
        total_impact = 0.0
        
        for _, injury in team_injuries.iterrows():
            if injury['status'].lower() in ['out', 'doubtful']:
                impact = self._get_player_impact(injury['player'])
                total_impact += impact
            elif injury['status'].lower() == 'questionable':
                impact = self._get_player_impact(injury['player']) * 0.5
                total_impact += impact
            elif injury['status'].lower() == 'probable':
                impact = self._get_player_impact(injury['player']) * 0.25
                total_impact += impact
        
        return min(total_impact, 1.0)  # Cap at 1.0
    
    def _get_player_impact(self, player_name):
        """Get impact value for a player"""
        if player_name in self.player_values:
            return self.player_values[player_name]
        
        # TODO: Implement more sophisticated player value calculation
        # For now, use a simple tier-based system
        if self._is_superstar(player_name):
            impact = self.player_tiers['superstar']
        elif self._is_star(player_name):
            impact = self.player_tiers['star']
        elif self._is_starter(player_name):
            impact = self.player_tiers['starter']
        elif self._is_rotation(player_name):
            impact = self.player_tiers['rotation']
        else:
            impact = self.player_tiers['bench']
        
        self.player_values[player_name] = impact
        return impact
    
    def _is_superstar(self, player_name):
        """Check if player is a superstar"""
        superstars = {
            'Giannis Antetokounmpo', 'Nikola Jokic', 'Joel Embiid',
            'Kevin Durant', 'Stephen Curry', 'Luka Doncic',
            'LeBron James', 'Jayson Tatum', 'Devin Booker'
        }
        return player_name in superstars
    
    def _is_star(self, player_name):
        """Check if player is a star"""
        stars = {
            'Donovan Mitchell', 'Damian Lillard', 'Ja Morant',
            'Trae Young', 'DeMar DeRozan', 'Jimmy Butler',
            'Paul George', 'Kawhi Leonard', 'Anthony Edwards'
        }
        return player_name in stars
    
    def _is_starter(self, player_name):
        """Check if player is a regular starter"""
        # This would ideally be based on current season starting lineups
        # For now, return a conservative estimate
        return False
    
    def _is_rotation(self, player_name):
        """Check if player is a rotation player"""
        # This would ideally be based on current season minutes
        # For now, return a conservative estimate
        return False
    
    def get_injury_report(self, team=None):
        """Get formatted injury report for a team or all teams"""
        if not self.injuries.empty:
            if team:
                report = self.injuries[self.injuries['team'] == team].copy()
            else:
                report = self.injuries.copy()
            
            report['impact'] = report['player'].apply(self._get_player_impact)
            return report.sort_values('impact', ascending=False)
        
        return pd.DataFrame() 