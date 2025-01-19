import pandas as pd
import requests
from datetime import datetime
import logging
from bs4 import BeautifulSoup
import time

class InjuryTracker:
    """
    Tracks NBA player injuries from multiple sources.
    
    Features:
    - Daily injury updates from ESPN and Rotoworld
    - Impact scoring based on player importance
    - Historical injury tracking
    - Team-level injury impact assessment
    """
    
    def __init__(self):
        self.injuries = pd.DataFrame(columns=[
            'player', 'team', 'status', 'details', 'return_date',
            'impact', 'source', 'last_update'
        ])
        self.sources = ['espn', 'rotoworld']
        self.status_impact = {
            'Out': 1.0,
            'Doubtful': 0.8,
            'Questionable': 0.5,
            'Probable': 0.2,
            'Available': 0.0
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def update_injuries(self):
        """Update injury data from all sources"""
        new_injuries = []
        
        for source in self.sources:
            try:
                if source == 'espn':
                    injuries = self._scrape_espn_injuries()
                elif source == 'rotoworld':
                    injuries = self._scrape_rotoworld_injuries()
                
                if injuries:
                    new_injuries.extend(injuries)
            except Exception as e:
                logging.error(f"Error updating injuries from {source}: {str(e)}")
        
        if new_injuries:
            new_df = pd.DataFrame(new_injuries)
            # Update existing entries and add new ones
            self.injuries = pd.concat([
                self.injuries[~self.injuries['player'].isin(new_df['player'])],
                new_df
            ]).reset_index(drop=True)
            
            logging.info(f"Updated injuries: {len(new_df)} entries")
    
    def _scrape_espn_injuries(self):
        """Scrape injury data from ESPN"""
        injuries = []
        url = "https://www.espn.com/nba/injuries"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for team_section in soup.find_all('div', class_='ResponsiveTable'):
                team_name = team_section.find('div', class_='Table__Title').text
                
                for row in team_section.find_all('tr')[1:]:  # Skip header row
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        player = cols[0].text.strip()
                        status = cols[1].text.strip()
                        details = cols[2].text.strip()
                        date = cols[3].text.strip()
                        
                        injuries.append({
                            'player': player,
                            'team': team_name,
                            'status': status,
                            'details': details,
                            'return_date': date,
                            'impact': self._calculate_impact(player, status),
                            'source': 'espn',
                            'last_update': datetime.now().strftime('%Y-%m-%d')
                        })
            
            return injuries
        except Exception as e:
            logging.error(f"Error scraping ESPN injuries: {str(e)}")
            return []
    
    def _scrape_rotoworld_injuries(self):
        """Scrape injury data from Rotoworld"""
        injuries = []
        url = "https://www.nbcsportsedge.com/basketball/nba/injury-report"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for row in soup.find_all('div', class_='player-news-item'):
                player_elem = row.find('span', class_='player-name')
                status_elem = row.find('span', class_='player-status')
                details_elem = row.find('div', class_='player-news-content')
                
                if player_elem and status_elem:
                    player = player_elem.text.strip()
                    status = status_elem.text.strip()
                    details = details_elem.text.strip() if details_elem else ''
                    team = self._extract_team(player)
                    
                    injuries.append({
                        'player': player,
                        'team': team,
                        'status': status,
                        'details': details,
                        'return_date': None,  # Rotoworld doesn't provide specific return dates
                        'impact': self._calculate_impact(player, status),
                        'source': 'rotoworld',
                        'last_update': datetime.now().strftime('%Y-%m-%d')
                    })
            
            return injuries
        except Exception as e:
            logging.error(f"Error scraping Rotoworld injuries: {str(e)}")
            return []
    
    def _calculate_impact(self, player, status):
        """Calculate injury impact based on player importance and status"""
        base_impact = self.status_impact.get(status, 0.0)
        
        # TODO: Enhance with player importance factors
        # For now, using a simple status-based impact
        return base_impact
    
    def _extract_team(self, player_name):
        """Extract team name from player context"""
        # TODO: Implement team extraction from player name
        # For now, returning None as team will be mapped later
        return None
    
    def get_team_injury_impact(self, team):
        """Calculate total injury impact for a team"""
        team_injuries = self.injuries[self.injuries['team'] == team]
        if team_injuries.empty:
            return 0.0
        
        # Sum of individual impacts, capped at 1.0
        total_impact = min(1.0, team_injuries['impact'].sum())
        return total_impact
    
    def get_injury_report(self, team):
        """Get detailed injury report for a team"""
        return self.injuries[self.injuries['team'] == team].copy() 