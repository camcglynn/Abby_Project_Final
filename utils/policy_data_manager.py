import logging
import json
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class PolicyDataManager:
    """Manage policy data for different states"""
    
    def __init__(self, policy_data_file="data/policy_data.json"):
        """
        Initialize the policy data manager
        
        Args:
            policy_data_file (str): File containing policy data
        """
        logger.info("Initializing PolicyDataManager")
        self.policy_data_file = policy_data_file
        self.policy_data = self._load_policy_data()
        self.last_updated = self._get_last_updated()
    
    def _load_policy_data(self):
        """
        Load policy data from file
        
        Returns:
            dict: Policy data by state
        """
        try:
            if not os.path.exists(self.policy_data_file):
                logger.warning(f"Policy data file not found: {self.policy_data_file}")
                return {}
                
            with open(self.policy_data_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded policy data for {len(data)} states")
                return data
        except Exception as e:
            logger.error(f"Error loading policy data: {str(e)}")
            return {}
    
    def _get_last_updated(self):
        """
        Get the last updated timestamp for policy data
        
        Returns:
            datetime: Last updated timestamp
        """
        try:
            if os.path.exists(self.policy_data_file):
                return datetime.fromtimestamp(os.path.getmtime(self.policy_data_file))
            return None
        except Exception as e:
            logger.error(f"Error getting last updated timestamp: {str(e)}")
            return None
    
    def get_state_policy(self, state):
        """
        Get policy data for a specific state
        
        Args:
            state (str): State name or abbreviation
            
        Returns:
            dict: Policy data for the state or None if not found
        """
        try:
            # Normalize state name
            state = self._normalize_state(state)
            if not state:
                return None
                
            # Find policy by state name or abbreviation
            for state_key, policy in self.policy_data.items():
                if state_key.lower() == state.lower() or \
                   self._get_state_abbreviation(state_key).lower() == state.lower():
                    return policy
                    
            logger.warning(f"No policy data found for state: {state}")
            return None
        except Exception as e:
            logger.error(f"Error getting state policy: {str(e)}")
            return None
    
    def get_all_states(self):
        """
        Get list of all states with policy data
        
        Returns:
            list: List of state names
        """
        return list(self.policy_data.keys())
    
    def update_state_policy(self, state, policy_data):
        """
        Update policy data for a specific state
        
        Args:
            state (str): State name
            policy_data (dict): Updated policy data
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            # Normalize state name
            state = self._normalize_state(state)
            if not state:
                return False
                
            # Update policy data
            self.policy_data[state] = policy_data
            
            # Save to file
            os.makedirs(os.path.dirname(self.policy_data_file), exist_ok=True)
            with open(self.policy_data_file, 'w') as f:
                json.dump(self.policy_data, f, indent=2)
                
            # Update last_updated timestamp
            self.last_updated = datetime.now()
            
            logger.info(f"Updated policy data for state: {state}")
            return True
        except Exception as e:
            logger.error(f"Error updating state policy: {str(e)}")
            return False
    
    def get_policy_summary(self, state):
        """
        Get a summary of policy data for a specific state
        
        Args:
            state (str): State name or abbreviation
            
        Returns:
            str: Summary of policy for the state
        """
        policy = self.get_state_policy(state)
        if not policy:
            return f"No policy information available for {state}."
            
        try:
            # Create a summary based on the policy data
            summary = f"Reproductive Health Policy for {state}:\n\n"
            
            # Add abortion policy details
            if "abortion" in policy:
                summary += "Abortion: " + policy["abortion"]["summary"] + "\n"
                
                if "restrictions" in policy["abortion"]:
                    summary += "Restrictions:\n"
                    for restriction in policy["abortion"]["restrictions"]:
                        summary += f"- {restriction}\n"
                    
                if "exceptions" in policy["abortion"]:
                    summary += "Exceptions:\n"
                    for exception in policy["abortion"]["exceptions"]:
                        summary += f"- {exception}\n"
            
            # Add contraception policy details
            if "contraception" in policy:
                summary += "\nContraception: " + policy["contraception"]["summary"] + "\n"
            
            # Add resources
            if "resources" in policy:
                summary += "\nResources:\n"
                for resource in policy["resources"]:
                    summary += f"- {resource['name']}: {resource['contact']}\n"
            
            # Add last updated date
            if "last_updated" in policy:
                summary += f"\nLast updated: {policy['last_updated']}"
            
            return summary
        except Exception as e:
            logger.error(f"Error generating policy summary: {str(e)}")
            return f"Error generating policy summary for {state}: {str(e)}"
    
    def _normalize_state(self, state):
        """
        Normalize state name
        
        Args:
            state (str): State name or abbreviation
            
        Returns:
            str: Normalized state name or None if invalid
        """
        if not state:
            return None
            
        # Remove special characters and extra spaces
        state = re.sub(r'[^\w\s]', '', state).strip()
        
        # Check if it's an abbreviation
        if len(state) == 2:
            # Convert to full state name if possible
            abbr_to_name = {
                "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
                "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
                "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
                "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
                "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
                "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
                "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
                "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
                "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
                "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
                "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
                "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
            }
            return abbr_to_name.get(state.upper(), state)
        
        # Capitalize first letter of each word
        return ' '.join(word.capitalize() for word in state.split())
    
    def _get_state_abbreviation(self, state):
        """
        Get state abbreviation from full name
        
        Args:
            state (str): State name
            
        Returns:
            str: State abbreviation or original string if not found
        """
        name_to_abbr = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
            "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
            "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
            "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
            "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
            "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
            "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
            "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
            "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
            "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
            "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
            "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
            "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"
        }
        return name_to_abbr.get(state, state) 