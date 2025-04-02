import logging
import json
import os
import re
import zipcodes
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class ClinicFinder:
    """Find reproductive health clinics based on location"""
    
    def __init__(self, clinics_data_file="data/clinics_data.json"):
        """
        Initialize the clinic finder
        
        Args:
            clinics_data_file (str): File containing clinics data
        """
        logger.info("Initializing ClinicFinder")
        self.clinics_data_file = clinics_data_file
        self.clinics_data = self._load_clinics_data()
    
    def _load_clinics_data(self):
        """
        Load clinics data from file
        
        Returns:
            list: List of clinic data
        """
        try:
            if not os.path.exists(self.clinics_data_file):
                logger.warning(f"Clinics data file not found: {self.clinics_data_file}")
                return []
                
            with open(self.clinics_data_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded data for {len(data)} clinics")
                return data
        except Exception as e:
            logger.error(f"Error loading clinics data: {str(e)}")
            return []
    
    def find_clinics_by_zip(self, zip_code, max_distance=50):
        """
        Find clinics near a zip code
        
        Args:
            zip_code (str): ZIP code to search near
            max_distance (int): Maximum distance in miles
            
        Returns:
            dict: Dictionary with search results, including clinics found
        """
        try:
            # Validate zip code
            zip_info = zipcodes.matching(zip_code)
            if not zip_info:
                logger.warning(f"Invalid ZIP code: {zip_code}")
                return {"success": False, "error": "Invalid ZIP code", "clinics": []}
                
            zip_info = zip_info[0]  # Get the first match
            
            # Extract coordinates
            lat1, lon1 = float(zip_info['lat']), float(zip_info['long'])
            
            # Find clinics within range
            nearby_clinics = []
            for clinic in self.clinics_data:
                try:
                    lat2, lon2 = float(clinic['latitude']), float(clinic['longitude'])
                    distance = self._calculate_distance(lat1, lon1, lat2, lon2)
                    
                    if distance <= max_distance:
                        clinic_info = clinic.copy()
                        clinic_info['distance_miles'] = round(distance, 1)
                        nearby_clinics.append(clinic_info)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error processing clinic data: {str(e)}")
                    continue
            
            # Sort by distance
            nearby_clinics.sort(key=lambda x: x.get('distance_miles', float('inf')))
            
            return {
                "success": True,
                "location": {
                    "zip": zip_code,
                    "city": zip_info.get('city', ''),
                    "state": zip_info.get('state', '')
                },
                "search_radius_miles": max_distance,
                "clinics_found": len(nearby_clinics),
                "clinics": nearby_clinics
            }
        except Exception as e:
            logger.error(f"Error finding clinics by ZIP: {str(e)}")
            return {"success": False, "error": str(e), "clinics": []}
    
    def find_clinics_by_state(self, state):
        """
        Find clinics in a specific state
        
        Args:
            state (str): State name or abbreviation
            
        Returns:
            dict: Dictionary with search results
        """
        try:
            # Normalize state name/abbreviation
            state = self._normalize_state(state)
            if not state:
                return {"success": False, "error": "Invalid state", "clinics": []}
            
            # Find clinics in state
            state_clinics = []
            for clinic in self.clinics_data:
                try:
                    clinic_state = clinic.get('state', '')
                    if self._normalize_state(clinic_state) == state:
                        state_clinics.append(clinic.copy())
                except Exception as e:
                    logger.error(f"Error processing clinic in state search: {str(e)}")
                    continue
            
            # Get state abbreviation
            state_abbr = self._get_state_abbreviation(state)
            
            return {
                "success": True,
                "location": {
                    "state": state,
                    "state_abbr": state_abbr
                },
                "clinics_found": len(state_clinics),
                "clinics": state_clinics
            }
        except Exception as e:
            logger.error(f"Error finding clinics by state: {str(e)}")
            return {"success": False, "error": str(e), "clinics": []}
    
    def get_clinic_details(self, clinic_id):
        """
        Get detailed information for a specific clinic
        
        Args:
            clinic_id (str): ID of the clinic
            
        Returns:
            dict: Clinic details or None if not found
        """
        try:
            for clinic in self.clinics_data:
                if clinic.get('id') == clinic_id:
                    return clinic
            
            logger.warning(f"Clinic not found with ID: {clinic_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting clinic details: {str(e)}")
            return None
    
    def add_clinic(self, clinic_data):
        """
        Add a new clinic to the database
        
        Args:
            clinic_data (dict): Clinic data to add
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            # Validate required fields
            required_fields = ['name', 'address', 'city', 'state', 'zip', 'phone', 'latitude', 'longitude']
            for field in required_fields:
                if field not in clinic_data:
                    logger.error(f"Missing required field '{field}' for clinic")
                    return False
            
            # Generate ID if not provided
            if 'id' not in clinic_data:
                clinic_data['id'] = f"clinic_{len(self.clinics_data) + 1}"
            
            # Add timestamp
            clinic_data['added_at'] = datetime.now().isoformat()
            
            # Add to clinics data
            self.clinics_data.append(clinic_data)
            
            # Save to file
            return self._save_clinics_data()
        except Exception as e:
            logger.error(f"Error adding clinic: {str(e)}")
            return False
    
    def update_clinic(self, clinic_id, clinic_data):
        """
        Update an existing clinic
        
        Args:
            clinic_id (str): ID of the clinic to update
            clinic_data (dict): Updated clinic data
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            # Find clinic by ID
            for i, clinic in enumerate(self.clinics_data):
                if clinic.get('id') == clinic_id:
                    # Update data while preserving ID
                    updated_clinic = clinic_data.copy()
                    updated_clinic['id'] = clinic_id
                    updated_clinic['updated_at'] = datetime.now().isoformat()
                    
                    # Replace in list
                    self.clinics_data[i] = updated_clinic
                    
                    # Save to file
                    return self._save_clinics_data()
            
            logger.warning(f"Clinic not found with ID: {clinic_id}")
            return False
        except Exception as e:
            logger.error(f"Error updating clinic: {str(e)}")
            return False
    
    def delete_clinic(self, clinic_id):
        """
        Delete a clinic from the database
        
        Args:
            clinic_id (str): ID of the clinic to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            # Find clinic by ID
            for i, clinic in enumerate(self.clinics_data):
                if clinic.get('id') == clinic_id:
                    # Remove from list
                    del self.clinics_data[i]
                    
                    # Save to file
                    return self._save_clinics_data()
            
            logger.warning(f"Clinic not found with ID: {clinic_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting clinic: {str(e)}")
            return False
    
    def _save_clinics_data(self):
        """
        Save clinics data to file
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.clinics_data_file), exist_ok=True)
            
            # Save to file
            with open(self.clinics_data_file, 'w') as f:
                json.dump(self.clinics_data, f, indent=2)
                
            logger.info(f"Saved data for {len(self.clinics_data)} clinics")
            return True
        except Exception as e:
            logger.error(f"Error saving clinics data: {str(e)}")
            return False
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two coordinates using Haversine formula
        
        Args:
            lat1 (float): Latitude of first point
            lon1 (float): Longitude of first point
            lat2 (float): Latitude of second point
            lon2 (float): Longitude of second point
            
        Returns:
            float: Distance in miles
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 3956  # Radius of Earth in miles
        
        return c * r
    
    def _normalize_state(self, state):
        """
        Normalize state name or abbreviation
        
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