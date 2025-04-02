// Google Maps initialization
let map;
let markers = [];

// Initialize the Google Maps component
function initMap(mapElementId, locations) {
    // Check if map element exists
    const mapElement = document.getElementById(mapElementId);
    if (!mapElement) {
        console.error(`Map element with ID ${mapElementId} not found`);
        return;
    }

    // Default center (US)
    const center = {lat: 37.0902, lng: -95.7129};
    
    // Create map
    map = new google.maps.Map(mapElement, {
        zoom: 4,
        center: center,
        mapTypeControl: true,
        streetViewControl: false,
        fullscreenControl: true,
        zoomControl: true
    });
    
    // Add markers if locations are provided
    if (locations && locations.length > 0) {
        addMarkers(locations);
        
        // If we have locations, center on the first one
        map.setCenter(locations[0].position);
        map.setZoom(11);
    }
}

// Add markers to the map
function addMarkers(locations) {
    // Clear any existing markers
    clearMarkers();
    
    // Create new markers
    locations.forEach(location => {
        const marker = new google.maps.Marker({
            position: location.position,
            map: map,
            title: location.name
        });
        
        // Add info window with clinic details
        const infoWindow = new google.maps.InfoWindow({
            content: `
                <div class="info-window">
                    <h4>${location.name}</h4>
                    <p>${location.address}</p>
                    <p>${location.phone || ''}</p>
                    <a href="${location.website || '#'}" target="_blank" rel="noopener">Visit website</a>
                    <a href="https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(location.address)}" 
                       target="_blank" class="directions-link">Get directions</a>
                </div>
            `
        });
        
        // Open info window on marker click
        marker.addListener('click', () => {
            infoWindow.open(map, marker);
        });
        
        markers.push(marker);
    });
}

// Clear all markers from the map
function clearMarkers() {
    markers.forEach(marker => marker.setMap(null));
    markers = [];
}

// Process clinic data from backend and display on map
function displayClinics(clinicData) {
    if (!clinicData || !clinicData.length) {
        console.log('No clinic data to display');
        return;
    }
    
    // Map clinic data to marker format
    const locations = clinicData.map(clinic => {
        // Handle both lat/lng and latitude/longitude formats
        const lat = clinic.lat || clinic.latitude || 0;
        const lng = clinic.lng || clinic.longitude || 0;
        
        return {
            position: {lat, lng},
            name: clinic.name,
            address: clinic.address,
            phone: clinic.phone,
            website: clinic.website
        };
    }).filter(location => location.position.lat && location.position.lng); // Filter out invalid coordinates
    
    if (locations.length === 0) {
        console.log('No valid clinic locations found');
        displayNoResults();
        return;
    }
    
    // Initialize or update map
    if (!map) {
        initMap('clinic-map', locations);
    } else {
        addMarkers(locations);
        
        // Center on first location
        if (locations.length > 0) {
            map.setCenter(locations[0].position);
            map.setZoom(11);
        }
    }
    
    // Ensure the map is visible 
    const mapContainer = document.getElementById('map-container');
    if (mapContainer) {
        mapContainer.style.display = 'block';
    }
}

// Function to show map when user provides a location
function showClinicMap(zipCode) {
    // Show the map container
    const mapContainer = document.getElementById('map-container');
    if (mapContainer) {
        mapContainer.style.display = 'block';
    }
    
    // Fetch clinic data based on zip code
    fetch(`/api/clinics?zip=${zipCode}`)
        .then(response => response.json())
        .then(data => {
            if (data.clinics && data.clinics.length > 0) {
                displayClinics(data.clinics);
            } else {
                displayNoResults();
            }
        })
        .catch(error => {
            console.error('Error fetching clinic data:', error);
            displayNoResults();
        });
}

// Display message when no clinics are found
function displayNoResults() {
    const mapElement = document.getElementById('clinic-map');
    if (mapElement) {
        mapElement.innerHTML = '<div class="no-results">No clinics found for this location. Please try another zip code or contact Planned Parenthood directly at 1-800-230-PLAN.</div>';
    }
}

// Export functions for use in other scripts
window.mapsApi = {
    initMap,
    showClinicMap,
    displayClinics
};

document.addEventListener('DOMContentLoaded', function() {
    // Add event listener for the close map button
    const closeMapBtn = document.getElementById('close-map');
    if (closeMapBtn) {
        closeMapBtn.addEventListener('click', function() {
            const mapContainer = document.getElementById('map-container');
            if (mapContainer) {
                mapContainer.style.display = 'none';
            }
        });
    }
}); 