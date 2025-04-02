/**
 * Quick Exit Functionality
 * Allows users to quickly navigate away from the site for privacy/safety reasons
 */
document.addEventListener('DOMContentLoaded', function() {
    const quickExitButton = document.getElementById('quickExit');
    
    if (quickExitButton) {
        // Handle quick exit button click
        quickExitButton.addEventListener('click', function() {
            performQuickExit();
        });
        
        // Also add keyboard shortcut (Escape key twice in quick succession)
        let escapeKeyPressTime = 0;
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                const currentTime = new Date().getTime();
                
                // If Escape was pressed within the last 500ms, trigger quick exit
                if (currentTime - escapeKeyPressTime < 500) {
                    performQuickExit();
                }
                
                escapeKeyPressTime = currentTime;
            }
        });
    }
    
    function performQuickExit() {
        // First, clear any locally stored data
        try {
            localStorage.clear();
            sessionStorage.clear();
        } catch (e) {
            console.error('Failed to clear storage', e);
        }
        
        // Navigate to a neutral site
        window.location.replace('https://www.google.com');
    }
}); 