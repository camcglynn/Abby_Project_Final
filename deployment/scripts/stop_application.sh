#!/bin/bash
set -e

# Check if the service is running
if systemctl is-active --quiet abby-chatbot; then
    echo "Stopping abby-chatbot service..."
    sudo systemctl stop abby-chatbot
else
    echo "abby-chatbot service is not running"
fi

# Disable the service
if systemctl is-enabled --quiet abby-chatbot; then
    echo "Disabling abby-chatbot service..."
    sudo systemctl disable abby-chatbot
fi

# Remove the service file
if [ -f /etc/systemd/system/abby-chatbot.service ]; then
    echo "Removing service file..."
    sudo rm /etc/systemd/system/abby-chatbot.service
    sudo systemctl daemon-reload
fi

echo "Application stop script completed successfully" 