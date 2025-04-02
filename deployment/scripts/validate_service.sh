#!/bin/bash

# Exit on error
set -e

# Logging configuration
exec > >(tee /var/log/codedeploy-validate-service.log) 2>&1
echo "Starting service validation script at $(date)"

# Wait for the application to start
sleep 10

# Check if the service is running
if ! systemctl is-active --quiet abby-chatbot; then
    echo "ERROR: abby-chatbot service is not running"
    exit 1
fi

# Check if the port is listening
if ! netstat -tulpn | grep :5006 > /dev/null; then
    echo "ERROR: Application is not listening on port 5006"
    exit 1
fi

# Check the health endpoint
HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5006/health)
if [ "$HEALTH_CHECK" != "200" ]; then
    echo "ERROR: Health check failed with status $HEALTH_CHECK"
    exit 1
fi

# Check system resources
MEMORY_USAGE=$(free | awk '/Mem:/ {print $3/$2 * 100.0}')
if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
    echo "WARNING: High memory usage: $MEMORY_USAGE%"
fi

CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
if (( $(echo "$CPU_USAGE > 90" | bc -l) )); then
    echo "WARNING: High CPU usage: $CPU_USAGE%"
fi

# Check log files for errors
if grep -i "error\|exception\|failed" /var/log/abby-chatbot/error.log > /dev/null; then
    echo "WARNING: Found errors in application log"
fi

echo "Service validation completed successfully" 