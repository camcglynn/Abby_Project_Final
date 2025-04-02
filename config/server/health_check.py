#!/usr/bin/env python3
"""
Health check endpoint for the Abby Chatbot application.
This file will be included in the deployment package.
"""

from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to be used by the load balancer"""
    return jsonify({
        'status': 'healthy',
        'message': 'Abby Chatbot is running'
    }) 