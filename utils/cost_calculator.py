"""
Cost Calculator for OpenAI API Usage

This module calculates costs for OpenAI API usage based on token consumption
for the GPT-4 mini model.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CostCalculator:
    """Calculates OpenAI API costs based on token usage"""
    
    # GPT-4 Mini Pricing (per 1M tokens)
    GPT4_MINI_PRICING = {
        'input': 0.150,      # $0.150 per 1M input tokens
        'output': 0.600,     # $0.600 per 1M output tokens
        'cached_input': 0.075  # $0.075 per 1M cached input tokens
    }
    
    @staticmethod
    def calculate_costs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate API costs from logs
        
        Args:
            logs: List of log entries containing token usage information
            
        Returns:
            Dict containing cost breakdown
        """
        try:
            # Initialize cost tracking
            costs = CostCalculator._get_empty_costs()
            
            # Process each log entry
            for log in logs:
                # Get token counts
                input_tokens = log.get('input_tokens', 0)
                output_tokens = log.get('output_tokens', 0)
                cached_input_tokens = log.get('cached_input_tokens', 0)
                
                # Calculate costs for each token type
                input_cost = (input_tokens / 1_000_000) * CostCalculator.GPT4_MINI_PRICING['input']
                output_cost = (output_tokens / 1_000_000) * CostCalculator.GPT4_MINI_PRICING['output']
                cached_input_cost = (cached_input_tokens / 1_000_000) * CostCalculator.GPT4_MINI_PRICING['cached_input']
                
                # Update total costs
                costs['gpt4_mini']['input_tokens'] += input_tokens
                costs['gpt4_mini']['output_tokens'] += output_tokens
                costs['gpt4_mini']['cached_input_tokens'] += cached_input_tokens
                costs['gpt4_mini']['input_cost'] += input_cost
                costs['gpt4_mini']['output_cost'] += output_cost
                costs['gpt4_mini']['cached_input_cost'] += cached_input_cost
                costs['gpt4_mini']['total_cost'] += input_cost + output_cost + cached_input_cost
            
            # Update overall total
            costs['total_cost'] = costs['gpt4_mini']['total_cost']
            
            logger.info("Costs calculated successfully")
            return costs
            
        except Exception as e:
            logger.error(f"Error calculating costs: {str(e)}", exc_info=True)
            return CostCalculator._get_empty_costs()
    
    @staticmethod
    def _get_empty_costs() -> Dict[str, Any]:
        """Get empty costs structure"""
        return {
            'gpt4_mini': {
                'input_tokens': 0,
                'output_tokens': 0,
                'cached_input_tokens': 0,
                'input_cost': 0.0,
                'output_cost': 0.0,
                'cached_input_cost': 0.0,
                'total_cost': 0.0
            },
            'total_cost': 0.0
        } 