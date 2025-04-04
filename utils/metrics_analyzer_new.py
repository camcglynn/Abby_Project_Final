"""
Metrics Analyzer for the reproductive health chatbot

This module analyzes evaluation logs to generate performance metrics
for the chatbot dashboard.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Import advanced metrics
from utils.advanced_metrics import AdvancedMetricsCalculator, generate_performance_report

logger = logging.getLogger(__name__)

class MetricsAnalyzer:
    """Analyzes chatbot evaluation logs for performance metrics"""
    
    def __init__(self, log_dir="logs"):
        """
        Initialize the metrics analyzer and start automatic calculation
        
        Args:
            log_dir (str): Directory containing evaluation logs
        """
        self.log_dir = log_dir
        self.evaluation_log_file = os.path.join(log_dir, "evaluation_logs.json")
        self.conversation_log_file = os.path.join(log_dir, "conversation_logs.json")
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize empty logs if files don't exist
        if not os.path.exists(self.evaluation_log_file):
            self._init_log_file(self.evaluation_log_file)
        if not os.path.exists(self.conversation_log_file):
            self._init_log_file(self.conversation_log_file)
            
        # Load logs
        self.evaluation_logs = self._load_logs(self.evaluation_log_file)
        self.conversation_logs = self._load_logs(self.conversation_log_file)
        
        # Initialize metrics storage
        self.metrics = self._get_empty_metrics()
        
        # Calculate initial metrics
        self._calculate_all_metrics()
        
    def _init_log_file(self, file_path: str) -> None:
        """Initialize an empty log file"""
        with open(file_path, 'w') as f:
            json.dump([], f)

    def _load_logs(self, file_path: str) -> List[Dict[str, Any]]:
        """Load logs from file with error handling"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading logs from {file_path}: {str(e)}")
            return []

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Get empty metrics structure"""
        return {
            'core': self._get_empty_core_metrics(),
            'conversation': self._get_empty_conversation_metrics(),
            'daily': self._get_empty_daily_metrics(),
            'performance': {
                'response_time': {
                    'average_ms': 0,
                    'min_ms': 0,
                    'max_ms': 0
                },
                'tokens': {
                    'average': 0,
                    'min': 0,
                    'max': 0
                },
                'memory': {
                    'average_mb': 0,
                    'min_mb': 0,
                    'max_mb': 0
                }
            },
            'rouge_metrics': {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            },
            'bleu_metrics': {
                'score': 0.0,
                'details': {
                    'precisions': [0.0, 0.0, 0.0, 0.0],
                    'bp': 0.0,
                    'ratio': 0.0,
                    'sys_len': 0,
                    'ref_len': 0
                }
            },
            'bert_score': {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            },
            'ragas_metrics': {
                'faithfulness': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0
            },
            'retrieval_metrics': {
                'precision_at_k': {'p@1': 0.0, 'p@3': 0.0, 'p@5': 0.0, 'p@10': 0.0},
                'recall_at_k': {'r@1': 0.0, 'r@3': 0.0, 'r@5': 0.0, 'r@10': 0.0},
                'mrr': 0.0
            },
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'total_logs': 0
            },
            # Additional fields for charts
            'avg_relevance': 0.0,
            'avg_accuracy': 0.0,
            'avg_completeness': 0.0,
            'avg_clarity': 0.0,
            'avg_empathy': 0.0,
            'dates': [],
            'daily_scores': [],
            'daily_safety': []
        }

    def _calculate_all_metrics(self) -> None:
        """Calculate all metrics automatically"""
        try:
            # Calculate core metrics
            self.metrics['core'] = self._calculate_core_metrics(self.evaluation_logs)
            
            # Calculate conversation metrics
            self.metrics['conversation'] = self._calculate_conversation_metrics(self.conversation_logs)
            
            # Calculate daily metrics
            self.metrics['daily'] = self._calculate_daily_metrics(self.evaluation_logs)
            
            # Calculate performance metrics
            self.metrics['performance'] = self._calculate_performance_metrics(self.evaluation_logs)
            
            # Calculate advanced metrics if there are evaluation logs
            if self.evaluation_logs:
                calculator = AdvancedMetricsCalculator()
                
                # Extract data for metrics calculation
                references = []
                predictions = []
                questions = []
                contexts = []
                
                for log in self.evaluation_logs:
                    if 'question' in log and 'response' in log:
                        references.append(log['question'])
                        predictions.append(log['response'])
                        questions.append(log['question'])
                        contexts.append(log.get('contexts', []))
                
                if references and predictions:
                    # Calculate text similarity metrics
                    text_similarity = calculator.calculate_text_similarity_metrics(references, predictions)
                    self.metrics['rouge_metrics'] = text_similarity.get('rouge', {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0})
                    self.metrics['bleu_metrics'] = text_similarity.get('bleu', {'score': 0.0, 'details': {}})
                    self.metrics['bert_score'] = text_similarity.get('bert_score', {'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
                    
                    # Calculate RAGAS metrics
                    if questions and contexts and predictions:
                        ragas_metrics = calculator.calculate_ragas_metrics(questions, contexts, predictions)
                        self.metrics['ragas_metrics'] = ragas_metrics
                    
                    # Calculate retrieval metrics
                    if contexts:
                        retrieval_metrics = calculator.calculate_retrieval_metrics(
                            retrieved_docs=contexts,
                            relevant_docs=[[{'id': i} for i in range(len(ctx))] for ctx in contexts]
                        )
                        self.metrics['retrieval_metrics'] = retrieval_metrics
            else:
                # Set empty advanced metrics
                self.metrics['rouge_metrics'] = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
                self.metrics['bleu_metrics'] = {'score': 0.0, 'details': {}}
                self.metrics['bert_score'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                self.metrics['ragas_metrics'] = {'faithfulness': 0.0, 'context_precision': 0.0, 'context_recall': 0.0}
                self.metrics['retrieval_metrics'] = {
                    'precision_at_k': {'p@1': 0.0, 'p@3': 0.0, 'p@5': 0.0, 'p@10': 0.0},
                    'recall_at_k': {'r@1': 0.0, 'r@3': 0.0, 'r@5': 0.0, 'r@10': 0.0},
                    'mrr': 0.0
                }
            
            # Calculate additional chart metrics
            avg_scores = self.metrics['core']['average_scores']
            self.metrics['avg_relevance'] = avg_scores.get('relevance', 0.0)
            self.metrics['avg_accuracy'] = avg_scores.get('accuracy', 0.0)
            self.metrics['avg_completeness'] = avg_scores.get('completeness', 0.0)
            self.metrics['avg_clarity'] = avg_scores.get('clarity', 0.0)
            self.metrics['avg_empathy'] = avg_scores.get('empathy', 0.0)
            
            # Calculate daily chart data
            daily_data = self.metrics['daily']
            self.metrics['dates'] = sorted(daily_data.keys())
            self.metrics['daily_scores'] = [
                daily_data[date]['average_scores'].get('quality', 0.0)
                for date in self.metrics['dates']
            ]
            self.metrics['daily_safety'] = [
                daily_data[date]['average_scores'].get('safety', 0.0)
                for date in self.metrics['dates']
            ]
            
            # Update metadata
            self.metrics['metadata'] = {
                'last_updated': datetime.now().isoformat(),
                'total_logs': len(self.evaluation_logs) + len(self.conversation_logs)
            }
            
            logger.info("All metrics calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            self.metrics = self._get_empty_metrics()

    def _calculate_core_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate core evaluation metrics"""
        if not logs:
            return self._get_empty_core_metrics()
        
        total = len(logs)
        scores = defaultdict(list)
        
        for log in logs:
            eval_data = log.get('evaluation', {})
            scores['relevance'].append(eval_data.get('relevance_score', 0))
            scores['quality'].append(eval_data.get('quality_score', 0))
            scores['safety'].append(eval_data.get('safety', {}).get('score', 0))
            scores['empathy'].append(eval_data.get('empathy_score', 0))
            scores['clarity'].append(eval_data.get('clarity_score', 0))
        
        return {
            'total_evaluations': total,
            'average_scores': {
                metric: sum(values)/len(values) if values else 0
                for metric, values in scores.items()
            },
            'improvement_rate': sum(1 for log in logs 
                                  if log.get('evaluation', {}).get('improved', False)) / total if total else 0
        }

    def _calculate_conversation_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate conversation-level metrics"""
        if not logs:
            return self._get_empty_conversation_metrics()
        
        total = len(logs)
        sessions = defaultdict(list)
        
        for log in logs:
            session_id = log.get('session_id')
            if session_id:
                sessions[session_id].append(log)
        
        return {
            'total_conversations': len(sessions),
            'avg_messages_per_conversation': sum(len(msgs) for msgs in sessions.values()) / len(sessions) if sessions else 0,
            'total_messages': total
        }

    def _calculate_daily_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate daily aggregated metrics"""
        if not logs:
            return {}
        
        daily_data = defaultdict(lambda: defaultdict(list))
        
        for log in logs:
            date = datetime.fromisoformat(log.get('timestamp', '')).date().isoformat()
            eval_data = log.get('evaluation', {})
            
            daily_data[date]['relevance'].append(eval_data.get('relevance_score', 0))
            daily_data[date]['quality'].append(eval_data.get('quality_score', 0))
            daily_data[date]['safety'].append(eval_data.get('safety', {}).get('score', 0))
            daily_data[date]['count'] += 1
        
        return {
            date: {
                'average_scores': {
                    metric: sum(scores)/len(scores) if scores else 0
                    for metric, scores in metrics.items() if metric != 'count'
                },
                'total_evaluations': metrics['count']
            }
            for date, metrics in daily_data.items()
        }

    def _calculate_performance_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics including ROUGE scores"""
        if not logs:
            return self._get_empty_performance_metrics()
            
        try:
            # Calculate response times
            response_times = []
            for log in logs:
                if 'response_time' in log:
                    response_times.append(log['response_time'])
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            min_response_time = min(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0
            
            # Calculate token usage
            token_counts = [log.get('token_count', 0) for log in logs]
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            min_tokens = min(token_counts) if token_counts else 0
            max_tokens = max(token_counts) if token_counts else 0
            
            # Calculate memory usage
            memory_usage = [log.get('memory_usage', 0) for log in logs]
            avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
            min_memory = min(memory_usage) if memory_usage else 0
            max_memory = max(memory_usage) if memory_usage else 0
            
            return {
                'response_time': {
                    'average_ms': avg_response_time,
                    'min_ms': min_response_time,
                    'max_ms': max_response_time
                },
                'tokens': {
                    'average': avg_tokens,
                    'min': min_tokens,
                    'max': max_tokens
                },
                'memory': {
                    'average_mb': avg_memory,
                    'min_mb': min_memory,
                    'max_mb': max_memory
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}", exc_info=True)
            return self._get_empty_performance_metrics()

    def get_metrics(self, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None,
                   session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all metrics, optionally filtered by date range and session
        
        Args:
            start_date (str, optional): Start date in ISO format
            end_date (str, optional): End date in ISO format
            session_id (str, optional): Session ID to filter by
            
        Returns:
            Dict[str, Any]: Dictionary containing all metrics
        """
        try:
            # Filter logs if needed
            filtered_eval_logs = self._filter_logs(self.evaluation_logs, start_date, end_date, session_id)
            filtered_conv_logs = self._filter_logs(self.conversation_logs, start_date, end_date, session_id)
            
            # Calculate metrics with filtered logs
            metrics = self._get_empty_metrics()
            
            # Calculate core metrics
            metrics['core'] = self._calculate_core_metrics(filtered_eval_logs)
            
            # Calculate conversation metrics
            metrics['conversation'] = self._calculate_conversation_metrics(filtered_conv_logs)
            
            # Calculate daily metrics
            metrics['daily'] = self._calculate_daily_metrics(filtered_eval_logs)
            
            # Calculate performance metrics
            metrics['performance'] = self._calculate_performance_metrics(filtered_eval_logs)
            
            # Calculate advanced metrics if there are evaluation logs
            if filtered_eval_logs:
                calculator = AdvancedMetricsCalculator()
                
                # Extract data for metrics calculation
                references = []
                predictions = []
                questions = []
                contexts = []
                
                for log in filtered_eval_logs:
                    if 'question' in log and 'response' in log:
                        references.append(log['question'])
                        predictions.append(log['response'])
                        questions.append(log['question'])
                        contexts.append(log.get('contexts', []))
                
                if references and predictions:
                    # Calculate text similarity metrics
                    text_similarity = calculator.calculate_text_similarity_metrics(references, predictions)
                    metrics['rouge_metrics'] = text_similarity.get('rouge', {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0})
                    metrics['bleu_metrics'] = text_similarity.get('bleu', {'score': 0.0, 'details': {}})
                    metrics['bert_score'] = text_similarity.get('bert_score', {'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
                    
                    # Calculate RAGAS metrics
                    if questions and contexts and predictions:
                        ragas_metrics = calculator.calculate_ragas_metrics(questions, contexts, predictions)
                        metrics['ragas_metrics'] = ragas_metrics
                    
                    # Calculate retrieval metrics
                    if contexts:
                        retrieval_metrics = calculator.calculate_retrieval_metrics(
                            retrieved_docs=contexts,
                            relevant_docs=[[{'id': i} for i in range(len(ctx))] for ctx in contexts]
                        )
                        metrics['retrieval_metrics'] = retrieval_metrics
            
            # Calculate additional chart metrics
            avg_scores = metrics['core']['average_scores']
            metrics['avg_relevance'] = avg_scores.get('relevance', 0.0)
            metrics['avg_accuracy'] = avg_scores.get('accuracy', 0.0)
            metrics['avg_completeness'] = avg_scores.get('completeness', 0.0)
            metrics['avg_clarity'] = avg_scores.get('clarity', 0.0)
            metrics['avg_empathy'] = avg_scores.get('empathy', 0.0)
            
            # Calculate daily chart data
            daily_data = metrics['daily']
            metrics['dates'] = sorted(daily_data.keys())
            metrics['daily_scores'] = [
                daily_data[date]['average_scores'].get('quality', 0.0)
                for date in metrics['dates']
            ]
            metrics['daily_safety'] = [
                daily_data[date]['average_scores'].get('safety', 0.0)
                for date in metrics['dates']
            ]
            
            # Update metadata
            metrics['metadata'] = {
                'last_updated': datetime.now().isoformat(),
                'total_logs': len(filtered_eval_logs) + len(filtered_conv_logs)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}", exc_info=True)
            return self._get_empty_metrics()

    def _filter_logs(self, logs: List[Dict[str, Any]], 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter logs by date range and session ID"""
        filtered_logs = logs
        
        if start_date:
            start = datetime.fromisoformat(start_date)
            filtered_logs = [log for log in filtered_logs 
                           if datetime.fromisoformat(log.get('timestamp', '')) >= start]
            
        if end_date:
            end = datetime.fromisoformat(end_date)
            filtered_logs = [log for log in filtered_logs 
                           if datetime.fromisoformat(log.get('timestamp', '')) <= end]
            
        if session_id:
            filtered_logs = [log for log in filtered_logs 
                           if log.get('session_id') == session_id]
            
        return filtered_logs

    def _get_empty_core_metrics(self) -> Dict[str, Any]:
        """Get empty core metrics structure"""
        return {
            'total_evaluations': 0,
            'average_scores': {
                'relevance': 0.0,
                'quality': 0.0,
                'safety': 0.0,
                'empathy': 0.0,
                'clarity': 0.0
            },
            'improvement_rate': 0.0
        }

    def _get_empty_conversation_metrics(self) -> Dict[str, Any]:
        """Get empty conversation metrics structure"""
        return {
            'total_conversations': 0,
            'avg_messages_per_conversation': 0,
            'total_messages': 0
        }

    def _get_empty_performance_metrics(self) -> Dict[str, Any]:
        """Get empty performance metrics structure"""
        return {
            'response_time': {
                'average_ms': 0,
                'min_ms': 0,
                'max_ms': 0
            },
            'tokens': {
                'average': 0,
                'min': 0,
                'max': 0
            },
            'memory': {
                'average_mb': 0,
                'min_mb': 0,
                'max_mb': 0
            }
        } 