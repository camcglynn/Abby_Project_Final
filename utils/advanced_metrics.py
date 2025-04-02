"""
Advanced Metrics for the reproductive health chatbot

This module provides advanced metrics calculations for evaluating chatbot responses:
- BLEU, ROUGE, BERTScore for text similarity evaluation
- Ragas metrics for RAG quality
- Precision@K, Recall@K, MRR for retrieval accuracy
- Faithfulness evaluation
- System performance metrics
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import evaluate
from ragas.metrics import faithfulness
from ragas.metrics import context_precision, context_recall

# Importing TrecTools for MRR calculation
from trectools import TrecRun, TrecQrel, TrecEval

logger = logging.getLogger(__name__)

class AdvancedMetricsCalculator:
    """
    Calculates advanced metrics for chatbot evaluation
    """
    
    def __init__(self, log_file="evaluation_logs.json"):
        """
        Initialize the advanced metrics calculator
        
        Args:
            log_file (str): Path to the evaluation logs file
        """
        self.log_file = log_file
        self.logs = self._load_logs()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        
        # Initialize Ragas metrics
        self.faithfulness_metric = faithfulness
        self.context_precision_metric = context_precision
        self.context_recall_metric = context_recall
        
        # For system performance tracking
        self.performance_data = {
            "inference_times": [],
            "tokens_per_response": [],
            "memory_usage": []
        }
    
    def _load_logs(self):
        """Load evaluation logs from the log file"""
        try:
            if not os.path.exists(self.log_file):
                logger.warning(f"Log file not found: {self.log_file}")
                return []
                
            logs = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        # Parse timestamp string to datetime
                        if 'timestamp' in log_entry:
                            log_entry['timestamp'] = datetime.fromisoformat(log_entry['timestamp'])
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in log file: {line}")
                    except Exception as e:
                        logger.warning(f"Error processing log entry: {str(e)}")
            
            return logs
        except Exception as e:
            logger.error(f"Error loading logs: {str(e)}")
            return []
    
    def calculate_text_similarity_metrics(self, references: List[str], 
                                         predictions: List[str],
                                         record_realtime_metrics: bool = True) -> Dict[str, Any]:
        """
        Calculate text similarity metrics (BLEU, ROUGE, BERTScore)
        
        Args:
            references (List[str]): List of reference/ground truth texts
            predictions (List[str]): List of generated/predicted texts
            record_realtime_metrics (bool): Whether to record metrics for real-time tracking
            
        Returns:
            Dict[str, Any]: Dictionary with calculated metrics
        """
        metrics = {}
        
        try:
            # Calculate BLEU score
            bleu_score = self.bleu.corpus_score(predictions, [references])
            metrics['bleu'] = {
                'score': bleu_score.score,
                'details': {
                    'precisions': bleu_score.precisions,
                    'bp': bleu_score.bp,
                    'ratio': bleu_score.ratio,
                    'sys_len': bleu_score.sys_len,
                    'ref_len': bleu_score.ref_len
                }
            }
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {str(e)}")
            metrics['bleu'] = {'score': 0.0, 'error': str(e)}
        
        try:
            # Calculate ROUGE scores
            rouge_scores = {
                'rouge1': [],
                'rouge2': [],
                'rougeL': []
            }
            
            for ref, pred in zip(references, predictions):
                scores = self.rouge_scorer.score(ref, pred)
                for key in rouge_scores:
                    rouge_scores[key].append(scores[key].fmeasure)
                
                # Record real-time metrics for individual item evaluation
                if record_realtime_metrics:
                    try:
                        from utils.metrics import record_rouge_metrics
                        record_rouge_metrics(
                            rouge1=scores['rouge1'].fmeasure,
                            rouge2=scores['rouge2'].fmeasure,
                            rougeL=scores['rougeL'].fmeasure
                        )
                    except ImportError:
                        logger.warning("Failed to import metrics module for real-time tracking")
                    except Exception as e:
                        logger.warning(f"Failed to record real-time ROUGE metrics: {str(e)}")
            
            metrics['rouge'] = {
                'rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0,
                'rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0,
                'rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            metrics['rouge'] = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'error': str(e)}
        
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score(predictions, references, lang="en", return_hash=False, rescale_with_baseline=True)
            metrics['bert_score'] = {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {str(e)}")
            metrics['bert_score'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'error': str(e)}
        
        return metrics
        
    def calculate_single_rouge_score(self, reference: str, prediction: str, 
                                    record_realtime: bool = True) -> Dict[str, float]:
        """
        Calculate ROUGE scores for a single text pair and optionally record for real-time tracking
        
        Args:
            reference (str): Reference/ground truth text
            prediction (str): Generated/predicted text
            record_realtime (bool): Whether to record for real-time tracking
            
        Returns:
            Dict[str, float]: Dictionary with ROUGE scores
        """
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            result = {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
            
            # Record metrics for real-time tracking
            if record_realtime:
                try:
                    from utils.metrics import record_rouge_metrics
                    record_rouge_metrics(
                        rouge1=result['rouge1'],
                        rouge2=result['rouge2'],
                        rougeL=result['rougeL']
                    )
                except ImportError:
                    logger.warning("Failed to import metrics module for real-time tracking")
                except Exception as e:
                    logger.warning(f"Failed to record real-time ROUGE metrics: {str(e)}")
            
            return result
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'error': str(e)}
    
    def calculate_retrieval_metrics(self, retrieved_docs: List[List[Dict]], 
                                   relevant_docs: List[List[Dict]],
                                   k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Calculate retrieval accuracy metrics (Precision@K, Recall@K, MRR)
        
        Args:
            retrieved_docs (List[List[Dict]]): List of lists of retrieved documents for each query
            relevant_docs (List[List[Dict]]): List of lists of relevant documents for each query
            k_values (List[int]): List of K values for Precision/Recall@K
            
        Returns:
            Dict[str, Any]: Dictionary with calculated metrics
        """
        metrics = {}
        
        try:
            # Calculate Precision@K and Recall@K
            precision_at_k = {k: [] for k in k_values}
            recall_at_k = {k: [] for k in k_values}
            
            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                # Get document IDs
                retrieved_ids = [doc.get('id') for doc in retrieved]
                relevant_ids = [doc.get('id') for doc in relevant]
                
                for k in k_values:
                    # Precision@K
                    if k <= len(retrieved_ids):
                        retrieved_at_k = retrieved_ids[:k]
                        relevant_retrieved = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_ids]
                        precision = len(relevant_retrieved) / k if k > 0 else 0
                        precision_at_k[k].append(precision)
                    
                    # Recall@K
                    if len(relevant_ids) > 0:
                        retrieved_at_k = retrieved_ids[:k] if k <= len(retrieved_ids) else retrieved_ids
                        relevant_retrieved = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_ids]
                        recall = len(relevant_retrieved) / len(relevant_ids) if len(relevant_ids) > 0 else 0
                        recall_at_k[k].append(recall)
            
            # Average Precision@K and Recall@K
            metrics['precision_at_k'] = {
                f'p@{k}': sum(precision_at_k[k]) / len(precision_at_k[k]) if precision_at_k[k] else 0
                for k in k_values
            }
            
            metrics['recall_at_k'] = {
                f'r@{k}': sum(recall_at_k[k]) / len(recall_at_k[k]) if recall_at_k[k] else 0
                for k in k_values
            }
            
            # Calculate MRR (Mean Reciprocal Rank)
            mrr_values = []
            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                retrieved_ids = [doc.get('id') for doc in retrieved]
                relevant_ids = [doc.get('id') for doc in relevant]
                
                # Find rank of first relevant document
                for i, doc_id in enumerate(retrieved_ids):
                    if doc_id in relevant_ids:
                        mrr_values.append(1.0 / (i + 1))
                        break
                else:
                    mrr_values.append(0.0)
            
            metrics['mrr'] = sum(mrr_values) / len(mrr_values) if mrr_values else 0
            
        except Exception as e:
            logger.error(f"Error calculating retrieval metrics: {str(e)}")
            metrics['retrieval_error'] = str(e)
        
        return metrics
    
    def calculate_ragas_metrics(self, questions: List[str], 
                               retrieved_contexts: List[List[str]],
                               generated_answers: List[str]) -> Dict[str, Any]:
        """
        Calculate RAG quality metrics using Ragas
        
        Args:
            questions (List[str]): List of questions
            retrieved_contexts (List[List[str]]): Retrieved contexts for each question
            generated_answers (List[str]): Generated answers for each question
            
        Returns:
            Dict[str, Any]: Dictionary with calculated metrics
        """
        metrics = {}
        
        try:
            # Ensure we have at least some valid data to work with
            if not questions or not generated_answers:
                logger.warning("No questions or answers provided for Ragas evaluation")
                metrics['ragas_error'] = "No questions or answers provided"
                return metrics
                
            # Make sure all lists have the same length
            min_length = min(len(questions), len(generated_answers))
            if len(retrieved_contexts) != min_length:
                logger.warning(f"Context list length ({len(retrieved_contexts)}) doesn't match questions length ({min_length})")
                # Adjust to the smallest length
                questions = questions[:min_length]
                generated_answers = generated_answers[:min_length]
                if len(retrieved_contexts) > min_length:
                    retrieved_contexts = retrieved_contexts[:min_length]
                else:
                    # Pad with empty contexts if needed
                    retrieved_contexts.extend([[] for _ in range(min_length - len(retrieved_contexts))])
            
            # Ensure each question has at least one context (even if empty)
            for i in range(len(retrieved_contexts)):
                if not retrieved_contexts[i]:
                    retrieved_contexts[i] = ["No context available"]
            
            # Set up the data for Ragas
            logger.info(f"Preparing Ragas evaluation for {len(questions)} questions")
            logger.info(f"Sample question: {questions[0][:50]}...")
            logger.info(f"Sample answer: {generated_answers[0][:50]}...")
            logger.info(f"Sample context count: {len(retrieved_contexts[0])}")
            
            data = {
                "question": questions,
                "contexts": retrieved_contexts,
                "answer": generated_answers
            }
            
            # Calculate faithfulness score
            logger.info("Calculating faithfulness score...")
            faithfulness_result = self.faithfulness_metric.score(data)
            metrics['faithfulness'] = faithfulness_result['faithfulness']
            
            # Calculate context precision and recall
            logger.info("Calculating context precision...")
            precision_result = self.context_precision_metric.score(data)
            metrics['context_precision'] = precision_result['context_precision']
            
            logger.info("Calculating context recall...")
            recall_result = self.context_recall_metric.score(data)
            metrics['context_recall'] = recall_result['context_recall']
            
            logger.info("Ragas metrics calculation complete")
            
        except Exception as e:
            logger.error(f"Error calculating Ragas metrics: {str(e)}", exc_info=True)
            metrics['ragas_error'] = str(e)
        
        return metrics
    
    def track_performance_metrics(self, inference_time: float, tokens: int, memory_mb: float):
        """
        Track system performance metrics
        
        Args:
            inference_time (float): Time taken for inference in milliseconds
            tokens (int): Number of tokens in the response
            memory_mb (float): Memory usage in MB
        """
        self.performance_data["inference_times"].append(inference_time)
        self.performance_data["tokens_per_response"].append(tokens)
        self.performance_data["memory_usage"].append(memory_mb)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get system performance metrics
        
        Returns:
            Dict[str, Any]: Dictionary with performance metrics
        """
        metrics = {}
        
        if self.performance_data["inference_times"]:
            metrics["average_inference_time_ms"] = sum(self.performance_data["inference_times"]) / len(self.performance_data["inference_times"])
            metrics["max_inference_time_ms"] = max(self.performance_data["inference_times"])
            metrics["min_inference_time_ms"] = min(self.performance_data["inference_times"])
        else:
            metrics["average_inference_time_ms"] = 0
            metrics["max_inference_time_ms"] = 0
            metrics["min_inference_time_ms"] = 0
        
        if self.performance_data["tokens_per_response"]:
            metrics["average_tokens_per_response"] = sum(self.performance_data["tokens_per_response"]) / len(self.performance_data["tokens_per_response"])
            metrics["max_tokens_per_response"] = max(self.performance_data["tokens_per_response"])
            metrics["min_tokens_per_response"] = min(self.performance_data["tokens_per_response"])
        else:
            metrics["average_tokens_per_response"] = 0
            metrics["max_tokens_per_response"] = 0
            metrics["min_tokens_per_response"] = 0
        
        if self.performance_data["memory_usage"]:
            metrics["average_memory_usage_mb"] = sum(self.performance_data["memory_usage"]) / len(self.performance_data["memory_usage"])
            metrics["max_memory_usage_mb"] = max(self.performance_data["memory_usage"])
            metrics["min_memory_usage_mb"] = min(self.performance_data["memory_usage"])
        else:
            metrics["average_memory_usage_mb"] = 0
            metrics["max_memory_usage_mb"] = 0
            metrics["min_memory_usage_mb"] = 0
        
        return metrics
    
    def calculate_all_metrics(self, references: List[str] = None, 
                             predictions: List[str] = None,
                             log_entries: List[Dict] = None) -> Dict[str, Any]:
        """
        Calculate all available metrics
        
        Args:
            references (List[str], optional): List of reference texts
            predictions (List[str], optional): List of predicted texts
            log_entries (List[Dict], optional): List of log entries to analyze
            
        Returns:
            Dict[str, Any]: Dictionary with all calculated metrics
        """
        metrics = {}
        
        # If log entries are provided, extract required data
        if log_entries is not None:
            # Extract data for text similarity metrics
            pairs = []
            for entry in log_entries:
                question = entry.get('question', '')
                original_response = entry.get('original_response', '')
                improved_response = entry.get('evaluation', {}).get('improved_response', '')
                
                if question and original_response:
                    pairs.append((question, original_response, improved_response if improved_response else original_response))
            
            if pairs:
                references = [pair[0] for pair in pairs]  # Questions as references
                predictions = [pair[1] for pair in pairs]  # Original responses as predictions
                improved_predictions = [pair[2] for pair in pairs]  # Improved responses
                
                # Calculate text similarity metrics
                metrics['text_similarity'] = self.calculate_text_similarity_metrics(references, predictions)
                metrics['improved_text_similarity'] = self.calculate_text_similarity_metrics(references, improved_predictions)
        
        # If explicit references and predictions are provided
        elif references is not None and predictions is not None:
            metrics['text_similarity'] = self.calculate_text_similarity_metrics(references, predictions)
        
        # Calculate performance metrics if available
        performance_metrics = self.get_performance_metrics()
        if performance_metrics:
            metrics['performance'] = performance_metrics
        
        return metrics
    
    def get_metrics_over_time(self, start_date=None, end_date=None, interval='day'):
        """
        Get metrics aggregated over time periods
        
        Args:
            start_date (datetime, optional): Start date for filtering
            end_date (datetime, optional): End date for filtering
            interval (str): Aggregation interval ('day', 'week', 'month')
            
        Returns:
            Dict[str, Any]: Dictionary with metrics over time
        """
        # Filter logs by date range
        filtered_logs = self.logs
        if start_date or end_date:
            filtered_logs = [
                log for log in filtered_logs 
                if 'timestamp' in log and 
                (not start_date or log['timestamp'] >= start_date) and
                (not end_date or log['timestamp'] <= end_date)
            ]
        
        # Group logs by time interval
        time_periods = {}
        
        for log in filtered_logs:
            timestamp = log.get('timestamp')
            if not timestamp:
                continue
            
            # Get period key based on interval
            if interval == 'day':
                period_key = timestamp.strftime('%Y-%m-%d')
            elif interval == 'week':
                period_key = f"{timestamp.strftime('%Y')}-W{timestamp.isocalendar()[1]}"
            elif interval == 'month':
                period_key = timestamp.strftime('%Y-%m')
            else:
                period_key = timestamp.strftime('%Y-%m-%d')
            
            if period_key not in time_periods:
                time_periods[period_key] = []
            
            time_periods[period_key].append(log)
        
        # Calculate metrics for each time period
        metrics_over_time = {}
        
        for period, logs in time_periods.items():
            metrics_over_time[period] = self.calculate_all_metrics(log_entries=logs)
        
        return metrics_over_time

# Function to generate a comprehensive performance report
def generate_performance_report(log_file="evaluation_logs.json", 
                               start_date=None, 
                               end_date=None) -> Dict[str, Any]:
    """
    Generate a comprehensive performance report with all metrics
    
    Args:
        log_file (str): Path to the evaluation logs file
        start_date (datetime, optional): Start date for filtering
        end_date (datetime, optional): End date for filtering
        
    Returns:
        Dict[str, Any]: Dictionary with comprehensive performance metrics
    """
    calculator = AdvancedMetricsCalculator(log_file)
    
    # Get metrics for the specified date range
    metrics = calculator.calculate_all_metrics(
        log_entries=[
            log for log in calculator.logs 
            if (not start_date or log.get('timestamp', datetime.min) >= start_date) and
               (not end_date or log.get('timestamp', datetime.max) <= end_date)
        ]
    )
    
    # Get metrics over time
    metrics_over_time = calculator.get_metrics_over_time(start_date, end_date, interval='day')
    
    return {
        'summary': metrics,
        'over_time': metrics_over_time
    }