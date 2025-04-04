import multiprocessing
import os
import sys
import logging

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    '%Y-%m-%d %H:%M:%S %z'
))
logger.addHandler(handler)

# Bind to 0.0.0.0:8080
bind = '0.0.0.0:8080'

# Number of worker processes
workers = int(os.environ.get('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))

# Worker class type
worker_class = 'sync'

# Maximum number of simultaneous clients
worker_connections = 1000

# Maximum requests before worker restart
max_requests = 1000
max_requests_jitter = 50

# Timeout (seconds)
timeout = 120

# Disable request line rewriting in logs
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %({X-Forwarded-For}i)s'

# Enable logging
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('LOG_LEVEL', 'info').lower()

# Statsd monitoring (if available)
statsd_host = os.environ.get('STATSD_HOST')
statsd_prefix = 'reproductive-health-chatbot'

# Called after the server is started
def on_starting(server):
    logger.info("Starting Gunicorn server for Reproductive Health Chatbot")
    
    # Initializing application resources
    logger.info("Initializing application resources...")
    
    # Log environment
    env = os.environ.get('FLASK_ENV', 'development')
    logger.info(f"Environment: {env}")
    
    # Check API keys
    if not os.environ.get('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY environment variable not set")
    
    if not os.environ.get('ABORTION_POLICY_API_KEY'):
        logger.warning("ABORTION_POLICY_API_KEY environment variable not set")

# Called when a worker is exiting
def worker_exit(server, worker):
    logger.info(f"Worker {worker.pid} exiting")

# Called just before the master process is terminated
def on_exit(server):
    logger.info("Shutting down Gunicorn server")
    
    # Clean up any resources if needed
    logger.info("Cleaning up resources...")
    
    logger.info("Shutdown complete")

# Called when a worker receives SIGTERM
def worker_abort(worker):
    logger.info(f"Worker {worker.pid} was aborted")

# Called every time when a new worker process is forked
def post_fork(server, worker):
    logger.info(f"Worker {worker.pid} spawned")
