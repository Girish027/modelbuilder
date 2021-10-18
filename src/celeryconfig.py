BROKER_URL = 'redis-sentinel://redis-sentinel:26379/0'
BROKER_TRANSPORT_OPTIONS = {
    'sentinels': [('localhost', 26379)],
    'service_name': 'mymaster',
    'socket_timeout': 1.0,
}

CELERY_RESULT_BACKEND = 'redis-sentinel://redis-sentinel:26379/0'
CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS = BROKER_TRANSPORT_OPTIONS