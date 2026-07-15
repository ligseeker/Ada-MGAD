GAIA_SERVICES = [
    'dbservice1', 'dbservice2',
    'logservice1', 'logservice2',
    'mobservice1', 'mobservice2',
    'redisservice1', 'redisservice2',
    'webservice1', 'webservice2',
]

GAIA_IP_MAP = {
    '0.0.0.4': ['dbservice1', 'mobservice2', 'system'],
    '0.0.0.2': ['dbservice2', 'logservice2', 'redisservice2', 'system'],
    '0.0.0.3': ['logservice1', 'redis', 'webservice2'],
    '0.0.0.1': ['mobservice1', 'redisservice1', 'system', 'webservice1', 'zookeeper'],
}

GAIA_ANOMALY_TYPES = [
    'login failure',
    'memory_anomalies',
    'cpu_anomalies',
    'file moving program',
    'normal memory freed label',
    'access permission denied exception',
]

GAIA_SAMPLE_INTERVAL = 30

GAIA_MULTI_CORE_PREFIXES = [
    'docker_cpu_core_',
]
