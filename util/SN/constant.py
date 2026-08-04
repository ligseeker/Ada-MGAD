SN_SERVICES = [
    'social-graph-service',
    'compose-post-service',
    'post-storage-service',
    'user-timeline-service',
    'url-shorten-service',
    'user-service',
    'media-service',
    'text-service',
    'unique-id-service',
    'user-mention-service',
    'home-timeline-service',
    'nginx-web-server',
]

SN_SERVICE2NID = {name: idx for idx, name in enumerate(SN_SERVICES)}
SN_NUM_NODES = len(SN_SERVICES)

SN_METRICS = [
    'cpu_usage_system',
    'cpu_usage_total',
    'cpu_usage_user',
    'memory_usage',
    'memory_working_set',
    'rx_bytes',
    'tx_bytes',
]

FAULT_CONTAINER_TO_SERVICE = {
    'socialnetwork-text-service-1': 'text-service',
    'socialnetwork-home-timeline-service-1': 'home-timeline-service',
    'socialnetwork-media-service-1': 'media-service',
    'socialnetwork-user-timeline-service-1': 'user-timeline-service',
    'socialnetwork-user-service-1': 'user-service',
    'socialnetwork-compose-post-service-1': 'compose-post-service',
    'socialnetwork-nginx-web-server-1': 'nginx-web-server',
    'socialnetwork-post-storage-service-1': 'post-storage-service',
    'socialnetwork-social-graph-service-1': 'social-graph-service',
    'socialnetwork-url-shorten-service-1': 'url-shorten-service',
    'socialnetwork-user-mention-service-1': 'user-mention-service',
    'socialnetwork-unique-id-service-1': 'unique-id-service',
}

SN_EDGES = [
    ('compose-post-service', 'home-timeline-service'),
    ('compose-post-service', 'media-service'),
    ('compose-post-service', 'post-storage-service'),
    ('compose-post-service', 'text-service'),
    ('compose-post-service', 'unique-id-service'),
    ('compose-post-service', 'user-service'),
    ('compose-post-service', 'user-timeline-service'),
    ('home-timeline-service', 'post-storage-service'),
    ('home-timeline-service', 'social-graph-service'),
    ('social-graph-service', 'user-service'),
    ('text-service', 'url-shorten-service'),
    ('text-service', 'user-mention-service'),
    ('nginx-web-server', 'compose-post-service'),
    ('nginx-web-server', 'home-timeline-service'),
    ('nginx-web-server', 'social-graph-service'),
    ('nginx-web-server', 'user-service'),
]
