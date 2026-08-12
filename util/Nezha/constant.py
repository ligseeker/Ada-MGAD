import re

NEZHA_DAYS = ['2022-08-22', '2022-08-23']

NEZHA_SERVICES = [
    'adservice',
    'cartservice',
    'checkoutservice',
    'currencyservice',
    'emailservice',
    'frontend',
    'paymentservice',
    'productcatalogservice',
    'recommendationservice',
    'shippingservice',
]

NEZHA_SERVICE2NID = {name: idx for idx, name in enumerate(NEZHA_SERVICES)}
NEZHA_NUM_NODES = len(NEZHA_SERVICES)

NEZHA_KPIS = [
    'CpuUsageRate(%)',
    'MemoryUsageRate(%)',
    'SyscallRead',
    'SyscallWrite',
    'NetworkReceiveBytes',
    'NetworkTransmitBytes',
    'PodServerLatencyP99(s)',
    'PodSuccessRate(%)',
    'PodWorkload(Ops)',
    'PodClientLatencyP99(s)',
]

FAULT_DURATION_SEC = 180
BUCKET_SEC = 60

# Go services emit plain-text container logs (json['log'] is the raw message);
# all other services emit structured logs where json['log'] is itself JSON with
# 'message' and 'severity' fields.
GO_SERVICES = {'adservice', 'cartservice'}

# Log-level channels appended after the template counts: [ERROR, WARNING, INFO, total].
LOG_LEVELS = ['ERROR', 'WARNING', 'INFO']
NUM_LEVEL_CHANNELS = len(LOG_LEVELS) + 1  # +1 for log_total

_LEVEL_PATTERN = re.compile(r'\b(INFO|WARNING|WARN|ERROR|ERR)\b')
_LEVEL_ALIAS = {'WARN': 'WARNING', 'ERR': 'ERROR'}


def normalize_level(token):
    """Map a raw level token to one of LOG_LEVELS, or '' if it is not one."""
    if not token:
        return ''
    token = str(token).strip().upper()
    token = _LEVEL_ALIAS.get(token, token)
    return token if token in LOG_LEVELS else ''


def level_from_text(text):
    """Extract a level from a free-text (Go) log message, or '' if absent."""
    match = _LEVEL_PATTERN.search(str(text))
    return normalize_level(match.group(1)) if match else ''


def pod_to_service(pod):
    return '-'.join(str(pod).split('-')[:-2])
