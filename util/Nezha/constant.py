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
]

FAULT_DURATION_SEC = 120
BUCKET_SEC = 60


def pod_to_service(pod):
    return '-'.join(str(pod).split('-')[:-2])
