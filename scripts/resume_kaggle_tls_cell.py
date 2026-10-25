# Run in the existing Colab runtime after the failed cell, then rerun cell 3.
import time
import requests
import kaggle_output_transfer as transfer

def retry_tls_alert(operation, **options):
    for attempt in range(4):
        try:
            return transfer._before_tls_retry(operation, **options)
        except requests.exceptions.SSLError as error:
            reason = str(error).upper()
            certificate = any(value in reason for value in (
                'CERTIFICATE_VERIFY_FAILED', 'CERTIFICATE VERIFICATION',
                'HOSTNAME MISMATCH', 'CERTIFICATE EXPIRED'))
            if certificate or 'TLSV1_ALERT_DECODE_ERROR' not in reason or attempt == 3:
                raise
            print(f'TLS decode alert: retry {attempt+1}/3; cache retained', flush=True)
            time.sleep(2**attempt)


retry_tls_alert._handles_tls_decode = True
if not getattr(transfer.retry_kaggle, '_handles_tls_decode', False):
    transfer._before_tls_retry = transfer.retry_kaggle
    transfer.retry_kaggle = retry_tls_alert
KAGGLE_NOTEBOOK_OUTPUT = 'btammm/notebookd47a60658f'
print('Đã vá TLS retry. Chạy lại riêng cell 3; giữ runtime và cache.', flush=True)
