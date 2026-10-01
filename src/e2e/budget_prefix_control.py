"""Original early-stopping rule on a fixed realised training trajectory."""


def prefix_control(history, patience=8):
    if patience != 8 or not history:
        raise ValueError('only the preregistered nonempty patience8 prefix is supported')
    best, worse = None, 0
    stop_reason = 'max_epochs'
    for position, row in enumerate(history):
        if row['epoch'] != position:
            raise ValueError('epoch sequence is not contiguous')
        key = tuple(row['selection_key'])
        if best is None or key > tuple(best['selection_key']):
            best, worse = row, 0
        else:
            worse += 1
        if worse >= patience:
            stop_reason = 'validation_event_f1_patience'
            break
    return {'selected_epoch': best['epoch'], 'selected_validation_threshold': best['validation_threshold'],
            'selected_validation_metrics': best['validation_metrics'], 'epochs_completed': position + 1,
            'stop_reason': stop_reason, 'history': history[:position + 1],
            'control_kind': 'patience8 prefix of the same realised stochastic training trajectory',
            'independent_training_run': False}
