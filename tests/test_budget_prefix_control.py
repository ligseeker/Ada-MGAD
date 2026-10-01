import copy

from src.e2e.budget_prefix_control import prefix_control


def test_early_stop_excludes_late_recovery_and_preserves_threshold():
    history = [{'epoch': epoch, 'selection_key': [f1, .7, .9], 'validation_threshold': .9,
                'validation_metrics': {'event_f1': f1}}
               for epoch, f1 in enumerate([.5, .8] + [.6] * 8 + [.9] * 20)]
    original = copy.deepcopy(history)
    result = prefix_control(history)
    assert result['epochs_completed'] == 10
    assert result['selected_epoch'] == 1
    assert result['selected_validation_threshold'] == .9
    assert result['stop_reason'] == 'validation_event_f1_patience'
    assert history == original


def test_original_lexicographic_tie_break_resets_patience():
    history = [{'epoch': epoch, 'selection_key': [.8, .7, threshold], 'validation_threshold': threshold,
                'validation_metrics': {'event_f1': .8}}
               for epoch, threshold in enumerate([.8] * 7 + [.9] + [.8] * 12)]
    result = prefix_control(history)
    assert result['selected_epoch'] == 7
    assert result['epochs_completed'] == 16
