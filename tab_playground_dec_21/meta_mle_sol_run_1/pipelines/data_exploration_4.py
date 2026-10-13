import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = '/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv'
CLASS_COUNTS = {1: 1468136, 2: 2262087, 3: 195712, 4: 377, 5: 1, 6: 11426, 7: 62261}

def audit_setup_entry():
    # Reproduce the locked row-selection graph without evaluation marks.
    # Exploration compares populations, rather than fitting models.
    data = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    data = data.sort_values('Id').reset_index(drop=True)
    rng = skrub.as_data_op(42).skb.apply_func(np.random.default_rng)
    pieces = []
    for label, count in CLASS_COUNTS.items():
        sample_size = max(1, int(np.floor(0.1 * count + 0.5)))
        positions = rng.choice(count, size=sample_size, replace=False)
        class_rows = data[data['Cover_Type'] == label]
        pieces.append(class_rows.iloc[positions])
    rows = pieces[0].skb.concat(pieces[1:], axis=0)
    rows = rows.sort_values('Id').reset_index(drop=True)
    return {
        'y': rows['Cover_Type'],
        'row_keys': rows['Id'],
        'audit': {
            'class_counts': rows['Cover_Type'].value_counts().sort_index(),
            'sample_shape': rows.shape,
            'id_bounds': rows['Id'].agg(['min', 'max']),
        },
    }

def build():
    first = audit_setup_entry()
    second = audit_setup_entry()
    keys_first = first['row_keys']
    keys_second = second['row_keys']
    return {
        'fingerprint_status': skrub.as_data_op(
            'UNVERIFIABLE: The exact harness fingerprint procedure is not supplied. '
            'The expected fingerprint is '
            '3ac027239c0967110753312cdc236925add4de5587bfeea9434ff482fd25b251. '
            'No substitute hashing procedure is used.'
        ),
        'artifact_status': skrub.as_data_op(
            'UNAVAILABLE: train.csv is the only authorized task source. '
            'Full artifacts and fitted estimators for probe_26b59caa9b89 and '
            'probe_bc6cfbb66a12 are not available as inputs. Prior outputs are '
            'not permitted plan inputs. Actual classes_, class-aligned OOF '
            'confusion matrices and paired errors therefore cannot be verified.'
        ),
        'repeat_test_scope': skrub.as_data_op(
            'Two replicas of the unchanged locked row-selection graph are '
            'constructed and their complete ordered row keys are compared within '
            'one evaluation pass. Evaluation marks are omitted to allow both '
            'replicas in an exploration graph. Graph deduplication may share '
            'nodes, so equality is not evidence of independent-execution '
            'reproducibility. Independent executions and actual harness fold '
            'assignments remain unverified. With identical ordered targets, '
            'StratifiedKFold(n_splits=3, shuffle=True, random_state=42) would '
            'produce identical splits; this is not a fingerprint verification.'
        ),
        'first_row_manifest': keys_first,
        'second_row_manifest': keys_second,
        'same_ordered_row_manifest': keys_first.equals(keys_second),
        'different_row_positions': (keys_first != keys_second).sum(),
        'same_ordered_targets': first['y'].equals(second['y']),
        'first_id_bounds': first['audit']['id_bounds'],
        'second_id_bounds': second['audit']['id_bounds'],
        'first_class_counts': first['audit']['class_counts'],
        'second_class_counts': second['audit']['class_counts'],
        'first_shape': first['audit']['sample_shape'],
        'second_shape': second['audit']['sample_shape'],
    }