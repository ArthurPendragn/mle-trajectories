import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = '/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv'
CLASS_COUNTS = {1: 1468136, 2: 2262087, 3: 195712, 4: 377, 5: 1, 6: 11426, 7: 62261}


def build():
    data = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    data = data.sort_values('Id').reset_index(drop=True)
    rng = skrub.as_data_op(42).skb.apply_func(np.random.default_rng)
    pieces = []
    outputs = {}
    for label, count in CLASS_COUNTS.items():
        sample_size = max(1, int(np.floor(0.1 * count + 0.5)))
        positions = rng.choice(count, size=sample_size, replace=False)
        class_rows = data[data['Cover_Type'] == label]
        piece = class_rows.iloc[positions]
        pieces.append(piece)
        outputs['selected_head_class_' + str(label)] = piece[['Id', 'Cover_Type']].sort_values('Id').head(10)

    rows = pieces[0].skb.concat(pieces[1:], axis=0)
    rows = rows.sort_values('Id').reset_index(drop=True)
    outputs['sample_shape'] = rows.shape
    outputs['class_counts'] = rows['Cover_Type'].value_counts().sort_index()
    outputs['id_bounds'] = rows['Id'].agg(['min', 'max'])
    outputs['sample_head'] = rows[['Id', 'Cover_Type']].head(30)
    outputs['probe_id_two_membership'] = rows[rows['Id'] == 2][['Id', 'Cover_Type']]
    outputs['singleton'] = rows[rows['Cover_Type'] == 5][['Id', 'Cover_Type']]

    X = rows.drop(columns=['Id', 'Cover_Type'])
    y = rows['Cover_Type']
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = skrub.as_data_op(cv).split(X, y).skb.apply_func(list)
    for fold in range(3):
        train_indices = folds[fold][0]
        test_indices = folds[fold][1]
        train_rows = rows.iloc[train_indices]
        test_rows = rows.iloc[test_indices]
        outputs['train_classes_fold_' + str(fold)] = train_rows['Cover_Type'].value_counts().sort_index()
        outputs['validation_classes_fold_' + str(fold)] = test_rows['Cover_Type'].value_counts().sort_index()
        outputs['validation_head_fold_' + str(fold)] = test_rows[['Id', 'Cover_Type']].head(20)
        outputs['probability_column_labels_fold_' + str(fold)] = train_rows['Cover_Type'].drop_duplicates().sort_values().reset_index(drop=True)

    wilderness = ['Wilderness_Area' + str(i) for i in range(1, 5)]
    soil = ['Soil_Type' + str(i) for i in range(1, 41)]
    audited = rows.assign(
        wilderness_count=rows[wilderness].sum(axis=1),
        soil_count=rows[soil].sum(axis=1),
    )
    outputs['wilderness_activation_counts'] = audited['wilderness_count'].value_counts().sort_index()
    outputs['soil_activation_counts'] = audited['soil_count'].value_counts().sort_index()
    for column in [
        'Elevation',
        'Horizontal_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
    ]:
        outputs[column + '_quantiles'] = rows[column].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

    outputs['analysis_limitations'] = skrub.as_data_op(
        'No OOF predictions or fitted fold estimators are task sources. '
        'Consequently this exploration cannot compute confusion matrices, recall, '
        'error confidence, error rates or paired model errors without a new probe. '
        'The probability column labels above are expected sorted training labels, '
        'not inspection of fitted classes_. The fold without class 5 should have '
        'six columns labeled 1, 2, 3, 4, 6, 7. '
        'The sample heads and fold heads allow comparison with the reported probe, '
        'but do not independently verify the harness fold fingerprint. '
        'The locked sampling graph shares a mutable Generator across class nodes; '
        'execution order may affect which class receives each RNG draw sequence. '
        'Any discrepancy between these rows, the prior audit and probe remains '
        'unresolved and should be investigated before attributing modeling gains. '
        'Do not change the locked sampling graph to repair this in an experiment.'
    )
    return outputs