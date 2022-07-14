"""
Optuna example that optimizes a classifier configuration for Iris dataset using sklearn.

In this example, we optimize a classifier configuration for Iris dataset. Classifiers are from
scikit-learn. We optimize both the choice of classifier (among SVC and RandomForest) and their
hyperparameters.

"""

from comet_ml import Experiment

import optuna

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


def log_trial_to_comet(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Logs a single trial for the given study to Comet."""
    project_name = f"sklearn_simple_comet_sdk_{study.study_name}"
    experiment = Experiment(project_name=project_name)
    experiment.set_name(f"trail_{trial.number}")

    experiment.log_parameters(trial.params)
    experiment.log_metrics(
        dic={
            "value": trial.value,
        },
        step=0)
    experiment.log_others({
        "trial_number": trial.number,
        "trial_datetime_start": trial.datetime_start,
        "trial_system_attrs": trial.system_attrs,
        "trial_user_attrs": trial.user_attrs,
        "study_direction": study.direction,
    })
    experiment.end()


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)

    for trial in study.get_trials(deepcopy=False):
        log_trial_to_comet(study, trial)
