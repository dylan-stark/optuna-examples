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


def log_study_trial(
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
    metric_name: str = "value"
) -> None:
    """Logs a single trial for the given study to Comet."""

    def add_flattened_with_index(source: dict, prefix: str, target: dict) -> dict:
        """Adds flattened dictionary with indexed set of (name, value) pairs."""
        for i, key in enumerate(source.keys()):
            target[f"{prefix}_{i}_name"] = key
            target[f"{prefix}_{i}_value"] = source[key]
        return target

    def add_flattened(source: dict, prefix: str, target: dict) -> dict:
        """Adds flattened dictionary with (name, value) pairs."""
        for key, value in source.items():
            target[f"{prefix}_{key}"] = value
        return target

    experiment = Experiment(project_name=study.study_name)
    experiment.set_name(f"trail_{trial.number}")

    study_info = {
        "study_best_trial_datetime_complete": study.best_trial.datetime_complete,
        "study_best_trial_datetime_start": study.best_trial.datetime_start,
        "study_best_trial_duration": study.best_trial.duration,
        "study_best_trial_number": study.best_trial.number,
        "study_best_trial_state": study.best_trial.state,
        f"study_best_trial_{metric_name}": study.best_trial.value,
        # TODO: Add support for values from best trial
        # TODO: Add support for best trials
        f"study_best_{metric_name}": study.best_value,
        "study_direction": study.direction,
        # TODO: Add support for directions
        "study_pruner": study.pruner,
        "study_sampler": study.sampler,
        "study_name": study.study_name,
    }
    study_info = add_flattened_with_index(study.best_params, "study_best_params", study_info)
    study_info = add_flattened_with_index(study.best_trial.distributions, "study_best_trial_distributions", study_info)
    study_info = add_flattened_with_index(study.best_trial.params, "study_best_trial_params", study_info)
    study_info = add_flattened(study.best_trial.system_attrs, "study_best_trial_system_attrs", study_info)
    study_info = add_flattened(study.best_trial.user_attrs, "study_best_trial_user_attrs", study_info)
    study_info = add_flattened(study.system_attrs, "study_system_attrs", study_info)
    study_info = add_flattened(study.user_attrs, "study_user_attrs", study_info)

    trial_info = {
        "trial_datetime_complete": trial.datetime_complete,
        "trial_datetime_start": trial.datetime_start,
        "trial_duration": trial.duration,
        "trial_number": trial.number,
        "trial_state": trial.state,
        f"trial_{metric_name}": trial.value,
    }
    trial_info = add_flattened_with_index(trial.distributions, "trial_distributions", trial_info)
    trial_info = add_flattened_with_index(trial.params, "trial_params", trial_info)
    trial_info = add_flattened(trial.system_attrs, "trial_system_attrs", trial_info)
    trial_info = add_flattened(trial.user_attrs, "trial_user_attrs", trial_info)

    experiment.log_parameters(trial.params)
    experiment.log_metrics(
        dic={
            metric_name: trial.value,
        },
        step=0)
    experiment.log_others(study_info)
    experiment.log_others(trial_info)
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
    study.optimize(objective, n_trials=10)
    print(study.best_trial)

    for trial in study.get_trials(deepcopy=False):
        log_study_trial(study, trial, metric_name="accuracy")
