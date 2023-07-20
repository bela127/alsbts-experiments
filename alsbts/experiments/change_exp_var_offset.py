import numpy as np
from alts.core.experiment_runner import ExperimentRunner

from alsbts.modules.blueprint import SbBlueprint

from alsbts.modules.estimator import PassThroughEstimator
from alts.core.query.query_selector import StreamQuerySelector


from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler
from alsbts.modules.query.query_decider import EmptyQueryDecider


from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider

from alsbts.modules.selection_criteria import ChangeSelectionCriteria

from alsbts.modules.change_detector import NoisyChangeDetector

blueprints = []
for offset in np.arange(0, 15, 1):

    bp = SbBlueprint(
        repeat=50,

        experiment_modules=StreamExperiment(
            query_selector=StreamQuerySelector(
                query_optimizer=NoQueryOptimizer(
                    selection_criteria= ChangeSelectionCriteria(
                        change_detector=NoisyChangeDetector(
                            change_offset_std=offset,
                            wrong_detection_ratio=0.015,
                            missed_detection_ratio=0.015,
                        ),
                    ),
                    query_sampler=StreamQuerySampler(),
                ),
                query_decider=ThresholdQueryDecider(threshold=0.0),
                ),
            estimator=PassThroughEstimator(),
        ),
        exp_name=f"change_exp_var_offset{offset}",
        exp_path="./eval/change_exp_var_offset",
    )
    blueprints.append(bp)


if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments_parallel()

    
