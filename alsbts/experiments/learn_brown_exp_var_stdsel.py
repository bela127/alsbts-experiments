import numpy as np
from alts.core.experiment_runner import ExperimentRunner

from alsbts.modules.blueprint import SbBlueprint

from alsbts.modules.query.query_selector import StreamQuerySelector


from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler


from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider


from alsbts.modules.estimator import BrownGPAdaptEstimator
from alsbts.modules.selection_criteria import STDSelectionCriteria




blueprints = []
for var in np.logspace(-3,-1, 20):

    bp = SbBlueprint(
        repeat=20,

        experiment_modules=StreamExperiment(
            query_selector=StreamQuerySelector(
                query_optimizer=NoQueryOptimizer(
                    selection_criteria= STDSelectionCriteria(std_threshold=var),
                    query_sampler=StreamQuerySampler(),
                ),
                query_decider=ThresholdQueryDecider(threshold=0.0),
                ),
            estimator=BrownGPAdaptEstimator(length_scale = 0.4),
        ),
        exp_name=f"learn_brown_exp_var_stdsel{var}",
        exp_path="./eval/learn_brown_exp_var_stdsel",
    )
    blueprints.append(bp)


if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments_parallel()

    
