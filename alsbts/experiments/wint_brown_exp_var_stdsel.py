import numpy as np
from alts.core.experiment_runner import ExperimentRunner

from alsbts.modules.blueprint import SbBlueprint

from alts.core.query.query_selector import StreamQuerySelector


from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler


from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider


from alsbts.modules.estimator import IntBrownGPEstimator
from alsbts.modules.selection_criteria import STDSelectionCriteria

from alts.modules.oracle.data_source import BrownianDriftDataSource
from alts.modules.data_process.process import DataSourceProcess, IntegratingDSProcess, WindowDSProcess



xblueprints = []
for std in np.logspace(-2.5,-1, 15):

    bp = SbBlueprint(
        repeat=10,

        process=WindowDSProcess(data_source=BrownianDriftDataSource(reinit=True)),

        experiment_modules=StreamExperiment(
            query_selector=StreamQuerySelector(
                query_optimizer=NoQueryOptimizer(
                    selection_criteria= STDSelectionCriteria(std_threshold=std),
                    query_sampler=StreamQuerySampler(),
                ),
                query_decider=ThresholdQueryDecider(threshold=0.0),
                ),
            estimator=IntBrownGPEstimator(length_scale = 0.4),
        ),
        exp_name=f"wint_brown_exp_var_stdsel{std}",
        exp_path="./eval/wint_brown_exp_var_stdsel",
    )
    xblueprints.append(bp)


if __name__ == '__main__':
    er = ExperimentRunner(xblueprints)
    er.run_experiments_parallel()

    
