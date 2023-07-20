import numpy as np

from alts.core.experiment_runner import ExperimentRunner

from alsbts.modules.estimator import PassThroughEstimator
from alts.core.query.query_selector import StreamQuerySelector


from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler
from alsbts.modules.query.query_decider import EmptyQueryDecider


from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider

from alsbts.modules.selection_criteria import FixedIntervalSelectionCriteria

from alsbts.modules.blueprint import SbBlueprint
from alts.modules.oracle.data_source import BrownianDriftDataSource

from alsbts.modules.estimator import IntBrownGPEstimator
from alsbts.modules.selection_criteria import STDSelectionCriteria

from alts.modules.oracle.data_source import BrownianDriftDataSource
from alts.modules.data_process.process import DataSourceProcess, IntegratingDSProcess, WindowDSProcess



xblueprints = []
for i in np.arange(1, 5, 1):
    for var in np.logspace(-2.5,-1, 15):

        bp = SbBlueprint(
            repeat=10,

            process=WindowDSProcess(data_source=BrownianDriftDataSource(reinit=True, rbf_leng=0.3), window_size=i),

            experiment_modules=StreamExperiment(
                query_selector=StreamQuerySelector(
                    query_optimizer=NoQueryOptimizer(
                        selection_criteria= STDSelectionCriteria(std_threshold=var),
                        query_sampler=StreamQuerySampler(),
                    ),
                    query_decider=ThresholdQueryDecider(threshold=0.0),
                    ),
                estimator=IntBrownGPEstimator(length_scale = 0.3),
            ),
            exp_name=f"wint_brown_var_wint{i}_sdtsel{var}",
            exp_path="./eval/wint_brown_var_wint_sdtsel",
        )
        xblueprints.append(bp)


if __name__ == '__main__':
    er = ExperimentRunner(xblueprints)
    er.run_experiments_parallel()
