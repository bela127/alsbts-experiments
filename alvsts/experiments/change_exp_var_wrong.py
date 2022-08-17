import numpy as np
from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.core.oracle.oracle import Oracle
from alts.modules.query.query_sampler import FixedQuerySampler, AllQuerySampler
from alts.core.oracle.augmentation import NoAugmentation
from alts.core.blueprint import Blueprint
from alts.modules.query.query_optimizer import MaxMCQueryOptimizer
from alts.core.experiment_runner import ExperimentRunner
from alts.modules.evaluator import PrintExpTimeEvaluator

from alvsts.modules.data_source import VSSimulationDataSource

from alvsts.modules.experiment_setup import ExperimentSetup
from alvsts.modules.stopping_criteria import SimEndStoppingCriteria
from alvsts.modules.selection_criteria import ChangeSelectionCriteria
from alvsts.modules.matlab_engin import MatLabEngin
from alvsts.modules.consumer_behavior import RandomTimeUniformKpBehavior
from alvsts.modules.estimator import PassThroughEstimator
from alsbts.core.experiment_modules import StreamExperiment
from alvsts.modules.evaluator import PlotVSEvaluator, LogAllEvaluator
from alvsts.modules.rvs_estimator import NoisyGaussianRVSEstimator
from alvsts.modules.change_detector import NoisyChangeDetector

blueprints = []
with MatLabEngin() as eng:
    for wrong in np.arange(0, 0.10, 0.005):

        bp = Blueprint(
            repeat=10,
            stopping_criteria= SimEndStoppingCriteria(),
            oracle = Oracle(
                data_source=VSSimulationDataSource(
                    exp_setup=ExperimentSetup(
                        eng=eng,
                        consumer_behavior=RandomTimeUniformKpBehavior(),
                        rvs_estimator = NoisyGaussianRVSEstimator(),
                        change_detector = NoisyChangeDetector(
                            change_offset_std=3,
                            wrong_detection_ratio=wrong,
                            missed_detection_ratio=0.025,
                            )
                        )
                    ),
                augmentation= NoAugmentation()
            ),
            queried_data_pool=FlatQueriedDataPool(),
            initial_query_sampler=FixedQuerySampler(fixed_query = np.asarray([0, 1])),
            query_optimizer=MaxMCQueryOptimizer(
                selection_criteria=ChangeSelectionCriteria(),
                num_queries=1,
                query_sampler=AllQuerySampler(),
                num_tries=2,
            ),
            experiment_modules=StreamExperiment(
                estimator=PassThroughEstimator()
            ),
            evaluators=[PrintExpTimeEvaluator(),PlotVSEvaluator(),LogAllEvaluator()],
            exp_name=f"wrong_detection{wrong}",
            exp_path="./eval/wrong_detection",
        )
        blueprints.append(bp)

    er = ExperimentRunner(blueprints)

    er.run_experiments()
    input()
    
