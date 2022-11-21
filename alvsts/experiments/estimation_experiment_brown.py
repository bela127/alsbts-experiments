import numpy as np
from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.core.oracle.oracle import Oracle
from alts.modules.query.query_sampler import FixedQuerySampler, OptimalQuerySampler, AllQuerySampler
from alts.core.oracle.augmentation import NoAugmentation
from alts.core.blueprint import Blueprint
from alts.modules.query.query_optimizer import MaxMCQueryOptimizer
from alts.core.experiment_runner import ExperimentRunner
from alts.modules.evaluator import PrintExpTimeEvaluator

from alvsts.modules.data_source import VSSimulationDataSource

from alvsts.modules.experiment_setup import ExperimentSetup
from alvsts.modules.stopping_criteria import SimEndStoppingCriteria
from alvsts.modules.selection_criteria import STDPreTrainSelectionCriteria
from alvsts.modules.matlab_engin import MatLabEngin
from alvsts.modules.consumer_behavior import RandomTimeBrownKpBehavior
from alvsts.modules.estimator import GPBrownIntEstimator
from alsbts.core.experiment_modules import StreamExperiment
from alvsts.modules.evaluator import PlotVSEvaluator, LogAllEvaluator
from alvsts.modules.rvs_estimator import NoisyGaussianRVSEstimator
from alvsts.modules.change_detector import NoisyChangeDetector, OptimalChangeDetector


with MatLabEngin() as eng:
        

    blueprint = Blueprint(
        repeat=10,
        stopping_criteria= SimEndStoppingCriteria(),
        oracle = Oracle(
            data_source=VSSimulationDataSource(
                exp_setup=ExperimentSetup(
                    eng=eng,
                    consumer_behavior=RandomTimeBrownKpBehavior(),
                    rvs_estimator = NoisyGaussianRVSEstimator(
                        noise_var=0.2,
                        length_scale=0.4,
                        probability_of_function_change=-1,#0.02,
                        change_size_proportion=0,#0.50,
                        ),
                    change_detector = OptimalChangeDetector(),
                    sim_stop_time=900
                    )
                ),
            augmentation= NoAugmentation()
        ),
        queried_data_pool=FlatQueriedDataPool(),
        initial_query_sampler=FixedQuerySampler(fixed_query = np.asarray([0, 1])),
        query_optimizer=MaxMCQueryOptimizer(
            selection_criteria=STDPreTrainSelectionCriteria(),
            num_queries=1,
            query_sampler=AllQuerySampler(),
            num_tries=2,
        ),
        experiment_modules=StreamExperiment(
            estimator=GPBrownIntEstimator()
        ),
        evaluators=[PrintExpTimeEvaluator(),PlotVSEvaluator(),LogAllEvaluator()],
        exp_name="estimation_brown"
    )

    er = ExperimentRunner([blueprint])

    er.run_experiments()
    input()
    
