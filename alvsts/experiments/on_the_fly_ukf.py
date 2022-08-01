import numpy as np
from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.core.oracle.oracle import Oracle
from alts.modules.query.query_sampler import FixedQuerySampler, OptimalQuerySampler
from alts.core.oracle.augmentation import NoAugmentation
from alts.core.blueprint import Blueprint
from alts.core.experiment_modules import ExperimentModules
from alts.modules.query.query_optimizer import MaxMCQueryOptimizer
from alts.core.experiment_runner import ExperimentRunner

from alvsts.modules.data_source import VSSimulationDataSource

from alvsts.modules.experiment_setup import ExperimentSetup
from alvsts.modules.stopping_criteria import SimEndStoppingCriteria
from alvsts.modules.selection_criteria import FixedIntervalSelectionCriteria
from alvsts.modules.matlab_engin import MatLabEngin
from alvsts.modules.consumer_behavior import RandomTimeUniformKpBehavior

with MatLabEngin() as eng:
        

    blueprint = Blueprint(
        repeat=1,
        stopping_criteria= SimEndStoppingCriteria(),
        oracle = Oracle(
            data_source=VSSimulationDataSource(
                exp_setup=ExperimentSetup(
                    eng=eng,
                    consumer_behavior=RandomTimeUniformKpBehavior(),
                    )
                ),
            augmentation= NoAugmentation()
        ),
        queried_data_pool=FlatQueriedDataPool(),
        initial_query_sampler=FixedQuerySampler(fixed_query = np.asarray([np.nan, 1])),
        query_optimizer=MaxMCQueryOptimizer(
            selection_criteria=FixedIntervalSelectionCriteria(),
            num_queries=1,
            query_sampler=OptimalQuerySampler(optimal_queries=(np.asarray([[np.nan, 0],[np.nan, 1]]),)),
            num_tries=2,
        ),
        experiment_modules=ExperimentModules(),
        evaluators=[ ],
    )

    er = ExperimentRunner([blueprint])
    er.run_experiments()
    input()
    exp_setup.stop_simulation()

    
