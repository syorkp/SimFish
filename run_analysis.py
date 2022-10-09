import sys
import os
from Analysis.Neural.MEI.estimate_mei_direct import produce_meis

try:
    run_config = sys.argv[1]
except IndexError:
    run_config = None

produce_meis("dqn_scaffold_26-2", "conv3l", True, 1000)




