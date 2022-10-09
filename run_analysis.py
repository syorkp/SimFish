import sys
import os
from Analysis.Neural.MEI.estimate_mei_direct import produce_meis, produce_meis_extended

try:
    run_config = sys.argv[1]
except IndexError:
    run_config = None

# produce_meis("dqn_scaffold_26-2", "rnn_in", True, 1000, conv=False)
produce_meis_extended("dqn_scaffold_26-2", "conv1l", True, 1000)
# produce_meis_extended("dqn_scaffold_26-2", "conv2l", True, 1000)
# produce_meis_extended("dqn_scaffold_26-2", "conv3l", True, 1000)
# produce_meis_extended("dqn_scaffold_26-2", "conv4l", True, 1000)




