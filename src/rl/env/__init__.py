from gym.envs.registration import register

# create a holdout set
from sklearn.model_selection import train_test_split
import numpy as np
from tools.interface import *
from env.malware_env import MalwareEnv

np.random.seed(123)
# sha256 = interface.get_available_sha256()
# sha256_train, sha256_holdout = train_test_split(sha256, test_size=2000)

interface = Interface()
sha256_train = interface.get_available_sha256()  #
interface.test = True
sha256_holdout = interface.get_available_sha256()  #

MAXTURNS = 60

register(
    id='malware-v0',
    entry_point='env:MalwareEnv',
    kwargs={'random_sample': False, 'maxturns': MAXTURNS, 'sha256list': sha256_train, 'test': False}
)

register(
    id='malware-test-v0',
    entry_point='env:MalwareEnv',
    kwargs={'random_sample': False, 'maxturns': MAXTURNS, 'sha256list': sha256_holdout, 'test': True}
)