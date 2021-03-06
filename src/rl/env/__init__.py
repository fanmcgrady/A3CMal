
from gym.envs.registration import register

# create a holdout set
from tools.interface import *


np.random.seed(123)
# sha256 = interface.get_available_sha256()
# sha256_train, sha256_holdout = train_test_split(sha256, test_size=2000)
from tools import interface

# interface = Interface()
sha256_train = interface.get_available_sha256()  #
# interface.test = True
sha256_holdout = interface.get_available_sha256(test=True)  #

MAXTURNS = 20

register(
    id='malware-v0',
    entry_point='env.malware:MalwareEnv',
    kwargs={'random_sample': False, 'maxturns': MAXTURNS, 'sha256list': sha256_train, 'test': False}
)

register(
    id='malware-test-v0',
    entry_point='env.malware:MalwareEnv',
    kwargs={'random_sample': False, 'maxturns': MAXTURNS, 'sha256list': sha256_holdout, 'test': True}
)
