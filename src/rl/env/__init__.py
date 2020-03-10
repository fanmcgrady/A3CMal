from gym.envs.registration import register

MAXTURNS = 60

register(
    id='malware-v0',
    entry_point='gym_malware.envs:MalwareEnv',
    kwargs={'random_sample': False, 'maxturns': MAXTURNS, 'sha256list': sha256_train, 'test': False}
)

register(
    id='malware-test-v0',
    entry_point='gym_malware.envs:MalwareEnv',
    kwargs={'random_sample': False, 'maxturns': MAXTURNS, 'sha256list': sha256_holdout, 'test': True}
)