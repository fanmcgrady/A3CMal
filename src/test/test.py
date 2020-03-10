import lief
import sys
sys.path.append('../rl/')
from action.action import MalwareManipulator

pe = '../../Dataset/pe/backdoor/Backdoor.Win32.Rbot.pzu'
binary = lief.PE.parse(pe)

def test_overlay_append():
    with open(pe, 'rb') as f:
        bytez = f.read()
    bytez2 = MalwareManipulator(pe).ARBE()
    if len(bytez) == len(bytez2):
        return 0
    else:
        return 1
