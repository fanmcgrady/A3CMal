import lief
import sys
sys.path.append('../rl')
sys.path.append('../kaggle_Microsoft_malware_full/')
# add novel
sys.path.append('../novel_feature')
from tools.interface import *
from action.action import *
interface = Interface()

bytez = interface.fetch_file('Backdoor.Win32.Bifrose.vxg')
print(test_overlay_append(bytez))
print(test_imports_append(bytez))
print(test_section_add(bytez))
print(test_section_rename(bytez))
print(test_section_append(bytez))
print(test_create_new_entry(bytez))