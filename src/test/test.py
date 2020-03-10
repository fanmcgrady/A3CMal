import lief

pe = '../../Dataset/pe/backdoor/Backdoor.Win32.Rbot.pzu'
binary = lief.PE.parse(pe)
print(binary)