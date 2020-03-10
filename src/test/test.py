import lief

pe = '../../Dataset/pe/backdoor/Backdoor.Win32.Rbot.pzu'


def fetch_file():
    with open(pe, 'rb') as infile:
        bytez = infile.read()

    return bytez


binary = lief.PE.parse(fetch_file())
print(binary)
