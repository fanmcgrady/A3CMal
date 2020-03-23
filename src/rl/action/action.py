# encoding=utf-8
# TODO:
# * modify exports using lief
# * zero out rich header (if it exists) -->
# requires updating OptionalHeader's checksum ("Rich Header" only in Microsoft-produced executables)
# * tinker with resources: https://lief.quarkslab.com/doc/tutorials/07_pe_resource.html
# also in our project dir. : /test/lief-tutorials/PE_resource

import array
import functools
import json
import multiprocessing
import os
import random
import signal
import struct  # byte manipulations
import subprocess
import sys
import tempfile

import lief  # pip install https://github.com/lief-project/LIEF/releases/download/0.7.0/linux_lief-0.7.0_py3.6.tar.gz

module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]

COMMON_SECTION_NAMES = open(os.path.join(
    module_path, 'section_names.txt'), 'r').read().rstrip().split('\n')
COMMON_IMPORTS = json.load(
    open(os.path.join(module_path, 'small_dll_imports.json'), 'r'))

######################
# explicitly list so that these may be used externally
ACTION_TABLE = {
    'ARBE': 'ARBE', # 末尾加随机字符串
    'imports_append': 'imports_append', # 加一个指定范围内随机加lib、function，10次之内直到加入为止
    # 'ARI': 'ARI',   # 加一个随机的lib，随机的function
    'ARS': 'ARS',   # 加节
    'section_rename': 'section_rename',   # 加节
    'section_append': 'section_append',   # 加节
    'create_new_entry': 'create_new_entry'   # 加节
    # 'ARS_BSS': 'ARS_BSS', # 加BSS节
    # 'ARS_UNKNOWN': 'ARS_UNKNOWN', # 加UNKNOWN节
    # 'ARS_EXPORT': 'ARS_EXPORT', # 加EXPORT节
    # 'ARS_IDATA': 'ARS_IDATA', # 加IDATA节
    # 'ARS_RELOCATION': 'ARS_RELOCATION', # 加RELOCATION节
    # 'ARS_RESOURCE': 'ARS_RESOURCE', # 加RESOURCE节
    # 'ARS_TEXT': 'ARS_TEXT', # 加TEXT'节
    # 'ARS_TLS': 'ARS_TLS', # 加TLS节
}

# action 操作类
class MalwareManipulator(object):
    def __init__(self, bytez):
        self.bytez = bytez

        self.min_append_log2 = 5
        self.max_append_log2 = 8

    # 随机生成数据
    def __random_length(self):
        return 2 ** random.randint(self.min_append_log2, self.max_append_log2)

    # 把lief结果build成bytez
    def __binary_to_bytez(self, binary, dos_stub=False, imports=False, overlay=False, relocations=False,
                          resources=False, tls=False):
        # write the file back as bytez
        builder = lief.PE.Builder(binary)
        builder.build_dos_stub(dos_stub)  # rebuild DOS stub

        builder.build_imports(imports)  # rebuild IAT in another section
        builder.patch_imports(imports)  # patch original import table with trampolines to new import table

        builder.build_overlay(overlay)  # rebuild overlay
        builder.build_relocations(relocations)  # rebuild relocation table in another section
        builder.build_resources(resources)  # rebuild resources in another section
        builder.build_tls(tls)  # rebuilt TLS object in another section

        builder.build()  # perform the build process

        # return bytestring
        return array.array('B', builder.get_build()).tobytes()

    def generate_random_import_libname(self, minlength=5, maxlength=7):
        length = random.randint(minlength, maxlength)
        suffix = random.choice(['.dll', '.exe'])
        return "".join(chr(random.randrange(ord('.'), ord('z'))) for _ in range(length)) + suffix

    def generate_random_name(self, minlength=5, maxlength=7):
        length = random.randint(minlength, maxlength)
        return "".join(chr(random.randrange(ord('.'), ord('z'))) for _ in range(length))

    def has_random_lib(self, imports, lowerlibname):
        for im in imports:
            if im.name.lower() == lowerlibname:
                return True

        return False

    # ==============================================
    # append bytes to the overlay (end of PE file)
    def ARBE(self, seed=None):  # random加的？？？
        random.seed(seed)
        L = self.__random_length() * (2 ** 3)
        # choose the upper bound for a uniform distribution in [0,upper]
        upper = random.randrange(256)
        # upper chooses the upper bound on uniform distribution:
        # upper=0 would append with all 0s
        # upper=126 would append with "printable ascii"
        # upper=255 would append with any character
        return self.bytez + bytes([random.randint(0, upper) for _ in range(L)])

    # add a function to the import address table that is never used
    def imports_append(self, seed=None):
        # add (unused) imports
        random.seed(seed)
        binary = lief.parse(self.bytez)

        importslist = binary.imports
        # draw a library at random
        libname = random.choice(list(COMMON_IMPORTS.keys()))
        funcname = random.choice(list(COMMON_IMPORTS[libname]))
        lowerlibname = libname.lower()

        count_limit = 0

        while self.has_random_lib(importslist, lowerlibname):
            # draw a library at random
            libname = random.choice(list(COMMON_IMPORTS.keys()))
            funcname = random.choice(list(COMMON_IMPORTS[libname]))
            lowerlibname = libname.lower()
            count_limit += 1
            if count_limit > 10:
                break

        # add a new library
        lib = binary.add_library(libname)

        # get current names
        names = set([e.name for e in lib.entries])
        if not funcname in names:
            lib.add_entry(funcname)

        self.bytez = self.__binary_to_bytez(binary, imports=True)

        return self.bytez

        # 生成随机的import name

    # add a function to the import address table that is random name
    # 加一个随机的lib，随机的function
    def ARI(self, seed=None):
        # add (unused) imports
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # draw a library at random
        libname = self.generate_random_import_libname()
        funcname = self.generate_random_name()
        lowerlibname = libname.lower()
        # append this lib in the imports
        lib = binary.add_library(lowerlibname)
        lib.add_entry(funcname)

        self.bytez = self.__binary_to_bytez(binary, imports=True)

        return self.bytez

    # create a new(unused) sections
    def ARS(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)  # section含content、虚拟地址、type
        L = self.__random_length() * (2 ** 3)
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section,
                           random.choice([
                               lief.PE.SECTION_TYPES.BSS,
                               lief.PE.SECTION_TYPES.DATA,
                               lief.PE.SECTION_TYPES.EXPORT,
                               lief.PE.SECTION_TYPES.IDATA,
                               lief.PE.SECTION_TYPES.RELOCATION,
                               lief.PE.SECTION_TYPES.RESOURCE,
                               lief.PE.SECTION_TYPES.TEXT,
                               lief.PE.SECTION_TYPES.TLS_,
                               lief.PE.SECTION_TYPES.UNKNOWN,
                           ]))

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    # manipulate existing section names
    def section_rename(self, seed=None):
        # rename a random section
        random.seed(seed)
        binary = lief.parse(self.bytez)

        # 所有section全部改名
        for targeted_section in binary.sections:
            targeted_section.name = random.choice(COMMON_SECTION_NAMES)[
                                    :7]  # current version of lief not allowing 8 chars?

        # 随机改一次名字
        # targeted_section = random.choice(binary.sections)
        # targeted_section.name = random.choice(COMMON_SECTION_NAMES)[:7]  # current version of lief not allowing 8 chars?

        self.bytez = self.__binary_to_bytez(binary)

        return self.bytez

    # append bytes to extra space at the end of sections
    def section_append(self, seed=None):
        # append to a section (changes size and entropy)
        random.seed(seed)
        binary = lief.parse(self.bytez)
        for targeted_section in binary.sections:
            L = self.__random_length()
            available_size = targeted_section.size - len(targeted_section.content)
            # print("available_size:{}".format(available_size))
            if available_size == 0:
                continue

            if L > available_size:
                L = available_size

            upper = random.randrange(256)
            targeted_section.content = targeted_section.content + \
                                   [random.randint(0, upper) for _ in range(L)]
            break

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    # create a new entry point which immediately jumps to the original entry point
    def create_new_entry(self, seed=None):
        # create a new section with jump to old entry point, and change entry point
        # DRAFT: this may have a few technical issues with it (not accounting for relocations),
        # but is a proof of concept for functionality
        random.seed(seed)

        binary = lief.parse(self.bytez)

        # get entry point
        entry_point = binary.optional_header.addressof_entrypoint

        # get name of section
        entryname = binary.section_from_rva(entry_point).name

        # create a new section
        new_section = lief.PE.Section(entryname + "".join(chr(random.randrange(
            ord('.'), ord('z'))) for _ in range(3)))  # e.g., ".text" + 3 random characters
        # push [old_entry_point]; ret
        new_section.content = [
                                  0x68] + list(struct.pack("<I", entry_point + 0x10000)) + [0xc3]
        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])
        # TO DO: account for base relocation (this is just a proof of concepts)

        # add new section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.TEXT)

        # redirect entry point
        binary.optional_header.addressof_entrypoint = new_section.virtual_address

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    # create a new(unused) sections
    def ARS_BSS(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.BSS)

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    def ARS_UNKNOWN(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.UNKNOWN)

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    def ARS_EXPORT(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.EXPORT)

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    def ARS_IDATA(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.IDATA)

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    def ARS_RELOCATION(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.RELOCATION)

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    def ARS_RESOURCE(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.RESOURCE)

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    def ARS_TEXT(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.TEXT)

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    def ARS_TLS(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.TLS_)

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

##############################
def identity(bytez, seed=None):
    return bytez

# def modify_without_breaking(bytez, action):
#     _action = ACTION_TABLE[action]
#
#     _action = MalwareManipulator(bytez).__getattribute__(_action)
#
#     # redirect standard out only in this queue
#     try:
#         bytez = _action()
#     except Exception as e:
#         # some exceptions that have yet to be handled by public release of LIEF
#         print("==== exception in process ===")
#         print(e)
#         print("return unmodified bytez to make sure training process continue!")
#         return bytez
#
#     import hashlib
#     m = hashlib.sha256()
#     m.update(bytez)
#     return bytez

def modify_without_breaking(bytez, action=None, seed=None):
    _action = ACTION_TABLE[action]

    # we run manipulation in a child process to shelter
    # our malware model from rare parsing errors in LIEF that
    # may segfault or timeout
    def helper(_action, shared_list):
        # TODO: LIEF is chatty. redirect stdout and stderr to /dev/null

        # for this process, change segfault of the child process
        # to a RuntimeEror
        def sig_handler(signum, frame):
            raise RuntimeError

        signal.signal(signal.SIGSEGV, sig_handler)

        bytez = array.array('B', shared_list[:]).tobytes()
        # TODO: LIEF is chatty. redirect output to /dev/null
        if type(_action) is str:
            _action = MalwareManipulator(bytez).__getattribute__(_action)
        else:
            _action = functools.partial(_action, bytez)

        # redirect standard out only in this queue
        try:
            shared_list[:] = _action(seed)
        except (RuntimeError, UnicodeDecodeError, TypeError, lief.not_found) as e:
            # some exceptions that have yet to be handled by public release of LIEF
            print("==== exception in child process ===")
            print(e)
            # shared_bytez remains unchanged

    # communicate with the subprocess through a shared list
    # can't use multiprocessing.Array since the subprocess may need to
    # change the size
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    shared_list[:] = bytez  # copy bytez to shared array
    # define process
    p = multiprocessing.Process(target=helper, args=(_action, shared_list))
    p.start()  # start the process
    try:
        p.join(5)  # allow this to take up to 5 seconds...
    except multiprocessing.TimeoutError:  # ..then become petulant
        print('==== timeouterror ')
        p.terminate()

    bytez = array.array('B', shared_list[:]).tobytes()  # copy result from child process

    import hashlib
    m = hashlib.sha256()
    m.update(bytez)
    return bytez

# test ARBE
def test_overlay_append(bytez):
    binary = lief.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.ARBE()
    binary2 = lief.parse(bytez2)
    if len(binary.overlay) == len(binary2.overlay):
        return 0
    else:
        return 1
        # assert len(binary.overlay) != len(binary2.overlay), "modification failed"

# test imports_append
def test_imports_append(bytez):
    binary = lief.parse(bytez)
    # SUCCEEDS, but note that lief builder also adds a new ".l1" section for each patch of the imports
    manip = MalwareManipulator(bytez)
    bytez2 = manip.imports_append(bytez)
    # bytez2 = manip.imports_append_org(bytez)
    binary2 = lief.parse(bytez2)
    # set1 = set(binary.imported_functions)
    # set2 = set(binary2.imported_functions)
    # diff = set2.difference(set1)
    # print(list(diff))
    if len(binary.imported_functions) == len(binary2.imported_functions):
        return 0
    else:
        return 1
        # assert len(binary.imported_functions) != len(binary2.imported_functions), "no new imported functions"

# test ARS
def test_section_add(bytez):
    binary = lief.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.ARS(bytez)
    # bytez2 = manip.section_add_org(bytez)
    binary2 = lief.parse(bytez2)
    oldsections = [s.name for s in binary.sections]
    newsections = [s.name for s in binary2.sections]
    # print(oldsections)
    # print(newsections)
    if len(newsections) == len(oldsections):
        return 0
    else:
        return 1
        # assert len(newsections) != len(oldsections), "no new sections"

# test section_rename
def test_section_rename(bytez):
    binary = lief.parse(bytez)
    # SUCCEEDS
    manip = MalwareManipulator(bytez)
    bytez2 = manip.section_rename(bytez)
    binary2 = lief.parse(bytez2)
    oldsections = [s.name for s in binary.sections]
    newsections = [s.name for s in binary2.sections]
    # print(oldsections)
    # print(newsections)
    if " ".join(newsections) == " ".join(oldsections):
        return 0
    else:
        return 1
        # assert " ".join(newsections) != " ".join(oldsections), "no modified sections"

# test section_append
def test_section_append(bytez):
    binary = lief.parse(bytez)
    # FAILS if there's insufficient room to add to the section 
    manip = MalwareManipulator(bytez)
    bytez2 = manip.section_append(bytez)
    binary2 = lief.parse(bytez2)

    # oldsections = [len(s.content) for s in binary.sections]
    # newsections = [len(s.content) for s in binary2.sections]
    # print(oldsections)
    # print(newsections)
    if binary == binary2:
        return 0
    else:
        return 1
        # assert sum(newsections) != sum(oldsections), "no appended section"

# test create_new_entry
def test_create_new_entry(bytez):
    binary = lief.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.create_new_entry(bytez)
    binary2 = lief.parse(bytez2)
    # print(binary.entrypoint)
    # print(binary2.entrypoint)
    if binary.entrypoint == binary2.entrypoint:
        return 0
    else:
        return 1
        # assert binary.entrypoint != binary2.entrypoint, "no new entry point"