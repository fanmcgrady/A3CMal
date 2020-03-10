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
    'ARBE': 'ARBE',
    'ARI': 'ARI',
    'ARS_BSS': 'ARS_BSS',
    'ARS_UNKNOWN': 'ARS_UNKNOWN',
    'ARS_EXPORT': 'ARS_EXPORT',
    'ARS_IDATA': 'ARS_IDATA',
    'ARS_RELOCATION': 'ARS_RELOCATION',
    'ARS_RESOURCE': 'ARS_RESOURCE',
    'ARS_TEXT': 'ARS_TEXT',
    'ARS_TLS': 'ARS_TLS',
    'RS': 'RS',
    'ARS': 'ARS'
}

# action 操作类
class MalwareManipulator(object):
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'rb') as infile:
            self.bytez = infile.read()

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

    # append bytes to the overlay (end of PE file)
    def ARBE(self, seed=None):  # random加的？？？
        random.seed(seed)
        L = self.__random_length()
        # choose the upper bound for a uniform distribution in [0,upper]
        upper = random.randrange(256)
        # upper chooses the upper bound on uniform distribution:
        # upper=0 would append with all 0s
        # upper=126 would append with "printable ascii"
        # upper=255 would append with any character
        return self.bytez + bytes([random.randint(0, upper) for _ in range(L)])

    def has_random_lib(self, imports, lowerlibname):
        for im in imports:
            if im.name.lower() == lowerlibname:
                return True

        return False

    # add a function to the import address table that is never used
    def imports_append(self, seed=None):
        # add (unused) imports
        random.seed(seed)
        binary = lief.PE.parse(self.bytez)

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

    def generate_random_import_libname(self, minlength=5, maxlength=7):
        length = random.randint(minlength, maxlength)
        suffix = random.choice(['.dll', '.exe'])
        return "".join(chr(random.randrange(ord('.'), ord('z'))) for _ in range(length)) + suffix

    def generate_random_name(self, minlength=5, maxlength=7):
        length = random.randint(minlength, maxlength)
        return "".join(chr(random.randrange(ord('.'), ord('z'))) for _ in range(length))

    # add a function to the import address table that is random name
    def ARI2(self, seed=None):
        # add (unused) imports
        random.seed(seed)
        binary = lief.PE.parse(self.bytez)
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
        binary = lief.PE.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.generate_random_name())

        # fill with random content
        upper = random.randrange(256)  # section含content、虚拟地址、type
        L = self.__random_length()
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

    # create a new(unused) sections
    def ARS_BSS(self, seed=None):
        random.seed(seed)
        binary = lief.PE.parse(self.bytez)
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
        binary = lief.PE.parse(self.bytez)
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
        binary = lief.PE.parse(self.bytez)
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
        binary = lief.PE.parse(self.bytez)
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
        binary = lief.PE.parse(self.bytez)
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
        binary = lief.PE.parse(self.bytez)
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
        binary = lief.PE.parse(self.bytez)
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
        binary = lief.PE.parse(self.bytez)
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

    # add a function to the import address table that is never used
    def ARI(self, seed=None):
        # add (unused) imports
        random.seed(seed)
        binary = lief.PE.parse(self.bytez)
        # draw a library at random
        libname = random.choice(list(COMMON_IMPORTS.keys()))  # 随机选择？
        funcname = random.choice(list(COMMON_IMPORTS[libname]))  # 随机选择？
        lowerlibname = libname.lower()
        # find this lib in the imports, if it exists
        lib = None
        for im in binary.imports:
            if im.name.lower() == lowerlibname:
                lib = im
                break
        if lib is None:
            # add a new library
            lib = binary.add_library(libname)
        # get current names
        names = set([e.name for e in lib.entries])  # 一个lib + lib里的entry
        if not funcname in names:
            lib.add_entry(funcname)

        self.bytez = self.__binary_to_bytez(binary, imports=True)

        return self.bytez

    # manipulate existing section names
    def section_rename(self, seed=None):
        # rename a random section
        random.seed(seed)
        binary = lief.PE.parse(self.bytez)
        targeted_section = random.choice(binary.sections)
        targeted_section.name = random.choice(COMMON_SECTION_NAMES)[:7]  # current version of lief not allowing 8 chars?

        self.bytez = self.__binary_to_bytez(binary)

        return self.bytez

    # append bytes to extra space at the end of sections
    def section_append(self, seed=None):
        # append to a section (changes size and entropy)
        random.seed(seed)
        binary = lief.PE.parse(self.bytez)
        targeted_section = random.choice(binary.sections)
        L = self.__random_length()
        available_size = targeted_section.size - len(targeted_section.content)
        print("available_size:{}".format(available_size))
        if L > available_size:
            L = available_size

        upper = random.randrange(256)
        targeted_section.content = targeted_section.content + \
                                   [random.randint(0, upper) for _ in range(L)]

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    # create a new entry point which immediately jumps to the original entry point
    def create_new_entry(self, seed=None):
        # create a new section with jump to old entry point, and change entry point
        # DRAFT: this may have a few technical issues with it (not accounting for relocations),
        # but is a proof of concept for functionality
        random.seed(seed)

        binary = lief.PE.parse(self.bytez)

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

    def upx_pack(self, seed=None):
        # tested with UPX 3.91
        random.seed(seed)
        tmpfilename = os.path.join(
            tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))

        # dump bytez to a temporary file
        with open(tmpfilename, 'wb') as outfile:
            outfile.write(self.bytez)

        options = ['--force', '--overlay=copy']
        compression_level = random.randint(1, 9)
        options += ['-{}'.format(compression_level)]
        # --exact
        # compression levels -1 to -9
        # --overlay=copy [default]

        # optional things:
        # --compress-exports=0/1
        # --compress-icons=0/1/2/3
        # --compress-resources=0/1
        # --strip-relocs=0/1
        options += ['--compress-exports={}'.format(random.randint(0, 1))]
        options += ['--compress-icons={}'.format(random.randint(0, 3))]
        options += ['--compress-resources={}'.format(random.randint(0, 1))]
        options += ['--strip-relocs={}'.format(random.randint(0, 1))]

        with open(os.devnull, 'w') as DEVNULL:
            retcode = subprocess.call(
                ['upx'] + options + [tmpfilename, '-o', tmpfilename + '_packed'], stdout=DEVNULL, stderr=DEVNULL)

        os.unlink(tmpfilename)

        if retcode == 0:  # successfully packed

            with open(tmpfilename + '_packed', 'rb') as infile:
                self.bytez = infile.read()

            os.unlink(tmpfilename + '_packed')

        return self.bytez

    def upx_unpack(self, seed=None):
        # dump bytez to a temporary file
        tmpfilename = os.path.join(
            tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))

        with open(tmpfilename, 'wb') as outfile:
            outfile.write(self.bytez)

        with open(os.devnull, 'w') as DEVNULL:
            retcode = subprocess.call(
                ['upx', tmpfilename, '-d', '-o', tmpfilename + '_unpacked'], stdout=DEVNULL, stderr=DEVNULL)

        os.unlink(tmpfilename)

        if retcode == 0:  # sucessfully unpacked
            with open(tmpfilename + '_unpacked', 'rb') as result:
                self.bytez = result.read()

            os.unlink(tmpfilename + '_unpacked')

        return self.bytez

    # manipulate (break) signature
    def RS(self, seed=None):  # signature 是 certificate table中的数据
        random.seed(seed)
        binary = lief.PE.parse(self.bytez)

        if binary.has_signature:
            for i, e in enumerate(binary.data_directories):
                if e.type == lief.PE.DATA_DIRECTORY.CERTIFICATE_TABLE:
                    # remove signature from certificate table
                    e.rva = 0
                    e.size = 0
                    return self.__binary_to_bytez(binary)
        # if no signature found, self.bytez is unmodified
        return self.bytez

    # manipulate debug info
    def remove_debug(self, seed=None):
        random.seed(seed)
        binary = lief.PE.parse(self.bytez)

        if binary.has_debug:
            for i, e in enumerate(binary.data_directories):
                if e.type == lief.PE.DATA_DIRECTORY.DEBUG:
                    # remove signature from certificate table
                    e.rva = 0
                    e.size = 0
                    return self.__binary_to_bytez(binary)
        # if no signature found, self.bytez is unmodified
        return self.bytez

    # modify (break) header checksum
    def break_optional_header_checksum(self, seed=None):
        binary = lief.PE.parse(self.bytez)
        binary.optional_header.checksum = 0
        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

        # def exports_append(self,seed=None):
        # TODO: when LIEF has a way to create this
        #     random.seed(seed)
        #     binary = lief.PE.parse( self.bytez )

        #     if not binary.has_exports:
        #         return self.bytez
        #         # TO DO: add a lief.PE.DATA_DIRECTORY.EXPORT_TABLE to the data directory

        #     # find the data directory
        #     for i,e in enumerate(binary.data_directories):
        #         if e.type == lief.PE.DATA_DIRECTORY.EXPORT_TABLE:
        #             break

        # def exports_reorder(self,seed=None):
        #   # reorder exports
        #   pass


##############################
def identity(bytez, seed=None):
    return bytez


def modify_without_breaking(bytez, actions=None, seed=None):
    if actions is None:
        actions = []
    for action in actions:

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


def test_overlay_append(bytez):
    binary = lief.PE.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.overlay_append()
    binary2 = lief.PE.parse(bytez2)
    if len(binary.overlay) == len(binary2.overlay):
        return 0
    else:
        return 1
        # assert len(binary.overlay) != len(binary2.overlay), "modification failed"


def test_imports_append(bytez):
    binary = lief.PE.parse(bytez)
    # SUCCEEDS, but note that lief builder also adds a new ".l1" section for each patch of the imports
    manip = MalwareManipulator(bytez)
    bytez2 = manip.imports_append2(bytez)
    # bytez2 = manip.imports_append_org(bytez)
    binary2 = lief.PE.parse(bytez2)
    # set1 = set(binary.imported_functions)
    # set2 = set(binary2.imported_functions)
    # diff = set2.difference(set1)
    # print(list(diff))
    if len(binary.imported_functions) == len(binary2.imported_functions):
        return 0
    else:
        return 1
        # assert len(binary.imported_functions) != len(binary2.imported_functions), "no new imported functions"


def test_section_rename(bytez):
    binary = lief.PE.parse(bytez)
    # SUCCEEDS
    manip = MalwareManipulator(bytez)
    bytez2 = manip.section_rename(bytez)
    binary2 = lief.PE.parse(bytez2)
    oldsections = [s.name for s in binary.sections]
    newsections = [s.name for s in binary2.sections]
    # print(oldsections)
    # print(newsections)
    if " ".join(newsections) == " ".join(oldsections):
        return 0
    else:
        return 1
        # assert " ".join(newsections) != " ".join(oldsections), "no modified sections"


def test_section_add(bytez):
    binary = lief.PE.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.section_add3(bytez)
    # bytez2 = manip.section_add_org(bytez)
    binary2 = lief.PE.parse(bytez2)
    oldsections = [s.name for s in binary.sections]
    newsections = [s.name for s in binary2.sections]
    # print(oldsections)
    # print(newsections)
    if len(newsections) == len(oldsections):
        return 0
    else:
        return 1
        # assert len(newsections) != len(oldsections), "no new sections"


def test_section_append(bytez):
    binary = lief.PE.parse(bytez)
    # FAILS if there's insufficient room to add to the section 
    manip = MalwareManipulator(bytez)
    bytez2 = manip.section_append(bytez)
    binary2 = lief.PE.parse(bytez2)
    oldsections = [len(s.content) for s in binary.sections]
    newsections = [len(s.content) for s in binary2.sections]
    print(oldsections)
    print(newsections)
    if sum(newsections) == sum(oldsections):
        return 0
    else:
        return 1
        # assert sum(newsections) != sum(oldsections), "no appended section"


def test_create_new_entry(bytez):
    binary = lief.PE.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.create_new_entry(bytez)
    binary2 = lief.PE.parse(bytez2)
    # print(binary.entrypoint)
    # print(binary2.entrypoint)
    if binary.entrypoint == binary2.entrypoint:
        return 0
    else:
        return 1
        # assert binary.entrypoint != binary2.entrypoint, "no new entry point"


def test_remove_signature(bytez):
    binary = lief.PE.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.remove_signature(bytez)
    binary2 = lief.PE.parse(bytez2)
    if binary.has_signature and binary2.has_signature:
        return 0
    else:
        return 1
        # assert binary2.has_signature == False, "failed to remove signature"


def test_remove_debug(bytez):
    binary = lief.PE.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.remove_debug(bytez)
    binary2 = lief.PE.parse(bytez2)
    if binary.has_debug and binary2.has_debug:
        return 0
    else:
        return 1
        # assert binary2.has_debug == False, "failed to remove debug"


def test_break_optional_header_checksum(bytez):
    binary = lief.PE.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.break_optional_header_checksum(bytez)
    binary2 = lief.PE.parse(bytez2)
    if binary2.optional_header.checksum != 0:
        return 0
    else:
        return 1
        # assert binary2.optional_header.checksum == 0, "checksum not zero :("
