#!/usr/bin/env python

""" Create an installer environment """

# Standard lib
import os
import math
import shutil
import random
import pathlib
import zipfile
import platform
import textwrap
import argparse
import subprocess
from collections import namedtuple

# 3rd party
import numpy as np

from PIL import Image

from vendor import patch

# Our own imports
from utils import parse_image_name, parse_tile_name, guess_channel_dir

# Constants

UNAME = platform.system().lower()

DATADIR = pathlib.Path('/data/Experiment')

MAX_TIMEPOINT = 12  # Maximum timepoint to pull in
MAX_TILES = 10  # Total number of tiles to pull in
SEQUENCE_LEN = 2  # How many timepoints in a row per tile

TILE_MAP = {
    '2017-01-30': {
        1: 100,
        2: 100,
        3: 100,
        4: 100,
        5: 100,
        6: 100,
        7: 100,
        8: 100,
        9: 100,
        10: 100,
        11: 100,
        12: 100,
        13: 100,
        14: 100,
        15: 100,
        16: 100,
    },
    '2017-03-03': {
        2: 30,
        3: 100,
        4: 30,
        7: 10,
        8: 10,
        9: 30,
        10: 10,
        12: 10,
        13: 100,
        14: 100,
        15: 30,
        16: 100,
    },
    '2017-05-03': {
        18: 100,
        19: 10,
        20: 30,
        21: 30,
        22: 100,
        23: 100,
        25: 30,
        27: 30,
        28: 10,
        30: 100,
        31: 10,
        32: 10,
    },
}


# Classes


DataRecord = namedtuple('DataRecord', 'path,rot90,flip_lr')


# Functions


def read_requirements(requirements_file):
    """ Read in the requirements file """

    if not requirements_file.is_file():
        raise OSError('Cannot find requirements file: {}'.format(requirements_file))

    requirements = []

    with requirements_file.open('rt') as fp:
        for line in fp:
            line = line.split('#', 1)[0].strip()
            if line == '':
                continue
            requirements.append(line)
    return requirements


def clean_build_files(thisdir, keep_env=False, regen_data=False):
    """ Clean the old build files from the build directory """

    print('Cleaning old build files...')

    install_env_root = thisdir / 'install_env'
    data_dir = thisdir / 'data'
    database_file = data_dir / 'RegionDB.sqlite3'
    spec_file = thisdir / 'cell_locator.spec'
    build_dir = thisdir / 'build'
    dist_dir = thisdir / 'dist'

    if install_env_root.is_dir():
        if not keep_env:
            shutil.rmtree(str(install_env_root))

    if data_dir.is_dir():
        if regen_data:
            shutil.rmtree(str(data_dir))

    if spec_file.is_file():
        spec_file.unlink()

    if database_file.is_file():
        database_file.unlink()

    if build_dir.is_dir():
        shutil.rmtree(str(build_dir))

    if dist_dir.is_dir():
        shutil.rmtree(str(dist_dir))


def create_data_file(data_file, data_source, tile_map,
                     channel='gfp',
                     drop_categories=None,
                     max_tiles=MAX_TILES,
                     max_timepoint=MAX_TIMEPOINT,
                     sequence_len=SEQUENCE_LEN):
    """ Create a data file

    :param Path data_file:
        The data file to write
    :param Path data_source:
        The base experiment directory to load
    :param dict tile_map:
        A map of experiment data to tile numbers and categories
    :param int max_tiles:
        If not None, the maximum number of tiles to select (rounded up by category)
    :param int max_timepoint:
        If not None, the maximum timepoint to use for a pair
    :param int sequence_len:
        How many sequential files to load for a given tile
    """

    if drop_categories is None:
        drop_categories = [0]

    # Collect all the tile data and group by category
    categories = {}
    for experiment in sorted(tile_map):
        category_map = tile_map[experiment]
        experiment_dir = data_source / experiment

        if not experiment_dir.is_dir():
            print('Skipping {}'.format(experiment_dir))
            continue
        channel_dir = guess_channel_dir(experiment_dir / 'Corrected', channel)[1]

        for tiledir in channel_dir.iterdir():
            if not tiledir.is_dir():
                continue
            tile_num = parse_tile_name(tiledir.name)['tile']
            if tile_num in category_map:
                categories.setdefault(category_map[tile_num], []).append(tiledir)

    # Drop the bad categories
    for drop_category in drop_categories:
        if drop_category in categories:
            del categories[drop_category]

    print('Number of tiles/category:')
    for category in sorted(categories):
        print('* {}: {}'.format(category, len(categories[category])))

    # How many tiles per category for a balanced selection
    min_tiles_per_cat = min([len(c) for c in categories.values()])
    if max_tiles is None:
        tiles_per_cat = min_tiles_per_cat
    else:
        tiles_per_cat = math.ceil(max_tiles / len(categories))
        tiles_per_cat = min([min_tiles_per_cat, tiles_per_cat])
    print('Loading {} tiles/category'.format(tiles_per_cat))

    # Select evenly from each category, with sequential timepoints, without replacement
    final_images = []
    for category in categories.values():
        target_tiles = random.choices(category, k=tiles_per_cat)
        for target_tile in target_tiles:
            target_images = [t for t in target_tile.iterdir()
                             if t.is_file() and t.suffix in ('.tif', )]
            target_images = [t for t in sorted(target_images)
                             if parse_image_name(t.name)['timepoint'] <= max_timepoint]
            target_index = random.randint(0, len(target_images) - sequence_len)
            final_images.extend(target_images[target_index:target_index+sequence_len])

    # Finally, shuffle everything one last time
    random.shuffle(final_images)

    # Write the results to a data file
    data_file.parent.mkdir(exist_ok=True, parents=True)
    with data_file.open('wt') as fp:
        fp.write('#Image,Rotation,HorizontalFlip' + os.linesep)
        for final_image in final_images:
            fp.write('{},{},{}{}'.format(final_image, random.randint(0, 3), random.randint(0, 1), os.linesep))


def read_data_file(data_file, regen_data=False):
    """ Read the data file in

    :param Path data_file:
        The tile mapping file to load
    :param bool regen_data:
        If True, regenerate the data mapping table
    :returns:
        A list of DataRecords mapping input files to targets
    """

    if not data_file.is_file() or regen_data:
        create_data_file(data_file, DATADIR, TILE_MAP)

    data_files = []

    with data_file.open('rt') as fp:
        for line in fp:
            line = line.split('#', 1)[0].strip()
            if line == '':
                continue
            rec = [r.strip() for r in line.split(',')]
            data_files.append(DataRecord(pathlib.Path(rec[0]), int(rec[1]), rec[2] == '1'))
    return data_files


def write_spec_file(thisdir, onefile=False):
    """ Write the spec file expected by pyinstaller """

    if onefile:
        template = textwrap.dedent("""
            # -*- mode: python -*-

            block_cipher = None

            a = Analysis(['cell_locator.py'],
                         pathex=["{pathex}"],
                         binaries=[],
                         datas=[],
                         hiddenimports=[],
                         hookspath=[],
                         runtime_hooks=[],
                         excludes=[],
                         win_no_prefer_redirects=False,
                         win_private_assemblies=False,
                         cipher=block_cipher)
            pyz = PYZ(a.pure, a.zipped_data,
                         cipher=block_cipher)
            exe = EXE(pyz,
                      a.scripts,
                      a.binaries,
                      a.zipfiles,
                      a.datas,
                      name='cell_locator',
                      debug=False,
                      strip=False,
                      upx=True,
                      runtime_tmpdir=None,
                      console=False )
        """).format(pathex=str(thisdir))
    else:
        template = textwrap.dedent("""
            # -*- mode: python -*-

            block_cipher = None

            a = Analysis(['cell_locator.py'],
                         pathex=["{pathex}"],
                         binaries=[],
                         datas=[("{datadir}", "data"), ("{imagedir}", "images")],
                         hiddenimports=[],
                         hookspath=[],
                         runtime_hooks=[],
                         excludes=[],
                         win_no_prefer_redirects=False,
                         win_private_assemblies=False,
                         cipher=block_cipher)
            pyz = PYZ(a.pure, a.zipped_data,
                         cipher=block_cipher)
            exe = EXE(pyz,
                      a.scripts,
                      exclude_binaries=True,
                      name='cell_locator',
                      debug=False,
                      strip=False,
                      upx=True,
                      console=False )
            coll = COLLECT(exe,
                           a.binaries,
                           a.zipfiles,
                           a.datas,
                           strip=False,
                           upx=True,
                           name='cell_locator')
        """).format(pathex=str(thisdir),
                    datadir=str(thisdir / 'data/*'),
                    imagedir=str(thisdir / 'images/*'))

    spec_file = thisdir / 'cell_locator.spec'

    with spec_file.open('wt') as fp:
        for line in template.splitlines():
            fp.write(line + os.linesep)


def copy_image_file(indata, outfile, delete_column=86):
    """ Copy the image file with flips and rotations """

    img = np.asarray(Image.open(str(indata.path)))

    # Bug in the normalization code creates a 1-pixel strip
    img = np.concatenate([img[:, :delete_column], img[:, delete_column+1:]], axis=1)

    img = np.rot90(img, indata.rot90)
    if indata.flip_lr:
        img = np.fliplr(img)
    oimg = Image.fromarray(img)
    oimg.save(str(outfile))


def patch_environment(install_env_root):
    """ Apply patches to fix brokenish code """

    patch_data = textwrap.dedent("""
        diff -u a/hook-sysconfig.py b/hook-sysconfig.py
        --- a/hook-sysconfig.py	2018-01-26 14:47:14.000000000 -0800
        +++ b/hook-sysconfig.py	2018-01-26 14:48:40.000000000 -0800
        @@ -39,4 +39,7 @@
             # https://github.com/python/cpython/blob/3.6/Lib/sysconfig.py#L417
             # Note: Some versions of Anaconda backport this feature to before 3.6.
             # See issue #3105
        -    hiddenimports = [sysconfig._get_sysconfigdata_name()]
        +    try:
        +        hiddenimports = [sysconfig._get_sysconfigdata_name(None)]
        +    except TypeError:
        +        hiddenimports = [sysconfig._get_sysconfigdata_name()]
    """)
    if UNAME == 'windows':
        input_path = install_env_root / 'Lib\\site-packages\\PyInstaller\\hooks'
    else:
        input_path = install_env_root / 'lib/python3.6/site-packages/PyInstaller/hooks'
    assert input_path.is_dir()
    patch_set = patch.fromstring(patch_data.encode('utf-8'))
    patch_set.apply(root=str(input_path))


def write_zip_archive(rootdir, archive_file):
    """ Write a zip archive of the final directory """

    archive_file.parent.mkdir(exist_ok=True, parents=True)

    print('Archiving {} to {}'.format(rootdir, archive_file))

    paths = [p for p in rootdir.iterdir()]
    with zipfile.ZipFile(str(archive_file), 'w') as fp:
        while paths != []:
            p = paths.pop(0)
            if p.name.startswith('.'):
                continue
            if p.is_dir():
                paths.extend(p.iterdir())
                continue
            relp = p.relative_to(rootdir)
            print(relp)
            fp.write(str(p), arcname=str(relp))


def figure_out_python():
    """ Work out which python to use """
    for python_cmd in ['python3', 'python']:
        print('Trying python command: {}'.format(python_cmd))
        try:
            proc = subprocess.Popen([python_cmd, '--version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        except FileNotFoundError:
            continue
        if out.decode('utf-8').startswith('Python 3.'):
            if UNAME == 'windows':
                return python_cmd + '.exe'
            return python_cmd
    raise FileNotFoundError('Cannot find a working python3')
    

# Command-Line Interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep-env', action='store_true',
                        help='Keep the build environment')
    parser.add_argument('--regen-data', action='store_true',
                        help='Regenerate the data files')
    parser.add_argument('--onefile', action='store_true')
    parser.add_argument('command', nargs='*', choices=('clean', 'build'))
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    
    python_cmd = figure_out_python()

    if args.command == []:
        command = ['clean', 'build']
    else:
        command = args.command

    thisdir = pathlib.Path.cwd()

    install_env_root = thisdir / 'install_env'
    requirements_file = thisdir / 'requirements.txt'
    data_dir = thisdir / 'data'
    data_file = thisdir / 'data.txt'

    requirements = read_requirements(requirements_file)
    data_files = read_data_file(data_file, regen_data=args.regen_data)

    # Clean all the old data files out
    if 'clean' in command:
        clean_build_files(thisdir,
                          keep_env=args.keep_env,
                          regen_data=args.regen_data)

    if 'build' in command:
        # Reinstall all prereqs
        if not install_env_root.is_dir():
            print('Installing prereqs: {}'.format(requirements))
            cmd = ['virtualenv', '-p', python_cmd, str(install_env_root)]
            subprocess.check_call(cmd)
        assert install_env_root.is_dir()

        if UNAME == 'windows':
            pip_path = install_env_root / 'Scripts' / 'pip.exe'
            python_path = install_env_root / 'Scripts' / 'python.exe'
        else:
            pip_path = install_env_root / 'bin' / 'pip'
            python_path = install_env_root / 'bin' / 'python'
        assert pip_path.is_file()
        assert python_path.is_file()

        for requirement in requirements:
            cmd = [str(pip_path), 'install', requirement]
            subprocess.check_call(cmd)

        # Apply patches to the environment
        patch_environment(install_env_root)

        # Reload all the data
        if not data_dir.is_dir():
            data_dir.mkdir()
            for i, data_file in enumerate(data_files):
                copy_image_file(data_file, data_dir / '{:03d}.tif'.format(i))

        # Write the spec file out
        write_spec_file(thisdir, onefile=args.onefile)

        # Bundle everything into a blob
        if UNAME == 'windows':
            pyinstaller_path = install_env_root / 'Scripts' / 'pyinstaller.exe'
        else:
            pyinstaller_path = install_env_root / 'bin' / 'pyinstaller'
        assert pyinstaller_path.is_file()

        if UNAME == 'windows':
            pyinstaller_cmd = [pyinstaller_path]
        else:
            pyinstaller_cmd = [python_path, pyinstaller_path]
        pyinstaller_cmd.extend(['--windowed', thisdir / 'cell_locator.spec'])

        pyinstaller_cmd = [str(c) for c in pyinstaller_cmd]
        print('Final pyinstaller cmd: "{}"'.format(' '.join(pyinstaller_cmd)))

        subprocess.check_call(pyinstaller_cmd, cwd=thisdir)

        if args.onefile:
            assert (thisdir / 'dist' / 'cell_locator').is_file()
            shutil.copytree(str(thisdir / 'data'), str(thisdir / 'dist' / 'data'))
            shutil.copytree(str(thisdir / 'images'), str(thisdir / 'dist' / 'images'))
            distdir = thisdir / 'dist'
        else:
            assert (thisdir / 'dist' / 'cell_locator').is_dir()
            distdir = thisdir / 'dist'

        # Pack up the resulting directory
        write_zip_archive(distdir, thisdir / 'dist' / 'cell_locator.zip')


if __name__ == '__main__':
    main()
