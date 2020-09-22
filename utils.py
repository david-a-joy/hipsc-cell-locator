# Imports
import re
import pathlib

# Constants

reTILE = re.compile(r'^s(?P<tile>[0-9]+)(-(?P<condition>[a-z0-9\.\-]+))?$', re.IGNORECASE)

reFILE4 = re.compile(r'''^
    (?P<prefix>[a-z0-9-\. ]+)_
    s(?P<tile>[0-9]+)
    t(?P<timepoint>[0-9]+)_
    (?P<channel_name>[a-z][a-z0-9 _+-]+)_ORG
    (?P<suffix>((InverseWarp)|(Warp)|(Affine))?\.[a-z]+(\.[a-z]+)*)
$''', re.IGNORECASE | re.VERBOSE)
reFILE3 = re.compile(r'''^
    (?P<prefix>[a-z0-9-\. ]+)_
    s(?P<tile>[0-9]+)
    t(?P<timepoint>[0-9]+)
    (?P<channel>[a-z][a-z0-9 _+-]+?)?(_ORG)?
    (?P<suffix>((InverseWarp)|(Warp)|(Affine))?\.[a-z]+(\.[a-z]+)*)
$''', re.IGNORECASE | re.VERBOSE)
reFILE2 = re.compile(r'''^
    (?P<prefix>[a-z0-9-\. ]+)-
    (?P<timepoint>[0-9]+)_
    s(?P<tile>[0-9]+)
    (c(?P<channel>[0-9]+))?(_ORG)?
    (?P<suffix>\.tif)
$''', re.IGNORECASE | re.VERBOSE)
reFILE1 = re.compile(r'''^
    (?P<channel_name>[a-z0-9_ ]+)-
    s(?P<tile>[0-9]+)
    t(?P<timepoint>[0-9]+)
    (?P<suffix>((InverseWarp)|(Warp)|(Affine))?\.[a-z]+(\.[a-z]+)*)
$''', re.IGNORECASE | re.VERBOSE)

CHANNEL_ALIASES = {
    1: 'TL Brightfield',
    2: 'mCherry',
    3: 'EGFP',
}
INV_CHANNEL_ALIASES = {v.lower(): k for k, v in CHANNEL_ALIASES.items()}

CHANNEL_NAME_ALIASES = {
    'GFP': ['EGFP', 'Alexa Fluor 488'],
    'MKATE': ['Alexa Fluor 568'],
}

# Functions



def parse_tile_name(name: str):
    """ Parse the tile """
    match = reTILE.match(name)
    if match is None:
        return None
    groups = match.groupdict()  # type: Dict[str, Any]
    groups['tile'] = int(groups['tile'])
    return groups


def parse_image_name(name: str):
    """ Try several image name parser formatters

    The returned dictionary has the following keys:

    * "tile" - int - The image tile number
    * "timepoint" - int - The image timepoint
    * "channel" - int - The channel number
    * "channel_name" - str - The channel name

    :param str name:
        The image file name
    :returns:
        The image file metadata
    """
    regs = [reFILE3, reFILE4, reFILE2, reFILE1]

    matchdict = None  # type: Optional[Dict[str, Any]]
    for reg in regs:
        match = reg.match(name)
        if match is None:
            continue
        matchdict = match.groupdict()
        break
    if matchdict is None:
        return None

    matchdict['tile'] = int(matchdict['tile'])
    matchdict['timepoint'] = int(matchdict['timepoint'])

    channel = matchdict.get('channel')
    if channel is None:
        channel_name = matchdict.get('channel_name')
        if channel_name is None:
            channel = 1
            channel_name = str(CHANNEL_ALIASES.get(channel, channel))
        else:
            channel = INV_CHANNEL_ALIASES.get(channel_name.lower(), 1)
    elif '_' in channel:
        channel_name, channel = channel.rsplit('_', 1)
        channel = int(channel)
    else:
        channel = int(channel)
        channel_name = str(CHANNEL_ALIASES.get(channel, channel))
    matchdict['channel'] = channel
    matchdict['channel_name'] = channel_name
    matchdict['key'] = '{:d}-{}'.format(matchdict['tile'], channel_name.lower().replace(' ', '_'))
    return matchdict


def guess_channel_dir(rootdir, channel_type):
    """ Guess the channel dir

    :param Path rootdir:
        The root image dir
    :param str channel_type:
        A key in the CHANNEL_NAME_ALIASES dict
    """
    channel_type = channel_type.upper()
    aliases = [channel_type]
    aliases += [a.upper() for a in CHANNEL_NAME_ALIASES.get(channel_type, [])]

    if not rootdir.is_dir():
        raise OSError('Input directory not found: {}'.format(rootdir))

    targets = []

    for subdir in rootdir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name.upper() in aliases:
            targets.append((subdir.name, subdir))

    if len(targets) == 0:
        raise OSError('No targets for {} under {}'.format(channel_type, rootdir))
    if len(targets) > 1:
        raise OSError('Multiple possible aliases for {} under {}'.format(channel_type, rootdir))
    return targets[0]
