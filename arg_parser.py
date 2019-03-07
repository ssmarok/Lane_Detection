import argparse

parser = argparse.ArgumentParser(description='Specify video File by number.')
parser.add_argument('-v', action="store", dest="video_num", help="Video Number", type=int, default=1)
parser.add_argument('-d', action="store", dest="video_delay", help="Video frame delay (ms)", type=int, default=10)
parser.add_argument('-dir', action="store", dest="video_dir", help="Video directory (e.g.: 'data/')", default='data/')

args = parser.parse_args()
