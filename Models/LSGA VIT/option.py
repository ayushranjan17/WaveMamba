import sys
import argparse

# Ensure sys.argv has at least one element
if not sys.argv:
    sys.argv.append("")

# Filter out unrecognized arguments in Jupyter/IPython environment
if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = [arg for arg in sys.argv if not arg.startswith(('-f', '/'))]

parser = argparse.ArgumentParser()
parser.add_argument('--datasetname', default="IP", help='IP,KSC,PU,SA,Houston')
parser.add_argument('--numtrain', type=float, default=0.1, help='the number of train sets')
parser.add_argument('--batchSize', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=70, help='number of epochs to train for')
parser.add_argument("--spectrumnum", type=int, default=36, help="number of spectral after PCA")
parser.add_argument('--inputsize', type=int, default=9, help='size of input')
parser.add_argument('--windowsize', type=int, default=3, help='size of windows')
parser.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode (random sampling or disjoint or fixed, default: random)",
    default="random",
)
parser.add_argument("--input3D", action="store_true", default=False)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--D_lr', type=float, default=0.0005, help='learning rate, default=0.001')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, default=531, help='manual seed')
parser.add_argument("--random_seed", type=int, default=5, help="random seed")  # Random seed
opt = parser.parse_args()
