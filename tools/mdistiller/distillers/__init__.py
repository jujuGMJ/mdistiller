from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .MLLD import MLLD
from .TKD import TKD
from .KD_DATS import KD_DATS
from .DKD_DATS import DKD_DATS
from .TKD_DATS import TKD_DATS
from .UKD import UKD
from .DUKD import DUKD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "MLLD": MLLD,
    "TKD": TKD,
    "KD_DATS": KD_DATS,
    "DKD_DATS": DKD_DATS,
    "TKD_DATS": TKD_DATS,
    "UKD": UKD,
    "DUKD": DUKD,
}
