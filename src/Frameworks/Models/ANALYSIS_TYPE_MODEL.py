from enum import Enum
from pathlib import Path

class ANALYSIS_TYPE_MODEL(Enum):

    BIDUM = Path(__file__).parent.parent.parent.parent.joinpath("models/BIDUM.h5")
    FIJI = Path(__file__).parent.parent.parent.parent.joinpath("models/FIJI.h5")
    HONGBAN = Path(__file__).parent.parent.parent.parent.joinpath("models/HONGBAN.h5")
    MISE = Path(__file__).parent.parent.parent.parent.joinpath("models/MISE.h5")
    NONGPO = Path(__file__).parent.parent.parent.parent.joinpath("models/NONGPO.h5")
    TALMO = Path(__file__).parent.parent.parent.parent.joinpath("models/TALMO.h5")
