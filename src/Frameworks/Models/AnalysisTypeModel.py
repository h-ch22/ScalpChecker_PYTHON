from enum import Enum
from pathlib import Path


class AnalysisTypeModel(Enum):
    BIDUM = Path(__file__).parent.parent.parent.parent.joinpath("models/BIDUM.h5")
    FIJI = Path(__file__).parent.parent.parent.parent.joinpath("models/FIJI.h5")
    HONGBAN = Path(__file__).parent.parent.parent.parent.joinpath("models/HONGBAN.h5")
    MISE = Path(__file__).parent.parent.parent.parent.joinpath("models/MISE.h5")
    NONGPO = Path(__file__).parent.parent.parent.parent.joinpath("models/NONGPO.h5")
    TALMO = Path(__file__).parent.parent.parent.parent.joinpath("models/TALMO.h5")

    BIDUM_ViT = Path(__file__).parent.parent.parent.parent.joinpath("models/BIDUM_ViT.h5")
    FIJI_ViT = Path(__file__).parent.parent.parent.parent.joinpath("models/FIJI_ViT.h5")
    HONGBAN_ViT = Path(__file__).parent.parent.parent.parent.joinpath("models/HONGBAN_ViT.h5")
    MISE_ViT = Path(__file__).parent.parent.parent.parent.joinpath("models/MISE_ViT.h5")
    NONGPO_ViT = Path(__file__).parent.parent.parent.parent.joinpath("models/NONGPO_ViT.h5")
    TALMO_ViT = Path(__file__).parent.parent.parent.parent.joinpath("models/TALMO_ViT.h5")
