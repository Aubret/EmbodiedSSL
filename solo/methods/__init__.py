# Copyright 2023 solo-learn development team.
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.methods.mocov3 import MoCoV3
from solo.methods.mocov3aa import AAMoCoV3
from solo.methods.mocov3aaplus import AAMoCoV3Plus
from solo.methods.simclr import SimCLR
from solo.methods.simclraa import AASimCLR
from solo.methods.simclraaplus import AASimCLRPlus
from solo.methods.vicreg import VICReg
from solo.methods.vicregaa import AAVICReg
from solo.methods.vicregaaplus import AAVICRegPlus



METHODS = {
    "base": BaseMethod,
    "linear": LinearModel,
    "mocov3": MoCoV3,
    "aamocov3": AAMoCoV3,
    "aamocov3+": AAMoCoV3Plus,
    "simclr": SimCLR,
    "aasimclr": AASimCLR,
    "aasimclrplus": AASimCLRPlus,
    "vicreg": VICReg,
    "aavicreg": AAVICReg,
    "aavicreg+": AAVICRegPlus,
}
__all__ = [
    "BaseMethod",
    "LinearModel",
    "MoCoV3",
    "SimCLR",
    "VICReg",
    "AAMoCoV3",
    "AAMoCoV3Plus",
    "AASimCLR",
    "AAVICReg",
]
