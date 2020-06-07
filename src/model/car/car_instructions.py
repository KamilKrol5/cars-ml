from dataclasses import dataclass


@dataclass
class CarInstruction:
    acceleration: float
    turning_rate: float
