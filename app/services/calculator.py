import math
from typing import Optional

def calculate_bri(waist_cm: Optional[float], height_cm: Optional[float]) -> float:
    """
    Body Roundness Index (BRI) calculation.
    BRI = 364.2 - 365.5 * sqrt(1 - (waist / (2*pi) / (0.5 * height))^2)
    """
    if not waist_cm or not height_cm or height_cm == 0:
        return 0.0
    
    # Standard formula
    # Conversion to meters if needed, but the ratio stays same
    try:
        eccentricity = math.sqrt(1 - ((waist_cm / (2 * math.pi)) / (0.5 * height_cm))**2)
        bri = 364.2 - 365.5 * eccentricity
        return round(bri, 4)
    except (ValueError, ZeroDivisionError):
        return 0.0

def calculate_nlr(neutrophils: Optional[float], lymphocytes: Optional[float]) -> float:
    """Neutrophil-to-Lymphocyte Ratio (NLR)"""
    if not neutrophils or not lymphocytes or lymphocytes == 0:
        return 0.0
    return round(neutrophils / lymphocytes, 4)

def estimate_dii_proxy(fiber_g: float) -> float:
    """
    Simplified proxy for DII based on Fiber (one of its strongest components).
    Fiber is protective (negative weight in DII).
    This is a dummy proxy for demonstration.
    """
    # DII usually ranges from -5 to +5. Fiber reduces DII.
    # Mean fiber intake ~15g. 
    return round(2.0 - (fiber_g / 10.0), 4)
