"""Vehicle brand/model dictionaries and color mappings."""

import re

# --------------------------------------------------------------------------- #
# Time budget
# --------------------------------------------------------------------------- #
MAX_TIME: float = 100.0  # seconds

# --------------------------------------------------------------------------- #
# YOLO plate detection
# --------------------------------------------------------------------------- #
PLATE_MODEL_PATH = "app/models/plate_detector.onnx"
PLATE_DETECT_CONF = 0.25
BADGE_DET_THRESH = 0.15

# --------------------------------------------------------------------------- #
# License plate regex — Peruvian formats:
#   Old: ABC-123  (3 letters + 3 digits)
#   New: A1A-123  (letter + digit + letter + 3 digits)
# --------------------------------------------------------------------------- #
PLATE_RE = re.compile(r"(?:[A-Z]{3}|[A-Z]\d[A-Z])-?\d{3}")

# --------------------------------------------------------------------------- #
# Model → Brand lookup (uppercase, no hyphens/spaces)
# --------------------------------------------------------------------------- #
MODEL_TO_BRAND: dict[str, str] = {
    # Toyota
    "COROLLA": "TOYOTA",
    "YARIS": "TOYOTA",
    "HILUX": "TOYOTA",
    "RAV4": "TOYOTA",
    "FORTUNER": "TOYOTA",
    "LANDCRUISER": "TOYOTA",
    "RUSH": "TOYOTA",
    "AVANZA": "TOYOTA",
    "PRADO": "TOYOTA",
    "CAMRY": "TOYOTA",
    "ETIOS": "TOYOTA",
    "AGYA": "TOYOTA",
    "VIOS": "TOYOTA",
    # Mazda
    "CX5": "MAZDA",
    "CX3": "MAZDA",
    "CX30": "MAZDA",
    "CX50": "MAZDA",
    "CX9": "MAZDA",
    "MAZDA2": "MAZDA",
    "MAZDA3": "MAZDA",
    "MAZDA6": "MAZDA",
    "BT50": "MAZDA",
    # Hyundai
    "TUCSON": "HYUNDAI",
    "ACCENT": "HYUNDAI",
    "SANTAFE": "HYUNDAI",
    "CRETA": "HYUNDAI",
    "VENUE": "HYUNDAI",
    "ELANTRA": "HYUNDAI",
    "STAREX": "HYUNDAI",
    "I10": "HYUNDAI",
    "I20": "HYUNDAI",
    "I30": "HYUNDAI",
    "SONATA": "HYUNDAI",
    # Kia
    "SPORTAGE": "KIA",
    "SELTOS": "KIA",
    "SORENTO": "KIA",
    "PICANTO": "KIA",
    "RIO": "KIA",
    "CERATO": "KIA",
    "FORTE": "KIA",
    "CARNIVAL": "KIA",
    "SOUL": "KIA",
    "STONIC": "KIA",
    # Nissan
    "SENTRA": "NISSAN",
    "VERSA": "NISSAN",
    "XTRAIL": "NISSAN",
    "FRONTIER": "NISSAN",
    "KICKS": "NISSAN",
    "MARCH": "NISSAN",
    "NAVARA": "NISSAN",
    "MURANO": "NISSAN",
    "QASHQAI": "NISSAN",
    "PATHFINDER": "NISSAN",
    "VDRIVE": "NISSAN",
    # Chevrolet
    "SPARK": "CHEVROLET",
    "SAIL": "CHEVROLET",
    "CRUZE": "CHEVROLET",
    "TRACKER": "CHEVROLET",
    "ONIX": "CHEVROLET",
    "GROOVE": "CHEVROLET",
    "DMAX": "CHEVROLET",
    "CAPTIVA": "CHEVROLET",
    # Suzuki
    "SWIFT": "SUZUKI",
    "VITARA": "SUZUKI",
    "ERTIGA": "SUZUKI",
    "JIMNY": "SUZUKI",
    "DZIRE": "SUZUKI",
    "CIAZ": "SUZUKI",
    "SCROSS": "SUZUKI",
    "CELERIO": "SUZUKI",
    # Volkswagen
    "GOL": "VOLKSWAGEN",
    "TIGUAN": "VOLKSWAGEN",
    "TCROSS": "VOLKSWAGEN",
    "TAOS": "VOLKSWAGEN",
    "JETTA": "VOLKSWAGEN",
    "POLO": "VOLKSWAGEN",
    "AMAROK": "VOLKSWAGEN",
    # Mitsubishi
    "OUTLANDER": "MITSUBISHI",
    "ASX": "MITSUBISHI",
    "L200": "MITSUBISHI",
    "MONTERO": "MITSUBISHI",
    "MIRAGE": "MITSUBISHI",
    "XPANDER": "MITSUBISHI",
    # Renault
    "DUSTER": "RENAULT",
    "KWID": "RENAULT",
    "LOGAN": "RENAULT",
    "STEPWAY": "RENAULT",
    "KOLEOS": "RENAULT",
    "SANDERO": "RENAULT",
    # Honda
    "CIVIC": "HONDA",
    "CRV": "HONDA",
    "HRV": "HONDA",
    "FIT": "HONDA",
    "CITY": "HONDA",
    "WRV": "HONDA",
    "ACCORD": "HONDA",
    "PILOT": "HONDA",
    # Subaru
    "FORESTER": "SUBARU",
    "XV": "SUBARU",
    "OUTBACK": "SUBARU",
    "IMPREZA": "SUBARU",
    "WRX": "SUBARU",
    # Ford
    "ECOSPORT": "FORD",
    "ESCAPE": "FORD",
    "RANGER": "FORD",
    "EXPLORER": "FORD",
    "BRONCO": "FORD",
    "TERRITORY": "FORD",
    # Chery
    "TIGGO2": "CHERY",
    "TIGGO3": "CHERY",
    "TIGGO4": "CHERY",
    "TIGGO5": "CHERY",
    "TIGGO7": "CHERY",
    "TIGGO8": "CHERY",
    "TIGGO4PRO": "CHERY",
    "TIGGO7PRO": "CHERY",
    "TIGGO8PRO": "CHERY",
    "ARRIZO5": "CHERY",
    "ARRIZO6": "CHERY",
    "ARRIZO8": "CHERY",
    # JAC
    "S2": "JAC",
    "S3": "JAC",
    "S4": "JAC",
    "S7": "JAC",
    "T6": "JAC",
    "T8": "JAC",
    "T9": "JAC",
    "JS4": "JAC",
    "JS6": "JAC",
    "J7": "JAC",
    # MG
    "ZS": "MG",
    "MG3": "MG",
    "MG5": "MG",
    "MG6": "MG",
    "MG7": "MG",
    "HS": "MG",
    "RX5": "MG",
    "MARVEL": "MG",
    # BYD (expanded)
    "ATTO3": "BYD",
    "ATTO": "BYD",
    "SEAGULL": "BYD",
    "SHARK": "BYD",
    "DESTROYER": "BYD",
    "FRIGATE": "BYD",
    "KINGKONG": "BYD",
    # Geely
    "COOLRAY": "GEELY",
    "AZKARRA": "GEELY",
    "OKAVANGO": "GEELY",
    "EMGRAND": "GEELY",
    "TUGELLA": "GEELY",
    "PREFACE": "GEELY",
    "MONJARO": "GEELY",
    # Changan
    "CS15": "CHANGAN",
    "CS35": "CHANGAN",
    "CS55": "CHANGAN",
    "CS75": "CHANGAN",
    "CS85": "CHANGAN",
    "CS95": "CHANGAN",
    "UNI-T": "CHANGAN",
    "UNIT": "CHANGAN",
    "UNIK": "CHANGAN",
    "UNI-K": "CHANGAN",
    "UNIV": "CHANGAN",
    "UNI-V": "CHANGAN",
    "ALSVIN": "CHANGAN",
    "HUNTER": "CHANGAN",
    # GAC (Trumpchi)
    "GS3": "GAC",
    "GS4": "GAC",
    "GS5": "GAC",
    "GS8": "GAC",
    "GM6": "GAC",
    "GM8": "GAC",
    "GN6": "GAC",
    "GN8": "GAC",
    "EMKOO": "GAC",
    "EMPOW": "GAC",
    # DFSK
    "GLORY": "DFSK",
    "GLORY500": "DFSK",
    "GLORY580": "DFSK",
    "GLORY600": "DFSK",
    "K01": "DFSK",
    "K02": "DFSK",
    "C31": "DFSK",
    "C35": "DFSK",
    # Foton
    "TUNLAND": "FOTON",
    "SAUVANA": "FOTON",
    "GRATOUR": "FOTON",
    "AUMARK": "FOTON",
    # Jetour
    "X70": "JETOUR",
    "X90": "JETOUR",
    "X95": "JETOUR",
    "DASHING": "JETOUR",
    "TRAVELER": "JETOUR",
    # Omoda (Chery sub-brand)
    "C5": "OMODA",
    "E5": "OMODA",
    # Jeep
    "COMPASS": "JEEP",
    "RENEGADE": "JEEP",
    "WRANGLER": "JEEP",
    "CHEROKEE": "JEEP",
    "GLADIATOR": "JEEP",
    # Peugeot
    "208": "PEUGEOT",
    "2008": "PEUGEOT",
    "3008": "PEUGEOT",
    "5008": "PEUGEOT",
    # Fiat
    "ARGO": "FIAT",
    "CRONOS": "FIAT",
    "PULSE": "FIAT",
    "STRADA": "FIAT",
    # BYD
    "DOLPHIN": "BYD",
    "SEAL": "BYD",
    "SONG": "BYD",
    "YUAN": "BYD",
    "HAN": "BYD",
    "TANG": "BYD",
    "SEALION": "BYD",
    "SEABIRD": "BYD",
    # Great Wall / Haval
    "H6": "HAVAL",
    "JOLION": "HAVAL",
    "DARGO": "HAVAL",
    "POER": "GREAT WALL",
    "WINGLE": "GREAT WALL",
}

BRANDS: list[str] = sorted(
    {
        "TOYOTA", "MAZDA", "HYUNDAI", "KIA", "NISSAN", "CHEVROLET",
        "SUZUKI", "VOLKSWAGEN", "MITSUBISHI", "RENAULT", "HONDA",
        "SUBARU", "FORD", "CHERY", "JAC", "MG", "JEEP", "PEUGEOT",
        "FIAT", "BYD", "HAVAL", "GREAT WALL", "BMW", "MERCEDES",
        "AUDI", "VOLVO", "LEXUS", "INFINITI", "ACURA", "LAND ROVER",
        "DODGE", "RAM", "CHRYSLER", "GEELY", "CHANGAN", "GAC",
        "DFSK", "FOTON", "SSANGYONG", "MAHINDRA", "TATA",
        "JETOUR", "OMODA",
    }
)

COLOR_RANGES: list[tuple[int, int, int, int, int, int, str]] = [
    # Reds wrap around 0/180
    (0, 10, 70, 255, 50, 255, "ROJO"),
    (170, 180, 70, 255, 50, 255, "ROJO"),
    # Orange
    (10, 25, 70, 255, 50, 255, "NARANJA"),
    # Yellow
    (25, 35, 70, 255, 50, 255, "AMARILLO"),
    # Green
    (35, 85, 40, 255, 40, 255, "VERDE"),
    # Blue
    (85, 130, 40, 255, 40, 255, "AZUL"),
    # Purple / Violet
    (130, 170, 40, 255, 40, 255, "MORADO"),
]

WHITE_V_MIN = 200
WHITE_S_MAX = 40
BLACK_V_MAX = 60
SILVER_V_RANGE = (120, 200)
SILVER_S_MAX = 50
GRAY_V_RANGE = (60, 120)
GRAY_S_MAX = 50
