"""Monitoramento de bebe com deteccao de posicao prona e brinquedos."""

from config import *
from prone_detector import check_prone, cleanup
from prone_timer import ProneTimer
from absence_timer import AbsenceTimer
from drawing import draw_prone_alert, draw_absence_alert
