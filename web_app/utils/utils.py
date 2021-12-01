import enum 


class EventRenderingType(enum.IntEnum):
    RED_BLUE_OVERLAP = enum.auto()
    RED_BLUE_NO_OVERLAP = enum.auto()
    BLACK_WHITE_NO_OVERLAP = enum.auto()
    TIME_SURFACE = enum.auto()
    EVENT_FRAME = enum.auto()

