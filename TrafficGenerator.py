"""
TrafficGenerator
Generate synthetic network traffic.
Trying to modularize traffic generation vs link type.
Behavior of the traffic generator may depend on how previous packets were handled by the link (eg. TCP),
so the API will be to generate one packet at a time, and be informed what happened to the last packet..
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from LinkProperties import LinkProperties, link_properties_library


@dataclass
class PacketInfo:
    size_kbyte: float
    tx_time_ms: float
    inter_packet_time_ms: float


class TrafficGenerator(ABC):
    _registry = {}

    def __init__(self, link_properties:LinkProperties):
        self.link_properties = link_properties
        self.packet_count = 0
        self.byte_count = 0
        self.last_pkt_time_ms = 0.0

    def __init_subclass__(cls, traffic_type, **kwargs):
        print(f"__init_subclass__(traffic_type={traffic_type})")
        super().__init_subclass__(**kwargs)
        # Automatically adds the subclass to the dictionary when it's imported
        cls._registry[traffic_type] = cls
        #cls.link_properties:LinkProperties = link_properties
        # instantiate the appropriate subclass
        # perhaps this is a silly way to do it.
        # avoid creating a loop in the constructor
        #if type(self) is TrafficGenerator:
        #    self.traffic_generator = SUBCLASS_MAPPING[link_properties.traffic_generator](link_properties)

    @staticmethod
    def create(link_properties:LinkProperties, traffic_type):
        return TrafficGenerator._registry[traffic_type](link_properties)

    @abstractmethod
    def next_packet(self, last_latency:float=0.0, last_drop:bool=False):
        pass

