from typing import Dict
from numpy import typing


class tracklet:
    @staticmethod
    def is_small(bbox: typing.ArrayLike) -> bool:
        """
        Decides whether an object is considered 'small'

        :param bbox: Bbox in YOLO format (0-1 scale)
        :return: Boolean. True if the box is small
        """
        # if bbox[2] < 0.07 or bbox[3] < 0.035:
        if bbox[2] < 0.04 or bbox[3] < 0.025:
            return True
        return False

    def __init__(self, uid: int, f_num: int):
        """
        Initializes an auxiliary tracklet object with a given uid

        :param uid: Unique identifier. Should be the initialization id when the tracklet is initializing.
                    If the tracklet has been initialized, then uid is set to its id.
        """
        self.uid: int = uid
        self.background: bool = True
        self.obj_type: Dict[str, int] = {}
        self.estimates = []
        self.final_type: str = ''
        self.latest_frame = -1
        self.first_frame = f_num
        self.narrow_frames = 0

    def finalize_obj_type(self) -> None:
        max_count: int = -1
        for obj_name, count in self.obj_type.items():
            if count > max_count:
                self.final_type = obj_name
                max_count = count

    def update(self, obj_type: str, bbox: typing.NDArray, frame_num: int) -> None:
        if not tracklet.is_small(bbox):
            self.background = False
            weight = 2
        else:
            weight = 1
        self.obj_type[obj_type] = self.obj_type.get(obj_type, 0) + weight
        # TODO: Experimental
        self.latest_frame = frame_num

        if bbox[2] == 0:
            self.narrow_frames += 5
        elif bbox[3]/bbox[2] > 5 or (bbox[3]/bbox[2] > 3 and obj_type in {2, 5, 7}):
            if self.uid == 2:
                print(frame_num)
                print( obj_type in {2, 5, 7})
            self.narrow_frames += 1
        else:
            self.narrow_frames = 0

    def add_estimate(self, entry:list):
        self.estimates.append(entry)

    def clear_estimates(self):
        self.estimates.clear()

    def update_id(self, id):
        self.uid = id
        for e in self.estimates:
            e[-1] = id