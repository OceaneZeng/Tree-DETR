"""Author-defined OWOD category orders and immutable released image lists.

M: Joseph et al., CVPR 2021, Sec. 5.1/5.2.
S: Gupta et al., CVPR 2022, Appendix A.2/Table 7.
The ImageSets are vendored from the pinned OW-DETR author repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPEC_ROOT = Path(__file__).resolve().parents[2] / 'configs' / 'owod'
COCO_NAMES = (
    'person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,'
    'fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,'
    'elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,'
    'skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,'
    'surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,'
    'apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,'
    'potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,'
    'microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,'
    'hair drier,toothbrush'
).split(',')
COCO_IDS = (list(range(1, 12)) + [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90])
NAME_TO_ID = dict(zip(COCO_NAMES, COCO_IDS))
ALIASES = dict(zip(['aeroplane', 'diningtable', 'motorbike', 'pottedplant',
                   'sofa', 'tvmonitor'],
                  ['airplane', 'dining table', 'motorcycle', 'potted plant',
                   'couch', 'tv']))
TASK_NAMES = {
    'm-owodb': [
        'airplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,dining table,dog,'
        'horse,motorcycle,person,potted plant,sheep,couch,train,tv',
        'truck,traffic light,fire hydrant,stop sign,parking meter,bench,elephant,'
        'bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,microwave,oven,'
        'toaster,sink,refrigerator',
        'frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,'
        'skateboard,surfboard,tennis racket,banana,apple,sandwich,orange,'
        'broccoli,carrot,hot dog,pizza,donut,cake',
        'bed,toilet,laptop,mouse,remote,keyboard,cell phone,book,clock,vase,'
        'scissors,teddy bear,hair drier,toothbrush,wine glass,cup,fork,knife,spoon,bowl',
    ],
    's-owodb': [
        'airplane,bicycle,bird,boat,bus,car,cat,cow,dog,horse,motorcycle,sheep,'
        'train,elephant,bear,zebra,giraffe,truck,person',
        'traffic light,fire hydrant,stop sign,parking meter,bench,chair,dining table,'
        'potted plant,backpack,umbrella,handbag,tie,suitcase,microwave,oven,toaster,'
        'sink,refrigerator,bed,toilet,couch',
        'frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,'
        'skateboard,surfboard,tennis racket,banana,apple,sandwich,orange,'
        'broccoli,carrot,hot dog,pizza,donut,cake',
        'laptop,mouse,remote,keyboard,cell phone,book,clock,vase,scissors,teddy bear,'
        'hair drier,toothbrush,wine glass,cup,fork,knife,spoon,bowl,tv,bottle',
    ],
}


def specification(protocol: str) -> dict:
    if protocol not in TASK_NAMES:
        raise ValueError(f'Unsupported OWOD benchmark: {protocol}')
    return {
        'protocol': protocol,
        'task_class_ids': [[NAME_TO_ID[n] for n in task.split(',')]
                           for task in TASK_NAMES[protocol]],
        'ap_mode': 'integral' if protocol == 'm-owodb' else 'voc2007_11point',
        'wi_recall': 0.8,
        'iou_threshold': 0.5,
        'test_split': 'all_task_test' if protocol == 'm-owodb' else 'test',
        'source_paper': ('Joseph et al. CVPR 2021 Sec.5.1/5.2' if protocol == 'm-owodb'
                         else 'Gupta et al. CVPR 2022 Appendix A.2/Table 7'),
    }


def official_split(protocol: str, name: str) -> list[str]:
    """Keep released ordering AND multiplicity; never randomly rebuild a split."""
    metadata = json.loads((SPEC_ROOT / 'sources.json').read_text(encoding='utf-8'))
    record = metadata[protocol]['files'][name]
    data = (SPEC_ROOT / record['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError(f'Official split was modified: {protocol}/{name}')
    return data.decode('utf-8').split()


def categories() -> list[dict]:
    return [{'id': i, 'name': n} for i, n in zip(COCO_IDS, COCO_NAMES)]
