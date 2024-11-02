import os
import json  # better to use "imports ujson as json" for the best performance
import tempfile

import uuid
import logging
from pathlib import Path

from PIL import Image

from label_studio_converter import Converter
from label_studio_converter.imports.label_config import generate_label_config
from label_studio_converter.utils import get_json_root_type
from collections import defaultdict
from copy import deepcopy

logger = logging.getLogger('root')

class YOLOAnnotationConverter():
    def __init__(self, dataset_dir, classes = [], to_name='image', from_name='label', label_type='bbox'):
        """Instantiate YOLO Annotation Convertert
        """
        self.ann_type = "YOLO"
        self.dataset_dir = dataset_dir
        self.classes = classes
        self.url_col = "dagshub_download_url"
        self.to_name = to_name
        self.from_name = from_name
        self.label_type = label_type

        # build categories=>labels dict
        if len(self.classes) == 0:
            if not self._update_classes_from_file():
                logger.warning(
                    'No classes.txt file found and now classes array supplied,'
                    'this might result in errors due to missing classes'
                )

        categories = {i: line for i, line in enumerate(self.classes)}
        logger.info(f'Found {len(categories)} categories')

        # generate and save labeling config
        self.config = generate_label_config(
            categories,
            {from_name: 'RectangleLabels'},
            to_name,
            from_name
        )

        self.ls_converter = DagsConverter(self.config, self.dataset_dir, download_resources=False)

    def _update_classes_from_file(self):
        notes_file = os.path.join(self.dataset_dir, 'classes.txt')
        if os.path.exists(notes_file):
            with open(notes_file) as f:
                self.classes = [line.strip() for line in f.readlines()]
            return True
        return False

    def _create_bbox(self, line, image_width, image_height):
        label_id, x, y, width, height = line.split()
        x, y, width, height = (
            float(x),
            float(y),
            float(width),
            float(height),
        )
        item = {
            "id": uuid.uuid4().hex[0:10],
            "type": "rectanglelabels",
            "value": {
                "x": (x - width / 2) * 100,
                "y": (y - height / 2) * 100,
                "width": width * 100,
                "height": height * 100,
                "rotation": 0,
                "rectanglelabels": [self.classes[int(label_id)]],
            },
            "to_name": self.to_name,
            "from_name": self.from_name,
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height,
        }
        return item

    def _create_segmentation(self, line, image_width, image_height):
        label_id = line.split()[0]
        points = [[float(x[0]), float(x[1])] for x in zip(*[iter(line.split()[1:])]*2)]

        for i in range(len(points)):
            points[i][0] = points[i][0] * 100.0
            points[i][1] = points[i][1] * 100.0

        item = {
            "id": uuid.uuid4().hex[0:10],
            "type": "polygonlabels",
            "value": {
                "closed": True,
                "points": points,
                "polygonlabels": [self.classes[int(label_id)]],
            },
            "to_name": self.to_name,
            "from_name": self.from_name,
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height,
        }
        return item

    def to_de(self, row, out_type="annotations"):
        """Convert YOLO labeling to Label Studio JSON
        :param out_type: annotation type - "annotations" or "predictions"
        """
        # define coresponding label file and check existence
        image_path = row["path"]
        if not "images/" in image_path:
            image_path = os.path.join("images", image_path)
        label_path = image_path.replace(image_path.split(".")[-1], "txt")
        if "/images/" in label_path or label_path.startswith("images/"):
            label_path = label_path.replace("images/","labels/")
        else:
            label_path = os.path.join("labels", label_path)

        label_file = os.path.join(self.dataset_dir, label_path)
        image_file = os.path.join(self.dataset_dir, image_path)
        image_width = 0
        image_height = 0


        task = {
            "data": {
                # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                "image": row[self.url_col]
            }
        }

        if os.path.exists(label_file):
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # read image sizes
            if not (image_width and image_height):
                # default to opening file if we aren't given image dims. slow!
                with Image.open(os.path.join(image_file)) as img:
                    if img.mode == "L":
                        im = img.copy()
                    elif img.layers == 3:
                        im = img.convert("RGB")
                    elif img.layers == 4:
                        im = img.convert("CMYK")
                    else:
                        raise ValueError(f"Unsupported number of layers: {img.layers}")
                    image_width, image_height = im.size

            with open(label_file) as file:
                # convert all bounding boxes to Label Studio Results
                lines = file.readlines()
                for line in lines:
                    if 'bbox' in self.label_type:
                      item = self._create_bbox(line, image_width, image_height)
                      task[out_type][0]['result'].append(item)
                    if 'segmentation' in self.label_type:
                      item = self._create_segmentation(line, image_width, image_height)
                      task[out_type][0]['result'].append(item)
                      task['is_labeled'] = True

        if task:
            return json.dumps(task).encode()

    def from_de(self, row):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f_path = os.path.join(tmp_dir, Path(row['path']).name)
            with open(f_path, "wb") as f:
                annotation_data = row["annotation"].to_ls_task()
                if annotation_data is None:
                    return
                f.write(annotation_data)
            ls_converter = DagsConverter(self.config, self.dataset_dir, download_resources=False)
            ls_converter.convert_to_yolo(input_data=f_path,
                                         output_dir=self.dataset_dir,
                                         output_label_dir=os.path.join(self.dataset_dir, "labels", *(row['path'].split("/")[:-1])),
                                         is_dir=False)

            # Add new classes to converter config
            ls_converter._get_labels()
            self._update_classes_from_file()

class DagsConverter(Converter):
    def iter_from_json_file(self, json_file):
        """Extract annotation results from json file

        param json_file: path to task dict with annotations
        """
        with open(json_file) as j:
            data = json.load(j)
            if not id in data:
                data['id'] = 0
            for item in self.annotation_result_from_task(data):
                yield item

    def _get_labels(self):
        labels = set()
        categories = list()
        category_name_to_id = dict()
        for name, info in self._schema.items():
            labels |= set(info['labels'])
            attrs = info['labels_attrs']
            for label in attrs:
                # DagsHub: add handling for case where there is no category attribute but we want to use the order in the schema
                label_num = attrs[label].get('category') if attrs[label].get('category') else info['labels'].index(label)
                categories.append(
                    {'id': label_num, 'name': label}
                )
                category_name_to_id[label] = label_num
        labels_to_add = set(labels) - set(list(category_name_to_id.keys()))
        labels_to_add = sorted(list(labels_to_add))
        idx = 0
        while idx in list(category_name_to_id.values()):
            idx += 1
        for label in labels_to_add:
            categories.append({'id': idx, 'name': label})
            category_name_to_id[label] = idx
            idx += 1
            while idx in list(category_name_to_id.values()):
                idx += 1
        return categories, category_name_to_id