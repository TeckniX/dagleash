import os
import yaml
import json

from .yolo_converter import YOLOAnnotationConverter

class DataFunctions():
    def __init__(self, dataset_dir, classes = [], to_name='image', from_name='label', label_type='bbox'):
        self.conv = YOLOAnnotationConverter(
            dataset_dir=dataset_dir,
            classes=classes,
            to_name=to_name,
            from_name=from_name,
            label_type=label_type)


    def create_yolo_v8_dataset_yaml(self, dataset):
        path = str(dataset.all().download_files(target_dir="."))
        for dp in dataset.all().get_blob_fields("annotation"):
            self.conv.from_de(dp)

        train = "data/images/train"
        val = "data/images/val"

        yaml_dict = {
            'path': os.getcwd(),
            'train': train,
            'val': val,
            'names': {i: name for i, name in enumerate(self.conv.classes)}
        }
        with open("custom_coco.yaml", "w") as file:
            file.write(yaml.dump(yaml_dict))

    def create_categories_COCO(self, annotations):
        categories = set()
        json_annotation = json.loads(annotations.decode())
        if 'annotations' in json_annotation:
            for annotation in json_annotation["annotations"]:
                for result in annotation['result']:
                    categories.add(result['value'][result['type']][0])
        return ', '.join(str(item) for item in categories)

    def create_metadata(self, s):
        if s['path'].startswith(('val', 'test', 'train')):
            s["valid_datapoint"] = True
            s['year'] = 2017
            path = s['path'].split('/')
            s["split"] = path[0]
            # Add annotations where relevant
            if not s['path'].startswith('test'):
                if not ('annotation' in s and s['annotation']):
                    s['annotation'] = self.conv.to_de(s)
                s['categories'] = self.create_categories_COCO(s["annotation"])
        else:
            print("Data paths must start with 'val'/'test'/'train'")
        return s