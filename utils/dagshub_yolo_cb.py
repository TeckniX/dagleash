import mlflow
import ultralytics

import os

class CustomYOLO(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = ultralytics.YOLO(context.artifacts['path'], task='segment')

    def predict(self, context, img):
        preds = self.model(img)

        return preds

def custom_on_train_end(trainer):
    """Called at end of train loop to log model artifact info."""
    if mlflow:
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or trainer.args.project or '/Shared/YOLOv8'
        mlflow.log_artifact(trainer.save_dir)
        mlflow.pyfunc.log_model(artifact_path=experiment_name,
                                code_paths=['utils'],
                                registered_model_name='YOLOv8 Custom',
                                artifacts={'path': str(trainer.best)},
                                python_model=CustomYOLO())

def custom_callbacks_fn(instance):
    from ultralytics.utils.callbacks.mlflow import callbacks as mlflow_cb
    mlflow_cb['on_train_end'] = custom_on_train_end
    for k, v in mlflow_cb.items():
        if v not in instance.callbacks[k]:  # prevent duplicate callbacks addition
            instance.callbacks[k].append(v)