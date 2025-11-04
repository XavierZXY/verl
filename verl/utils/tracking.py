# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import json
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: str | list[str] = "console",
        config=None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn(
                    "`tracking` logger is deprecated. use `wandb` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                settings=settings,
            )
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get(
                "MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db"
            )
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(
                experiment_id=experiment.experiment_id, run_name=experiment_name
            )
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(
                    SWANLAB_API_KEY
                )  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(
                project_name, experiment_name
            )

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(
                project_name, experiment_name, config
            )

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "file" in self.logger:
            self.logger["file"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
            print(f"Creating file logger at {self.filepath}")
        self.fp = open(self.filepath, "w")

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(json.dumps(data) + "\n")

    def finish(self):
        self.fp.close()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get(
            "TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}"
        )
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(
        _transform_params_to_json_serializable(params, convert_list_to_dict=True),
        sep="/",
    )


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(
        _transform_params_to_json_serializable,
        convert_list_to_dict=convert_list_to_dict,
    )

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {
                f"{i}": _transform(v) for i, v in enumerate(x)
            }
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table"""

        # Create column names for all samples
        columns = ["step"] + sum(
            [
                [f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"]
                for i in range(len(samples))
            ],
            [],
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]

        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(
                f"WARNING: save validation generation file to mlflow failed with error {e}"
            )

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join(
                    "tensorboard_log", self.project_name, self.experiment_name
                )
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()

    def log_multiturn_generations(self, loggers, multiturn_data, step, phase="val"):
        """Log multi-turn conversation data with tool responses and images.

        Args:
            loggers: List of logger names to use
            multiturn_data: List of dicts, each containing:
                - messages: List of message dicts with 'role' and 'content'
                - multi_modal_inputs: Dict with 'image' key containing PIL images
                - score: Final score for this rollout
                - uid: Unique identifier for the sample
            step: Current training step
            phase: Phase of logging, either "train" or "val" (default: "val")
        """
        if "wandb" in loggers:
            self._log_multiturn_to_wandb(multiturn_data, step, phase)
        if "swanlab" in loggers:
            self._log_multiturn_to_swanlab(multiturn_data, step, phase)

    def _log_multiturn_to_wandb(self, multiturn_data, step, phase="val"):
        """Log multi-turn conversations to wandb as a detailed table.

        Creates a table where each row represents a conversation turn, allowing
        users to see the full flow of user → assistant → tool → assistant interactions.

        Args:
            multiturn_data: List of conversation data
            step: Current training step
            phase: "train" or "val" to use different tables
        """
        import wandb

        # Create table columns - added tool_name, original_image, cropped_image and reward components
        columns = [
            "step",
            "sample_id",
            "turn_num",
            "role",
            "content",
            "tool_name",
            "original_image",
            "cropped_image",
            "tool_reward",
            "score",
            "bbox_iou",
            "acc_reward",
        ]

        # Use different table instances for train and val
        table_attr_name = f"multiturn_table_{phase}"
        if not hasattr(self, table_attr_name):
            setattr(self, table_attr_name, wandb.Table(columns=columns))

        # Create new table with existing data
        existing_table = getattr(self, table_attr_name)
        new_table = wandb.Table(columns=columns, data=existing_table.data)

        # Process each sample's multi-turn conversation
        for sample_idx, sample_data in enumerate(multiturn_data):
            # Support both formats: conversation_history (from training) and messages (from validation)
            conversation_history = sample_data.get("conversation_history", [])
            messages = sample_data.get("messages", [])
            multi_modal_inputs = sample_data.get("multi_modal_inputs", {})
            bbox_iou = sample_data.get("bbox_iou", None)
            score = sample_data.get("score", None)
            uid = sample_data.get("uid", f"sample_{sample_idx}")

            # If conversation_history is available (training rollout), use it
            if conversation_history:
                # print(f"[DEBUG] Processing conversation_history with {len(conversation_history)} entries for sample {uid}")
                for turn_num, turn_data in enumerate(conversation_history):
                    role = turn_data.get("role", "unknown")
                    content = turn_data.get("content", "")
                    tool_name = turn_data.get("tool_name", "")
                    tool_reward = sample_data.get("tool_reward", None)
                    tool_success = turn_data.get("tool_success", True)

                    # Truncate long content for readability
                    if len(content) > 500:
                        content_display = content[:497] + "..."
                    else:
                        content_display = content

                    # Get original and cropped images for tool responses
                    original_image = turn_data.get("original_image", None)
                    cropped_images = turn_data.get("cropped_images", [])

                    # print(f"[DEBUG] Turn {turn_num}: role={role}, has_original_image={original_image is not None}, "
                    #       f"original_image_type={type(original_image)}, has_cropped_images={len(cropped_images) > 0}, "
                    #       f"cropped_images_count={len(cropped_images)}")

                    # Process original image
                    original_image_obj = None
                    if role == "tool" and original_image is not None:
                        print(
                            f"[DEBUG] Converting original_image to wandb.Image: type={type(original_image)}, "
                            f"is_PIL={hasattr(original_image, 'size')}, size={getattr(original_image, 'size', None)}"
                        )
                        try:
                            original_image_obj = wandb.Image(
                                original_image, caption=f"Original - {tool_name}"
                            )
                            print(
                                f"[DEBUG] Successfully converted original_image to wandb.Image"
                            )
                        except Exception as e:
                            print(
                                f"[DEBUG] ERROR: Failed to convert original image to wandb.Image: {e}"
                            )
                            import traceback

                            traceback.print_exc()

                    # Process cropped images
                    cropped_image_obj = None
                    if role == "tool" and cropped_images:
                        try:
                            # Create a wandb Image from the first cropped image
                            # If multiple images, we could create a caption or montage
                            if len(cropped_images) == 1:
                                cropped_image_obj = wandb.Image(
                                    cropped_images[0],
                                    caption=f"Cropped - {tool_name}: {content_display[:100]}",
                                )
                            else:
                                # For multiple images, create a grid or log them separately
                                import numpy as np
                                from PIL import Image

                                # Create a simple horizontal concatenation
                                try:
                                    widths, heights = zip(
                                        *(i.size for i in cropped_images)
                                    )
                                    total_width = sum(widths)
                                    max_height = max(heights)
                                    new_im = Image.new("RGB", (total_width, max_height))
                                    x_offset = 0
                                    for im in cropped_images:
                                        new_im.paste(im, (x_offset, 0))
                                        x_offset += im.width
                                    cropped_image_obj = wandb.Image(
                                        new_im,
                                        caption=f"Cropped - {tool_name}: {len(cropped_images)} images",
                                    )
                                except Exception as e:
                                    # Fallback to first image
                                    cropped_image_obj = wandb.Image(
                                        cropped_images[0],
                                        caption=f"Cropped - {tool_name}: {len(cropped_images)} images",
                                    )
                        except Exception as e:
                            print(
                                f"Warning: Failed to convert cropped image to wandb.Image: {e}"
                            )

                    # Add score only on the last turn
                    turn_score = (
                        score if turn_num == len(conversation_history) - 1 else None
                    )

                    # Add row to table
                    # print(f"[DEBUG] Adding row to table: turn={turn_num}, role={role}, "
                    #       f"has_original_image_obj={original_image_obj is not None}, "
                    #       f"has_cropped_image_obj={cropped_image_obj is not None}")

                    # Add reward components only on the last turn
                    turn_bbox_iou = (
                        bbox_iou if turn_num == len(conversation_history) - 1 else None
                    )
                    turn_acc_reward = (
                        sample_data.get("acc_reward", None)
                        if turn_num == len(conversation_history) - 1
                        else None
                    )

                    new_table.add_data(
                        step,
                        str(uid)[:12],  # Truncate uid for readability
                        turn_num,
                        role,
                        content_display,
                        tool_name if tool_name else None,  # Use None instead of ""
                        original_image_obj,  # Original image before tool processing
                        cropped_image_obj,  # Cropped/processed image after tool
                        tool_reward
                        if tool_reward is not None
                        else None,  # Use None instead of ""
                        turn_score,
                        turn_bbox_iou,  # bbox_iou
                        turn_acc_reward,  # acc_reward
                    )

            # Otherwise use messages format (validation)
            elif messages:
                # Get images if available from multi_modal_inputs
                images = (
                    multi_modal_inputs.get("image", []) if multi_modal_inputs else []
                )
                image_idx = 0

                # Process each turn in the conversation
                for turn_num, message in enumerate(messages):
                    role = message.get("role", "unknown")
                    content = message.get("content", "")

                    # Format content for display
                    if isinstance(content, list):
                        # Multi-modal content (e.g., [{"type": "image"}, {"type": "text", "text": "..."}])
                        text_parts = [
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        content_str = (
                            " ".join(text_parts)
                            if text_parts
                            else "[multimodal content]"
                        )
                    elif isinstance(content, dict):
                        content_str = str(content)
                    else:
                        content_str = str(content)

                    # Truncate long content for readability
                    if len(content_str) > 500:
                        content_str = content_str[:497] + "..."

                    # Get associated image for tool responses
                    image_obj = None
                    if role == "tool" and images and image_idx < len(images):
                        try:
                            # Convert PIL image to wandb Image
                            image_obj = wandb.Image(images[image_idx])
                            image_idx += 1
                        except Exception as e:
                            print(
                                f"Warning: Failed to convert image to wandb.Image: {e}"
                            )

                    # Add score only on the last turn
                    turn_score = score if turn_num == len(messages) - 1 else None

                    # Add reward components only on the last turn
                    turn_bbox_iou = bbox_iou if turn_num == len(messages) - 1 else None
                    turn_acc_reward = (
                        sample_data.get("acc_reward", None)
                        if turn_num == len(messages) - 1
                        else None
                    )

                    # Add row to table
                    new_table.add_data(
                        step,
                        str(uid)[:12] if uid else f"sample_{sample_idx}",
                        turn_num,
                        role,
                        content_str,
                        None,  # tool_name - use None instead of ""
                        None,  # original_image - not available in legacy format
                        image_obj,  # cropped_image (or tool result image)
                        None,  # tool_reward - use None instead of ""
                        turn_score,
                        turn_bbox_iou,  # bbox_iou
                        turn_acc_reward,  # acc_reward
                    )

        # Log the table with phase-specific name
        table_name = f"{phase}/multiturn_generations"
        wandb.log({table_name: new_table}, step=step)
        setattr(self, table_attr_name, new_table)

    def _log_multiturn_to_swanlab(self, multiturn_data, step, phase="val"):
        """Log multi-turn conversations to swanlab as a table.

        Args:
            multiturn_data: List of conversation data
            step: Current training step
            phase: "train" or "val" to use different tables
        """
        import swanlab

        swanlab_table = swanlab.echarts.Table()
        headers = [
            "step",
            "sample_id",
            "turn_num",
            "role",
            "content",
            "score",
            "bbox_iou",
            "acc_reward",
        ]

        rows = []
        for sample_idx, sample_data in enumerate(multiturn_data):
            messages = sample_data.get("messages", [])
            score = sample_data.get("score", None)
            bbox_iou = sample_data.get("bbox_iou", None)
            acc_reward = sample_data.get("acc_reward", None)

            for turn_num, message in enumerate(messages):
                role = message.get("role", "unknown")
                content = message.get("content", "")

                # Format content
                if isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    content_str = (
                        " ".join(text_parts) if text_parts else "[multimodal content]"
                    )
                elif isinstance(content, dict):
                    content_str = str(content)
                else:
                    content_str = str(content)

                # Truncate long content
                if len(content_str) > 300:
                    content_str = content_str[:297] + "..."

                # Add score and reward components only on last turn
                turn_score = score if turn_num == len(messages) - 1 else ""
                turn_bbox_iou = bbox_iou if turn_num == len(messages) - 1 else ""
                turn_acc_reward = acc_reward if turn_num == len(messages) - 1 else ""

                rows.append(
                    [
                        step,
                        sample_idx,
                        turn_num,
                        role,
                        content_str,
                        turn_score,
                        turn_bbox_iou,
                        turn_acc_reward,
                    ]
                )

        swanlab_table.add(headers=headers, rows=rows)
        table_name = f"{phase}/multiturn_generations"
        swanlab.log({table_name: swanlab_table}, step=step)
