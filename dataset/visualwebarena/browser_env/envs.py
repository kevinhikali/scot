import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(4):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union
from io import BytesIO, StringIO

import pandas as pd
import numpy as np
import numpy.typing as npt
import requests
from beartype import beartype
from gymnasium import Env
from gymnasium.spaces import Box, Text
from playwright.sync_api import (
    CDPSession,
    Page,
    Playwright,
    ViewportSize,
    expect,
    sync_playwright,
)

DATASET = os.environ["DATASET"]
if DATASET == "visualwebarena":
    from browser_env.env_config import (
        CLASSIFIEDS,
        CLASSIFIEDS_RESET_TOKEN,
    )

from .actions import Action, execute_action, get_action_space
from .processors import ObservationHandler, ObservationMetadata
from .utils import (
    AccessibilityTree,
    DetachedPage,
    Observation,
    png_bytes_to_numpy,
)


@dataclass
class PlaywrightScript:
    function: str  # goto, get_by_role
    destination: str  # https://www.google.com/, combobox
    name: str | None = None  # Search, Avatar 2009
    operation: str | None = None  # click, fill, press
    value: str | None = None  # avatar movie, Enter


def parse_action(action: str) -> PlaywrightScript:
    splitted = action.strip().split(" ")
    assert len(splitted) >= 2
    match splitted[:2]:
        case ["goto", url]:
            assert len(splitted) == 2
            return PlaywrightScript("goto", url)
        case ["get_by_role", destination]:
            assert len(splitted) >= 4
            match splitted[2:]:
                case [name, operation]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation
                    )
                case [name, operation, value]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation, value
                    )
                case _:
                    raise ValueError("Invalid action")
        case _:
            raise ValueError(f"Invalid action {action}")


class ScriptBrowserEnv(Env[dict[str, Observation], Action]):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """

    @beartype
    def __init__(
        self,
        max_page_length: int = 8192,
        headless: bool = True,
        slow_mo: int = 0,
        observation_type: str = "html",
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.0,
        captioning_fn=None,
    ):
        # TODO: make Space[Action] = ActionSpace
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution

        match observation_type:
            case "html" | "accessibility_tree" | "accessibility_tree_with_captioner":
                self.text_observation_type = observation_type
                self.image_observation_type = ""
                self.main_observation_type = "text"
            case "image":
                self.image_observation_type = observation_type
                self.text_observation_type = ""  # type: ignore[assignment]
                self.main_observation_type = "image"
            case "image_som":
                self.image_observation_type = observation_type
                self.text_observation_type = observation_type  # type: ignore[assignment]
                self.main_observation_type = "image"
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )

        self.observation_handler = ObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
            captioning_fn,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )

    @beartype
    def setup(self, config_file: Path | None = None) -> None:
        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.__enter__()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo
        )

        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}

        # Reset site if needed. Currently only supported for Classifieds.
        # TODO(jykoh): Add reset functionality for Shopping/Reddit.
        if instance_config.get("require_reset", False):
            if "classifieds" in instance_config["sites"]:
                # Send POST request to __CLASSIFIEDS__/index.php?page=reset with token=CLASSIFIEDS_TOKEN
                response = requests.post(
                    f"{CLASSIFIEDS}/index.php?page=reset",
                    data={"token": CLASSIFIEDS_RESET_TOKEN},
                )

                # Check if the request was successful
                if response.status_code == 200:
                    print("Reset Classifieds site.")
                else:
                    print(
                        "Failed to reset Classifieds site:",
                        response.status_code,
                    )
            else:
                print(
                    "WARNING: Reset is not supported for this site. Please manually reset the site."
                )

        storage_state = instance_config.get("storage_state", None)
        start_url = instance_config.get("start_url", None)
        geolocation = instance_config.get("geolocation", None)

        # Use custom viewport size if specified in the config, otherwise use the default.
        viewport_size = self.viewport_size.copy()
        viewport_size.update(instance_config.get("viewport_size", {}))
        self.observation_handler.viewport_size = viewport_size

        self.context = self.browser.new_context(
            viewport=viewport_size,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
        )
        if self.save_trace_enabled:
            self.context.tracing.start(screenshots=True, snapshots=True)

        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                page = self.context.new_page()
                if self.text_observation_type in [
                    "accessibility_tree",
                    "accessibility_tree_with_captioner",
                ]:
                    client = page.context.new_cdp_session(page)
                    client.send("Accessibility.enable")
                    client.detach()
                page.goto(url)
            # set the first page as the current page
            self.page = self.context.pages[0]
            self.page.bring_to_front()
        else:
            self.page = self.context.new_page()
            if self.text_observation_type in [
                "accessibility_tree",
                "accessibility_tree_with_captioner",
            ]:
                client = self.page.context.new_cdp_session(self.page)
                client.send("Accessibility.enable")
                client.detach()

    def _get_obs(self) -> dict[str, Observation]:
        obs = self.observation_handler.get_observation(self.page)
        return obs

    def _get_obs_metadata(self) -> dict[str, ObservationMetadata]:
        metadata = self.observation_handler.get_observation_metadata()
        return metadata

    def get_som_img(self):
        return self.observation_handler.action_processor.get_som_img(self.page)

    def get_bboxes(self) -> dict:
        browser_info = self.observation_handler.action_processor.fetch_browser_info(self.page)
        browser_config = browser_info["config"]
        data_string = self.observation_handler.action_processor.get_page_bboxes(self.page)
        df = pd.read_csv(StringIO(data_string), delimiter=",", quotechar='"')
        df["Area"] = df["Width"] * df["Height"]
        # Remove bounding boxes that are clipped.
        b_x, b_y = (browser_config["win_left_bound"], browser_config["win_upper_bound"])
        df = df[
            (df["Bottom"] - b_y >= 0)
            & (df["Top"] - b_y <= self.viewport_size["height"])
            & (df["Right"] - b_x >= 0)
            & (df["Left"] - b_x <= self.viewport_size["width"])
        ]
        viewport_area = self.viewport_size["width"] * self.viewport_size["height"]
        # Filter out bounding boxes that too large (more than 80% of the viewport)
        df = df[df["Area"] <= 0.8 * viewport_area]

        bboxes = {}
        index = 0
        for _, row in df.iterrows():
            if not row["Interactable"]:
                content = ""
                # Add image alt-text to the text representation.
                if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                    content += row["Alt"]
                # Add HTML textContent (if any) to the text representation.
                if pd.notna(row["TextContent"]):
                    content += (
                        row["TextContent"].strip().replace("\n", "").replace("\t", "")
                    )[
                        :200
                    ]  # Limit to 200 characters to avoid having too much text
                continue

            unique_id = str(index + 1)
            top, right, bottom, left, width, height = (
                row["Top"],
                row["Right"],
                row["Bottom"],
                row["Left"],
                row["Width"],
                row["Height"],
            )
            left, right, top, bottom = left - b_x, right - b_x, top - b_y, bottom - b_y

            ori_idx = row['ID']
            element = row['Element']
            alt = row['Alt']
            cls = row['Class']
            idx2 = row['Id']
            text_content = row['TextContent']
            interactable = row['Interactable']
            area = row['Area']

            bboxes[unique_id] = {
                'idx': ori_idx,
                'element': element,
                'top': top,
                'right': right,
                'bottom': bottom,
                'left': left,
                'width': width,
                'height': height,
                'alt': alt,
                'cls': cls,
                'idx2': idx2,
                'text_content': text_content,
                'interactable': interactable,
                'area': area,
            }

            index += 1
        return bboxes

    @beartype
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        super().reset(seed=seed, options=options)
        if self.reset_finished:
            self.context_manager.__exit__()

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                self.setup(config_file=config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            self.setup()
        self.reset_finished = True

        self.page.wait_for_timeout(int(self.sleep_after_execution * 1000))

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
        }

        return observation, info

    def save_trace(self, trace_path: str | Path) -> None:
        if self.save_trace_enabled:
            self.context.tracing.stop(path=trace_path)

    def close(self) -> None:
        if self.reset_finished:
            self.context_manager.__exit__()

    def step(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        success = False
        fail_error = ""
        # try:
        self.page = execute_action(
            action,
            self.page,
            self.context,
            self.observation_handler.action_processor,
            self.sleep_after_execution,
        )
        success = True
        # except Exception as e:
        #     fail_error = str(e)
        #     ERROR(fail_error)

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, self.page.content()),
            "fail_error": fail_error,
            "observation_metadata": observation_metadata,
        }
        msg = (
            observation,
            float(success),  # reward
            False,  # terminated
            False,  # truncated
            info,
        )
        return msg
