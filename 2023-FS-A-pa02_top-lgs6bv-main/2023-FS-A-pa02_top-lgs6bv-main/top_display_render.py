#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
import mujoco  # type: ignore
from mujoco import viewer
import mediapy as media
from top_evolve import xml_string_builder

with open(file="evolved_pop.json", mode="r") as fh:
    population = json.load(fp=fh)

top_str = xml_string_builder(genome=population[0]["genome"])
model = mujoco.MjModel.from_xml_string(xml=top_str)
data = mujoco.MjData(model)
viewer.launch(model)

video = False
if video:
    renderer = mujoco.Renderer(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    frames = []
    framerate = 60  # (Hz)
    quit_rot_vel = 50
    while quit_rot_vel < data.qvel[5]:
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="closeup")
        pixels = renderer.render()
        frames.append(pixels)
    media.write_video("top.mp4", frames, fps=framerate)
